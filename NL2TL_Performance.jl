using PyCall
using JSON3
using OrderedCollections

include("LTLEquivalence.jl")

# =================================================================================================
# Configuration
# =================================================================================================

const DEFAULT_DATASET_PATH = joinpath(@__DIR__, "dataset", "DatasetWithNaturalNL_plus_simplified.json")
const DEFAULT_RESULTS_PATH = joinpath(@__DIR__, "results", "NL2TL.json")
const DEFAULT_INPUT_FIELDS = [
    "natural_paraphrase",
    "paraphrase_gpt5.4-mini",
    "paraphrase_gemini-2.5-flash",
    "paraphrase_deepseek",
]

# Change these if your fine-tuned checkpoint or weights directory are elsewhere.
const DEFAULT_MODEL_CHECKPOINT = "t5-base"
const DEFAULT_OUTPUT_DIR = "/Users/abayat/Dio/Dionysos.jl_1/NL2TL/checkpoint-62500"
const DEFAULT_PREFIX = "Transform the following sentence into Signal Temporal logic: "
const DEFAULT_MAX_INPUT_LENGTH = 1024
const DEFAULT_MAX_TARGET_LENGTH = 256

# =================================================================================================
# Python / Transformers setup
# =================================================================================================

if !isdefined(@__MODULE__, :transformers)
    global transformers = pyimport("transformers")
end
if !isdefined(@__MODULE__, :torch)
    global torch = pyimport("torch")
end
if !isdefined(@__MODULE__, :T5Tokenizer)
    global T5Tokenizer = transformers.T5Tokenizer
end
if !isdefined(@__MODULE__, :T5ForConditionalGeneration)
    global T5ForConditionalGeneration = transformers.T5ForConditionalGeneration
end

function resolve_weight_source(output_dir::String)
    pytorch_path = joinpath(output_dir, "pytorch_model.bin")
    safetensors_path = joinpath(output_dir, "model.safetensors")

    if isfile(pytorch_path)
        return (:pytorch_bin, output_dir, pytorch_path)
    elseif isfile(safetensors_path)
        return (:pretrained_dir, output_dir, safetensors_path)
    end

    throw(ArgumentError(
        "Fine-tuned weights could not be found in `$(output_dir)`. Expected either `pytorch_model.bin` or `model.safetensors`."
    ))
end

function load_finetuned_t5_model(
    ;
    model_checkpoint::String = DEFAULT_MODEL_CHECKPOINT,
    output_dir::String = DEFAULT_OUTPUT_DIR,
)
    weight_mode, resolved_dir, resolved_path = resolve_weight_source(output_dir)

    tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)

    if weight_mode == :pretrained_dir
        model = T5ForConditionalGeneration.from_pretrained(resolved_dir)
    else
        model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)
        state_dict = torch.load(resolved_path, map_location="cpu")
        model.load_state_dict(state_dict)
    end

    device = torch.cuda.is_available() ? torch.device("cuda") : torch.device("cpu")
    model.to(device)
    model.eval()

    println("Loaded fine-tuned weights from: ", resolved_path)
    return tokenizer, model, device
end

# =================================================================================================
# Formula normalization
# =================================================================================================


function map_token(tok::AbstractString)::String
    cleaned = strip(String(tok), [' ', ',', '.', ';', ':'])
    w = lowercase(cleaned)
    if w == "globally" || w == "always"
        return "G"
    elseif w == "finally" || w == "eventually"
        return "F"
    elseif w == "next"
        return "X"
    elseif w == "until"
        return "U"
    elseif w == "release" || w == "releases"
        return "R"
    elseif w == "and"
        return "&"
    elseif w == "or"
        return "|"
    elseif w == "not" || w == "negation" || w == "negate"
        return "!"
    elseif w == "imply" || w == "implies" || w == "implie" || w == "implicate"
        return "->"
    elseif w == "equivalent" || w == "equivalence" || w == "equal" || w == "equals"
        return "<->"
    elseif w == "true" || w == "tt"
        return "true"
    elseif w == "false" || w == "ff"
        return "false"
    else
        return cleaned
    end
end

function canonicalize_simple_unary_forms(t::String)::String
    return strip(t)
end

function wrap_simple_unary_operands(t::String)::String
    s = t
    s = replace(s, r"\bG\s+(prop_\d+|true|false)\b" => s"G(\1)")
    s = replace(s, r"\bF\s+(prop_\d+|true|false)\b" => s"F(\1)")
    s = replace(s, r"\bX\s+(prop_\d+|true|false)\b" => s"X(\1)")
    s = replace(s, r"!\s+(prop_\d+|true|false)\b" => s"!(\1)")
    return s
end

function drop_extra_closing_parentheses(t::String)::String
    io = IOBuffer()
    balance = 0
    for c in t
        if c == '('
            balance += 1
            print(io, c)
        elseif c == ')'
            if balance > 0
                balance -= 1
                print(io, c)
            end
        else
            print(io, c)
        end
    end
    return String(take!(io))
end

function add_missing_closing_parentheses(t::String)::String
    opens = count(==( '(' ), t)
    closes = count(==( ')' ), t)
    if opens > closes
        return t * repeat(")", opens - closes)
    end
    return t
end

function repair_ltl_for_ltlfilt(t::AbstractString)::String
    repaired = strip(String(t))
    repaired = replace(repaired, r"\s+" => " ")
    repaired = wrap_simple_unary_operands(repaired)
    repaired = replace(repaired, r"\s+" => " ")
    return strip(repaired)
end

function llm_to_spot_ltl(s::AbstractString)::String
    t = strip(String(s))

    if startswith(t, "LTL:")
        t = strip(t[5:end])
    elseif startswith(lowercase(t), "ltl:")
        t = strip(t[5:end])
    end

    if startswith(t, "```")
        parts = split(t, "```")
        for part in parts
            cleaned = strip(replace(part, "ltl" => "", "LTL" => ""))
            if !isempty(cleaned)
                t = cleaned
                break
            end
        end
    end

    t = replace(t, "¬" => "!", "∧" => "&", "∨" => "|", "→" => "->", "↔" => "<->")

    # Map common textual operators before token splitting.
    t = replace(t, r"(?i)\b(globally|always)\b" => "G")
    t = replace(t, r"(?i)\b(finally|eventually)\b" => "F")
    t = replace(t, r"(?i)\bnext\b" => "X")
    t = replace(t, r"(?i)\buntil\b" => "U")
    t = replace(t, r"(?i)\b(release|releases)\b" => "R")
    t = replace(t, r"(?i)\band\b" => "&")
    t = replace(t, r"(?i)\bor\b" => "|")
    t = replace(t, r"(?i)\b(not|negation|negate)\b" => "!")
    t = replace(t, r"(?i)\b(imply|implies|implie|implicate)\b" => "->")
    t = replace(t, r"(?i)\b(equivalent|equivalence)\b" => "<->")
    t = replace(t, r"(?i)\btt\b" => "true")
    t = replace(t, r"(?i)\bff\b" => "false")

    t = replace(t, r"\s+" => " ")

    tokens = split(strip(t), ' ')
    mapped = [map_token(tok) for tok in tokens if !isempty(tok)]
    normalized = strip(join(mapped, " "))
    normalized = replace(normalized, r"\s+" => " ")
    normalized = canonicalize_simple_unary_forms(normalized)
    return strip(normalized)
end

function ltl_is_parseable(formula::AbstractString)::Bool
    ltlfilt_path = Sys.which("ltlfilt")
    isnothing(ltlfilt_path) && throw(ArgumentError("Spot's `ltlfilt` was not found in PATH."))

    cmd = `$(ltlfilt_path) -f $(String(formula)) -q`
    process = run(cmd; wait=false)
    wait(process)
    return process.exitcode == 0
end

# =================================================================================================
# Dataset / results helpers
# =================================================================================================

function paraphrase_source_model(field::AbstractString)
    if field == "natural_paraphrase"
        return "unknown"
    elseif field == "paraphrase_gpt5.4-mini"
        return "gpt-5.4-mini"
    elseif field == "paraphrase_gemini-2.5-flash"
        return "gemini-2.5-flash"
    elseif field == "paraphrase_deepseek"
        return "deepseek"
    else
        return "unknown"
    end
end

function load_dataset(dataset_path::String = DEFAULT_DATASET_PATH)
    if !isfile(dataset_path)
        throw(ArgumentError("Dataset file not found: $(dataset_path)"))
    end
    data = JSON3.read(read(dataset_path, String))
    return [OrderedDict{String,Any}(String(k) => v for (k, v) in pairs(record)) for record in data]
end

function ensure_parent_directory(path::String)
    parent = dirname(path)
    isdir(parent) || mkpath(parent)
end

function initial_results_object(
    ;
    dataset_path::String = DEFAULT_DATASET_PATH,
    model_checkpoint::String = DEFAULT_MODEL_CHECKPOINT,
    output_dir::String = DEFAULT_OUTPUT_DIR,
    input_fields::Vector{String} = DEFAULT_INPUT_FIELDS,
)
    return OrderedDict(
        "dataset_path" => dataset_path,
        "translation_model" => "finetuned-t5",
        "model_checkpoint" => model_checkpoint,
        "weights_dir" => output_dir,
        "input_fields" => input_fields,
        "results" => OrderedDict[],
    )
end

function load_results(
    results_path::String = DEFAULT_RESULTS_PATH;
    dataset_path::String = DEFAULT_DATASET_PATH,
    model_checkpoint::String = DEFAULT_MODEL_CHECKPOINT,
    output_dir::String = DEFAULT_OUTPUT_DIR,
    input_fields::Vector{String} = DEFAULT_INPUT_FIELDS,
)
    if !isfile(results_path)
        return initial_results_object(; dataset_path=dataset_path, model_checkpoint=model_checkpoint, output_dir=output_dir, input_fields=input_fields)
    end

    content = strip(read(results_path, String))
    isempty(content) && return initial_results_object(; dataset_path=dataset_path, model_checkpoint=model_checkpoint, output_dir=output_dir, input_fields=input_fields)

    parsed = JSON3.read(content)
    results_obj = OrderedDict{String,Any}(String(k) => v for (k, v) in pairs(parsed))

    if haskey(results_obj, "results")
        materialized_results = OrderedDict{String,Any}[]
        for entry in results_obj["results"]
            push!(materialized_results, OrderedDict{String,Any}(String(k) => v for (k, v) in pairs(entry)))
        end
        results_obj["results"] = materialized_results
    else
        results_obj["results"] = OrderedDict[]
    end

    return results_obj
end

function save_results(results_obj::OrderedDict{String,Any}, results_path::String = DEFAULT_RESULTS_PATH)
    ensure_parent_directory(results_path)
    open(results_path, "w") do io
        JSON3.pretty(io, results_obj)
        write(io, "\n")
    end
end

function result_key(record_id, field::AbstractString)
    return "$(record_id)||$(field)"
end

function latest_results_by_key(results_obj::OrderedDict{String,Any})
    latest = Dict{String,OrderedDict{String,Any}}()
    for entry_any in results_obj["results"]
        entry = OrderedDict{String,Any}(String(k) => v for (k, v) in pairs(entry_any))
        if haskey(entry, "record_id") && haskey(entry, "input_field")
            rid = entry["record_id"]
            field = String(entry["input_field"])
            latest[result_key(rid, field)] = entry
        end
    end
    return latest
end

function successful_result_keys(results_obj::OrderedDict{String,Any})
    keys_set = Set{String}()
    for (key, entry) in latest_results_by_key(results_obj)
        if haskey(entry, "status") && String(entry["status"]) == "ok"
            push!(keys_set, key)
        end
    end
    return keys_set
end

function append_result!(results_obj::OrderedDict{String,Any}, entry::OrderedDict{String,Any})
    entries = results_obj["results"]
    new_key = result_key(entry["record_id"], String(entry["input_field"]))

    filtered = OrderedDict{String,Any}[]
    for existing in entries
        existing_key = result_key(existing["record_id"], String(existing["input_field"]))
        if existing_key != new_key
            push!(filtered, existing)
        end
    end

    push!(filtered, entry)
    results_obj["results"] = filtered
    return results_obj
end

function make_result_entry(
    record::OrderedDict{String,Any},
    input_field::String,
    source_text::AbstractString,
    predicted_ltl::Union{Nothing,AbstractString},
    equivalent::Union{Nothing,Bool},
    status::String;
    model_checkpoint::String = DEFAULT_MODEL_CHECKPOINT,
    output_dir::String = DEFAULT_OUTPUT_DIR,
    error_message::Union{Nothing,String} = nothing,
)
    entry = OrderedDict{String,Any}(
        "record_id" => get(record, "id", nothing),
        "input_field" => input_field,
        "source_model" => paraphrase_source_model(input_field),
        "translation_model" => "finetuned-t5",
        "model_checkpoint" => model_checkpoint,
        "weights_dir" => output_dir,
        "source_text" => String(source_text),
        "original_ltl" => String(get(record, "LTL", "")),
        "predicted_ltl" => isnothing(predicted_ltl) ? nothing : String(predicted_ltl),
        "equivalent" => equivalent,
        "status" => status,
    )

    !isnothing(error_message) && (entry["error"] = error_message)
    return entry
end

# =================================================================================================
# Translation
# =================================================================================================

function translate_sentence_to_ltl(
    tokenizer,
    model,
    device,
    sentence::AbstractString;
    prefix::String = DEFAULT_PREFIX,
    max_input_length::Int = DEFAULT_MAX_INPUT_LENGTH,
    max_target_length::Int = DEFAULT_MAX_TARGET_LENGTH,
)::String
    local_inputs = [prefix * String(sentence)]
    tokenized = tokenizer(local_inputs; max_length=max_input_length, truncation=true, return_tensors="pt")
    tokenized = tokenized.to(device)

    output = model.generate(
        input_ids = tokenized["input_ids"],
        attention_mask = tokenized["attention_mask"],
        num_beams = 8,
        do_sample = true,
        max_length = max_target_length,
    )

    decoded = tokenizer.batch_decode(output, skip_special_tokens=true)[1]
    return llm_to_spot_ltl(decoded)
end

# =================================================================================================
# Evaluation
# =================================================================================================

function evaluate_nl2tl_on_benchmark(
    dataset_path::String = DEFAULT_DATASET_PATH;
    results_path::String = DEFAULT_RESULTS_PATH,
    input_fields::Vector{String} = DEFAULT_INPUT_FIELDS,
    model_checkpoint::String = DEFAULT_MODEL_CHECKPOINT,
    output_dir::String = DEFAULT_OUTPUT_DIR,
    prefix::String = DEFAULT_PREFIX,
    max_input_length::Int = DEFAULT_MAX_INPUT_LENGTH,
    max_target_length::Int = DEFAULT_MAX_TARGET_LENGTH,
)
    tokenizer, model, device = load_finetuned_t5_model(; model_checkpoint=model_checkpoint, output_dir=output_dir)

    dataset = load_dataset(dataset_path)
    results_obj = load_results(results_path; dataset_path=dataset_path, model_checkpoint=model_checkpoint, output_dir=output_dir, input_fields=input_fields)
    seen = successful_result_keys(results_obj)

    processed_count = 0
    skipped_count = 0

    for record in dataset
        haskey(record, "LTL") || continue

        for field in input_fields
            haskey(record, field) || continue
            source_text = strip(String(record[field]))
            isempty(source_text) && continue

            key = result_key(get(record, "id", nothing), field)
            if key in seen
                skipped_count += 1
                println("Skipping record ID ", get(record, "id", "?"), " field `", field, "` because it already has a successful result.")
                continue
            end

            println("Processing record ID ", get(record, "id", "?"), " field `", field, "`...")

            predicted_ltl = nothing
            repaired_ltl = nothing

            try
                predicted_ltl = translate_sentence_to_ltl(
                    tokenizer,
                    model,
                    device,
                    source_text;
                    prefix=prefix,
                    max_input_length=max_input_length,
                    max_target_length=max_target_length,
                )
                println("Predicted normalized LTL: ", predicted_ltl)
                original_ltl = String(record["LTL"])

                final_ltl = predicted_ltl
                if !ltl_is_parseable(final_ltl)
                    repaired_ltl = repair_ltl_for_ltlfilt(predicted_ltl)
                    println("Predicted repaired LTL: ", repaired_ltl)
                    if ltl_is_parseable(repaired_ltl)
                        final_ltl = repaired_ltl
                    else
                        error("Both raw and repaired LTL are unparsable. Raw=`$(predicted_ltl)` Repaired=`$(repaired_ltl)`")
                    end
                end

                equivalent = are_equivalent(final_ltl, original_ltl)

                entry = make_result_entry(
                    record,
                    field,
                    source_text,
                    final_ltl,
                    equivalent,
                    "ok";
                    model_checkpoint=model_checkpoint,
                    output_dir=output_dir,
                )
                append_result!(results_obj, entry)
                save_results(results_obj, results_path)
                push!(seen, key)
                processed_count += 1
                println("Saved result for record ID ", get(record, "id", "?"), " field `", field, "`. Equivalent: ", equivalent)
            catch err
                entry = make_result_entry(
                    record,
                    field,
                    source_text,
                    predicted_ltl,
                    nothing,
                    "error";
                    model_checkpoint=model_checkpoint,
                    output_dir=output_dir,
                    error_message=sprint(showerror, err),
                )
                append_result!(results_obj, entry)
                save_results(results_obj, results_path)
                processed_count += 1
                println("Saved error result for record ID ", get(record, "id", "?"), " field `", field, "`.")
            end
        end
    end

    println("Processed $(processed_count) evaluations.")
    println("Skipped $(skipped_count) evaluations already present in results.")
    println("Results path: $(results_path)")
end

function summarize_nl2tl_results(results_path::String = DEFAULT_RESULTS_PATH)
    results_obj = load_results(results_path)
    entries = results_obj["results"]

    total = length(entries)
    ok_count = count(entry -> haskey(entry, "status") && String(entry["status"]) == "ok", entries)
    equivalent_count = count(entry -> haskey(entry, "equivalent") && entry["equivalent"] === true, entries)
    success_count = count(entry -> (haskey(entry, "status") && String(entry["status"]) == "ok") && (haskey(entry, "equivalent") && entry["equivalent"] === true), entries)
    error_count = total - ok_count

    println("Results path: ", results_path)
    println("Total evaluated: ", total)
    println("Successful translations: ", ok_count)
    println("Semantically equivalent outputs: ", equivalent_count)
    println("Success count (ok AND equivalent): ", success_count)
    println("Error count: ", error_count)
    println("Success rate: ", round(total == 0 ? 0.0 : success_count / total; digits=4))
end

function main()
    println("Loaded NL2TL_Performance.jl")
    println("Run `evaluate_nl2tl_on_benchmark()` to evaluate the fine-tuned T5 model on the dataset.")
    println("Run `summarize_nl2tl_results()` to print a results summary.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end