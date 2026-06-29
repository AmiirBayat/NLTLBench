

using JSON3
using OrderedCollections
using Plots

const DEFAULT_DATASET_PATH = joinpath(@__DIR__, "dataset", "DatasetWithNaturalNL_plus_simplified.json")
const DEFAULT_RESULTS_DIR = joinpath(@__DIR__, "results")

const DEFAULT_OUTPUT_DIR = joinpath(@__DIR__, "results")

const USE_PAPER_MODEL_SUBSET = false

const PAPER_MODEL_SUBSET = Set([
    "mistral-medium-latest",
    "claude-opus-4-7",
    "gpt-5.4",
    "gpt-5.5",
])

const USE_ZERO_VS_FEWSHOT_SUBPLOTS = true
const ZERO_VS_FEWSHOT_LAYOUT_MODE = "zero_shot_only_four_panel"  # allowed: "six_panel", "zero_shot_only_three_panel", "zero_shot_only_four_panel"
const ZERO_VS_FEWSHOT_MODELS = [
    "gpt-5.4",
    "gpt-5.5",
    "claude-opus-4-7",
    "deepseek-v4-flash",
    "mistral-medium-latest",
]
const ZERO_VS_FEWSHOT_FEWSHOT_FILES = Set([
    "Claude_fewshot",
    "GPT55_fewshot",
    "GPT54_fewshot",
    "DeepSeek_fewshot",
    "Mistral_fewshot",
])
const MAX_PLOTTED_FORMULA_SIZE = 34
const DEFAULT_PLOT_DPI = 150

# =================================================================================================
# I/O helpers
# =================================================================================================

function ensure_directory(path::String)
    isdir(path) || mkpath(path)
end

function load_json_array(path::String)
    if !isfile(path)
        throw(ArgumentError("JSON file not found: $(path)"))
    end
    parsed = JSON3.read(read(path, String))
    return collect(parsed)
end

function load_json_object(path::String)
    if !isfile(path)
        throw(ArgumentError("JSON file not found: $(path)"))
    end
    content = strip(read(path, String))
    isempty(content) && throw(ArgumentError("JSON file is empty: $(path)"))
    parsed = JSON3.read(content)

    if parsed isa AbstractVector
        entries = OrderedDict{String,Any}[]
        for entry in parsed
            push!(entries, OrderedDict{String,Any}(String(k) => v for (k, v) in pairs(entry)))
        end
        return OrderedDict{String,Any}("results" => entries)
    end

    return OrderedDict{String,Any}(String(k) => v for (k, v) in pairs(parsed))
end

function available_result_files(results_dir::String = DEFAULT_RESULTS_DIR)
    if !isdir(results_dir)
        throw(ArgumentError("Results directory not found: $(results_dir)"))
    end
    files = sort(filter(f -> endswith(lowercase(f), ".json") && isfile(f) && filesize(f) > 0, readdir(results_dir; join=true)))
    isempty(files) && throw(ArgumentError("No non-empty JSON result files found in $(results_dir)"))
    return files
end

# =================================================================================================
# Dataset indexing
# =================================================================================================

function load_dataset_index(dataset_path::String = DEFAULT_DATASET_PATH)
    dataset = load_json_array(dataset_path)
    index = Dict{Int,OrderedDict{String,Any}}()

    for record in dataset
        dict_record = OrderedDict{String,Any}(String(k) => v for (k, v) in pairs(record))
        haskey(dict_record, "id") || continue
        index[Int(dict_record["id"])] = dict_record
    end

    return index
end


function tokenize_formula_size_fallback(formula::AbstractString)
    s = strip(String(formula))
    isempty(s) && return String[]

    tokens = String[]
    i = firstindex(s)

    while i <= lastindex(s)
        c = s[i]

        if isspace(c)
            i = nextind(s, i)
        elseif c in ('(', ')', '&', '|', '!')
            push!(tokens, string(c))
            i = nextind(s, i)
        elseif c == '<'
            j = nextind(s, i)
            k = j <= lastindex(s) ? nextind(s, j) : j
            if j <= lastindex(s) && k <= lastindex(s) && s[j] == '-' && s[k] == '>'
                push!(tokens, "<->")
                i = nextind(s, k)
            else
                push!(tokens, string(c))
                i = nextind(s, i)
            end
        elseif c == '-'
            j = nextind(s, i)
            if j <= lastindex(s) && s[j] == '>'
                push!(tokens, "->")
                i = nextind(s, j)
            else
                push!(tokens, string(c))
                i = nextind(s, i)
            end
        else
            j = i
            while j <= lastindex(s)
                cj = s[j]
                if isspace(cj) || cj in ('(', ')', '&', '|', '!', '<', '-')
                    break
                end
                j = nextind(s, j)
            end
            push!(tokens, s[i:prevind(s, j)])
            i = j
        end
    end

    return tokens
end

function get_record_formula_size(record::OrderedDict{String,Any})
    if haskey(record, "formula_size")
        return Int(record["formula_size"])
    elseif haskey(record, "ast_size")
        return Int(record["ast_size"])
    elseif haskey(record, "LTL")
        tokens = tokenize_formula_size_fallback(String(record["LTL"]))
        return count(tok -> tok != "(" && tok != ")", tokens)
    else
        throw(ArgumentError("Dataset record ID $(get(record, "id", "?")) does not contain `formula_size`, `ast_size`, or `LTL`."))
    end
end

function record_has_nnf_flag(record::OrderedDict{String,Any})
    return haskey(record, "is_nnf") && record["is_nnf"] === true
end

function build_benchmark_min_equiv_formula_size_map(dataset_path::String = DEFAULT_DATASET_PATH)
    dataset_index = load_dataset_index(dataset_path)
    min_size_map = Dict{Int,Int}()

    for (record_id, record) in dataset_index
        haskey(record, "LTL") || continue
        current_size = get_record_formula_size(record)

        if haskey(record, "source_record_id")
            source_id = try
                Int(record["source_record_id"])
            catch
                nothing
            end

            if !isnothing(source_id) && haskey(dataset_index, source_id) && haskey(dataset_index[source_id], "LTL")
                source_size = get_record_formula_size(dataset_index[source_id])
                min_size_map[record_id] = min(current_size, source_size)
            else
                min_size_map[record_id] = current_size
            end
        else
            min_size_map[record_id] = current_size
        end
    end

    return min_size_map
end

function canonicalized_size_bucket_statistics(
    entries::Vector{OrderedDict{String,Any}},
    dataset_index::Dict{Int,OrderedDict{String,Any}},
    min_size_map::Dict{Int,Int};
    ignore_errors::Bool = false,
)
    bucket_counts = OrderedDict{Int,OrderedDict{String,Int}}()

    for entry in entries
        haskey(entry, "record_id") || continue
        record_id = Int(entry["record_id"])
        haskey(dataset_index, record_id) || continue
        haskey(min_size_map, record_id) || continue
        size_value = min_size_map[record_id]

        if !haskey(bucket_counts, size_value)
            bucket_counts[size_value] = OrderedDict(
                "total" => 0,
                "success" => 0,
            )
        end

        status_ok = haskey(entry, "status") && String(entry["status"]) == "ok"
        equivalent_true = haskey(entry, "equivalent") && entry["equivalent"] === true

        if ignore_errors
            bucket_counts[size_value]["total"] += status_ok ? 1 : 0
        else
            bucket_counts[size_value]["total"] += 1
        end

        if status_ok && equivalent_true
            bucket_counts[size_value]["success"] += 1
        end
    end

    return bucket_counts
end

function canonicalized_size_bucket_success_rates(
    entries::Vector{OrderedDict{String,Any}},
    dataset_index::Dict{Int,OrderedDict{String,Any}},
    min_size_map::Dict{Int,Int};
    ignore_errors::Bool = false,
)
    bucket_stats = canonicalized_size_bucket_statistics(entries, dataset_index, min_size_map; ignore_errors=ignore_errors)
    sizes = sort(collect(keys(bucket_stats)))
    rates = Float64[]
    totals = Int[]

    for size_value in sizes
        total = bucket_stats[size_value]["total"]
        success = bucket_stats[size_value]["success"]
        push!(totals, total)
        push!(rates, total == 0 ? 0.0 : success / total)
    end

    return sizes, rates, totals
end

function temporal_depth_from_tokens(formula::AbstractString)
    tokens = tokenize_formula_size_fallback(formula)
    isempty(tokens) && return 0

    function parse_unary_depth(i::Int)
        i > length(tokens) && return 0, i
        tok = tokens[i]

        if tok == "!"
            return parse_unary_depth(i + 1)
        elseif tok in ("X", "F", "G")
            child_depth, next_i = parse_unary_depth(i + 1)
            return 1 + child_depth, next_i
        elseif tok == "("
            inner_depth, next_i = parse_binary_depth(i + 1)
            if next_i <= length(tokens) && tokens[next_i] == ")"
                return inner_depth, next_i + 1
            end
            return inner_depth, next_i
        else
            return 0, i + 1
        end
    end

    function parse_binary_depth(i::Int)
        left_depth, i = parse_unary_depth(i)
        best = left_depth

        while i <= length(tokens)
            tok = tokens[i]
            if tok == ")"
                break
            elseif tok in ("&", "|", "->", "<->")
                right_depth, next_i = parse_unary_depth(i + 1)
                best = max(best, right_depth)
                i = next_i
            elseif tok == "U"
                right_depth, next_i = parse_unary_depth(i + 1)
                best = max(best, left_depth + 1, right_depth + 1)
                left_depth = max(left_depth, right_depth) + 1
                i = next_i
            else
                break
            end
        end

        return max(best, left_depth), i
    end

    depth, _ = parse_binary_depth(1)
    return depth
end


function get_record_temporal_depth(record::OrderedDict{String,Any})
    if haskey(record, "temporal_depth")
        return Int(record["temporal_depth"])
    elseif haskey(record, "LTL")
        return temporal_depth_from_tokens(String(record["LTL"]))
    else
        throw(ArgumentError("Dataset record ID $(get(record, "id", "?")) does not contain `temporal_depth` or `LTL`."))
    end
end


function get_record_automaton_size(record::OrderedDict{String,Any})
    if haskey(record, "automaton_size")
        return Int(record["automaton_size"])
    elseif haskey(record, "automaton_num_states") && haskey(record, "automaton_num_transitions")
        return Int(record["automaton_num_states"]) + Int(record["automaton_num_transitions"])
    else
        throw(ArgumentError("Dataset record ID $(get(record, "id", "?")) does not contain `automaton_size` or both `automaton_num_states` and `automaton_num_transitions`."))
    end
end

# =================================================================================================
# NL phrase length helpers and statistics
# =================================================================================================

function get_entry_input_field(entry::OrderedDict{String,Any})
    for key in ("input_field", "field", "source_field")
        if haskey(entry, key)
            return String(entry[key])
        end
    end
    return nothing
end

function nl_phrase_word_count(text::AbstractString)
    s = strip(String(text))
    isempty(s) && return 0
    return length(split(s))
end

function get_entry_nl_phrase_length(entry::OrderedDict{String,Any}, dataset_index::Dict{Int,OrderedDict{String,Any}})
    haskey(entry, "record_id") || throw(ArgumentError("Entry does not contain `record_id`."))
    record_id = Int(entry["record_id"])
    haskey(dataset_index, record_id) || throw(ArgumentError("Dataset index does not contain record ID $(record_id)."))

    input_field = get_entry_input_field(entry)
    isnothing(input_field) && throw(ArgumentError("Entry for record ID $(record_id) does not contain `input_field`."))
    haskey(dataset_index[record_id], input_field) || throw(ArgumentError("Dataset record ID $(record_id) does not contain field `$(input_field)`."))

    text_value = String(dataset_index[record_id][input_field])
    return nl_phrase_word_count(text_value)
end

# =================================================================================================
# Result parsing
# =================================================================================================

function load_result_entries(results_path::String)
    results_obj = load_json_object(results_path)
    haskey(results_obj, "results") || throw(ArgumentError("Result file $(results_path) does not contain a `results` field."))

    entries = OrderedDict{String,Any}[]
    for entry in results_obj["results"]
        push!(entries, OrderedDict{String,Any}(String(k) => v for (k, v) in pairs(entry)))
    end
    return results_obj, entries
end

function result_model_name(results_obj::OrderedDict{String,Any}, results_path::String)
    model_name = if haskey(results_obj, "translation_model")
        String(results_obj["translation_model"])
    else
        splitext(basename(results_path))[1]
    end

    basename_name = splitext(basename(results_path))[1]
    basename_lower = lowercase(basename_name)
    if basename_lower == "t5" || startswith(basename_lower, "t5_") || endswith(basename_lower, "_t5") || occursin("_t5_", basename_lower) || occursin("finetuned-t5", basename_lower)
        return "finetuned-t5"
    end
    if basename_name == "GPT54_fewshot"
        return "gpt-5.4"
    elseif basename_name == "GPT55_fewshot"
        return "gpt-5.5"
    elseif basename_name == "Claude_fewshot"
        return "claude-opus-4-7"
    elseif basename_name == "DeepSeek_fewshot"
        return "deepseek-v4-flash"
    elseif basename_name == "Mistral_fewshot"
        return "mistral-medium-latest"
    end

    lowercase_model = lowercase(model_name)
    if occursin("deepseek", lowercase_model)
        return "deepseek-v4-flash"
    elseif occursin("claude", lowercase_model)
        return "claude-opus-4-7"
    elseif occursin("mistral", lowercase_model)
        return "mistral-medium-latest"
    elseif lowercase_model == "gpt54" || lowercase_model == "gpt-5.4" || lowercase_model == "gpt5.4" || occursin("gpt54_fewshot", lowercase_model) || occursin("gpt5.4", lowercase_model)
        return "gpt-5.4"
    elseif lowercase_model == "gpt55" || lowercase_model == "gpt-5.5" || lowercase_model == "gpt5.5" || occursin("gpt55_fewshot", lowercase_model) || occursin("gpt5.5", lowercase_model)
        return "gpt-5.5"
    elseif lowercase_model == "t5" || startswith(lowercase_model, "t5_") || endswith(lowercase_model, "_t5") || occursin("_t5_", lowercase_model) || occursin("finetuned-t5", lowercase_model)
        return "finetuned-t5"
    end

    return model_name
end


function result_provider_name(results_obj::OrderedDict{String,Any})
    if haskey(results_obj, "translation_provider")
        return String(results_obj["translation_provider"])
    end
    return "unknown"
end

function result_prompt_setting(results_obj::OrderedDict{String,Any}, results_path::String)
    if haskey(results_obj, "prompt_setting")
        return lowercase(String(results_obj["prompt_setting"]))
    end

    basename_lower = lowercase(basename(results_path))
    if occursin("fewshot", basename_lower) || occursin("few_shot", basename_lower)
        return "fewshot"
    end

    return "zeroshot"
end

# =================================================================================================
# Helper functions for equivalence annotation
# =================================================================================================

function get_entry_record_id(entry::OrderedDict{String,Any})
    for key in ("record_id", "id", "dataset_id")
        if haskey(entry, key)
            return Int(entry[key])
        end
    end
    return nothing
end

function get_entry_prediction(entry::OrderedDict{String,Any})
    for key in ("predicted_ltl", "predicted_LTL", "prediction", "predicted", "output_ltl", "output", "ltl", "LTL")
        if haskey(entry, key)
            return strip(String(entry[key]))
        end
    end
    return nothing
end

function spot_semantically_equivalent(predicted::AbstractString, ground_truth::AbstractString)
    pred = strip(String(predicted))
    gt = strip(String(ground_truth))
    isempty(pred) && return false
    isempty(gt) && return false

    try
        output = read(pipeline(IOBuffer(pred), `ltlfilt --equivalent-to=$gt`), String)
        return !isempty(strip(output))
    catch err
        return false
    end
end

function add_equivalence_to_result_file(
    results_path::String;
    dataset_path::String = DEFAULT_DATASET_PATH,
    output_path::String = replace(results_path, ".json" => "_with_equivalence.json"),
)
    dataset_index = load_dataset_index(dataset_path)
    results_obj, entries = load_result_entries(results_path)

    updated_entries = OrderedDict{String,Any}[]
    total = 0
    checked = 0
    equivalent_count = 0
    missing_ground_truth = 0
    missing_prediction = 0

    for entry in entries
        total += 1
        updated = OrderedDict{String,Any}(entry)
        record_id = get_entry_record_id(entry)
        prediction = get_entry_prediction(entry)

        if isnothing(record_id) || !haskey(dataset_index, record_id)
            updated["status"] = get(updated, "status", "error")
            updated["equivalent"] = false
            updated["equivalence_error"] = "missing_ground_truth"
            missing_ground_truth += 1
        elseif isnothing(prediction) || isempty(prediction)
            updated["status"] = get(updated, "status", "error")
            updated["equivalent"] = false
            updated["equivalence_error"] = "missing_prediction"
            missing_prediction += 1
        else
            ground_truth = String(dataset_index[record_id]["LTL"])
            is_equiv = spot_semantically_equivalent(prediction, ground_truth)
            updated["ground_truth_ltl"] = ground_truth
            updated["equivalent"] = is_equiv
            updated["status"] = get(updated, "status", "ok")
            checked += 1
            equivalent_count += is_equiv ? 1 : 0
        end

        push!(updated_entries, updated)
    end

    output_obj = OrderedDict{String,Any}()
    for (k, v) in results_obj
        k == "results" && continue
        output_obj[k] = v
    end
    output_obj["results"] = updated_entries

    open(output_path, "w") do io
        JSON3.pretty(io, output_obj)
    end

    println("Equivalence study written to: ", output_path)
    println("Total entries: ", total)
    println("Checked entries: ", checked)
    println("Equivalent entries: ", equivalent_count)
    println("Missing ground truth: ", missing_ground_truth)
    println("Missing prediction: ", missing_prediction)
    println("Success rate among checked entries: ", checked == 0 ? 0.0 : round(equivalent_count / checked; digits=4))

    return output_path
end

function use_zero_vs_fewshot_result_file(stats::OrderedDict{String,Any})
    prompt_setting = haskey(stats, "prompt_setting") ? String(stats["prompt_setting"]) : "zeroshot"
    if prompt_setting == "fewshot"
        result_base = splitext(basename(String(stats["results_path"])))[1]
        return result_base in ZERO_VS_FEWSHOT_FEWSHOT_FILES
    end
    return String(stats["model"]) in ZERO_VS_FEWSHOT_MODELS
end

function use_zero_shot_only_result_file(stats::OrderedDict{String,Any})
    prompt_setting = haskey(stats, "prompt_setting") ? String(stats["prompt_setting"]) : "zeroshot"
    return prompt_setting == "zeroshot" && (String(stats["model"]) in ZERO_VS_FEWSHOT_MODELS)
end

function source_label_from_field(field::String)
    if field == "natural_paraphrase"
        return "natural"
    elseif field == "paraphrase_gpt5.4-mini"
        return "gpt-5.4-mini"
    elseif field == "paraphrase_gemini-2.5-flash"
        return "gemini-2.5-flash"
    elseif field == "paraphrase_deepseek"
        return "deepseek-v4-flash"
    elseif field == "paraphrase_claude"
        return "claude"
    else
        return field
    end
end

function ignore_errors_in_success_rate(results_obj::OrderedDict{String,Any}, results_path::String)
    model_name = result_model_name(results_obj, results_path)
    basename_lower = lowercase(splitext(basename(results_path))[1])
    return model_name == "finetuned-t5" || basename_lower == "nl2tl" || occursin("t5", basename_lower)
end

function success_rate(entries::Vector{OrderedDict{String,Any}}; ignore_errors::Bool = false)
    success_count = 0
    denominator = 0

    for entry in entries
        status_ok = haskey(entry, "status") && String(entry["status"]) == "ok"
        equivalent_true = haskey(entry, "equivalent") && entry["equivalent"] === true
        success_count += (status_ok && equivalent_true) ? 1 : 0
        if ignore_errors
            denominator += status_ok ? 1 : 0
        else
            denominator += 1
        end
    end

    denominator == 0 && return 0.0
    return success_count / denominator
end

function fieldwise_statistics(entries::Vector{OrderedDict{String,Any}}; ignore_errors::Bool = false)
    stats = OrderedDict{String,OrderedDict{String,Int}}()

    for entry in entries
        field = haskey(entry, "input_field") ? String(entry["input_field"]) : "unknown"
        if !haskey(stats, field)
            stats[field] = OrderedDict(
                "total" => 0,
                "ok" => 0,
                "equivalent" => 0,
                "success" => 0,
                "error" => 0,
                "effective_total" => 0,
            )
        end

        stats[field]["total"] += 1

        status_ok = haskey(entry, "status") && String(entry["status"]) == "ok"
        equivalent_true = haskey(entry, "equivalent") && entry["equivalent"] === true

        if status_ok
            stats[field]["ok"] += 1
        else
            stats[field]["error"] += 1
        end

        if equivalent_true
            stats[field]["equivalent"] += 1
        end

        if status_ok && equivalent_true
            stats[field]["success"] += 1
        end

        if ignore_errors
            stats[field]["effective_total"] += status_ok ? 1 : 0
        else
            stats[field]["effective_total"] += 1
        end
    end

    return stats
end


function size_bucket_statistics(entries::Vector{OrderedDict{String,Any}}, dataset_index::Dict{Int,OrderedDict{String,Any}}; ignore_errors::Bool = false)
    bucket_counts = OrderedDict{Int,OrderedDict{String,Int}}()

    for entry in entries
        haskey(entry, "record_id") || continue
        record_id = Int(entry["record_id"])
        haskey(dataset_index, record_id) || continue
        size_value = get_record_formula_size(dataset_index[record_id])

        if !haskey(bucket_counts, size_value)
            bucket_counts[size_value] = OrderedDict(
                "total" => 0,
                "success" => 0,
            )
        end

        status_ok = haskey(entry, "status") && String(entry["status"]) == "ok"
        equivalent_true = haskey(entry, "equivalent") && entry["equivalent"] === true

        if ignore_errors
            bucket_counts[size_value]["total"] += status_ok ? 1 : 0
        else
            bucket_counts[size_value]["total"] += 1
        end

        if status_ok && equivalent_true
            bucket_counts[size_value]["success"] += 1
        end
    end

    return bucket_counts
end

function automaton_size_bucket_statistics(entries::Vector{OrderedDict{String,Any}}, dataset_index::Dict{Int,OrderedDict{String,Any}}; ignore_errors::Bool = false)
    bucket_counts = OrderedDict{Int,OrderedDict{String,Int}}()

    for entry in entries
        haskey(entry, "record_id") || continue
        record_id = Int(entry["record_id"])
        haskey(dataset_index, record_id) || continue
        automaton_size_value = get_record_automaton_size(dataset_index[record_id])

        if !haskey(bucket_counts, automaton_size_value)
            bucket_counts[automaton_size_value] = OrderedDict(
                "total" => 0,
                "success" => 0,
            )
        end

        status_ok = haskey(entry, "status") && String(entry["status"]) == "ok"
        equivalent_true = haskey(entry, "equivalent") && entry["equivalent"] === true

        if ignore_errors
            bucket_counts[automaton_size_value]["total"] += status_ok ? 1 : 0
        else
            bucket_counts[automaton_size_value]["total"] += 1
        end

        if status_ok && equivalent_true
            bucket_counts[automaton_size_value]["success"] += 1
        end
    end

    return bucket_counts
end


function temporal_depth_bucket_statistics(entries::Vector{OrderedDict{String,Any}}, dataset_index::Dict{Int,OrderedDict{String,Any}}; ignore_errors::Bool = false)
    bucket_counts = OrderedDict{Int,OrderedDict{String,Int}}()

    for entry in entries
        haskey(entry, "record_id") || continue
        record_id = Int(entry["record_id"])
        haskey(dataset_index, record_id) || continue
        depth_value = get_record_temporal_depth(dataset_index[record_id])

        if !haskey(bucket_counts, depth_value)
            bucket_counts[depth_value] = OrderedDict(
                "total" => 0,
                "success" => 0,
            )
        end

        status_ok = haskey(entry, "status") && String(entry["status"]) == "ok"
        equivalent_true = haskey(entry, "equivalent") && entry["equivalent"] === true

        if ignore_errors
            bucket_counts[depth_value]["total"] += status_ok ? 1 : 0
        else
            bucket_counts[depth_value]["total"] += 1
        end

        if status_ok && equivalent_true
            bucket_counts[depth_value]["success"] += 1
        end
    end

    return bucket_counts
end

function nl_phrase_length_bucket_statistics(entries::Vector{OrderedDict{String,Any}}, dataset_index::Dict{Int,OrderedDict{String,Any}}; ignore_errors::Bool = false)
    bucket_counts = OrderedDict{Int,OrderedDict{String,Int}}()

    for entry in entries
        haskey(entry, "record_id") || continue
        record_id = Int(entry["record_id"])
        haskey(dataset_index, record_id) || continue

        input_field = get_entry_input_field(entry)
        isnothing(input_field) && continue
        haskey(dataset_index[record_id], input_field) || continue

        nl_length_value = get_entry_nl_phrase_length(entry, dataset_index)

        if !haskey(bucket_counts, nl_length_value)
            bucket_counts[nl_length_value] = OrderedDict(
                "total" => 0,
                "success" => 0,
            )
        end

        status_ok = haskey(entry, "status") && String(entry["status"]) == "ok"
        equivalent_true = haskey(entry, "equivalent") && entry["equivalent"] === true

        if ignore_errors
            bucket_counts[nl_length_value]["total"] += status_ok ? 1 : 0
        else
            bucket_counts[nl_length_value]["total"] += 1
        end

        if status_ok && equivalent_true
            bucket_counts[nl_length_value]["success"] += 1
        end
    end

    return bucket_counts
end


function temporal_depth_bucket_success_rates(entries::Vector{OrderedDict{String,Any}}, dataset_index::Dict{Int,OrderedDict{String,Any}}; ignore_errors::Bool = false)
    bucket_stats = temporal_depth_bucket_statistics(entries, dataset_index; ignore_errors=ignore_errors)
    depths = sort(collect(keys(bucket_stats)))
    rates = Float64[]
    totals = Int[]

    for depth_value in depths
        total = bucket_stats[depth_value]["total"]
        success = bucket_stats[depth_value]["success"]
        push!(totals, total)
        push!(rates, total == 0 ? 0.0 : success / total)
    end

    return depths, rates, totals
end

function nl_phrase_length_bucket_success_rates(entries::Vector{OrderedDict{String,Any}}, dataset_index::Dict{Int,OrderedDict{String,Any}}; ignore_errors::Bool = false)
    bucket_stats = nl_phrase_length_bucket_statistics(entries, dataset_index; ignore_errors=ignore_errors)
    lengths = sort(collect(keys(bucket_stats)))
    rates = Float64[]
    totals = Int[]

    for length_value in lengths
        total = bucket_stats[length_value]["total"]
        success = bucket_stats[length_value]["success"]
        push!(totals, total)
        push!(rates, total == 0 ? 0.0 : success / total)
    end

    return lengths, rates, totals
end

function nl_phrase_length_distribution(entries::Vector{OrderedDict{String,Any}}, dataset_index::Dict{Int,OrderedDict{String,Any}})
    counts = OrderedDict{Int,Int}()

    for entry in entries
        haskey(entry, "record_id") || continue
        record_id = Int(entry["record_id"])
        haskey(dataset_index, record_id) || continue

        input_field = get_entry_input_field(entry)
        isnothing(input_field) && continue
        haskey(dataset_index[record_id], input_field) || continue

        nl_length_value = get_entry_nl_phrase_length(entry, dataset_index)
        counts[nl_length_value] = get(counts, nl_length_value, 0) + 1
    end

    lengths = sort(collect(keys(counts)))
    freqs = [counts[length_value] for length_value in lengths]
    return lengths, freqs
end


function size_bucket_success_rates(entries::Vector{OrderedDict{String,Any}}, dataset_index::Dict{Int,OrderedDict{String,Any}}; ignore_errors::Bool = false)
    bucket_stats = size_bucket_statistics(entries, dataset_index; ignore_errors=ignore_errors)
    sizes = sort(collect(keys(bucket_stats)))
    rates = Float64[]
    totals = Int[]

    for size_value in sizes
        total = bucket_stats[size_value]["total"]
        success = bucket_stats[size_value]["success"]
        push!(totals, total)
        push!(rates, total == 0 ? 0.0 : success / total)
    end

    return sizes, rates, totals
end

function automaton_size_bucket_success_rates(entries::Vector{OrderedDict{String,Any}}, dataset_index::Dict{Int,OrderedDict{String,Any}}; ignore_errors::Bool = false)
    bucket_stats = automaton_size_bucket_statistics(entries, dataset_index; ignore_errors=ignore_errors)
    sizes = sort(collect(keys(bucket_stats)))
    rates = Float64[]
    totals = Int[]

    for size_value in sizes
        total = bucket_stats[size_value]["total"]
        success = bucket_stats[size_value]["success"]
        push!(totals, total)
        push!(rates, total == 0 ? 0.0 : success / total)
    end

    return sizes, rates, totals
end

function binomial_standard_errors(rates::Vector{Float64}, totals::Vector{Int})
    errors = Float64[]
    for i in eachindex(rates)
        n = totals[i]
        p = rates[i]
        if n <= 0
            push!(errors, 0.0)
        else
            push!(errors, sqrt(p * (1.0 - p) / n))
        end
    end
    return errors
end

function claude_band_bounds(ys::Vector{Float64}, totals::Vector{Int})
    errs = binomial_standard_errors(ys, totals)
    lower = Float64[]
    upper = Float64[]
    for i in eachindex(ys)
        push!(lower, max(0.0, ys[i] - errs[i]))
        push!(upper, min(1.0, ys[i] + errs[i]))
    end
    return lower, upper
end

function overall_success_statistics(entries::Vector{OrderedDict{String,Any}}; ignore_errors::Bool = false)
    total = length(entries)
    success_count = count(entry -> (haskey(entry, "status") && String(entry["status"]) == "ok") && (haskey(entry, "equivalent") && entry["equivalent"] === true), entries)
    ok_count = count(entry -> haskey(entry, "status") && String(entry["status"]) == "ok", entries)
    equivalent_count = count(entry -> haskey(entry, "equivalent") && entry["equivalent"] === true, entries)
    error_count = total - ok_count
    effective_total = ignore_errors ? ok_count : total
    success_rate_value = effective_total == 0 ? 0.0 : success_count / effective_total

    return OrderedDict(
        "total" => total,
        "effective_total" => effective_total,
        "ok" => ok_count,
        "equivalent" => equivalent_count,
        "success" => success_count,
        "error" => error_count,
        "success_rate" => success_rate_value,
    )
end

function aggregate_result_file_statistics(results_path::String)
    results_obj, entries = load_result_entries(results_path)
    ignore_errors = ignore_errors_in_success_rate(results_obj, results_path)
    return OrderedDict(
        "results_path" => results_path,
        "provider" => result_provider_name(results_obj),
        "model" => result_model_name(results_obj, results_path),
        "prompt_setting" => result_prompt_setting(results_obj, results_path),
        "ignore_errors_in_success_rate" => ignore_errors,
        "entries" => entries,
        "overall" => overall_success_statistics(entries; ignore_errors=ignore_errors),
        "fieldwise" => fieldwise_statistics(entries; ignore_errors=ignore_errors),
    )
end

function aggregate_all_result_files(results_dir::String = DEFAULT_RESULTS_DIR)
    files = available_result_files(results_dir)
    aggregated = OrderedDict{String,Any}[]

    for path in files
        try
            push!(aggregated, aggregate_result_file_statistics(path))
        catch err
            println("Warning: skipping result file `", path, "` because it could not be parsed: ", err)
        end
    end

    isempty(aggregated) && throw(ArgumentError("No valid result JSON files could be loaded from $(results_dir)."))
    return aggregated
end


function maybe_filter_to_paper_models(aggregated_stats::Vector{OrderedDict{String,Any}}; enabled::Bool = USE_PAPER_MODEL_SUBSET)
    if !enabled
        return aggregated_stats
    end

    filtered = OrderedDict{String,Any}[]
    for stats in aggregated_stats
        model_name = String(stats["model"])
        if model_name in PAPER_MODEL_SUBSET
            push!(filtered, stats)
        end
    end

    isempty(filtered) && throw(ArgumentError("Paper-model filter is enabled, but none of the result files matched the selected models."))
    return filtered
end

# =================================================================================================
# Printing and plotting
# =================================================================================================

function print_model_comparison_table(aggregated_stats::Vector{OrderedDict{String,Any}})
    println("================================================================================")
    println("Overall success rate comparison")
    println("================================================================================")

    sorted_stats = sort(aggregated_stats; by = x -> -Float64(x["overall"]["success_rate"]))
    for stats in sorted_stats
        overall = stats["overall"]
        println("Model: ", stats["model"], " (provider: ", stats["provider"], ")")
        println("  total: ", overall["total"])
        println("  effective_total: ", overall["effective_total"])
        println("  success: ", overall["success"])
        println("  error: ", overall["error"])
        println("  success_rate: ", round(Float64(overall["success_rate"]); digits=4))
        println()
    end
end

function field_success_rate_matrix(aggregated_stats::Vector{OrderedDict{String,Any}})
    model_labels = [String(stats["model"]) for stats in aggregated_stats]
    field_order = [
        "natural_paraphrase",
        "paraphrase_gpt5.4-mini",
        "paraphrase_gemini-2.5-flash",
        "paraphrase_deepseek",
        "paraphrase_claude",
    ]
    present_fields = String[]

    for field in field_order
        for stats in aggregated_stats
            if haskey(stats["fieldwise"], field)
                push!(present_fields, field)
                break
            end
        end
    end

    matrix = fill(NaN, length(present_fields), length(aggregated_stats))

    for (j, stats) in enumerate(aggregated_stats)
        fieldwise = stats["fieldwise"]
        for (i, field) in enumerate(present_fields)
            if haskey(fieldwise, field)
                total = fieldwise[field]["total"]
                success = fieldwise[field]["success"]
                matrix[i, j] = total == 0 ? NaN : success / total
            end
        end
    end

    field_labels = [source_label_from_field(field) for field in present_fields]
    return model_labels, field_labels, matrix
end

function plot_llm_paraphrase_heatmap(
    aggregated_stats::Vector{OrderedDict{String,Any}};
    output_dir::String = DEFAULT_OUTPUT_DIR,
)
    ensure_directory(output_dir)
    model_labels, field_labels, matrix = field_success_rate_matrix(aggregated_stats)
    isempty(model_labels) && throw(ArgumentError("No result files available for heatmap plotting."))
    isempty(field_labels) && throw(ArgumentError("No paraphrase fields available for heatmap plotting."))

    output_path = joinpath(output_dir, "llm_vs_paraphrase_success_heatmap.png")

    p = heatmap(
        1:length(model_labels),
        1:length(field_labels),
        matrix;
        xticks=(1:length(model_labels), model_labels),
        yticks=(1:length(field_labels), field_labels),
        xrotation=25,
        xlabel="Translation model",
        ylabel="Paraphrase source",
        title="Success rate by translation model and paraphrase source",
        clims=(0.0, 1.0),
        colorbar_title="Success rate",
        aspect_ratio=:equal,
        size=(1900, 1200),
        dpi=DEFAULT_PLOT_DPI,
        bottom_margin=55Plots.mm,
        left_margin=35Plots.mm,
        right_margin=28Plots.mm,
        top_margin=20Plots.mm,
        guidefontsize=20,
        tickfontsize=15,
        titlefontsize=22,
        colorbar_tickfontsize=13,
        colorbar_titlefontsize=15,
    )

    for i in 1:size(matrix, 1)
        for j in 1:size(matrix, 2)
            value = matrix[i, j]
            if !isnan(value)
                annotate!(p, j, i, text(string(round(value; digits=2)), 12, :white, :center))
            end
        end
    end

    savefig(p, output_path)
    display(p)
    println("Saved heatmap: ", output_path)
    return output_path
end

function print_summary(results_path::String, entries::Vector{OrderedDict{String,Any}})
    results_obj = load_json_object(results_path)
    ignore_errors = ignore_errors_in_success_rate(results_obj, results_path)
    overall = overall_success_statistics(entries; ignore_errors=ignore_errors)

    println("Results file: ", results_path)
    println("Total evaluations: ", overall["total"])
    println("Effective total for success rate: ", overall["effective_total"])
    println("Successful translations (status = ok): ", overall["ok"])
    println("Semantically equivalent outputs: ", overall["equivalent"])
    println("Success count (ok AND equivalent): ", overall["success"])
    println("Error count: ", overall["error"])
    println("Success rate: ", round(Float64(overall["success_rate"]); digits=4))
    println()

    field_stats = fieldwise_statistics(entries; ignore_errors=ignore_errors)
    println("Field-wise statistics:")
    for (field, stats) in field_stats
        denom = stats["effective_total"]
        rate = denom == 0 ? 0.0 : stats["success"] / denom
        println("  Field: ", field)
        println("    total: ", stats["total"])
        println("    effective_total: ", stats["effective_total"])
        println("    ok: ", stats["ok"])
        println("    equivalent: ", stats["equivalent"])
        println("    success: ", stats["success"])
        println("    error: ", stats["error"])
        println("    success_rate: ", round(rate; digits=4))
    end
    println()
end

function sanitize_basename(path::String)
    base = splitext(basename(path))[1]
    return replace(base, r"[^A-Za-z0-9._-]" => "_")
end


function plot_success_rate_vs_formula_size(
    entries::Vector{OrderedDict{String,Any}},
    dataset_index::Dict{Int,OrderedDict{String,Any}};
    results_path::String,
    output_dir::String = DEFAULT_OUTPUT_DIR,
)
    results_obj = load_json_object(results_path)
    ignore_errors = ignore_errors_in_success_rate(results_obj, results_path)
    sizes, rates, totals = size_bucket_success_rates(entries, dataset_index; ignore_errors=ignore_errors)
    isempty(sizes) && throw(ArgumentError("No size statistics could be computed for $(results_path)."))

    ensure_directory(output_dir)
    output_path = joinpath(output_dir, sanitize_basename(results_path) * "_success_rate_vs_formula_size.png")

    p = plot(
        sizes,
        rates;
        seriestype=:scatter,
        xlabel="LTL formula size",
        ylabel="Success rate",
        title="",
        label="Success rate",
        legend=:topright,
        ylim=(0.0, 1.0),
        xlims=(minimum(sizes) - 0.5, maximum(sizes) + 0.5),
        marker=:circle,
        markersize=10,
        markerstrokewidth=0.0,
        linewidth=3.5,
        linecolor=:black,
        framestyle=:box,
        grid=true,
        size=(1200, 750),
        dpi=DEFAULT_PLOT_DPI,
        left_margin=14Plots.mm,
        right_margin=10Plots.mm,
        bottom_margin=12Plots.mm,
        top_margin=6Plots.mm,
        guidefontsize=16,
        tickfontsize=12,
        legendfontsize=12,
    )
    plot!(p, sizes, rates; label="", linewidth=2.5, linecolor=:black)

    annotations = [(sizes[i], min(rates[i] + 0.04, 0.98), text(string(totals[i]), 9, :center)) for i in eachindex(sizes)]
    annotate!(p, annotations)

    savefig(p, output_path)
    println("Saved figure: ", output_path)
    return output_path
end

function plot_success_rate_vs_benchmark_min_equiv_formula_size(
    entries::Vector{OrderedDict{String,Any}},
    dataset_index::Dict{Int,OrderedDict{String,Any}};
    results_path::String,
    dataset_path::String = DEFAULT_DATASET_PATH,
    output_dir::String = DEFAULT_OUTPUT_DIR,
)
    results_obj = load_json_object(results_path)
    ignore_errors = ignore_errors_in_success_rate(results_obj, results_path)
    min_size_map = build_benchmark_min_equiv_formula_size_map(dataset_path)
    sizes, rates, totals = canonicalized_size_bucket_success_rates(entries, dataset_index, min_size_map; ignore_errors=ignore_errors)
    isempty(sizes) && throw(ArgumentError("No benchmark-minimal equivalent size statistics could be computed for $(results_path)."))

    ensure_directory(output_dir)
    output_path = joinpath(output_dir, sanitize_basename(results_path) * "_success_rate_vs_benchmark_min_equiv_formula_size.png")

    p = plot(
        sizes,
        rates;
        seriestype=:scatter,
        xlabel="Benchmark-minimal equivalent LTL size",
        ylabel="Success rate",
        title="",
        label="Success rate",
        legend=:topright,
        ylim=(0.0, 1.0),
        xlims=(minimum(sizes) - 0.5, maximum(sizes) + 0.5),
        marker=:circle,
        markersize=10,
        markerstrokewidth=0.0,
        linewidth=3.5,
        linecolor=:black,
        framestyle=:box,
        grid=true,
        size=(1200, 750),
        dpi=DEFAULT_PLOT_DPI,
        left_margin=14Plots.mm,
        right_margin=10Plots.mm,
        bottom_margin=12Plots.mm,
        top_margin=6Plots.mm,
        guidefontsize=16,
        tickfontsize=12,
        legendfontsize=12,
    )
    plot!(p, sizes, rates; label="", linewidth=2.5, linecolor=:black)

    annotations = [(sizes[i], min(rates[i] + 0.04, 0.98), text(string(totals[i]), 9, :center)) for i in eachindex(sizes)]
    annotate!(p, annotations)

    savefig(p, output_path)
    println("Saved figure: ", output_path)
    return output_path
end

function plot_success_rate_vs_automaton_size(
    entries::Vector{OrderedDict{String,Any}},
    dataset_index::Dict{Int,OrderedDict{String,Any}};
    results_path::String,
    output_dir::String = DEFAULT_OUTPUT_DIR,
)
    results_obj = load_json_object(results_path)
    ignore_errors = ignore_errors_in_success_rate(results_obj, results_path)
    sizes, rates, totals = automaton_size_bucket_success_rates(entries, dataset_index; ignore_errors=ignore_errors)
    isempty(sizes) && throw(ArgumentError("No automaton-size statistics could be computed for $(results_path)."))

    ensure_directory(output_dir)
    output_path = joinpath(output_dir, sanitize_basename(results_path) * "_success_rate_vs_automaton_size.png")

    p = plot(
        sizes,
        rates;
        seriestype=:scatter,
        xlabel="Automaton size",
        ylabel="Success rate",
        title="",
        label="Success rate",
        legend=:topright,
        ylim=(0.0, 1.0),
        xlims=(minimum(sizes) - 0.5, maximum(sizes) + 0.5),
        marker=:circle,
        markersize=10,
        markerstrokewidth=1.0,
        linewidth=2.5,
        linecolor=:black,
        framestyle=:box,
        grid=true,
        size=(1200, 750),
        dpi=DEFAULT_PLOT_DPI,
        left_margin=14Plots.mm,
        right_margin=10Plots.mm,
        bottom_margin=12Plots.mm,
        top_margin=6Plots.mm,
        guidefontsize=16,
        tickfontsize=12,
        legendfontsize=11,
    )
    plot!(p, sizes, rates; label="", linewidth=2.5, linecolor=:black)

    annotations = [(sizes[i], min(rates[i] + 0.04, 0.98), text(string(totals[i]), 9, :center)) for i in eachindex(sizes)]
    annotate!(p, annotations)

    savefig(p, output_path)
    println("Saved figure: ", output_path)
    return output_path
end


function plot_success_rate_vs_temporal_depth(
    entries::Vector{OrderedDict{String,Any}},
    dataset_index::Dict{Int,OrderedDict{String,Any}};
    results_path::String,
    output_dir::String = DEFAULT_OUTPUT_DIR,
)
    results_obj = load_json_object(results_path)
    ignore_errors = ignore_errors_in_success_rate(results_obj, results_path)
    depths, rates, totals = temporal_depth_bucket_success_rates(entries, dataset_index; ignore_errors=ignore_errors)
    isempty(depths) && throw(ArgumentError("No temporal-depth statistics could be computed for $(results_path)."))

    ensure_directory(output_dir)
    output_path = joinpath(output_dir, sanitize_basename(results_path) * "_success_rate_vs_temporal_depth.png")

    p = plot(
        depths,
        rates;
        seriestype=:scatter,
        xlabel="Temporal depth",
        ylabel="Success rate",
        title="",
        label="Success rate",
        legend=:topright,
        ylim=(0.0, 1.0),
        xlims=(minimum(depths) - 0.5, maximum(depths) + 0.5),
        marker=:circle,
        markersize=10,
        markerstrokewidth=0,
        linewidth=2.5,
        linecolor=:black,
        framestyle=:box,
        grid=true,
        size=(1200, 750),
        dpi=DEFAULT_PLOT_DPI,
        left_margin=14Plots.mm,
        right_margin=10Plots.mm,
        bottom_margin=12Plots.mm,
        top_margin=6Plots.mm,
        guidefontsize=16,
        tickfontsize=12,
        legendfontsize=11,
    )
    plot!(p, depths, rates; label="", linewidth=2.5, linecolor=:black)

    annotations = [(depths[i], min(rates[i] + 0.04, 0.98), text(string(totals[i]), 9, :center)) for i in eachindex(depths)]
    annotate!(p, annotations)

    savefig(p, output_path)
    println("Saved figure: ", output_path)
    return output_path
end

function plot_success_rate_vs_nl_phrase_length(
    entries::Vector{OrderedDict{String,Any}},
    dataset_index::Dict{Int,OrderedDict{String,Any}};
    results_path::String,
    output_dir::String = DEFAULT_OUTPUT_DIR,
)
    results_obj = load_json_object(results_path)
    ignore_errors = ignore_errors_in_success_rate(results_obj, results_path)
    lengths, rates, totals = nl_phrase_length_bucket_success_rates(entries, dataset_index; ignore_errors=ignore_errors)
    keep_idx = [i for i in eachindex(lengths) if lengths[i] <= 75]
    isempty(keep_idx) && throw(ArgumentError("No NL-phrase-length statistics at or below 75 words could be computed for $(results_path)."))
    lengths = lengths[keep_idx]
    rates = rates[keep_idx]
    totals = totals[keep_idx]
    isempty(lengths) && throw(ArgumentError("No NL-phrase-length statistics could be computed for $(results_path)."))

    ensure_directory(output_dir)
    output_path = joinpath(output_dir, sanitize_basename(results_path) * "_success_rate_vs_nl_phrase_length.png")

    p = plot(
        lengths,
        rates;
        seriestype=:scatter,
        xlabel="NL phrase length (words)",
        ylabel="Success rate",
        title="",
        label="Success rate",
        legend=:topright,
        ylim=(0.0, 1.0),
        xlims=(minimum(lengths) - 0.5, maximum(lengths) + 0.5),
        marker=:circle,
        markersize=10,
        markerstrokewidth=0.0,
        linewidth=3.5,
        linecolor=:black,
        framestyle=:box,
        grid=true,
        size=(1200, 750),
        dpi=DEFAULT_PLOT_DPI,
        left_margin=14Plots.mm,
        right_margin=10Plots.mm,
        bottom_margin=12Plots.mm,
        top_margin=6Plots.mm,
        guidefontsize=16,
        tickfontsize=12,
        legendfontsize=12,
    )
    plot!(p, lengths, rates; label="", linewidth=2.5, linecolor=:black)

    annotations = [(lengths[i], min(rates[i] + 0.04, 0.98), text(string(totals[i]), 9, :center)) for i in eachindex(lengths)]
    annotate!(p, annotations)

    savefig(p, output_path)
    println("Saved figure: ", output_path)
    return output_path
end


function plot_nl_phrase_length_distribution(
    entries::Vector{OrderedDict{String,Any}},
    dataset_index::Dict{Int,OrderedDict{String,Any}};
    results_path::String,
    output_dir::String = DEFAULT_OUTPUT_DIR,
)
    lengths, freqs = nl_phrase_length_distribution(entries, dataset_index)
    keep_idx = [i for i in eachindex(lengths) if lengths[i] <= 75]
    isempty(keep_idx) && throw(ArgumentError("No NL-phrase-length distribution at or below 75 words could be computed for $(results_path)."))
    lengths = lengths[keep_idx]
    freqs = freqs[keep_idx]
    isempty(lengths) && throw(ArgumentError("No NL-phrase-length distribution could be computed for $(results_path)."))

    ensure_directory(output_dir)
    output_path = joinpath(output_dir, sanitize_basename(results_path) * "_nl_phrase_length_distribution.png")

    p = bar(
        lengths,
        freqs;
        xlabel="NL phrase length (words)",
        ylabel="Count",
        title="",
        label="",
        legend=false,
        framestyle=:box,
        grid=true,
        size=(1200, 750),
        dpi=DEFAULT_PLOT_DPI,
        left_margin=14Plots.mm,
        right_margin=10Plots.mm,
        bottom_margin=12Plots.mm,
        top_margin=6Plots.mm,
        guidefontsize=16,
        tickfontsize=12,
        bar_width=0.8,
    )

    annotations = [(lengths[i], freqs[i] + maximum(freqs) * 0.02, text(string(freqs[i]), 9, :center)) for i in eachindex(lengths)]
    annotate!(p, annotations)

    savefig(p, output_path)
    println("Saved figure: ", output_path)
    return output_path
end

function plot_nnf_formula_size_distribution(
    dataset_path::String = DEFAULT_DATASET_PATH;
    output_dir::String = DEFAULT_OUTPUT_DIR,
)
    ensure_directory(output_dir)
    dataset_index = load_dataset_index(dataset_path)

    counts = OrderedDict{Int,Int}()
    for record in values(dataset_index)
        record_has_nnf_flag(record) || continue
        size_value = get_record_formula_size(record)
        counts[size_value] = get(counts, size_value, 0) + 1
    end

    isempty(counts) && throw(ArgumentError("No NNF formulas with formula-size information were found in $(dataset_path)."))

    sizes = sort(collect(keys(counts)))
    freqs = [counts[size_value] for size_value in sizes]
    output_path = joinpath(output_dir, "nnf_formula_size_distribution.png")

    p = bar(
        sizes,
        freqs;
        xlabel="NNF formula size",
        ylabel="Count",
        title="",
        label="",
        legend=false,
        framestyle=:box,
        grid=true,
        size=(1200, 750),
        dpi=DEFAULT_PLOT_DPI,
        left_margin=14Plots.mm,
        right_margin=10Plots.mm,
        bottom_margin=12Plots.mm,
        top_margin=6Plots.mm,
        guidefontsize=16,
        tickfontsize=12,
        bar_width=0.8,
    )

    annotations = [(sizes[i], freqs[i] + maximum(freqs) * 0.02, text(string(freqs[i]), 9, :center)) for i in eachindex(sizes)]
    annotate!(p, annotations)

    savefig(p, output_path)
    display(p)
    println("Saved NNF formula-size distribution figure: ", output_path)
    return output_path
end

function plot_all_models_success_rate_vs_formula_size(
    results_dir::String = DEFAULT_RESULTS_DIR;
    dataset_path::String = DEFAULT_DATASET_PATH,
    output_dir::String = DEFAULT_OUTPUT_DIR,
)
    ensure_directory(output_dir)
    dataset_index = load_dataset_index(dataset_path)
    aggregated_stats_raw = aggregate_all_result_files(results_dir)
    aggregated_stats = USE_ZERO_VS_FEWSHOT_SUBPLOTS ? aggregated_stats_raw : maybe_filter_to_paper_models(aggregated_stats_raw)

    if USE_ZERO_VS_FEWSHOT_SUBPLOTS
        filtered_stats = OrderedDict{String,Any}[]
        for stats in aggregated_stats
            if use_zero_vs_fewshot_result_file(stats)
                push!(filtered_stats, stats)
            end
        end

        isempty(filtered_stats) && throw(ArgumentError("Zero-vs-few-shot subplot mode is enabled, but no matching result files were found for the selected models."))

        series_data = Vector{Tuple{String,String,Vector{Int},Vector{Float64},Vector{Int}}}()
        all_sizes = Int[]

        for stats in filtered_stats
            model_name = String(stats["model"])
            prompt_setting = haskey(stats, "prompt_setting") ? String(stats["prompt_setting"]) : "zeroshot"
            entries = Vector{OrderedDict{String,Any}}(stats["entries"])
            ignore_errors = haskey(stats, "ignore_errors_in_success_rate") ? Bool(stats["ignore_errors_in_success_rate"]) : false
            sizes, rates, totals = size_bucket_success_rates(entries, dataset_index; ignore_errors=ignore_errors)
            keep_idx = [i for i in eachindex(sizes) if 1 <= sizes[i] <= MAX_PLOTTED_FORMULA_SIZE]
            isempty(keep_idx) && continue
            sizes = sizes[keep_idx]
            rates = rates[keep_idx]
            totals = totals[keep_idx]
            push!(series_data, (model_name, prompt_setting, sizes, rates, totals))
            append!(all_sizes, sizes)
        end

        isempty(all_sizes) && throw(ArgumentError("No size statistics could be computed from the available zero-shot/few-shot result files."))

        x_values = sort(unique(all_sizes))
        xtick_step = maximum(x_values) <= 20 ? 1 : (maximum(x_values) <= 40 ? 2 : 5)
        xtick_values = collect(minimum(x_values):xtick_step:maximum(x_values))
        marker_shapes = Dict(
            "gpt-5.4" => :circle,
            "gpt-5.5" => :rect,
            "claude-opus-4-7" => :diamond,
            "deepseek-v4-flash" => :utriangle,
            "mistral-medium-latest" => :dtriangle,
        )

        subplot_list = Plots.Plot[]
        for prompt_setting in ["zeroshot", "fewshot"]
            prompt_series = [(model_name, sizes, rates, totals) for (model_name, ps, sizes, rates, totals) in series_data if ps == prompt_setting]

            local_p = plot(
                xlabel=(prompt_setting == "fewshot" ? "LTL formula size" : ""),
                ylabel=(prompt_setting == "zeroshot" ? "Zero-shot success rate" : "Few-shot success rate"),
                title="",
                legend=false,
                ylim=(0.0, 1.0),
                xlims=(minimum(x_values) - 0.5, maximum(x_values) + 0.5),
                xticks=xtick_values,
                framestyle=:box,
                grid=true,
                guidefontsize=25,
                tickfontsize=15,
                titlefontsize=18,
                legendfontsize=14,
                guidefontweight=:bold,
                left_margin=14Plots.mm,
                right_margin=10Plots.mm,
                bottom_margin=10Plots.mm,
                top_margin=6Plots.mm,
            )

            for model_name in ZERO_VS_FEWSHOT_MODELS
                matching = [item for item in prompt_series if item[1] == model_name]
                isempty(matching) && continue
                _, sizes, rates, totals = matching[1]
                marker_shape = get(marker_shapes, model_name, :circle)
                plot!(
                    local_p,
                    sizes,
                    rates;
                    marker=marker_shape,
                    markersize=5,
                    markerstrokewidth=0.4,
                    linewidth=2.3,
                    label="",
                )
            end

            push!(subplot_list, local_p)
        end

        p = plot(
            subplot_list...;
            layout=(2, 1),
            size=(1700, 1250),
            dpi=DEFAULT_PLOT_DPI,
        )
    else
        series_data = Vector{Tuple{String,Vector{Int},Vector{Float64},Vector{Int},Float64}}()
        all_sizes = Int[]

        for stats in aggregated_stats
            model_name = String(stats["model"])
            entries = Vector{OrderedDict{String,Any}}(stats["entries"])
            ignore_errors = haskey(stats, "ignore_errors_in_success_rate") ? Bool(stats["ignore_errors_in_success_rate"]) : false
            sizes, rates, totals = size_bucket_success_rates(entries, dataset_index; ignore_errors=ignore_errors)
            keep_idx = [i for i in eachindex(sizes) if 1 <= sizes[i] <= MAX_PLOTTED_FORMULA_SIZE]
            isempty(keep_idx) && continue
            sizes = sizes[keep_idx]
            rates = rates[keep_idx]
            totals = totals[keep_idx]
            overall_rate = haskey(stats, "overall") && haskey(stats["overall"], "success_rate") ? Float64(stats["overall"]["success_rate"]) : 0.0
            push!(series_data, (model_name, sizes, rates, totals, overall_rate))
            append!(all_sizes, sizes)
        end

        isempty(all_sizes) && throw(ArgumentError("No size statistics could be computed from the available result files."))

        sorted_series = sort(series_data; by = item -> -item[5])
        x_values = sort(unique(all_sizes))
        xtick_step = maximum(x_values) <= 20 ? 1 : (maximum(x_values) <= 40 ? 2 : 5)
        xtick_values = collect(minimum(x_values):xtick_step:maximum(x_values))

        n_models = length(sorted_series)
        ncols = 2
        nrows = ceil(Int, n_models / ncols)

        plots_list = Plots.Plot[]
        marker_shapes = [:circle, :rect, :diamond, :utriangle, :dtriangle, :star5, :hexagon, :xcross]

        for (idx, (model_name, sizes, rates, totals, overall_rate)) in enumerate(sorted_series)
            marker_shape = marker_shapes[mod1(idx, length(marker_shapes))]

            local_p = plot(
                sizes,
                rates;
                seriestype=:scatter,
                xlabel=(idx > n_models - ncols ? "LTL formula size" : ""),
                ylabel=(mod1(idx, ncols) == 1 ? "Success rate" : ""),
                title=model_name,
                label="",
                ylim=(0.0, 1.0),
                xlims=(minimum(x_values) - 0.5, maximum(x_values) + 0.5),
                xticks=xtick_values,
                marker=marker_shape,
                markersize=5,
                markerstrokewidth=0.4,
                linewidth=2.3,
                linecolor=:black,
                framestyle=:box,
                grid=true,
                guidefontsize=13,
                tickfontsize=10,
                titlefontsize=13,
                left_margin=8Plots.mm,
                right_margin=4Plots.mm,
                bottom_margin=6Plots.mm,
                top_margin=4Plots.mm,
            )
            plot!(local_p, sizes, rates; label="", linewidth=2.3, linecolor=:black)
            push!(plots_list, local_p)
        end

        p = plot(
            plots_list...;
            layout=(nrows, ncols),
            size=(1600, 420 * nrows),
            dpi=DEFAULT_PLOT_DPI,
        )
    end

    output_path = joinpath(output_dir, "all_models_success_rate_vs_formula_size.png")
    savefig(p, output_path)
    display(p)
    println("Saved combined success-rate figure: ", output_path)
    return output_path
end

function plot_all_models_success_rate_vs_benchmark_min_equiv_formula_size(
    results_dir::String = DEFAULT_RESULTS_DIR;
    dataset_path::String = DEFAULT_DATASET_PATH,
    output_dir::String = DEFAULT_OUTPUT_DIR,
)
    ensure_directory(output_dir)
    dataset_index = load_dataset_index(dataset_path)
    min_size_map = build_benchmark_min_equiv_formula_size_map(dataset_path)
    aggregated_stats_raw = aggregate_all_result_files(results_dir)
    aggregated_stats = USE_ZERO_VS_FEWSHOT_SUBPLOTS ? aggregated_stats_raw : maybe_filter_to_paper_models(aggregated_stats_raw)

    if USE_ZERO_VS_FEWSHOT_SUBPLOTS
        filtered_stats = OrderedDict{String,Any}[]
        for stats in aggregated_stats
            if use_zero_vs_fewshot_result_file(stats)
                push!(filtered_stats, stats)
            end
        end

        isempty(filtered_stats) && throw(ArgumentError("Zero-vs-few-shot subplot mode is enabled, but no matching result files were found for the selected models."))

        series_data = Vector{Tuple{String,String,Vector{Int},Vector{Float64},Vector{Int}}}()
        all_sizes = Int[]

        for stats in filtered_stats
            model_name = String(stats["model"])
            prompt_setting = haskey(stats, "prompt_setting") ? String(stats["prompt_setting"]) : "zeroshot"
            entries = Vector{OrderedDict{String,Any}}(stats["entries"])
            ignore_errors = haskey(stats, "ignore_errors_in_success_rate") ? Bool(stats["ignore_errors_in_success_rate"]) : false
            sizes, rates, totals = canonicalized_size_bucket_success_rates(entries, dataset_index, min_size_map; ignore_errors=ignore_errors)
            keep_idx = [i for i in eachindex(sizes) if 1 <= sizes[i] <= MAX_PLOTTED_FORMULA_SIZE]
            isempty(keep_idx) && continue
            sizes = sizes[keep_idx]
            rates = rates[keep_idx]
            totals = totals[keep_idx]
            push!(series_data, (model_name, prompt_setting, sizes, rates, totals))
            append!(all_sizes, sizes)
        end

        isempty(all_sizes) && throw(ArgumentError("No benchmark-minimal equivalent size statistics could be computed from the available zero-shot/few-shot result files."))

        x_values = sort(unique(all_sizes))
        xtick_step = maximum(x_values) <= 20 ? 1 : (maximum(x_values) <= 40 ? 2 : 5)
        xtick_values = collect(minimum(x_values):xtick_step:maximum(x_values))
        marker_shapes = Dict(
            "gpt-5.4" => :circle,
            "gpt-5.5" => :rect,
            "claude-opus-4-7" => :diamond,
            "deepseek-v4-flash" => :utriangle,
            "mistral-medium-latest" => :dtriangle,
        )

        prompt_settings_to_plot = ZERO_VS_FEWSHOT_LAYOUT_MODE == "zero_shot_only_three_panel" ? ["zeroshot"] : ["zeroshot", "fewshot"]
        subplot_list = Plots.Plot[]
        for prompt_setting in prompt_settings_to_plot
            prompt_series = [(model_name, sizes, rates, totals) for (model_name, ps, sizes, rates, totals) in series_data if ps == prompt_setting]

            local_p = plot(
                xlabel=(ZERO_VS_FEWSHOT_LAYOUT_MODE == "zero_shot_only_three_panel" ? "Minimum AST size among equivalent LTLs in benchmark" : (prompt_setting == "fewshot" ? "Minimum AST size among equivalent LTLs in benchmark" : "")),
                ylabel=(prompt_setting == "zeroshot" ? "Zero-shot success rate" : "Few-shot success rate"),
                title="",
                legend=(ZERO_VS_FEWSHOT_LAYOUT_MODE == "zero_shot_only_three_panel" && prompt_setting == "zeroshot" ? :topright : false),
                ylim=(0.0, 1.0),
                xlims=(minimum(x_values) - 0.5, maximum(x_values) + 0.5),
                xticks=xtick_values,
                framestyle=:box,
                grid=true,
                guidefontsize=20,
                tickfontsize=15,
                titlefontsize=18,
                legendfontsize=14,
                guidefontweight=:bold,
                left_margin=14Plots.mm,
                right_margin=10Plots.mm,
                bottom_margin=10Plots.mm,
                top_margin=6Plots.mm,
            )

            for model_name in ZERO_VS_FEWSHOT_MODELS
                matching = [item for item in prompt_series if item[1] == model_name]
                isempty(matching) && continue
                _, sizes, rates, totals = matching[1]
                marker_shape = get(marker_shapes, model_name, :circle)
                plot!(
                    local_p,
                    sizes,
                    rates;
                    marker=marker_shape,
                    markersize=5,
                    markerstrokewidth=0.4,
                    linewidth=2.3,
                    label=(ZERO_VS_FEWSHOT_LAYOUT_MODE == "zero_shot_only_three_panel" && prompt_setting == "zeroshot" ? model_name : ""),
                )
            end

            push!(subplot_list, local_p)
        end

        if ZERO_VS_FEWSHOT_LAYOUT_MODE == "zero_shot_only_three_panel"
            p = plot(
                subplot_list...;
                layout=(1, 1),
                size=(1200, 560),
                dpi=DEFAULT_PLOT_DPI,
            )
        else
            p = plot(
                subplot_list...;
                layout=(2, 1),
                size=(1700, 1250),
                dpi=DEFAULT_PLOT_DPI,
            )
        end
    else
        series_data = Vector{Tuple{String,Vector{Int},Vector{Float64},Vector{Int},Float64}}()
        all_sizes = Int[]

        for stats in aggregated_stats
            model_name = String(stats["model"])
            entries = Vector{OrderedDict{String,Any}}(stats["entries"])
            ignore_errors = haskey(stats, "ignore_errors_in_success_rate") ? Bool(stats["ignore_errors_in_success_rate"]) : false
            sizes, rates, totals = canonicalized_size_bucket_success_rates(entries, dataset_index, min_size_map; ignore_errors=ignore_errors)
            keep_idx = [i for i in eachindex(sizes) if 1 <= sizes[i] <= MAX_PLOTTED_FORMULA_SIZE]
            isempty(keep_idx) && continue
            sizes = sizes[keep_idx]
            rates = rates[keep_idx]
            totals = totals[keep_idx]
            overall_rate = haskey(stats, "overall") && haskey(stats["overall"], "success_rate") ? Float64(stats["overall"]["success_rate"]) : 0.0
            push!(series_data, (model_name, sizes, rates, totals, overall_rate))
            append!(all_sizes, sizes)
        end

        isempty(all_sizes) && throw(ArgumentError("No benchmark-minimal equivalent size statistics could be computed from the available result files."))

        sorted_series = sort(series_data; by = item -> -item[5])
        x_values = sort(unique(all_sizes))
        xtick_step = maximum(x_values) <= 20 ? 1 : (maximum(x_values) <= 40 ? 2 : 5)
        xtick_values = collect(minimum(x_values):xtick_step:maximum(x_values))

        n_models = length(sorted_series)
        ncols = 2
        nrows = ceil(Int, n_models / ncols)

        plots_list = Plots.Plot[]
        marker_shapes = [:circle, :rect, :diamond, :utriangle, :dtriangle, :star5, :hexagon, :xcross]

        for (idx, (model_name, sizes, rates, totals, overall_rate)) in enumerate(sorted_series)
            marker_shape = marker_shapes[mod1(idx, length(marker_shapes))]

            local_p = plot(
                sizes,
                rates;
                seriestype=:scatter,
                xlabel=(idx > n_models - ncols ? "Benchmark-minimal equivalent AST size" : ""),
                ylabel=(mod1(idx, ncols) == 1 ? "Success rate" : ""),
                title=model_name,
                label="",
                ylim=(0.0, 1.0),
                xlims=(minimum(x_values) - 0.5, maximum(x_values) + 0.5),
                xticks=xtick_values,
                marker=marker_shape,
                markersize=5,
                markerstrokewidth=0.4,
                linewidth=2.3,
                linecolor=:black,
                framestyle=:box,
                grid=true,
                guidefontsize=11,
                tickfontsize=10,
                titlefontsize=13,
                left_margin=8Plots.mm,
                right_margin=4Plots.mm,
                bottom_margin=6Plots.mm,
                top_margin=4Plots.mm,
            )
            plot!(local_p, sizes, rates; label="", linewidth=2.3, linecolor=:black)
            push!(plots_list, local_p)
        end

        p = plot(
            plots_list...;
            layout=(nrows, ncols),
            size=(1600, 420 * nrows),
            dpi=DEFAULT_PLOT_DPI,
        )
    end

    output_path = joinpath(output_dir, "all_models_success_rate_vs_benchmark_min_equiv_formula_size.png")
    savefig(p, output_path)
    display(p)
    println("Saved combined benchmark-minimal equivalent size figure: ", output_path)
    return output_path
end


function plot_all_models_success_rate_vs_formula_size_nnf(
    results_dir::String = DEFAULT_RESULTS_DIR;
    dataset_path::String = DEFAULT_DATASET_PATH,
    output_dir::String = DEFAULT_OUTPUT_DIR,
)
    ensure_directory(output_dir)
    dataset_index = load_dataset_index(dataset_path)
    aggregated_stats_raw = aggregate_all_result_files(results_dir)
    aggregated_stats = USE_ZERO_VS_FEWSHOT_SUBPLOTS ? aggregated_stats_raw : maybe_filter_to_paper_models(aggregated_stats_raw)

    function filter_entries_to_nnf(entries)
        return [entry for entry in entries if haskey(entry, "record_id") &&
                                          haskey(dataset_index, Int(entry["record_id"])) &&
                                          record_has_nnf_flag(dataset_index[Int(entry["record_id"])])]
    end

    if USE_ZERO_VS_FEWSHOT_SUBPLOTS
        filtered_stats = OrderedDict{String,Any}[]
        for stats in aggregated_stats
            if use_zero_vs_fewshot_result_file(stats)
                push!(filtered_stats, stats)
            end
        end

        isempty(filtered_stats) && throw(ArgumentError("Zero-vs-few-shot subplot mode is enabled, but no matching result files were found for the selected models."))

        series_data = Vector{Tuple{String,String,Vector{Int},Vector{Float64},Vector{Int}}}()
        all_sizes = Int[]

        for stats in filtered_stats
            model_name = String(stats["model"])
            prompt_setting = haskey(stats, "prompt_setting") ? String(stats["prompt_setting"]) : "zeroshot"
            entries = filter_entries_to_nnf(Vector{OrderedDict{String,Any}}(stats["entries"]))
            isempty(entries) && continue
            ignore_errors = haskey(stats, "ignore_errors_in_success_rate") ? Bool(stats["ignore_errors_in_success_rate"]) : false
            sizes, rates, totals = size_bucket_success_rates(entries, dataset_index; ignore_errors=ignore_errors)
            keep_idx = [i for i in eachindex(sizes) if 1 <= sizes[i] <= MAX_PLOTTED_FORMULA_SIZE]
            isempty(keep_idx) && continue
            sizes = sizes[keep_idx]
            rates = rates[keep_idx]
            totals = totals[keep_idx]
            push!(series_data, (model_name, prompt_setting, sizes, rates, totals))
            append!(all_sizes, sizes)
        end

        isempty(all_sizes) && throw(ArgumentError("No NNF formula-size statistics could be computed from the available result files."))

        x_values = sort(unique(all_sizes))
        xtick_step = maximum(x_values) <= 20 ? 1 : (maximum(x_values) <= 40 ? 2 : 5)
        xtick_values = collect(minimum(x_values):xtick_step:maximum(x_values))
        marker_shapes = Dict(
            "gpt-5.4" => :circle,
            "gpt-5.5" => :rect,
            "claude-opus-4-7" => :diamond,
            "deepseek-v4-flash" => :utriangle,
            "mistral-medium-latest" => :dtriangle,
        )

        subplot_list = Plots.Plot[]
        for prompt_setting in ["zeroshot", "fewshot"]
            prompt_series = [(model_name, sizes, rates, totals) for (model_name, ps, sizes, rates, totals) in series_data if ps == prompt_setting]

            local_p = plot(
                xlabel=(prompt_setting == "fewshot" ? "NNF formula size" : ""),
                ylabel=(prompt_setting == "zeroshot" ? "Zero-shot success rate" : "Few-shot success rate"),
                title="",
                legend=false,
                ylim=(0.0, 1.0),
                xlims=(minimum(x_values) - 0.5, maximum(x_values) + 0.5),
                xticks=xtick_values,
                framestyle=:box,
                grid=true,
                guidefontsize=25,
                tickfontsize=15,
                titlefontsize=18,
                legendfontsize=14,
                guidefontweight=:bold,
                left_margin=14Plots.mm,
                right_margin=10Plots.mm,
                bottom_margin=10Plots.mm,
                top_margin=6Plots.mm,
            )

            for model_name in ZERO_VS_FEWSHOT_MODELS
                matching = [item for item in prompt_series if item[1] == model_name]
                isempty(matching) && continue
                _, sizes, rates, totals = matching[1]
                marker_shape = get(marker_shapes, model_name, :circle)
                plot!(
                    local_p,
                    sizes,
                    rates;
                    marker=marker_shape,
                    markersize=5,
                    markerstrokewidth=0.4,
                    linewidth=2.3,
                    label="",
                )
            end

            push!(subplot_list, local_p)
        end

        p = plot(
            subplot_list...;
            layout=(2, 1),
            size=(1700, 1250),
            dpi=DEFAULT_PLOT_DPI,
        )
    else
        series_data = Vector{Tuple{String,Vector{Int},Vector{Float64},Vector{Int},Float64}}()
        all_sizes = Int[]

        for stats in aggregated_stats
            model_name = String(stats["model"])
            entries = filter_entries_to_nnf(Vector{OrderedDict{String,Any}}(stats["entries"]))
            isempty(entries) && continue
            ignore_errors = haskey(stats, "ignore_errors_in_success_rate") ? Bool(stats["ignore_errors_in_success_rate"]) : false
            sizes, rates, totals = size_bucket_success_rates(entries, dataset_index; ignore_errors=ignore_errors)
            keep_idx = [i for i in eachindex(sizes) if 1 <= sizes[i] <= MAX_PLOTTED_FORMULA_SIZE]
            isempty(keep_idx) && continue
            sizes = sizes[keep_idx]
            rates = rates[keep_idx]
            totals = totals[keep_idx]
            overall_rate = haskey(stats, "overall") && haskey(stats["overall"], "success_rate") ? Float64(stats["overall"]["success_rate"]) : 0.0
            push!(series_data, (model_name, sizes, rates, totals, overall_rate))
            append!(all_sizes, sizes)
        end

        isempty(all_sizes) && throw(ArgumentError("No NNF formula-size statistics could be computed from the available result files."))

        sorted_series = sort(series_data; by = item -> -item[5])
        x_values = sort(unique(all_sizes))
        xtick_step = maximum(x_values) <= 20 ? 1 : (maximum(x_values) <= 40 ? 2 : 5)
        xtick_values = collect(minimum(x_values):xtick_step:maximum(x_values))

        n_models = length(sorted_series)
        ncols = 2
        nrows = ceil(Int, n_models / ncols)

        plots_list = Plots.Plot[]
        marker_shapes = [:circle, :rect, :diamond, :utriangle, :dtriangle, :star5, :hexagon, :xcross]

        for (idx, (model_name, sizes, rates, totals, overall_rate)) in enumerate(sorted_series)
            marker_shape = marker_shapes[mod1(idx, length(marker_shapes))]

            local_p = plot(
                sizes,
                rates;
                seriestype=:scatter,
                xlabel=(idx > n_models - ncols ? "NNF formula size" : ""),
                ylabel=(mod1(idx, ncols) == 1 ? "Success rate" : ""),
                title=model_name,
                label="",
                ylim=(0.0, 1.0),
                xlims=(minimum(x_values) - 0.5, maximum(x_values) + 0.5),
                xticks=xtick_values,
                marker=marker_shape,
                markersize=5,
                markerstrokewidth=0.4,
                linewidth=2.3,
                linecolor=:black,
                framestyle=:box,
                grid=true,
                guidefontsize=13,
                tickfontsize=10,
                titlefontsize=13,
                left_margin=8Plots.mm,
                right_margin=4Plots.mm,
                bottom_margin=6Plots.mm,
                top_margin=4Plots.mm,
            )
            plot!(local_p, sizes, rates; label="", linewidth=2.3, linecolor=:black)
            push!(plots_list, local_p)
        end

        p = plot(
            plots_list...;
            layout=(nrows, ncols),
            size=(1600, 420 * nrows),
            dpi=DEFAULT_PLOT_DPI,
        )
    end

    output_path = joinpath(output_dir, "all_models_success_rate_vs_formula_size_nnf.png")
    savefig(p, output_path)
    display(p)
    println("Saved combined NNF formula-size figure: ", output_path)
    return output_path
end


function plot_all_models_success_rate_vs_automaton_size(
    results_dir::String = DEFAULT_RESULTS_DIR;
    dataset_path::String = DEFAULT_DATASET_PATH,
    output_dir::String = DEFAULT_OUTPUT_DIR,
)
    ensure_directory(output_dir)
    dataset_index = load_dataset_index(dataset_path)
    aggregated_stats_raw = aggregate_all_result_files(results_dir)
    aggregated_stats = USE_ZERO_VS_FEWSHOT_SUBPLOTS ? aggregated_stats_raw : maybe_filter_to_paper_models(aggregated_stats_raw)

    if USE_ZERO_VS_FEWSHOT_SUBPLOTS
        filtered_stats = OrderedDict{String,Any}[]
        for stats in aggregated_stats
            if use_zero_vs_fewshot_result_file(stats)
                push!(filtered_stats, stats)
            end
        end

        isempty(filtered_stats) && throw(ArgumentError("Zero-vs-few-shot subplot mode is enabled, but no matching result files were found for the selected models."))

        series_data = Vector{Tuple{String,String,Vector{Int},Vector{Float64},Vector{Int}}}()
        all_sizes = Int[]

        for stats in filtered_stats
            model_name = String(stats["model"])
            prompt_setting = haskey(stats, "prompt_setting") ? String(stats["prompt_setting"]) : "zeroshot"
            entries = Vector{OrderedDict{String,Any}}(stats["entries"])
            ignore_errors = haskey(stats, "ignore_errors_in_success_rate") ? Bool(stats["ignore_errors_in_success_rate"]) : false
            sizes, rates, totals = automaton_size_bucket_success_rates(entries, dataset_index; ignore_errors=ignore_errors)
            isempty(sizes) && continue
            push!(series_data, (model_name, prompt_setting, sizes, rates, totals))
            append!(all_sizes, sizes)
        end

        isempty(all_sizes) && throw(ArgumentError("No automaton-size statistics could be computed from the available zero-shot/few-shot result files."))

        x_values = sort(unique(all_sizes))
        println("Automaton size x-values for plotting: ", x_values)
        println("Maximum automaton size for plotting: ", maximum(x_values))
        xtick_step = maximum(x_values) <= 20 ? 1 : (maximum(x_values) <= 40 ? 2 : (maximum(x_values) <= 80 ? 5 : 10))
        xtick_values = collect(minimum(x_values):xtick_step:min(maximum(x_values), 50))
        marker_shapes = Dict(
            "gpt-5.4" => :circle,
            "gpt-5.5" => :rect,
            "claude-opus-4-7" => :diamond,
            "deepseek-v4-flash" => :utriangle,
            "mistral-medium-latest" => :dtriangle,
        )

        subplot_list = Plots.Plot[]
        for prompt_setting in ["zeroshot", "fewshot"]
            prompt_series = [(model_name, sizes, rates, totals) for (model_name, ps, sizes, rates, totals) in series_data if ps == prompt_setting]

            local_p = plot(
                xlabel=(prompt_setting == "fewshot" ? "Automaton size" : ""),
                ylabel=(prompt_setting == "zeroshot" ? "Zero-shot success rate" : "Few-shot success rate"),
                title="",
                legend=false,
                ylim=(0.0, 1.0),
                xlims=(minimum(x_values) - 0.5, 50.5),
                xticks=xtick_values,
                framestyle=:box,
                grid=true,
                guidefontsize=25,
                tickfontsize=15,
                titlefontsize=18,
                legendfontsize=14,
                guidefontweight=:bold,
                left_margin=14Plots.mm,
                right_margin=10Plots.mm,
                bottom_margin=10Plots.mm,
                top_margin=6Plots.mm,
            )

            for model_name in ZERO_VS_FEWSHOT_MODELS
                matching = [item for item in prompt_series if item[1] == model_name]
                isempty(matching) && continue
                _, sizes, rates, totals = matching[1]
                marker_shape = get(marker_shapes, model_name, :circle)
                plot!(
                    local_p,
                    sizes,
                    rates;
                    marker=marker_shape,
                    markersize=5,
                    markerstrokewidth=0.4,
                    linewidth=2.3,
                    label="",
                )
            end

            push!(subplot_list, local_p)
        end

        p = plot(
            subplot_list...;
            layout=(2, 1),
            size=(1700, 1250),
            dpi=DEFAULT_PLOT_DPI,
        )
    else
        series_data = OrderedDict{String,Tuple{Vector{Int},Vector{Float64},Vector{Int}}}()
        all_sizes = Int[]

        for stats in aggregated_stats
            model_name = String(stats["model"])
            entries = Vector{OrderedDict{String,Any}}(stats["entries"])
            ignore_errors = haskey(stats, "ignore_errors_in_success_rate") ? Bool(stats["ignore_errors_in_success_rate"]) : false
            sizes, rates, totals = automaton_size_bucket_success_rates(entries, dataset_index; ignore_errors=ignore_errors)
            series_data[model_name] = (sizes, rates, totals)
            append!(all_sizes, sizes)
        end

        isempty(all_sizes) && throw(ArgumentError("No automaton-size statistics could be computed from the available result files."))

        x_values = sort(unique(all_sizes))
        println("Automaton size x-values for plotting: ", x_values)
        println("Maximum automaton size for plotting: ", maximum(x_values))
        xtick_step = maximum(x_values) <= 20 ? 1 : (maximum(x_values) <= 40 ? 2 : (maximum(x_values) <= 80 ? 5 : 10))
        xtick_values = collect(minimum(x_values):xtick_step:min(maximum(x_values), 100))
        p = plot(
            xlabel="Automaton size",
            ylabel="Success rate",
            title="",
            legend=:topright,
            ylim=(0.0, 1.0),
            xlims=(minimum(x_values) - 0.5, 50.5),
            xticks=xtick_values,
            framestyle=:box,
            grid=true,
            size=(1450, 850),
            dpi=DEFAULT_PLOT_DPI,
            left_margin=16Plots.mm,
            right_margin=12Plots.mm,
            bottom_margin=12Plots.mm,
            top_margin=6Plots.mm,
            guidefontsize=18,
            tickfontsize=13,
            legendfontsize=12,
        )

        for (model_name, (sizes, rates, totals)) in series_data
            rate_map = Dict{Int,Float64}(sizes[i] => rates[i] for i in eachindex(sizes))
            y_values = [get(rate_map, x, NaN) for x in x_values]
            plot!(p, x_values, y_values; marker=:circle, markersize=5, markerstrokewidth=0, linewidth=2.5, label=model_name)
        end
    end

    output_path = joinpath(output_dir, "all_models_success_rate_vs_automaton_size.png")
    savefig(p, output_path)
    display(p)
    println("Saved combined automaton-size figure: ", output_path)
    return output_path
end

function plot_all_models_success_rate_vs_temporal_depth(
    results_dir::String = DEFAULT_RESULTS_DIR;
    dataset_path::String = DEFAULT_DATASET_PATH,
    output_dir::String = DEFAULT_OUTPUT_DIR,
)
    ensure_directory(output_dir)
    dataset_index = load_dataset_index(dataset_path)
    aggregated_stats_raw = aggregate_all_result_files(results_dir)
    aggregated_stats = USE_ZERO_VS_FEWSHOT_SUBPLOTS ? aggregated_stats_raw : maybe_filter_to_paper_models(aggregated_stats_raw)

    if USE_ZERO_VS_FEWSHOT_SUBPLOTS
        filtered_stats = OrderedDict{String,Any}[]
        for stats in aggregated_stats
            if use_zero_vs_fewshot_result_file(stats)
                push!(filtered_stats, stats)
            end
        end

        isempty(filtered_stats) && throw(ArgumentError("Zero-vs-few-shot subplot mode is enabled, but no matching result files were found for the selected models."))

        series_data = Vector{Tuple{String,String,Vector{Int},Vector{Float64},Vector{Int}}}()
        all_depths = Int[]

        for stats in filtered_stats
            model_name = String(stats["model"])
            prompt_setting = haskey(stats, "prompt_setting") ? String(stats["prompt_setting"]) : "zeroshot"
            entries = Vector{OrderedDict{String,Any}}(stats["entries"])
            ignore_errors = haskey(stats, "ignore_errors_in_success_rate") ? Bool(stats["ignore_errors_in_success_rate"]) : false
            depths, rates, totals = temporal_depth_bucket_success_rates(entries, dataset_index; ignore_errors=ignore_errors)
            isempty(depths) && continue
            push!(series_data, (model_name, prompt_setting, depths, rates, totals))
            append!(all_depths, depths)
        end

        isempty(all_depths) && throw(ArgumentError("No temporal-depth statistics could be computed from the available zero-shot/few-shot result files."))

        x_values = sort(unique(all_depths))
        xtick_values = collect(minimum(x_values):maximum(x_values))
        marker_shapes = Dict(
            "gpt-5.4" => :circle,
            "gpt-5.5" => :rect,
            "claude-opus-4-7" => :diamond,
            "deepseek-v4-flash" => :utriangle,
            "mistral-medium-latest" => :dtriangle,
        )

        subplot_list = Plots.Plot[]
        for prompt_setting in ["zeroshot", "fewshot"]
            prompt_series = [(model_name, depths, rates, totals) for (model_name, ps, depths, rates, totals) in series_data if ps == prompt_setting]

            local_p = plot(
                xlabel=(prompt_setting == "fewshot" ? "Temporal depth" : ""),
                ylabel=(prompt_setting == "zeroshot" ? "Zero-shot success rate" : "Few-shot success rate"),
                title="",
                legend=(prompt_setting == "zeroshot" ? :topright : false),
                ylim=(0.0, 1.0),
                xlims=(minimum(x_values) - 0.5, maximum(x_values) + 0.5),
                xticks=xtick_values,
                framestyle=:box,
                grid=true,
                guidefontsize=28,
                tickfontsize=17,
                titlefontsize=18,
                legendfontsize=20,
                guidefontweight=:bold,
                left_margin=16Plots.mm,
                right_margin=16Plots.mm,
                bottom_margin=12Plots.mm,
                top_margin=8Plots.mm,
            )

            for model_name in ZERO_VS_FEWSHOT_MODELS
                matching = [item for item in prompt_series if item[1] == model_name]
                isempty(matching) && continue
                _, depths, rates, totals = matching[1]
                marker_shape = get(marker_shapes, model_name, :circle)
                plot!(
                    local_p,
                    depths,
                    rates;
                    marker=marker_shape,
                    markersize=5,
                    markerstrokewidth=0.4,
                    linewidth=2.3,
                    label=(prompt_setting == "zeroshot" ? model_name : ""),
                )
            end

            push!(subplot_list, local_p)
        end

        p = plot(
            subplot_list...;
            layout=(2, 1),
            size=(1700, 1250),
            dpi=DEFAULT_PLOT_DPI,
        )
    else
        series_data = OrderedDict{String,Tuple{Vector{Int},Vector{Float64},Vector{Int}}}()
        all_depths = Int[]

        for stats in aggregated_stats
            model_name = String(stats["model"])
            entries = Vector{OrderedDict{String,Any}}(stats["entries"])
            ignore_errors = haskey(stats, "ignore_errors_in_success_rate") ? Bool(stats["ignore_errors_in_success_rate"]) : false
            depths, rates, totals = temporal_depth_bucket_success_rates(entries, dataset_index; ignore_errors=ignore_errors)
            series_data[model_name] = (depths, rates, totals)
            append!(all_depths, depths)
        end

        isempty(all_depths) && throw(ArgumentError("No temporal-depth statistics could be computed from the available result files."))

        x_values = sort(unique(all_depths))
        p = plot(
            xlabel="Temporal depth",
            ylabel="Success rate",
            title="",
            legend=:topright,
            ylim=(0.0, 1.0),
            xlims=(minimum(x_values) - 0.5, maximum(x_values) + 0.5),
            framestyle=:box,
            grid=true,
            size=(1450, 850),
            dpi=DEFAULT_PLOT_DPI,
            left_margin=16Plots.mm,
            right_margin=12Plots.mm,
            bottom_margin=12Plots.mm,
            top_margin=6Plots.mm,
            guidefontsize=18,
            tickfontsize=13,
            legendfontsize=12,
        )

        for (model_name, (depths, rates, totals)) in series_data
            rate_map = Dict{Int,Float64}(depths[i] => rates[i] for i in eachindex(depths))
            y_values = [get(rate_map, x, NaN) for x in x_values]
            plot!(p, x_values, y_values; marker=:circle, markersize=5, markerstrokewidth=0, linewidth=2.5, label=model_name)
        end
    end

    output_path = joinpath(output_dir, "all_models_success_rate_vs_temporal_depth.png")
    savefig(p, output_path)
    display(p)
    println("Saved combined temporal-depth figure: ", output_path)
    return output_path
end

function plot_all_models_success_rate_vs_nl_phrase_length(
    results_dir::String = DEFAULT_RESULTS_DIR;
    dataset_path::String = DEFAULT_DATASET_PATH,
    output_dir::String = DEFAULT_OUTPUT_DIR,
)
    ensure_directory(output_dir)
    dataset_index = load_dataset_index(dataset_path)
    aggregated_stats_raw = aggregate_all_result_files(results_dir)
    aggregated_stats = USE_ZERO_VS_FEWSHOT_SUBPLOTS ? aggregated_stats_raw : maybe_filter_to_paper_models(aggregated_stats_raw)

    if USE_ZERO_VS_FEWSHOT_SUBPLOTS
        filtered_stats = OrderedDict{String,Any}[]
        for stats in aggregated_stats
            if use_zero_vs_fewshot_result_file(stats)
                push!(filtered_stats, stats)
            end
        end

        isempty(filtered_stats) && throw(ArgumentError("Zero-vs-few-shot subplot mode is enabled, but no matching result files were found for the selected models."))

        series_data = Vector{Tuple{String,String,Vector{Int},Vector{Float64},Vector{Int}}}()
        all_lengths = Int[]

        for stats in filtered_stats
            model_name = String(stats["model"])
            prompt_setting = haskey(stats, "prompt_setting") ? String(stats["prompt_setting"]) : "zeroshot"
            entries = Vector{OrderedDict{String,Any}}(stats["entries"])
            ignore_errors = haskey(stats, "ignore_errors_in_success_rate") ? Bool(stats["ignore_errors_in_success_rate"]) : false
            lengths, rates, totals = nl_phrase_length_bucket_success_rates(entries, dataset_index; ignore_errors=ignore_errors)
            keep_idx = [i for i in eachindex(lengths) if lengths[i] <= 75]
            isempty(keep_idx) && continue
            lengths = lengths[keep_idx]
            rates = rates[keep_idx]
            totals = totals[keep_idx]
            push!(series_data, (model_name, prompt_setting, lengths, rates, totals))
            append!(all_lengths, lengths)
        end

        isempty(all_lengths) && throw(ArgumentError("No NL-phrase-length statistics could be computed from the available zero-shot/few-shot result files."))

        x_values = sort(unique(all_lengths))
        xtick_step = maximum(x_values) <= 20 ? 1 : (maximum(x_values) <= 40 ? 2 : (maximum(x_values) <= 80 ? 5 : 10))
        xtick_values = collect(minimum(x_values):xtick_step:maximum(x_values))
        marker_shapes = Dict(
            "gpt-5.4" => :circle,
            "gpt-5.5" => :rect,
            "claude-opus-4-7" => :diamond,
            "deepseek-v4-flash" => :utriangle,
            "mistral-medium-latest" => :dtriangle,
        )

        subplot_list = Plots.Plot[]
        for prompt_setting in ["zeroshot", "fewshot"]
            prompt_series = [(model_name, lengths, rates, totals) for (model_name, ps, lengths, rates, totals) in series_data if ps == prompt_setting]

            local_p = plot(
                xlabel=(prompt_setting == "fewshot" ? "NL phrase length (words)" : ""),
                ylabel=(prompt_setting == "zeroshot" ? "Zero-shot success rate" : "Few-shot success rate"),
                title="",
                legend=(prompt_setting == "zeroshot" ? :topright : false),
                ylim=(0.0, 1.0),
                xlims=(minimum(x_values) - 0.5, maximum(x_values) + 0.5),
                xticks=xtick_values,
                framestyle=:box,
                grid=true,
                guidefontsize=28,
                tickfontsize=17,
                titlefontsize=18,
                legendfontsize=20,
                guidefontweight=:bold,
                left_margin=16Plots.mm,
                right_margin=16Plots.mm,
                bottom_margin=12Plots.mm,
                top_margin=8Plots.mm,
            )

            for model_name in ZERO_VS_FEWSHOT_MODELS
                matching = [item for item in prompt_series if item[1] == model_name]
                isempty(matching) && continue
                _, lengths, rates, totals = matching[1]
                marker_shape = get(marker_shapes, model_name, :circle)
                plot!(
                    local_p,
                    lengths,
                    rates;
                    marker=marker_shape,
                    markersize=5,
                    markerstrokewidth=0.4,
                    linewidth=2.3,
                    label=(prompt_setting == "zeroshot" ? model_name : ""),
                )
            end

            push!(subplot_list, local_p)
        end

        p = plot(
            subplot_list...;
            layout=(2, 1),
            size=(1700, 1250),
            dpi=DEFAULT_PLOT_DPI,
        )
    else
        series_data = OrderedDict{String,Tuple{Vector{Int},Vector{Float64},Vector{Int}}}()
        all_lengths = Int[]

        for stats in aggregated_stats
            model_name = String(stats["model"])
            entries = Vector{OrderedDict{String,Any}}(stats["entries"])
            ignore_errors = haskey(stats, "ignore_errors_in_success_rate") ? Bool(stats["ignore_errors_in_success_rate"]) : false
            lengths, rates, totals = nl_phrase_length_bucket_success_rates(entries, dataset_index; ignore_errors=ignore_errors)
            keep_idx = [i for i in eachindex(lengths) if lengths[i] <= 75]
            isempty(keep_idx) && continue
            lengths = lengths[keep_idx]
            rates = rates[keep_idx]
            totals = totals[keep_idx]
            series_data[model_name] = (lengths, rates, totals)
            append!(all_lengths, lengths)
        end

        isempty(all_lengths) && throw(ArgumentError("No NL-phrase-length statistics could be computed from the available result files."))

        x_values = sort(unique(all_lengths))
        xtick_step = maximum(x_values) <= 20 ? 1 : (maximum(x_values) <= 40 ? 2 : (maximum(x_values) <= 80 ? 5 : 10))
        xtick_values = collect(minimum(x_values):xtick_step:maximum(x_values))
        p = plot(
            xlabel="NL phrase length (words)",
            ylabel="Success rate",
            title="",
            legend=:topright,
            ylim=(0.0, 1.0),
            xlims=(minimum(x_values) - 0.5, maximum(x_values) + 0.5),
            xticks=xtick_values,
            framestyle=:box,
            grid=true,
            size=(1450, 850),
            dpi=DEFAULT_PLOT_DPI,
            left_margin=16Plots.mm,
            right_margin=12Plots.mm,
            bottom_margin=12Plots.mm,
            top_margin=6Plots.mm,
            guidefontsize=18,
            tickfontsize=13,
            legendfontsize=12,
        )

        for (model_name, (lengths, rates, totals)) in series_data
            rate_map = Dict{Int,Float64}(lengths[i] => rates[i] for i in eachindex(lengths))
            y_values = [get(rate_map, x, NaN) for x in x_values]
            plot!(p, x_values, y_values; marker=:circle, markersize=5, markerstrokewidth=0, linewidth=2.5, label=model_name)
        end
    end

    output_path = joinpath(output_dir, "all_models_success_rate_vs_nl_phrase_length.png")
    savefig(p, output_path)
    display(p)
    println("Saved combined NL-phrase-length figure: ", output_path)
    return output_path
end

# =================================================================================================
# Combined six-panel plot
# =================================================================================================

function plot_all_models_success_rate_combined_six_panel(
    results_dir::String = DEFAULT_RESULTS_DIR;
    dataset_path::String = DEFAULT_DATASET_PATH,
    output_dir::String = DEFAULT_OUTPUT_DIR,
)
    USE_ZERO_VS_FEWSHOT_SUBPLOTS || throw(ArgumentError("This combined six-panel plot is intended for USE_ZERO_VS_FEWSHOT_SUBPLOTS = true."))

    ensure_directory(output_dir)

    dataset_index = load_dataset_index(dataset_path)
    aggregated_stats_raw = aggregate_all_result_files(results_dir)
    aggregated_stats = OrderedDict{String,Any}[]
    for stats in aggregated_stats_raw
        if use_zero_vs_fewshot_result_file(stats)
            push!(aggregated_stats, stats)
        end
    end
    isempty(aggregated_stats) && throw(ArgumentError("No matching zero-shot/few-shot result files were found for the selected models."))

    marker_shapes = Dict(
        "gpt-5.4" => :circle,
        "gpt-5.5" => :rect,
        "claude-opus-4-7" => :diamond,
        "deepseek-v4-flash" => :utriangle,
        "mistral-medium-latest" => :dtriangle,
    )
    model_colors = Dict(
        "gpt-5.4" => :dodgerblue,
        "gpt-5.5" => :orangered,
        "claude-opus-4-7" => :forestgreen,
        "deepseek-v4-flash" => :mediumorchid,
        "mistral-medium-latest" => :darkgoldenrod,
    )

    function collect_metric_series(metric_fn)
        series_data = Vector{Tuple{String,String,Vector{Int},Vector{Float64}}}()
        all_x = Int[]
        for stats in aggregated_stats
            model_name = String(stats["model"])
            prompt_setting = haskey(stats, "prompt_setting") ? String(stats["prompt_setting"]) : "zeroshot"
            entries = Vector{OrderedDict{String,Any}}(stats["entries"])
            ignore_errors = haskey(stats, "ignore_errors_in_success_rate") ? Bool(stats["ignore_errors_in_success_rate"]) : false
            xs, ys, _ = metric_fn(entries, dataset_index; ignore_errors=ignore_errors)
            isempty(xs) && continue
            push!(series_data, (model_name, prompt_setting, xs, ys))
            append!(all_x, xs)
        end
        isempty(all_x) && throw(ArgumentError("No data available for one of the combined six-panel metrics."))
        return series_data, sort(unique(all_x))
    end

    formula_series, formula_x = collect_metric_series(size_bucket_success_rates)
    formula_x = [x for x in formula_x if 1 <= x <= MAX_PLOTTED_FORMULA_SIZE]
    formula_xtick_step = maximum(formula_x) <= 20 ? 1 : (maximum(formula_x) <= 40 ? 2 : 5)
    formula_xticks = collect(minimum(formula_x):formula_xtick_step:maximum(formula_x))

    automaton_series, automaton_x = collect_metric_series(automaton_size_bucket_success_rates)
    automaton_xtick_step = maximum(automaton_x) <= 20 ? 1 : (maximum(automaton_x) <= 40 ? 2 : (maximum(automaton_x) <= 80 ? 5 : 10))
    automaton_xticks = collect(minimum(automaton_x):automaton_xtick_step:min(maximum(automaton_x), 50))

    temporal_series, temporal_x = collect_metric_series(temporal_depth_bucket_success_rates)

    # Force GPT-5.4 and GPT-5.5 to appear in the zero-shot temporal-depth subplot if they are missing.
    existing_zero_temporal_models = Set(model_name for (model_name, prompt_setting, _, _) in temporal_series if prompt_setting == "zeroshot")
    for forced_model in ["gpt-5.4", "gpt-5.5"]
        if !(forced_model in existing_zero_temporal_models)
            for stats in aggregated_stats
                model_name = String(stats["model"])
                model_name == forced_model || continue
                entries = Vector{OrderedDict{String,Any}}(stats["entries"])
                ignore_errors = haskey(stats, "ignore_errors_in_success_rate") ? Bool(stats["ignore_errors_in_success_rate"]) : false
                xs, ys, _ = temporal_depth_bucket_success_rates(entries, dataset_index; ignore_errors=ignore_errors)
                isempty(xs) && continue
                push!(temporal_series, (forced_model, "zeroshot", xs, ys))
                append!(temporal_x, xs)
                break
            end
        end
    end
    temporal_x = sort(unique(temporal_x))
    temporal_xticks = collect(minimum(temporal_x):maximum(temporal_x))

    function make_panel(series_data, x_values, xticks, xlabel::String, ylabel::String; show_legend::Bool=false, x_upper=nothing, show_ylabel::Bool=true, show_xlabel::Bool=true)
        upper = isnothing(x_upper) ? maximum(x_values) + 0.5 : x_upper + 0.5
        p = plot(
            xguide=(show_xlabel ? xlabel : ""),
            yguide=(show_ylabel ? ylabel : ""),
            title="",
            legend=(show_legend ? :topright : false),
            ylim=(0.0, 1.0),
            xlims=(minimum(x_values) - 0.5, upper),
            xticks=xticks,
            framestyle=:box,
            grid=true,
            guidefontsize=(show_legend ? 38 : 34),
            tickfontsize=(show_legend ? 24 : 22),
            titlefontsize=20,
            legendfontsize=(show_legend ? 30 : 22),
            guidefontweight=:bold,
            guide_position=:outer,
            left_margin=(show_ylabel ? (show_legend ? 58Plots.mm : 52Plots.mm) : 8Plots.mm),
            right_margin=(show_legend ? 20Plots.mm : 6Plots.mm),
            bottom_margin=(show_xlabel ? 30Plots.mm : 8Plots.mm),
            top_margin=8Plots.mm,
        )
        return p
    end

    p1 = make_panel(formula_series, formula_x, formula_xticks, "", "Zero-shot success rate"; show_ylabel=true, show_xlabel=false)
    p2 = make_panel(automaton_series, automaton_x, automaton_xticks, "", ""; x_upper=50, show_ylabel=false, show_xlabel=false)
    p3 = make_panel(temporal_series, temporal_x, temporal_xticks, "", ""; show_legend=true, show_ylabel=false, show_xlabel=false)
    p4 = make_panel(formula_series, formula_x, formula_xticks, "LTL formula size", "Few-shot success rate"; show_ylabel=true, show_xlabel=true)
    p5 = make_panel(automaton_series, automaton_x, automaton_xticks, "Automaton size", ""; x_upper=50, show_ylabel=false, show_xlabel=true)
    p6 = make_panel(temporal_series, temporal_x, temporal_xticks, "Temporal depth", ""; show_ylabel=false, show_xlabel=true)

    plot!(p4; xguide="LTL formula size", yguide="Few-shot success rate")
    plot!(p5; xguide="Automaton size")
    plot!(p6; xguide="Temporal depth")
    plot!(p1; yguide="Zero-shot success rate")

    for (model_name, prompt_setting, xs, ys) in formula_series
        keep_idx = [i for i in eachindex(xs) if 1 <= xs[i] <= MAX_PLOTTED_FORMULA_SIZE]
        isempty(keep_idx) && continue
        xsf = xs[keep_idx]
        ysf = ys[keep_idx]
        target = prompt_setting == "zeroshot" ? p1 : p4
        plot!(
            target,
            xsf,
            ysf;
            markershape=get(marker_shapes, model_name, :circle),
            color=get(model_colors, model_name, :black),
            markercolor=get(model_colors, model_name, :black),
            markerstrokecolor=:black,
            markersize=10,
            markerstrokewidth=1.0,
            linewidth=3.0,
            label="",
        )
    end

    for (model_name, prompt_setting, xs, ys) in automaton_series
        keep_idx = [i for i in eachindex(xs) if xs[i] <= 50]
        isempty(keep_idx) && continue
        xsf = xs[keep_idx]
        ysf = ys[keep_idx]
        target = prompt_setting == "zeroshot" ? p2 : p5
        plot!(
            target,
            xsf,
            ysf;
            markershape=get(marker_shapes, model_name, :circle),
            color=get(model_colors, model_name, :black),
            markercolor=get(model_colors, model_name, :black),
            markerstrokecolor=:black,
            markersize=10,
            markerstrokewidth=1.0,
            linewidth=3.0,
            label="",
        )
    end

    for (model_name, prompt_setting, xs, ys) in temporal_series
        if prompt_setting == "zeroshot"
            plot!(
                p3,
                xs,
                ys;
                markershape=get(marker_shapes, model_name, :circle),
                color=get(model_colors, model_name, :black),
                markercolor=get(model_colors, model_name, :black),
                markerstrokecolor=:black,
                markersize=10,
                markerstrokewidth=1.0,
                linewidth=3.2,
                label=model_name,
            )
        else
            plot!(
                p6,
                xs,
                ys;
                markershape=get(marker_shapes, model_name, :circle),
                color=get(model_colors, model_name, :black),
                markercolor=get(model_colors, model_name, :black),
                markerstrokecolor=:black,
                markersize=10,
                markerstrokewidth=1.0,
                linewidth=3.0,
                label="",
            )
        end
    end

    combined = plot(
        p1, p2, p3, p4, p5, p6;
        layout=(2, 3),
        size=(5400, 2100),
        dpi=DEFAULT_PLOT_DPI,
        bottom_margin=24Plots.mm,
        left_margin=42Plots.mm,
        right_margin=12Plots.mm,
        top_margin=12Plots.mm,
    )

    output_path = joinpath(output_dir, "all_models_success_rate_six_panel.png")
    savefig(combined, output_path)
    display(combined)
    println("Saved combined six-panel success-rate figure: ", output_path)
    return output_path
end

function plot_all_models_success_rate_combined_zero_shot_three_panel(
    results_dir::String = DEFAULT_RESULTS_DIR;
    dataset_path::String = DEFAULT_DATASET_PATH,
    output_dir::String = DEFAULT_OUTPUT_DIR,
)
    ensure_directory(output_dir)
    dataset_index = load_dataset_index(dataset_path)
    aggregated_stats_raw = aggregate_all_result_files(results_dir)

    filtered_stats = OrderedDict{String,Any}[]
    for stats in aggregated_stats_raw
        if use_zero_shot_only_result_file(stats)
            push!(filtered_stats, stats)
        end
    end

    isempty(filtered_stats) && throw(ArgumentError("Zero-shot-only three-panel mode is enabled, but no matching zero-shot result files were found for the selected models."))

    size_series = Vector{Tuple{String,Vector{Int},Vector{Float64},Vector{Int}}}()
    automaton_series = Vector{Tuple{String,Vector{Int},Vector{Float64},Vector{Int}}}()
    depth_series = Vector{Tuple{String,Vector{Int},Vector{Float64},Vector{Int}}}()

    all_sizes = Int[]
    all_automaton_sizes = Int[]
    all_depths = Int[]

    for stats in filtered_stats
        model_name = String(stats["model"])
        entries = Vector{OrderedDict{String,Any}}(stats["entries"])
        ignore_errors = haskey(stats, "ignore_errors_in_success_rate") ? Bool(stats["ignore_errors_in_success_rate"]) : false

        sizes, size_rates, size_totals = size_bucket_success_rates(entries, dataset_index; ignore_errors=ignore_errors)
        keep_idx = [i for i in eachindex(sizes) if 1 <= sizes[i] <= MAX_PLOTTED_FORMULA_SIZE]
        if !isempty(keep_idx)
            sizes = sizes[keep_idx]
            size_rates = size_rates[keep_idx]
            size_totals = size_totals[keep_idx]
            push!(size_series, (model_name, sizes, size_rates, size_totals))
            append!(all_sizes, sizes)
        end

        automaton_sizes, automaton_rates, automaton_totals = automaton_size_bucket_success_rates(entries, dataset_index; ignore_errors=ignore_errors)
        if !isempty(automaton_sizes)
            push!(automaton_series, (model_name, automaton_sizes, automaton_rates, automaton_totals))
            append!(all_automaton_sizes, automaton_sizes)
        end

        depths, depth_rates, depth_totals = temporal_depth_bucket_success_rates(entries, dataset_index; ignore_errors=ignore_errors)
        if !isempty(depths)
            push!(depth_series, (model_name, depths, depth_rates, depth_totals))
            append!(all_depths, depths)
        end
    end

    isempty(all_sizes) && throw(ArgumentError("No zero-shot formula-size statistics could be computed."))
    isempty(all_automaton_sizes) && throw(ArgumentError("No zero-shot automaton-size statistics could be computed."))
    isempty(all_depths) && throw(ArgumentError("No zero-shot temporal-depth statistics could be computed."))

    size_x_values = sort(unique(all_sizes))
    size_xtick_step = maximum(size_x_values) <= 20 ? 1 : (maximum(size_x_values) <= 40 ? 2 : 5)
    size_xticks = collect(minimum(size_x_values):size_xtick_step:maximum(size_x_values))

    automaton_x_values = sort(unique(all_automaton_sizes))
    automaton_xtick_step = maximum(automaton_x_values) <= 20 ? 1 : (maximum(automaton_x_values) <= 40 ? 2 : (maximum(automaton_x_values) <= 80 ? 5 : 10))
    automaton_xticks = collect(minimum(automaton_x_values):automaton_xtick_step:min(maximum(automaton_x_values), 50))

    depth_x_values = sort(unique(all_depths))
    depth_xticks = collect(minimum(depth_x_values):maximum(depth_x_values))

    marker_shapes = Dict(
        "gpt-5.4" => :circle,
        "gpt-5.5" => :rect,
        "claude-opus-4-7" => :diamond,
        "deepseek-v4-flash" => :utriangle,
        "mistral-medium-latest" => :dtriangle,
    )

    p_size = plot(
        xlabel="LTL formula size",
        ylabel="Zero-shot success rate",
        title="",
        legend=false,
        ylim=(0.0, 1.0),
        xlims=(minimum(size_x_values) - 0.5, maximum(size_x_values) + 0.5),
        xticks=size_xticks,
        framestyle=:box,
        grid=true,
        guidefontsize=20,
        tickfontsize=14,
        titlefontsize=18,
        left_margin=18Plots.mm,
        right_margin=12Plots.mm,
        bottom_margin=18Plots.mm,
        top_margin=8Plots.mm,
    )
    for model_name in ZERO_VS_FEWSHOT_MODELS
        matching = [item for item in size_series if item[1] == model_name]
        isempty(matching) && continue
        _, xs, ys, _ = matching[1]
        plot!(p_size, xs, ys; marker=get(marker_shapes, model_name, :circle), markersize=5, markerstrokewidth=0.4, linewidth=2.3, label="")
    end

    p_automaton = plot(
        xlabel="Automaton size",
        ylabel="Zero-shot success rate",
        title="",
        legend=false,
        ylim=(0.0, 1.0),
        xlims=(minimum(automaton_x_values) - 0.5, 50.5),
        xticks=automaton_xticks,
        framestyle=:box,
        grid=true,
        guidefontsize=20,
        tickfontsize=14,
        titlefontsize=18,
        left_margin=18Plots.mm,
        right_margin=12Plots.mm,
        bottom_margin=16Plots.mm,
        top_margin=8Plots.mm,
    )
    for model_name in ZERO_VS_FEWSHOT_MODELS
        matching = [item for item in automaton_series if item[1] == model_name]
        isempty(matching) && continue
        _, xs, ys, _ = matching[1]
        if model_name == "claude-opus-4-7"
            lower, upper = claude_band_bounds(ys, matching[1][4])
            plot!(p_automaton, xs, upper; fillrange=lower, fillalpha=0.18, linewidth=0, label="")
            plot!(p_automaton, xs, ys; marker=get(marker_shapes, model_name, :circle), markersize=5, markerstrokewidth=0.4, linewidth=2.3, label="")
        else
            plot!(p_automaton, xs, ys; marker=get(marker_shapes, model_name, :circle), markersize=5, markerstrokewidth=0.4, linewidth=2.3, label="")
        end
    end

    p_depth = plot(
        xlabel="Temporal depth",
        ylabel="Zero-shot success rate",
        title="",
        legend=:topright,
        ylim=(0.0, 1.0),
        xlims=(minimum(depth_x_values) - 0.5, maximum(depth_x_values) + 0.5),
        xticks=depth_xticks,
        framestyle=:box,
        grid=true,
        guidefontsize=20,
        tickfontsize=14,
        titlefontsize=18,
        legendfontsize=15,
        left_margin=18Plots.mm,
        right_margin=14Plots.mm,
        bottom_margin=16Plots.mm,
        top_margin=8Plots.mm,
    )
    for model_name in ZERO_VS_FEWSHOT_MODELS
        matching = [item for item in depth_series if item[1] == model_name]
        isempty(matching) && continue
        _, xs, ys, _ = matching[1]
        if model_name == "claude-opus-4-7"
            lower, upper = claude_band_bounds(ys, matching[1][4])
            plot!(p_depth, xs, upper; fillrange=lower, fillalpha=0.18, linewidth=0, label="")
            plot!(p_depth, xs, ys; marker=get(marker_shapes, model_name, :circle), markersize=5, markerstrokewidth=0.4, linewidth=2.3, label=model_name)
        else
            plot!(p_depth, xs, ys; marker=get(marker_shapes, model_name, :circle), markersize=5, markerstrokewidth=0.4, linewidth=2.3, label=model_name)
        end
    end

    p = plot(
        p_size,
        p_automaton,
        p_depth;
        layout=(1, 3),
        size=(2400, 500),
        dpi=DEFAULT_PLOT_DPI,
    )

    output_path = joinpath(output_dir, "all_models_success_rate_combined_zero_shot_three_panel.png")
    savefig(p, output_path)
    display(p)
    println("Saved combined zero-shot three-panel figure: ", output_path)
    return output_path
end

function plot_all_models_success_rate_combined_zero_shot_four_panel(
    results_dir::String = DEFAULT_RESULTS_DIR;
    dataset_path::String = DEFAULT_DATASET_PATH,
    output_dir::String = DEFAULT_OUTPUT_DIR,
)
    ensure_directory(output_dir)
    dataset_index = load_dataset_index(dataset_path)
    min_size_map = build_benchmark_min_equiv_formula_size_map(dataset_path)
    aggregated_stats_raw = aggregate_all_result_files(results_dir)

    filtered_stats = OrderedDict{String,Any}[]
    for stats in aggregated_stats_raw
        if use_zero_shot_only_result_file(stats)
            push!(filtered_stats, stats)
        end
    end

    isempty(filtered_stats) && throw(ArgumentError("Zero-shot-only four-panel mode is enabled, but no matching zero-shot result files were found for the selected models."))

    canon_series = Vector{Tuple{String,Vector{Int},Vector{Float64},Vector{Int}}}()
    depth_series = Vector{Tuple{String,Vector{Int},Vector{Float64},Vector{Int}}}()
    automaton_series = Vector{Tuple{String,Vector{Int},Vector{Float64},Vector{Int}}}()
    nl_length_series = Vector{Tuple{String,Vector{Int},Vector{Float64},Vector{Int}}}()

    all_canon_sizes = Int[]
    all_depths = Int[]
    all_automaton_sizes = Int[]
    all_nl_lengths = Int[]

    for stats in filtered_stats
        model_name = String(stats["model"])
        entries = Vector{OrderedDict{String,Any}}(stats["entries"])
        ignore_errors = haskey(stats, "ignore_errors_in_success_rate") ? Bool(stats["ignore_errors_in_success_rate"]) : false

        canon_sizes, canon_rates, canon_totals = canonicalized_size_bucket_success_rates(entries, dataset_index, min_size_map; ignore_errors=ignore_errors)
        keep_canon_idx = [i for i in eachindex(canon_sizes) if 1 <= canon_sizes[i] <= MAX_PLOTTED_FORMULA_SIZE]
        if !isempty(keep_canon_idx)
            canon_sizes = canon_sizes[keep_canon_idx]
            canon_rates = canon_rates[keep_canon_idx]
            canon_totals = canon_totals[keep_canon_idx]
            push!(canon_series, (model_name, canon_sizes, canon_rates, canon_totals))
            append!(all_canon_sizes, canon_sizes)
        end

        depths, depth_rates, depth_totals = temporal_depth_bucket_success_rates(entries, dataset_index; ignore_errors=ignore_errors)
        if !isempty(depths)
            push!(depth_series, (model_name, depths, depth_rates, depth_totals))
            append!(all_depths, depths)
        end

        automaton_sizes, automaton_rates, automaton_totals = automaton_size_bucket_success_rates(entries, dataset_index; ignore_errors=ignore_errors)
        if !isempty(automaton_sizes)
            push!(automaton_series, (model_name, automaton_sizes, automaton_rates, automaton_totals))
            append!(all_automaton_sizes, automaton_sizes)
        end

        nl_lengths, nl_rates, nl_totals = nl_phrase_length_bucket_success_rates(entries, dataset_index; ignore_errors=ignore_errors)
        keep_nl_idx = [i for i in eachindex(nl_lengths) if nl_lengths[i] <= 75]
        if !isempty(keep_nl_idx)
            nl_lengths = nl_lengths[keep_nl_idx]
            nl_rates = nl_rates[keep_nl_idx]
            nl_totals = nl_totals[keep_nl_idx]
            push!(nl_length_series, (model_name, nl_lengths, nl_rates, nl_totals))
            append!(all_nl_lengths, nl_lengths)
        end
    end

    isempty(all_canon_sizes) && throw(ArgumentError("No zero-shot minimum-equivalent-AST statistics could be computed."))
    isempty(all_depths) && throw(ArgumentError("No zero-shot temporal-depth statistics could be computed."))
    isempty(all_automaton_sizes) && throw(ArgumentError("No zero-shot automaton-size statistics could be computed."))
    isempty(all_nl_lengths) && throw(ArgumentError("No zero-shot NL phrase length statistics could be computed."))

    canon_x_values = sort(unique(all_canon_sizes))
    canon_xtick_step = maximum(canon_x_values) <= 20 ? 1 : (maximum(canon_x_values) <= 40 ? 2 : 5)
    canon_xticks = collect(minimum(canon_x_values):canon_xtick_step:maximum(canon_x_values))

    depth_x_values = sort(unique(all_depths))
    depth_xticks = collect(minimum(depth_x_values):maximum(depth_x_values))

    automaton_x_values = sort(unique(all_automaton_sizes))
    automaton_xtick_step = maximum(automaton_x_values) <= 20 ? 1 : (maximum(automaton_x_values) <= 40 ? 2 : (maximum(automaton_x_values) <= 80 ? 5 : 10))
    automaton_xticks = collect(minimum(automaton_x_values):automaton_xtick_step:min(maximum(automaton_x_values), 50))

    nl_x_values = sort(unique(all_nl_lengths))
    nl_xtick_step = maximum(nl_x_values) <= 20 ? 1 : (maximum(nl_x_values) <= 40 ? 2 : (maximum(nl_x_values) <= 80 ? 5 : 10))
    nl_xticks = collect(minimum(nl_x_values):nl_xtick_step:maximum(nl_x_values))

    marker_shapes = Dict(
        "gpt-5.4" => :circle,
        "gpt-5.5" => :rect,
        "claude-opus-4-7" => :diamond,
        "deepseek-v4-flash" => :utriangle,
        "mistral-medium-latest" => :dtriangle,
    )

    p_canon = plot(
        xlabel="Minimum AST size among equivalent LTLs in benchmark",
        ylabel="Zero-shot success rate",
        title="",
        legend=false,
        ylim=(0.0, 1.0),
        xlims=(minimum(canon_x_values) - 0.5, maximum(canon_x_values) + 0.5),
        xticks=canon_xticks,
        framestyle=:box,
        grid=true,
        guidefontsize=26,
        tickfontsize=18,
        titlefontsize=18,
        left_margin=18Plots.mm,
        right_margin=12Plots.mm,
        bottom_margin=16Plots.mm,
        top_margin=8Plots.mm,
    )
    for model_name in ZERO_VS_FEWSHOT_MODELS
        matching = [item for item in canon_series if item[1] == model_name]
        isempty(matching) && continue
        _, xs, ys, _ = matching[1]
        if model_name == "claude-opus-4-7"
            lower, upper = claude_band_bounds(ys, matching[1][4])
            plot!(p_canon, xs, upper; fillrange=lower, fillalpha=0.18, linewidth=0, label="")
            plot!(p_canon, xs, ys; marker=get(marker_shapes, model_name, :circle), markersize=5, markerstrokewidth=0.4, linewidth=2.3, label="")
        else
            plot!(p_canon, xs, ys; marker=get(marker_shapes, model_name, :circle), markersize=5, markerstrokewidth=0.4, linewidth=2.3, label="")
        end
    end

    p_depth = plot(
        xlabel="Temporal depth",
        ylabel="Zero-shot success rate",
        title="",
        legend=:topright,
        ylim=(0.0, 1.0),
        xlims=(minimum(depth_x_values) - 0.5, maximum(depth_x_values) + 0.5),
        xticks=depth_xticks,
        framestyle=:box,
        grid=true,
        guidefontsize=26,
        legendfontsize=20,
        tickfontsize=18,
        titlefontsize=18,
        left_margin=18Plots.mm,
        right_margin=12Plots.mm,
        bottom_margin=16Plots.mm,
        top_margin=8Plots.mm,
    )
    for model_name in ZERO_VS_FEWSHOT_MODELS
        matching = [item for item in depth_series if item[1] == model_name]
        isempty(matching) && continue
        _, xs, ys, _ = matching[1]
        if model_name == "claude-opus-4-7"
            lower, upper = claude_band_bounds(ys, matching[1][4])
            plot!(p_depth, xs, upper; fillrange=lower, fillalpha=0.18, linewidth=0, label="Claude ±1 SE")
            plot!(p_depth, xs, ys; marker=get(marker_shapes, model_name, :circle), markersize=5, markerstrokewidth=0.4, linewidth=2.3, label=model_name)
        else
            plot!(p_depth, xs, ys; marker=get(marker_shapes, model_name, :circle), markersize=5, markerstrokewidth=0.4, linewidth=2.3, label=model_name)
        end
    end

    p_automaton = plot(
        xlabel="Minimized Büchi size",
        ylabel="Zero-shot success rate",
        title="",
        legend=false,
        ylim=(0.0, 1.0),
        xlims=(minimum(automaton_x_values) - 0.5, 50.5),
        xticks=automaton_xticks,
        framestyle=:box,
        grid=true,
        guidefontsize=26,
        tickfontsize=18,
        titlefontsize=18,
        left_margin=18Plots.mm,
        right_margin=12Plots.mm,
        bottom_margin=16Plots.mm,
        top_margin=8Plots.mm,
    )
    for model_name in ZERO_VS_FEWSHOT_MODELS
        matching = [item for item in automaton_series if item[1] == model_name]
        isempty(matching) && continue
        _, xs, ys, _ = matching[1]
        if model_name == "claude-opus-4-7"
            lower, upper = claude_band_bounds(ys, matching[1][4])
            plot!(p_automaton, xs, upper; fillrange=lower, fillalpha=0.18, linewidth=0, label="")
            plot!(p_automaton, xs, ys; marker=get(marker_shapes, model_name, :circle), markersize=5, markerstrokewidth=0.4, linewidth=2.3, label="")
        else
            plot!(p_automaton, xs, ys; marker=get(marker_shapes, model_name, :circle), markersize=5, markerstrokewidth=0.4, linewidth=2.3, label="")
        end
    end

    p_nl = plot(
        xlabel="NL phrase length (words)",
        ylabel="Zero-shot success rate",
        title="",
        legend=:false,
        ylim=(0.0, 1.0),
        xlims=(minimum(nl_x_values) - 0.5, maximum(nl_x_values) + 0.5),
        xticks=nl_xticks,
        framestyle=:box,
        grid=true,
        guidefontsize=26,
        tickfontsize=18,
        titlefontsize=18,
        legendfontsize=15,
        left_margin=18Plots.mm,
        right_margin=14Plots.mm,
        bottom_margin=16Plots.mm,
        top_margin=8Plots.mm,
    )
    for model_name in ZERO_VS_FEWSHOT_MODELS
        matching = [item for item in nl_length_series if item[1] == model_name]
        isempty(matching) && continue
        _, xs, ys, _ = matching[1]
        if model_name == "claude-opus-4-7"
            lower, upper = claude_band_bounds(ys, matching[1][4])
            plot!(p_nl, xs, upper; fillrange=lower, fillalpha=0.18, linewidth=0, label="Claude ±1 SE")
            plot!(p_nl, xs, ys; marker=get(marker_shapes, model_name, :circle), markersize=5, markerstrokewidth=0.4, linewidth=2.3, label=model_name)
        else
            plot!(p_nl, xs, ys; marker=get(marker_shapes, model_name, :circle), markersize=5, markerstrokewidth=0.4, linewidth=2.3, label=model_name)
        end
    end

    p = plot(
        p_canon,
        p_depth,
        p_automaton,
        p_nl;
        layout=(4, 1),
        size=(1300, 2400),
        dpi=DEFAULT_PLOT_DPI,
    )

    output_path = joinpath(output_dir, "all_models_success_rate_combined_zero_shot_four_panel.png")
    savefig(p, output_path)
    display(p)
    println("Saved combined zero-shot four-panel figure: ", output_path)
    return output_path
end


function compare_all_models_and_plot_heatmap(
    results_dir::String = DEFAULT_RESULTS_DIR;
    output_dir::String = DEFAULT_OUTPUT_DIR,
)
    aggregated_stats = maybe_filter_to_paper_models(aggregate_all_result_files(results_dir))
    print_model_comparison_table(aggregated_stats)
    heatmap_path = plot_llm_paraphrase_heatmap(aggregated_stats; output_dir=output_dir)
    println("Heatmap written to: ", heatmap_path)
    if USE_ZERO_VS_FEWSHOT_SUBPLOTS
        if ZERO_VS_FEWSHOT_LAYOUT_MODE == "six_panel"
            six_panel_path = plot_all_models_success_rate_combined_six_panel(results_dir; dataset_path=DEFAULT_DATASET_PATH, output_dir=output_dir)
            println("Combined six-panel plot written to: ", six_panel_path)
        elseif ZERO_VS_FEWSHOT_LAYOUT_MODE == "zero_shot_only_three_panel"
            three_panel_path = plot_all_models_success_rate_combined_zero_shot_three_panel(results_dir; dataset_path=DEFAULT_DATASET_PATH, output_dir=output_dir)
            println("Combined zero-shot three-panel plot written to: ", three_panel_path)
        elseif ZERO_VS_FEWSHOT_LAYOUT_MODE == "zero_shot_only_four_panel"
            four_panel_path = plot_all_models_success_rate_combined_zero_shot_four_panel(results_dir; dataset_path=DEFAULT_DATASET_PATH, output_dir=output_dir)
            println("Combined zero-shot four-panel plot written to: ", four_panel_path)
        else
            throw(ArgumentError("Unsupported ZERO_VS_FEWSHOT_LAYOUT_MODE=$(ZERO_VS_FEWSHOT_LAYOUT_MODE). Allowed values are `six_panel`, `zero_shot_only_three_panel`, and `zero_shot_only_four_panel`."))
        end
    else
        size_plot_path = plot_all_models_success_rate_vs_formula_size(results_dir; output_dir=output_dir)
        println("Combined size plot written to: ", size_plot_path)
        automaton_size_plot_path = plot_all_models_success_rate_vs_automaton_size(results_dir; output_dir=output_dir)
        println("Combined automaton-size plot written to: ", automaton_size_plot_path)
        temporal_depth_plot_path = plot_all_models_success_rate_vs_temporal_depth(results_dir; output_dir=output_dir)
        println("Combined temporal-depth plot written to: ", temporal_depth_plot_path)
        nl_length_plot_path = plot_all_models_success_rate_vs_nl_phrase_length(results_dir; dataset_path=DEFAULT_DATASET_PATH, output_dir=output_dir)
        println("Combined NL-phrase-length plot written to: ", nl_length_plot_path)
    end
    return nothing
end



function study_result_file(
    results_path::String;
    dataset_path::String = DEFAULT_DATASET_PATH,
    output_dir::String = DEFAULT_OUTPUT_DIR,
)
    dataset_index = load_dataset_index(dataset_path)
    _, entries = load_result_entries(results_path)
    print_summary(results_path, entries)
    plot_success_rate_vs_formula_size(entries, dataset_index; results_path=results_path, output_dir=output_dir)
    plot_success_rate_vs_benchmark_min_equiv_formula_size(entries, dataset_index; results_path=results_path, dataset_path=DEFAULT_DATASET_PATH, output_dir=output_dir)
    plot_success_rate_vs_automaton_size(entries, dataset_index; results_path=results_path, output_dir=output_dir)
    plot_success_rate_vs_temporal_depth(entries, dataset_index; results_path=results_path, output_dir=output_dir)
    plot_success_rate_vs_nl_phrase_length(entries, dataset_index; results_path=results_path, output_dir=output_dir)
    plot_nl_phrase_length_distribution(entries, dataset_index; results_path=results_path, output_dir=output_dir)
end

function study_all_result_files(
    results_dir::String = DEFAULT_RESULTS_DIR;
    dataset_path::String = DEFAULT_DATASET_PATH,
    output_dir::String = DEFAULT_OUTPUT_DIR,
)
    files = available_result_files(results_dir)
    for path in files
        println("================================================================================")
        study_result_file(path; dataset_path=dataset_path, output_dir=output_dir)
    end
end



function main()
    println("Loaded ResultStatistics.jl")
    println("Run `compare_all_models_and_plot_heatmap()` to compare overall success rates, plot the heatmap, and plot success rate vs LTL size, automaton size, temporal depth, and NL phrase length across models.")
    println("Run `plot_all_models_success_rate_vs_formula_size_nnf()` to generate the formula-size success-rate plot restricted to NNF formulas only.")
    println("Run `plot_success_rate_vs_benchmark_min_equiv_formula_size(...)` for one file, or `plot_all_models_success_rate_vs_benchmark_min_equiv_formula_size()` across models, to use the minimum size between a record and its `source_record_id` formula when that field exists.")
    println("This source-record-minimal size uses dataset metadata and avoids the previous pairwise equivalence pass over the full benchmark.")
    println("Run `plot_nnf_formula_size_distribution()` to plot the size distribution of only NNF formulas in the dataset.")
    println("Run `plot_nl_phrase_length_distribution(entries, dataset_index; results_path=...)` or `study_result_file(...)` to get a histogram of NL phrase length distribution for one result file.")
    println("Run `study_result_file(joinpath(@__DIR__, \"results\", \"GPT54.json\"))` for one detailed file report.")
    println("Run `add_equivalence_to_result_file(joinpath(@__DIR__, \"results\", \"t5_lletter_results.json\"))` before studying T5 result files that do not already contain equivalence labels.")
    println("Run `study_all_result_files()` to analyze all JSON files in the results folder one by one.")
    println("Set `USE_PAPER_MODEL_SUBSET = true` to plot only mistral-medium-latest, claude-opus-4-7, gpt-5.4, and gpt-5.5.")
    println("Set `USE_ZERO_VS_FEWSHOT_SUBPLOTS = true` to use the subplot mode.")
    println("Set `ZERO_VS_FEWSHOT_LAYOUT_MODE = \"six_panel\"` for the 2x3 zero-shot/few-shot figure, `ZERO_VS_FEWSHOT_LAYOUT_MODE = \"zero_shot_only_three_panel\"` for a 1x3 figure with only zero-shot results, or `ZERO_VS_FEWSHOT_LAYOUT_MODE = \"zero_shot_only_four_panel\"` for a 2x2 zero-shot figure with minimum equivalent AST size, temporal depth, minimized Büchi size, and NL phrase length.")
    
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end


