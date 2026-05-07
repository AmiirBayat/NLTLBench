include("LTLEquivalence.jl")

using JSON3
using OrderedCollections
using HTTP
using DotEnv
using Dates

const DEFAULT_DATASET_PATH = joinpath(@__DIR__, "dataset", "DatasetWithNaturalNL_plus_simplified.json")
const DEFAULT_RESULTS_PATH = joinpath(@__DIR__, "results", "GPT55.json")
const DEFAULT_MODEL = "gpt-5.5"
const DEFAULT_API_URL = "https://api.openai.com/v1/responses"
const DEFAULT_INPUT_FIELDS = [
    "natural_paraphrase",
    "paraphrase_gpt5.4-mini",
    "paraphrase_gemini-2.5-flash",
    "paraphrase_deepseek",
]

# ----------------------------------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------------------------------

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

function save_results(results_obj::OrderedDict{String,Any}, results_path::String = DEFAULT_RESULTS_PATH)
    ensure_parent_directory(results_path)
    open(results_path, "w") do io
        JSON3.pretty(io, results_obj)
        write(io, "\n")
    end
end

function initial_results_object(
    dataset_path::String = DEFAULT_DATASET_PATH;
    model::String = DEFAULT_MODEL,
    input_fields::Vector{String} = DEFAULT_INPUT_FIELDS,
)
    return OrderedDict(
        "dataset_path" => dataset_path,
        "translation_model" => model,
        "input_fields" => input_fields,
        "created_at" => string(now()),
        "updated_at" => string(now()),
        "results" => OrderedDict[],
    )
end

function load_results(
    results_path::String = DEFAULT_RESULTS_PATH;
    dataset_path::String = DEFAULT_DATASET_PATH,
    model::String = DEFAULT_MODEL,
    input_fields::Vector{String} = DEFAULT_INPUT_FIELDS,
)
    if !isfile(results_path)
        return initial_results_object(dataset_path; model=model, input_fields=input_fields)
    end

    content = strip(read(results_path, String))
    isempty(content) && return initial_results_object(dataset_path; model=model, input_fields=input_fields)

    parsed = JSON3.read(content)
    results_obj = OrderedDict{String,Any}(String(k) => v for (k, v) in pairs(parsed))
    haskey(results_obj, "results") || (results_obj["results"] = OrderedDict[])
    return results_obj
end

# ----------------------------------------------------------------------------------------------
# OpenAI API helpers
# ----------------------------------------------------------------------------------------------

function get_openai_api_key()
    isfile(joinpath(@__DIR__, ".env")) && DotEnv.load!(joinpath(@__DIR__, ".env"))

    api_key = get(ENV, "OPENAI_API_KEY", "")
    isempty(api_key) && throw(ArgumentError(
        "OPENAI_API_KEY is not set. Put it in a local .env file as OPENAI_API_KEY=... or export it in your shell before running this script."
    ))
    return api_key
end

function build_translation_prompt(sentence::AbstractString)
    return join([
        "You translate natural language statements into Linear Temporal Logic (LTL).",
        "",
        "Output rules:",
        "- Output only the LTL formula.",
        "- Do not add explanations, comments, labels, or code fences.",
        "- Keep proposition names exactly as they appear (prop_1, prop_2, ...).",
        "- Preserve the exact logical and temporal meaning.",
        "- Do not introduce domain assumptions.",
        "- Use standard LTL operators such as !, &, |, ->, <->, X, F, G, U when needed.",
        "",
        "Natural language statement:",
        String(sentence),
        "",
        "LTL:",
    ], "\n")
end

function extract_output_text(response_json)
    if haskey(response_json, :output_text)
        return String(response_json[:output_text])
    elseif haskey(response_json, "output_text")
        return String(response_json["output_text"])
    end

    key = haskey(response_json, :output) ? :output : (haskey(response_json, "output") ? "output" : nothing)
    isnothing(key) && throw(ErrorException("OpenAI response did not contain `output_text` or `output`."))

    outputs = response_json[key]
    for item in outputs
        if (haskey(item, :type) && item[:type] == "message") || (haskey(item, "type") && item["type"] == "message")
            content_key = haskey(item, :content) ? :content : "content"
            for part in item[content_key]
                part_type = haskey(part, :type) ? part[:type] : part["type"]
                if part_type in ("output_text", "text")
                    text_key = haskey(part, :text) ? :text : "text"
                    return String(part[text_key])
                end
            end
        end
    end

    throw(ErrorException("Could not extract model text from OpenAI response."))
end

function normalize_formula_text(text::AbstractString)
    s = strip(String(text))

    if startswith(s, "```")
        parts = split(s, "```")
        for part in parts
            cleaned = strip(replace(part, "ltl" => "", "LTL" => ""))
            if !isempty(cleaned)
                s = cleaned
                break
            end
        end
    end

    if startswith(s, "LTL:")
        s = strip(s[5:end])
    elseif startswith(s, "ltl:")
        s = strip(s[5:end])
    end

    return strip(s)
end

function request_gpt54_translation(
    sentence::AbstractString;
    model::String = DEFAULT_MODEL,
    api_url::String = DEFAULT_API_URL,
    temperature::Float64 = 0.0,
)
    api_key = get_openai_api_key()
    prompt = build_translation_prompt(sentence)

    body = Dict(
        "model" => model,
        "input" => prompt,
    )

    headers = [
        "Authorization" => "Bearer $(api_key)",
        "Content-Type" => "application/json",
    ]

    response = HTTP.post(api_url, headers, JSON3.write(body))
    response.status == 200 || throw(ErrorException(
        "OpenAI API request failed with status $(response.status): $(String(response.body))"
    ))

    parsed = JSON3.read(String(response.body))
    return normalize_formula_text(extract_output_text(parsed))
end

# ----------------------------------------------------------------------------------------------
# Result construction
# ----------------------------------------------------------------------------------------------

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

function result_key(record_id, field::AbstractString)
    return "$(record_id)||$(field)"
end

function existing_result_keys(results_obj::OrderedDict{String,Any})
    keys_set = Set{String}()
    for entry in results_obj["results"]
        status = haskey(entry, "status") ? String(entry["status"]) : (haskey(entry, :status) ? String(entry[:status]) : "")
        if status != "ok"
            continue
        end

        if (haskey(entry, "record_id") || haskey(entry, :record_id)) && (haskey(entry, "input_field") || haskey(entry, :input_field))
            rid = haskey(entry, "record_id") ? entry["record_id"] : entry[:record_id]
            field = haskey(entry, "input_field") ? String(entry["input_field"]) : String(entry[:input_field])
            push!(keys_set, result_key(rid, field))
        end
    end
    return keys_set
end

function append_result!(results_obj::OrderedDict{String,Any}, entry::OrderedDict{String,Any})
    existing_entries = OrderedDict{String,Any}[]
    for item in results_obj["results"]
        push!(existing_entries, OrderedDict{String,Any}(String(k) => v for (k, v) in pairs(item)))
    end

    push!(existing_entries, entry)
    results_obj["results"] = existing_entries
    results_obj["updated_at"] = string(now())
    return results_obj
end

function make_result_entry(
    record::OrderedDict{String,Any},
    input_field::String,
    source_text::AbstractString,
    predicted_ltl::Union{Nothing,AbstractString},
    equivalent::Union{Nothing,Bool},
    status::String;
    translation_model::String = DEFAULT_MODEL,
    error_message::Union{Nothing,String} = nothing,
)
    entry = OrderedDict{String,Any}(
        "record_id" => get(record, "id", nothing),
        "input_field" => input_field,
        "source_model" => paraphrase_source_model(input_field),
        "translation_model" => translation_model,
        "source_text" => String(source_text),
        "original_ltl" => String(get(record, "LTL", "")),
        "predicted_ltl" => isnothing(predicted_ltl) ? nothing : String(predicted_ltl),
        "equivalent" => equivalent,
        "status" => status,
        "timestamp" => string(now()),
    )

    !isnothing(error_message) && (entry["error"] = error_message)
    return entry
end

# ----------------------------------------------------------------------------------------------
# Evaluation loop
# ----------------------------------------------------------------------------------------------

function evaluate_gpt54_on_benchmark(
    dataset_path::String = DEFAULT_DATASET_PATH;
    results_path::String = DEFAULT_RESULTS_PATH,
    input_fields::Vector{String} = DEFAULT_INPUT_FIELDS,
    model::String = DEFAULT_MODEL,
)
    dataset = load_dataset(dataset_path)
    results_obj = load_results(results_path; dataset_path=dataset_path, model=model, input_fields=input_fields)
    seen = existing_result_keys(results_obj)

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
                println("Skipping record ID ", get(record, "id", "?"), " field `", field, "` because it already exists in results.")
                continue
            end

            println("Processing record ID ", get(record, "id", "?"), " field `", field, "`...")

            try
                predicted_ltl = request_gpt54_translation(source_text; model=model)
                original_ltl = String(record["LTL"])
                equivalent = are_equivalent(predicted_ltl, original_ltl)

                entry = make_result_entry(
                    record,
                    field,
                    source_text,
                    predicted_ltl,
                    equivalent,
                    "ok";
                    translation_model=model,
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
                    nothing,
                    nothing,
                    "error";
                    translation_model=model,
                    error_message=sprint(showerror, err),
                )
                append_result!(results_obj, entry)
                save_results(results_obj, results_path)
                push!(seen, key)
                processed_count += 1
                println("Saved error result for record ID ", get(record, "id", "?"), " field `", field, "`.")
            end
        end
    end

    println("Processed $(processed_count) evaluations.")
    println("Skipped $(skipped_count) evaluations already present in results.")
    println("Results path: $(results_path)")
end

function summarize_results(results_path::String = DEFAULT_RESULTS_PATH)
    results_obj = load_results(results_path)
    entries = results_obj["results"]

    total = length(entries)
    ok_count = 0
    error_count = 0
    equivalent_count = 0

    by_field = OrderedDict{String,OrderedDict{String,Int}}()

    for entry in entries
        field = haskey(entry, "input_field") ? String(entry["input_field"]) : String(entry[:input_field])
        status = haskey(entry, "status") ? String(entry["status"]) : String(entry[:status])
        equivalent = haskey(entry, "equivalent") ? entry["equivalent"] : get(entry, :equivalent, nothing)

        if !haskey(by_field, field)
            by_field[field] = OrderedDict("total" => 0, "ok" => 0, "error" => 0, "equivalent" => 0)
        end

        by_field[field]["total"] += 1
        if status == "ok"
            ok_count += 1
            by_field[field]["ok"] += 1
            if equivalent === true
                equivalent_count += 1
                by_field[field]["equivalent"] += 1
            end
        else
            error_count += 1
            by_field[field]["error"] += 1
        end
    end

    println("Results path: ", results_path)
    println("Total evaluated: ", total)
    println("Successful translations: ", ok_count)
    println("Errors: ", error_count)
    println("Equivalent among successful: ", equivalent_count)
    println()

    for (field, stats) in by_field
        println("Field: ", field)
        println("  total: ", stats["total"])
        println("  ok: ", stats["ok"])
        println("  error: ", stats["error"])
        println("  equivalent: ", stats["equivalent"])
        println()
    end
end

function main()
    println("Loaded GPT_Performance.jl")
    println("Run `evaluate_gpt54_on_benchmark()` to evaluate GPT-5.5 on the dataset.")
    println("Run `summarize_results()` to print a results summary.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end