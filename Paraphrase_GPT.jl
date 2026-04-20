using JSON3
using OrderedCollections
using HTTP
using DotEnv

const DEFAULT_DATASET_PATH = joinpath(@__DIR__, "dataset", "DatasetWithNaturalNL.json")
const DEFAULT_OUTPUT_FIELD = "paraphrase"
const DEFAULT_MODEL_FIELD = "paraphrase_model"
const DEFAULT_MODEL = "gpt-5.4-mini"
const DEFAULT_API_URL = "https://api.openai.com/v1/responses"

# ----------------------------------------------------------------------------------------------
# Dataset I/O
# ----------------------------------------------------------------------------------------------

function load_dataset(dataset_path::String = DEFAULT_DATASET_PATH)
    if !isfile(dataset_path)
        throw(ArgumentError("Dataset file not found: $(dataset_path)"))
    end
    data = JSON3.read(read(dataset_path, String))
    return [OrderedDict{String,Any}(String(k) => v for (k, v) in pairs(record)) for record in data]
end

function save_dataset(records, dataset_path::String = DEFAULT_DATASET_PATH)
    open(dataset_path, "w") do io
        JSON3.pretty(io, records)
        write(io, "\n")
    end
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

function build_paraphrase_prompt(natural_paraphrase::AbstractString)
    return join([
        "You rewrite natural language statements derived from LTL formulas.",
        "",
        "Your goal is to produce ONE paraphrase that:",
        "- preserves the exact logical meaning,",
        "- remains faithful to the original statement,",
        "- sounds natural in English,",
        "- uses noticeably different wording and, when possible, a different sentence structure.",
        "",
        "Strict rules:",
        "1. Do not change the meaning.",
        "2. Do not add or remove conditions.",
        "3. Do not introduce domain assumptions.",
        "4. Keep proposition names unchanged (prop_1, prop_2, ...).",
        "5. Prefer a genuine rewording instead of minor edits.",
        "6. Avoid copying the original sentence structure unless necessary.",
        "7. Output only the paraphrased sentence.",
        "",
        "Input sentence:",
        String(natural_paraphrase),
        "",
        "Paraphrase:",
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

function request_paraphrase(
    natural_paraphrase::AbstractString;
    model::String = DEFAULT_MODEL,
    api_url::String = DEFAULT_API_URL,
    temperature::Float64 = 0.5,
)
    api_key = get_openai_api_key()
    prompt = build_paraphrase_prompt(natural_paraphrase)

    body = Dict(
        "model" => model,
        "input" => prompt,
        "temperature" => temperature,
    )

    headers = [
        "Authorization" => "Bearer $(api_key)",
        "Content-Type" => "application/json",
    ]

    response = HTTP.post(api_url, headers, JSON3.write(body))
    response.status == 200 || throw(ErrorException("OpenAI API request failed with status $(response.status): $(String(response.body))"))

    parsed = JSON3.read(String(response.body))
    return strip(extract_output_text(parsed))
end

# ----------------------------------------------------------------------------------------------
# Dataset enrichment
# ----------------------------------------------------------------------------------------------

function update_dataset_with_paraphrases(
    dataset_path::String = DEFAULT_DATASET_PATH;
    input_field::String = "natural_paraphrase",
    output_field::String = DEFAULT_OUTPUT_FIELD,
    model_field::String = DEFAULT_MODEL_FIELD,
    overwrite::Bool = true,
    model::String = DEFAULT_MODEL,
)
    records = load_dataset(dataset_path)
    updated_count = 0
    skipped_count = 0

    for (i, record) in enumerate(records)
        haskey(record, input_field) || continue

        if !overwrite && haskey(record, output_field)
            skipped_count += 1
            continue
        end

        source_text = String(record[input_field])
        paraphrase = request_paraphrase(source_text; model=model)

        record[output_field] = paraphrase
        record[model_field] = model
        updated_count += 1
    end

    save_dataset(records, dataset_path)

    println("Updated $(updated_count) entries with paraphrases.")
    println("Skipped $(skipped_count) existing entries.")
    println("Dataset path: $(dataset_path)")
end

function preview_paraphrases(
    dataset_path::String = DEFAULT_DATASET_PATH;
    input_field::String = "natural_paraphrase",
    n::Int = 3,
    model::String = DEFAULT_MODEL,
)
    records = load_dataset(dataset_path)
    shown = 0

    for (i, record) in enumerate(records)
        haskey(record, input_field) || continue

        source_text = String(record[input_field])
        paraphrase = request_paraphrase(source_text; model=model)

        println("Source: ", source_text)
        println("Output: ", paraphrase)
        println()

        shown += 1
        shown >= n && break
    end
end

function main()
    println("Loaded Paraphrase.jl")
    println("Run `preview_paraphrases()` to inspect a few examples.")
    println("Run `update_dataset_with_paraphrases()` to write them into the dataset.")
    println("Set OPENAI_API_KEY in your shell or place it in a local .env file.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end