include("GenerateLTL.jl")
include("Filter.jl")

using JSON3
using OrderedCollections
using HTTP
using DotEnv

const DEFAULT_DATASET_PATH = joinpath(@__DIR__, "dataset", "DatasetWithNaturalNL.json")
const DEFAULT_OUTPUT_FIELD = "natural_paraphrase"
const DEFAULT_MODEL = "gpt-5.2"
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
    # Load from a local .env file if present, then fall back to the process environment.
    isfile(joinpath(@__DIR__, ".env")) && DotEnv.load!(joinpath(@__DIR__, ".env"))

    api_key = get(ENV, "OPENAI_API_KEY", "")
    isempty(api_key) && throw(ArgumentError(
        "OPENAI_API_KEY is not set. Put it in a local .env file as OPENAI_API_KEY=... or export it in your shell before running this script."
    ))
    return api_key
end

function build_paraphrase_prompt(ltl::AbstractString, back_translation::AbstractString)
    return join([
        "Rewrite the back-translation into natural English while preserving the exact meaning of the LTL formula.",
        "",
        "Important constraints:",
        "- Keep proposition names unchanged (e.g., prop_1, prop_2).",
        "- Do not introduce domain-specific interpretations.",
        "- Do not add, remove, weaken, or strengthen conditions.",
        "- Preserve the exact temporal meaning and logical structure.",
        "- Prefer clear and semantically transparent phrasing over highly idiomatic expressions.",
        "- Avoid ambiguity.",
        "",
        "Here are some examples:",
        "",
        "Example 1:",
        "Input:",
        "LTL: F(prop_1)",
        "Back-translation: eventually, (prop_1 is true)",
        "",
        "Output:",
        "prop_1 eventually becomes true.",
        "",
        "---",
        "",
        "Example 2:",
        "Input:",
        "LTL: G(F(prop_1))",
        "Back-translation: always, (eventually, (prop_1 is true))",
        "",
        "Output:",
        "prop_1 becomes true infinitely often.",
        "",
        "---",
        "",
        "Example 3:",
        "Input:",
        "LTL: !(F(prop_1))",
        "Back-translation: it is not the case that (eventually, (prop_1 is true))",
        "",
        "Output:",
        "prop_1 never becomes true.",
        "",
        "---",
        "",
        "Example 4:",
        "Input:",
        "LTL: (prop_1 -> G(prop_2))",
        "Back-translation: if (prop_1 is true), then (always, (prop_2 is true))",
        "",
        "Output:",
        "If prop_1 is true, then prop_2 must always remain true.",
        "",
        "---",
        "",
        "Example 5:",
        "Input:",
        "LTL: (prop_1 U prop_2)",
        "Back-translation: (prop_1 is true) holds until (prop_2 is true)",
        "",
        "Output:",
        "prop_1 remains true until prop_2 becomes true.",
        "",
        "---",
        "",
        "Example 6:",
        "Input:",
        "LTL: X(prop_1)",
        "Back-translation: in the next step, (prop_1 is true)",
        "",
        "Output:",
        "In the next step, prop_1 is true.",
        "",
        "Now rewrite the following case.",
        "",
        "Input:",
        "LTL: $(String(ltl))",
        "Back-translation: $(String(back_translation))",
        "",
        "Output:",
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

function request_natural_paraphrase(
    ltl::AbstractString,
    back_translation::AbstractString;
    model::String = DEFAULT_MODEL,
    api_url::String = DEFAULT_API_URL,
)
    api_key = get_openai_api_key()
    prompt = build_paraphrase_prompt(ltl, back_translation)

    body = Dict(
        "model" => model,
        "input" => prompt,
        "temperature" => 0.3,
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

function update_dataset_with_natural_paraphrases(
    dataset_path::String = DEFAULT_DATASET_PATH;
    output_field::String = DEFAULT_OUTPUT_FIELD,
    overwrite::Bool = true,
    model::String = DEFAULT_MODEL,
)
    records = load_dataset(dataset_path)
    updated_count = 0
    skipped_count = 0

    for record in records
        haskey(record, "LTL") || continue
        haskey(record, "back_translation") || continue

        if !overwrite && haskey(record, output_field)
            skipped_count += 1
            continue
        end

        ltl = String(record["LTL"])
        back_translation = String(record["back_translation"])

        record[output_field] = request_natural_paraphrase(ltl, back_translation; model=model)
        updated_count += 1
    end

    save_dataset(records, dataset_path)

    println("Updated $(updated_count) entries with natural paraphrases.")
    println("Skipped $(skipped_count) existing entries.")
    println("Dataset path: $(dataset_path)")
end

function preview_natural_paraphrases(
    dataset_path::String = DEFAULT_DATASET_PATH;
    n::Int = 3,
    model::String = DEFAULT_MODEL,
)
    records = load_dataset(dataset_path)
    shown = 0

    for record in records
        haskey(record, "LTL") || continue
        haskey(record, "back_translation") || continue

        ltl = String(record["LTL"])
        back_translation = String(record["back_translation"])
        natural = request_natural_paraphrase(ltl, back_translation; model=model)

        println("LTL: ", ltl)
        println("Back-translation: ", back_translation)
        println("Natural paraphrase: ", natural)
        println()

        shown += 1
        shown >= n && break
    end
end

function main()
    println("Loaded BackTranslationtToNaturalNL.jl")
    println("Run `preview_natural_paraphrases()` to inspect a few examples.")
    println("Run `update_dataset_with_natural_paraphrases()` to write them into the dataset.")
    println("Set OPENAI_API_KEY in your shell or place it in a local .env file.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end