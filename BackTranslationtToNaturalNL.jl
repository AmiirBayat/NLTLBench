include("GenerateLTL.jl")
include("Filter.jl")

using JSON3
using OrderedCollections
using HTTP
using DotEnv

const DEFAULT_DATASET_PATH = joinpath(@__DIR__, "dataset", "DatasetWithNaturalNL_plus_simplified.json")
const DEFAULT_OUTPUT_FIELD = "natural_paraphrase"
const DEFAULT_MODEL = "claude-sonnet-4-5"
const DEFAULT_API_URL = "https://api.anthropic.com/v1/messages"

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
# Anthropic API helpers
# ----------------------------------------------------------------------------------------------

function get_anthropic_api_key()
    # Load from a local .env file if present, then fall back to the process environment.
    isfile(joinpath(@__DIR__, ".env")) && DotEnv.load!(joinpath(@__DIR__, ".env"))

    api_key = get(ENV, "ANTHROPIC_API_KEY", "")
    isempty(api_key) && throw(ArgumentError(
        "ANTHROPIC_API_KEY is not set. Put it in a local .env file as ANTHROPIC_API_KEY=... or export it in your shell before running this script."
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
    content = if haskey(response_json, :content)
        response_json[:content]
    elseif haskey(response_json, "content")
        response_json["content"]
    else
        throw(ErrorException("Anthropic response did not contain `content`."))
    end

    text_chunks = String[]
    for part in content
        part_type = haskey(part, :type) ? String(part[:type]) : String(part["type"])
        if part_type == "text"
            if haskey(part, :text)
                push!(text_chunks, String(part[:text]))
            elseif haskey(part, "text")
                push!(text_chunks, String(part["text"]))
            end
        end
    end

    isempty(text_chunks) && throw(ErrorException("Anthropic response contained no text content."))
    return strip(join(text_chunks, "\n"))
end

function request_natural_paraphrase(
    ltl::AbstractString,
    back_translation::AbstractString;
    model::String = DEFAULT_MODEL,
    api_url::String = DEFAULT_API_URL,
)
    api_key = get_anthropic_api_key()
    prompt = build_paraphrase_prompt(ltl, back_translation)

    body = Dict(
        "model" => model,
        "max_tokens" => 512,
        "temperature" => 0.3,
        "messages" => [
            Dict(
                "role" => "user",
                "content" => prompt,
            )
        ],
    )

    headers = [
        "x-api-key" => api_key,
        "anthropic-version" => "2023-06-01",
        "content-type" => "application/json",
    ]

    response = HTTP.post(api_url, headers, JSON3.write(body))
    response.status == 200 || throw(ErrorException("Anthropic API request failed with status $(response.status): $(String(response.body))"))

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
    min_id::Int = 525,
)
    records = load_dataset(dataset_path)
    updated_count = 0
    skipped_count = 0
    error_count = 0

    for record in records
        if !haskey(record, "id")
            continue
        end
        record_id = try
            Int(record["id"])
        catch
            continue
        end
        if record_id < min_id
            continue
        end
        haskey(record, "LTL") || continue
        haskey(record, "back_translation") || continue

        already_has_output = haskey(record, output_field) && !isempty(strip(String(record[output_field])))
        if !overwrite && already_has_output
            skipped_count += 1
            continue
        end

        ltl = String(record["LTL"])
        back_translation = String(record["back_translation"])

        println("Processing record id ", record_id, "...")
        try
            record[output_field] = request_natural_paraphrase(ltl, back_translation; model=model)
            updated_count += 1
            save_dataset(records, dataset_path)
            println("Saved natural paraphrase for record id ", record_id)
        catch err
            error_count += 1
            println("Failed on record id ", record_id, ": ", sprint(showerror, err))
        end
    end

    println("Updated $(updated_count) entries with natural paraphrases.")
    println("Skipped $(skipped_count) existing entries.")
    println("Encountered $(error_count) errors.")
    println("Dataset path: $(dataset_path)")
end

function preview_natural_paraphrases(
    dataset_path::String = DEFAULT_DATASET_PATH;
    n::Int = 3,
    model::String = DEFAULT_MODEL,
    min_id::Int = 525,
)
    records = load_dataset(dataset_path)
    shown = 0

    for record in records
        if !haskey(record, "id")
            continue
        end
        record_id = try
            Int(record["id"])
        catch
            continue
        end
        if record_id < min_id
            continue
        end
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
    println("Run `preview_natural_paraphrases(min_id=525)` to inspect examples for records with id 525 and later.")
    println("Run `update_dataset_with_natural_paraphrases(overwrite=false, min_id=525)` to write them into DatasetWithNaturalNL_plus_simplified.json.")
    println("Set ANTHROPIC_API_KEY in your shell or place it in a local .env file.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end