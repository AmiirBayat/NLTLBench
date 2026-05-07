using JSON3
using OrderedCollections
using HTTP
using Random
using Dates
using DotEnv

const DEFAULT_DATASET_PATH = joinpath(@__DIR__, "dataset", "DatasetWithNaturalNL_plus_simplified.json")
const DEFAULT_INPUT_FIELD = "natural_paraphrase"
const DEFAULT_OUTPUT_FIELD = "paraphrase_gemini-2.5-flash"
const DEFAULT_MODEL = "gemini-2.5-flash"
const DEFAULT_API_URL = "https://generativelanguage.googleapis.com/v1beta/models"
const DEFAULT_MAX_RETRIES = 5
const DEFAULT_RETRY_BASE_SECONDS = 2.0

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
# Gemini API helpers
# ----------------------------------------------------------------------------------------------

function get_gemini_api_key()
    isfile(joinpath(@__DIR__, ".env")) && DotEnv.load!(joinpath(@__DIR__, ".env"))

    api_key = get(ENV, "GEMINI_API_KEY", "")
    isempty(api_key) && throw(ArgumentError(
        "GEMINI_API_KEY is not set. Put it in a local .env file as GEMINI_API_KEY=... or export it in your shell before running this script."
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

function extract_gemini_text(response_json)
    if haskey(response_json, :candidates)
        candidates = response_json[:candidates]
    elseif haskey(response_json, "candidates")
        candidates = response_json["candidates"]
    else
        throw(ErrorException("Gemini response did not contain `candidates`."))
    end

    isempty(candidates) && throw(ErrorException("Gemini response contained no candidates."))
    first_candidate = candidates[1]

    content = haskey(first_candidate, :content) ? first_candidate[:content] : first_candidate["content"]
    parts = haskey(content, :parts) ? content[:parts] : content["parts"]

    text_chunks = String[]
    for part in parts
        if haskey(part, :text)
            push!(text_chunks, String(part[:text]))
        elseif haskey(part, "text")
            push!(text_chunks, String(part["text"]))
        end
    end

    isempty(text_chunks) && throw(ErrorException("Gemini response contained no text parts."))
    return strip(join(text_chunks, "\n"))
end

function should_retry_gemini_status(status::Integer)
    return status in (429, 500, 502, 503, 504)
end

function retry_delay_seconds(attempt::Int; base::Float64 = DEFAULT_RETRY_BASE_SECONDS)
    deterministic = base * (2.0 ^ (attempt - 1))
    jitter = rand() * base
    return deterministic + jitter
end

function request_gemini_paraphrase(
    natural_paraphrase::AbstractString;
    model::String = DEFAULT_MODEL,
    api_url::String = DEFAULT_API_URL,
    temperature::Float64 = 0.6,
    max_retries::Int = DEFAULT_MAX_RETRIES,
)
    api_key = get_gemini_api_key()
    prompt = build_paraphrase_prompt(natural_paraphrase)

    endpoint = "$(api_url)/$(model):generateContent"
    body = Dict(
        "contents" => [
            Dict(
                "parts" => [
                    Dict("text" => prompt),
                ],
            ),
        ],
        "generationConfig" => Dict(
            "temperature" => temperature,
        ),
    )

    headers = [
        "x-goog-api-key" => api_key,
        "Content-Type" => "application/json",
    ]

    last_error = nothing

    for attempt in 1:max_retries
        response = try
            HTTP.post(endpoint, headers, JSON3.write(body); status_exception=false)
        catch err
            last_error = err
            if attempt == max_retries
                rethrow(err)
            end
            delay = retry_delay_seconds(attempt)
            println("Gemini request failed on attempt $(attempt)/$(max_retries). Retrying in $(round(delay; digits=1)) seconds...")
            sleep(delay)
            continue
        end

        if response.status == 200
            parsed = JSON3.read(String(response.body))
            return extract_gemini_text(parsed)
        elseif should_retry_gemini_status(response.status) && attempt < max_retries
            delay = retry_delay_seconds(attempt)
            println("Gemini returned status $(response.status) on attempt $(attempt)/$(max_retries). Retrying in $(round(delay; digits=1)) seconds...")
            sleep(delay)
            last_error = ErrorException("Gemini API request failed with status $(response.status): $(String(response.body))")
            continue
        else
            throw(ErrorException(
                "Gemini API request failed with status $(response.status): $(String(response.body))"
            ))
        end
    end

    isnothing(last_error) || throw(last_error)
    throw(ErrorException("Gemini API request failed after $(max_retries) attempts."))
end

# ----------------------------------------------------------------------------------------------
# Dataset enrichment
# ----------------------------------------------------------------------------------------------

function update_dataset_with_gemini_paraphrases(
    dataset_path::String = DEFAULT_DATASET_PATH;
    input_field::String = DEFAULT_INPUT_FIELD,
    output_field::String = DEFAULT_OUTPUT_FIELD,
    overwrite::Bool = false,
    model::String = DEFAULT_MODEL,
    min_id::Int = 525,
)
    records = load_dataset(dataset_path)
    updated_count = 0
    skipped_count = 0
    failed_count = 0

    for record in records
        haskey(record, "id") || continue
        record_id = try
            Int(record["id"])
        catch
            continue
        end
        if record_id < min_id
            continue
        end

        haskey(record, input_field) || continue

        already_has_output = haskey(record, output_field) && !isempty(strip(String(record[output_field])))
        if !overwrite && already_has_output
            skipped_count += 1
            println("Skipping record ID ", record_id, " because it already has a Gemini paraphrase.")
            continue
        end

        source_text = String(record[input_field])
        println("Processing record ID ", record_id, "...")

        try
            paraphrase = request_gemini_paraphrase(source_text; model=model)
            record[output_field] = paraphrase
            updated_count += 1
            save_dataset(records, dataset_path)
            println("Saved Gemini paraphrase for record ID ", record_id, ".")
            sleep(1.0)
        catch err
            failed_count += 1
            println("Failed record ID ", record_id, ": ", err)
            println("Continuing to the next record.")
        end
    end

    println("Updated $(updated_count) entries with Gemini paraphrases.")
    println("Skipped $(skipped_count) existing entries.")
    println("Failed $(failed_count) entries.")
    println("Dataset path: $(dataset_path)")
end

function preview_gemini_paraphrases(
    dataset_path::String = DEFAULT_DATASET_PATH;
    input_field::String = DEFAULT_INPUT_FIELD,
    n::Int = 3,
    model::String = DEFAULT_MODEL,
    min_id::Int = 525,
)
    records = load_dataset(dataset_path)
    shown = 0

    for record in records
        haskey(record, "id") || continue
        record_id = try
            Int(record["id"])
        catch
            continue
        end
        if record_id < min_id
            continue
        end

        haskey(record, input_field) || continue

        source_text = String(record[input_field])
        paraphrase = request_gemini_paraphrase(source_text; model=model)

        println("Record ID: ", record_id)
        println("Source: ", source_text)
        println("Output: ", paraphrase)
        println()

        shown += 1
        shown >= n && break
    end
end

function main()
    println("Loaded Paraphrase_gemini.jl")
    println("Run `preview_gemini_paraphrases(min_id=525)` to inspect a few examples from record id 525 onward.")
    println("Run `update_dataset_with_gemini_paraphrases(overwrite=false, min_id=525)` to write them into the dataset from record id 525 onward.")
    println("Set GEMINI_API_KEY in your shell or place it in a local .env file.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end