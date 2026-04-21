using JSON3
using OrderedCollections
using HTTP
using DotEnv
using Random

const DEFAULT_DATASET_PATH = joinpath(@__DIR__, "dataset", "DatasetWithNaturalNL.json")
const DEFAULT_INPUT_FIELD = "natural_paraphrase"
const DEFAULT_OUTPUT_FIELD = "paraphrase_deepseek"
const DEFAULT_MODEL = "deepseek-chat"
const DEFAULT_API_URL = "https://api.deepseek.com/chat/completions"
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
# DeepSeek API helpers
# ----------------------------------------------------------------------------------------------

function get_deepseek_api_key()
    isfile(joinpath(@__DIR__, ".env")) && DotEnv.load!(joinpath(@__DIR__, ".env"))

    api_key = get(ENV, "DEEPSEEK_API_KEY", "")
    isempty(api_key) && throw(ArgumentError(
        "DEEPSEEK_API_KEY is not set. Put it in a local .env file as DEEPSEEK_API_KEY=... or export it in your shell before running this script."
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

function extract_deepseek_text(response_json)
    choices = if haskey(response_json, :choices)
        response_json[:choices]
    elseif haskey(response_json, "choices")
        response_json["choices"]
    else
        throw(ErrorException("DeepSeek response did not contain `choices`."))
    end

    isempty(choices) && throw(ErrorException("DeepSeek response contained no choices."))
    first_choice = choices[1]
    message = haskey(first_choice, :message) ? first_choice[:message] : first_choice["message"]

    if haskey(message, :content)
        return strip(String(message[:content]))
    elseif haskey(message, "content")
        return strip(String(message["content"]))
    else
        throw(ErrorException("DeepSeek response did not contain `message.content`."))
    end
end

function should_retry_deepseek_status(status::Integer)
    return status in (408, 409, 429, 500, 502, 503, 504)
end

function retry_delay_seconds(attempt::Int; base::Float64 = DEFAULT_RETRY_BASE_SECONDS)
    deterministic = base * (2.0 ^ (attempt - 1))
    jitter = rand() * base
    return deterministic + jitter
end

function request_deepseek_paraphrase(
    natural_paraphrase::AbstractString;
    model::String = DEFAULT_MODEL,
    api_url::String = DEFAULT_API_URL,
    temperature::Float64 = 0.6,
    max_retries::Int = DEFAULT_MAX_RETRIES,
)
    api_key = get_deepseek_api_key()
    prompt = build_paraphrase_prompt(natural_paraphrase)

    body = Dict(
        "model" => model,
        "messages" => [
            Dict(
                "role" => "system",
                "content" => "You paraphrase natural language statements derived from LTL formulas while preserving their exact meaning.",
            ),
            Dict(
                "role" => "user",
                "content" => prompt,
            ),
        ],
        "temperature" => temperature,
        "stream" => false,
    )

    headers = [
        "Authorization" => "Bearer $(api_key)",
        "Content-Type" => "application/json",
    ]

    last_error = nothing

    for attempt in 1:max_retries
        response = try
            HTTP.post(api_url, headers, JSON3.write(body); status_exception=false)
        catch err
            last_error = err
            if attempt == max_retries
                rethrow(err)
            end
            delay = retry_delay_seconds(attempt)
            println("DeepSeek request failed on attempt $(attempt)/$(max_retries). Retrying in $(round(delay; digits=1)) seconds...")
            sleep(delay)
            continue
        end

        if response.status == 200
            parsed = JSON3.read(String(response.body))
            return extract_deepseek_text(parsed)
        elseif should_retry_deepseek_status(response.status) && attempt < max_retries
            delay = retry_delay_seconds(attempt)
            println("DeepSeek returned status $(response.status) on attempt $(attempt)/$(max_retries). Retrying in $(round(delay; digits=1)) seconds...")
            sleep(delay)
            last_error = ErrorException("DeepSeek API request failed with status $(response.status): $(String(response.body))")
            continue
        else
            throw(ErrorException(
                "DeepSeek API request failed with status $(response.status): $(String(response.body))"
            ))
        end
    end

    isnothing(last_error) || throw(last_error)
    throw(ErrorException("DeepSeek API request failed after $(max_retries) attempts."))
end

# ----------------------------------------------------------------------------------------------
# Dataset enrichment
# ----------------------------------------------------------------------------------------------

function update_dataset_with_deepseek_paraphrases(
    dataset_path::String = DEFAULT_DATASET_PATH;
    input_field::String = DEFAULT_INPUT_FIELD,
    output_field::String = DEFAULT_OUTPUT_FIELD,
    overwrite::Bool = false,
    model::String = DEFAULT_MODEL,
)
    records = load_dataset(dataset_path)
    updated_count = 0
    skipped_count = 0
    failed_count = 0

    for record in records
        haskey(record, input_field) || continue

        if !overwrite && haskey(record, output_field)
            skipped_count += 1
            println("Skipping record ID ", get(record, "id", "?"), " because it already has a DeepSeek paraphrase.")
            continue
        end

        source_text = String(record[input_field])
        println("Processing record ID ", get(record, "id", "?"), "...")

        try
            paraphrase = request_deepseek_paraphrase(source_text; model=model)
            record[output_field] = paraphrase
            updated_count += 1
            save_dataset(records, dataset_path)
            println("Saved DeepSeek paraphrase for record ID ", get(record, "id", "?"), ".")
            sleep(1.0)
        catch err
            failed_count += 1
            println("Failed record ID ", get(record, "id", "?"), ": ", err)
            println("Continuing to the next record.")
        end
    end

    println("Updated $(updated_count) entries with DeepSeek paraphrases.")
    println("Skipped $(skipped_count) existing entries.")
    println("Failed $(failed_count) entries.")
    println("Dataset path: $(dataset_path)")
end

function preview_deepseek_paraphrases(
    dataset_path::String = DEFAULT_DATASET_PATH;
    input_field::String = DEFAULT_INPUT_FIELD,
    n::Int = 3,
    model::String = DEFAULT_MODEL,
)
    records = load_dataset(dataset_path)
    shown = 0

    for record in records
        haskey(record, input_field) || continue

        source_text = String(record[input_field])
        paraphrase = request_deepseek_paraphrase(source_text; model=model)

        println("Source: ", source_text)
        println("Output: ", paraphrase)
        println()

        shown += 1
        shown >= n && break
    end
end

function main()
    println("Loaded Paraphrase_DeepSeek.jl")
    println("Run `preview_deepseek_paraphrases()` to inspect a few examples.")
    println("Run `update_dataset_with_deepseek_paraphrases()` to write them into the dataset.")
    println("Set DEEPSEEK_API_KEY in your shell or place it in a local .env file.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
