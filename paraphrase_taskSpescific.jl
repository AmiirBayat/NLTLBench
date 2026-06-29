using HTTP
using JSON3
using OrderedCollections
using DotEnv

const DEFAULT_ORIGINAL_INPUT_PATH = joinpath(@__DIR__, "dataset", "ltl_dataset_task_specific_compatible.json")
const DEFAULT_OUTPUT_PATH = joinpath(@__DIR__, "dataset", "ltl_dataset_task_specific_compatible_rewritten.json")
const DEFAULT_MODEL = "gpt-5.4-mini"
const DEFAULT_OUTPUT_FIELD = "rewritten_gpt54mini"
const DEFAULT_TEMPERATURE = 0.7
const DEFAULT_MAX_OUTPUT_TOKENS = 120
const DEFAULT_SLEEP_SECONDS = 0.2
DotEnv.load!()

const DEFAULT_SOURCE_FIELDS = [
    "natural_paraphrase",
    "paraphrase_gpt5.4-mini",
    "paraphrase_gemini-2.5-flash",
    "paraphrase_deepseek",
]

const REWRITE_PROMPT_TEMPLATE = """
You are given a natural-language description derived from an LTL formula for a robot navigation task.

Rewrite the description as a natural task instruction that a human might give to a robot.

The environment consists of colored regions. A proposition such as \"blue is true\" means that the robot is in the blue region. Likewise for brown, yellow, green, and purple.

Rules:
- Preserve the exact meaning of the original sentence.
- Do not introduce new goals, constraints, or assumptions.
- Do not mention LTL, propositions, truth values, formulas, or logic.
- Use natural navigation language.
- Keep the instruction concise (one sentence whenever possible).
- Keep some of the input sentence's phrasing style when possible, while still rewriting it as a natural robot instruction.
- Preserve stylistic diversity across different inputs: if two inputs are phrased differently but mean the same thing, do not automatically rewrite them using the exact same wording.
- Avoid always defaulting to the same verb pattern such as only \"reach\" or only \"visit\" when another equally correct natural phrasing is possible.
- Keep the rewriting natural, but retain differences in tone or structure when the input wording differs.
- Do not explain your reasoning.
- Output only the rewritten instruction.

Examples:

Input:
\"blue eventually becomes true.\"

Output:
\"Eventually reach the blue region.\"

Input:
\"brown becomes true infinitely often.\"

Output:
\"Visit the brown region infinitely often.\"

Input:
\"If green is true, then yellow must always be true.\"

Output:
\"If the robot is in the green region, then it must always stay in the yellow region.\"

Input:
\"Either brown is true, or purple eventually becomes true.\"

Output:
\"Either the robot is in the brown region, or it should eventually reach the purple region.\"

Input:
\"In the next step, green is true.\"

Output:
\"Move so that the robot is in the green region at the next step.\"

Rewrite the following sentence:

{natural_paraphrase}
"""

function load_json_records(path::String)
    isfile(path) || throw(ArgumentError("Input file not found: $(path)"))
    raw_records = collect(JSON3.read(read(path, String)))
    records = OrderedDict{String,Any}[]

    for raw_record in raw_records
        record = OrderedDict{String,Any}()
        for (k, v) in pairs(raw_record)
            record[String(k)] = v
        end
        push!(records, record)
    end

    return records
end

function save_json_records(path::String, records)
    open(path, "w") do io
        JSON3.pretty(io, records)
    end
    println("Saved rewritten dataset to: ", path)
end

function resolve_input_path(input_path::Union{Nothing,String}, output_path::String)
    if input_path !== nothing
        return String(input_path)
    end
    return isfile(output_path) ? output_path : DEFAULT_ORIGINAL_INPUT_PATH
end

function build_prompt(natural_paraphrase::AbstractString)
    return replace(REWRITE_PROMPT_TEMPLATE, "{natural_paraphrase}" => String(natural_paraphrase))
end

function build_prompt(natural_paraphrase::AbstractString, source_field::AbstractString)
    source_style_hint = if source_field == "natural_paraphrase"
        "Keep the rewrite plain and natural."
    elseif source_field == "paraphrase_gpt5.4-mini"
        "Keep the rewrite polished and concise, but not identical to other rewrites."
    elseif source_field == "paraphrase_gemini-2.5-flash"
        "Keep the rewrite slightly formal and well-structured, while remaining natural."
    elseif source_field == "paraphrase_deepseek"
        "Keep the rewrite natural and direct, while preserving some of the source phrasing style."
    else
        "Keep the rewrite natural while preserving some of the input phrasing style."
    end

    prompt = replace(REWRITE_PROMPT_TEMPLATE, "{natural_paraphrase}" => String(natural_paraphrase))
    return prompt * "\n\nAdditional style hint: " * source_style_hint
end

function extract_output_text(response_json)
    if haskey(response_json, "output_text") && !isnothing(response_json["output_text"])
        return strip(String(response_json["output_text"]))
    end

    if haskey(response_json, "output")
        output = response_json["output"]
        for item in output
            if haskey(item, "content")
                for content_item in item["content"]
                    if haskey(content_item, "text") && !isnothing(content_item["text"])
                        text = strip(String(content_item["text"]))
                        !isempty(text) && return text
                    end
                end
            end
        end
    end

    error("Could not extract model text from API response.")
end

function rewrite_with_openai(
    natural_paraphrase::AbstractString;
    model::String = DEFAULT_MODEL,
    source_field::String = "natural_paraphrase",
    temperature::Float64 = DEFAULT_TEMPERATURE,
    max_output_tokens::Int = DEFAULT_MAX_OUTPUT_TOKENS,
)
    api_key = get(ENV, "OPENAI_API_KEY", "")
    isempty(api_key) && error("OPENAI_API_KEY is not set.")

    prompt = build_prompt(natural_paraphrase, source_field)

    request_body = OrderedDict(
        "model" => model,
        "input" => prompt,
        "temperature" => temperature,
        "max_output_tokens" => max_output_tokens,
    )

    response = HTTP.post(
        "https://api.openai.com/v1/responses",
        [
            "Authorization" => "Bearer $(api_key)",
            "Content-Type" => "application/json",
        ],
        JSON3.write(request_body),
    )

    response.status == 200 || error("OpenAI API request failed with status $(response.status): $(String(response.body))")
    parsed = JSON3.read(String(response.body))
    return extract_output_text(parsed)
end

function rewrite_task_specific_paraphrases(
    input_path::Union{Nothing,String} = nothing;
    output_path::String = DEFAULT_OUTPUT_PATH,
    model::String = DEFAULT_MODEL,
    source_fields::Vector{String} = DEFAULT_SOURCE_FIELDS,
    output_suffix::String = "_gpt54mini_rewritten",
    overwrite_existing::Bool = false,
    temperature::Float64 = DEFAULT_TEMPERATURE,
    max_output_tokens::Int = DEFAULT_MAX_OUTPUT_TOKENS,
    sleep_seconds::Float64 = DEFAULT_SLEEP_SECONDS,
)
    resolved_input_path = resolve_input_path(input_path, output_path)
    records = load_json_records(resolved_input_path)
    rewritten_count = 0
    skipped_count = 0
    println("Loading records from: ", resolved_input_path)

    for (idx, record) in enumerate(records)
        for source_field in source_fields
            haskey(record, source_field) || begin
                @warn "Skipping missing source field" index=idx field=source_field
                skipped_count += 1
                continue
            end

            source_text = strip(String(record[source_field]))
            isempty(source_text) && begin
                @warn "Skipping empty source field" index=idx field=source_field
                skipped_count += 1
                continue
            end

            output_field = source_field * output_suffix
            if !overwrite_existing && haskey(record, output_field)
                existing_text = strip(String(record[output_field]))
                if !isempty(existing_text)
                    skipped_count += 1
                    continue
                end
            end

            rewritten = rewrite_with_openai(
                source_text;
                model=model,
                source_field=source_field,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )

            record[output_field] = rewritten
            rewritten_count += 1
            println("Rewritten record $(idx), field $(source_field): ", rewritten)
            save_json_records(output_path, records)
            sleep(sleep_seconds)
        end
    end

    println("Finished rewriting paraphrases.")
    println("Rewritten: ", rewritten_count)
    println("Skipped: ", skipped_count)
    return records
end

function main()
    rewrite_task_specific_paraphrases(input_path=nothing, output_path=DEFAULT_OUTPUT_PATH)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
