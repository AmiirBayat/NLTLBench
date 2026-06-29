using JSON3
using OrderedCollections
using Random

const DEFAULT_TEMPLATE_BENCHMARK_PATH = joinpath(@__DIR__, "dataset", "DatasetWithNaturalNL_plus_simplified.json")
const DEFAULT_TASK_SPECIFIC_OUTPUT_PATH = joinpath(@__DIR__, "dataset", "ltl_dataset_task_specific.json")
const TASK_SPECIFIC_APS = [ "blue", "brown", "purple", "green", "yellow"]
const PLACEHOLDER_AP_RE = r"\bprop_(\d+)\b"

function load_json_records(path::String)
    isfile(path) || throw(ArgumentError("Input benchmark file not found: $(path)"))
    return collect(JSON3.read(read(path, String)))
end

function save_json_records(path::String, records)
    open(path, "w") do io
        JSON3.pretty(io, records)
    end
    println("Saved task-specific benchmark to: ", path)
end

function placeholder_props_in_text(text::AbstractString)
    matches = collect(eachmatch(PLACEHOLDER_AP_RE, String(text)))
    isempty(matches) && return String[]

    indices = sort(unique(parse(Int, m.captures[1]) for m in matches))
    return ["prop_$(i)" for i in indices]
end

function placeholder_props_in_record(record)
    props = String[]
    fields_to_scan = [
        "LTL",
        "generated_formula",
        "simplified_formula",
        "back_translation",
        "natural_paraphrase",
        "natural_language",
        "nl",
        "sentence",
        "description",
    ]

    for field in fields_to_scan
        if haskey(record, field)
            append!(props, placeholder_props_in_text(String(record[field])))
        end
    end

    return sort(unique(props); by = prop -> parse(Int, split(prop, "_")[2]))
end

function random_k_permutations(items::Vector{String}, k::Int, n::Int; rng::AbstractRNG = Random.default_rng())
    k < 0 && throw(ArgumentError("k must be nonnegative."))
    n < 0 && throw(ArgumentError("n must be nonnegative."))
    k == 0 && return [String[] for _ in 1:n]
    k > length(items) && return Vector{Vector{String}}()

    seen = Set{Tuple{Vararg{String}}}()
    results = Vector{Vector{String}}()
    max_unique = factorial(length(items)) ÷ factorial(length(items) - k)
    target = min(n, max_unique)

    while length(results) < target
        perm = Random.randperm(rng, length(items))
        choice = [items[perm[i]] for i in 1:k]
        key = Tuple(choice)
        key in seen && continue
        push!(seen, key)
        push!(results, choice)
    end

    return results
end

function replace_placeholders(text::AbstractString, mapping::OrderedDict{String,String})
    result = String(text)
    for (source, target) in mapping
        result = replace(result, Regex("\\b" * source * "\\b") => target)
    end
    return result
end

function task_specific_mapping(placeholders::Vector{String}, chosen_aps::Vector{String})
    length(placeholders) == length(chosen_aps) || throw(ArgumentError("Placeholder/AP length mismatch."))
    mapping = OrderedDict{String,String}()
    for (placeholder, ap) in zip(placeholders, chosen_aps)
        mapping[placeholder] = ap
    end
    return mapping
end

function convert_record_to_task_specific_variants(
    record;
    available_aps::Vector{String} = TASK_SPECIFIC_APS,
    num_variants::Int = 2,
    rng::AbstractRNG = Random.default_rng(),
)
    placeholders = placeholder_props_in_record(record)
    isempty(placeholders) && return OrderedDict{String,Any}[OrderedDict{String,Any}(pairs(record))]

    assignments = random_k_permutations(available_aps, length(placeholders), num_variants; rng=rng)
    isempty(assignments) && return OrderedDict{String,Any}[]

    variants = OrderedDict{String,Any}[]

    for chosen_aps in assignments
        mapping = task_specific_mapping(placeholders, chosen_aps)
        variant = OrderedDict{String,Any}()

        for (key, value) in pairs(record)
            if value isa AbstractString
                variant[String(key)] = replace_placeholders(String(value), mapping)
            else
                variant[String(key)] = value
            end
        end

        variant["template_id"] = haskey(record, "id") ? Int(record["id"]) : nothing
        variant["task_specific_ap_mapping"] = OrderedDict{String,String}(mapping)
        push!(variants, variant)
    end

    return variants
end

function build_task_specific_benchmark(
    input_path::String = DEFAULT_TEMPLATE_BENCHMARK_PATH;
    output_path::String = DEFAULT_TASK_SPECIFIC_OUTPUT_PATH,
    available_aps::Vector{String} = TASK_SPECIFIC_APS,
    num_variants_per_template::Int = 2,
    rng::AbstractRNG = Random.default_rng(),
)
    template_records = load_json_records(input_path)
    task_specific_records = OrderedDict{String,Any}[]
    next_id = 1
    discarded_templates = 0

    for record in template_records
        placeholders = placeholder_props_in_record(record)
        if length(placeholders) > length(available_aps) - 1
            discarded_templates += 1
            continue
        end

        variants = convert_record_to_task_specific_variants(
            record;
            available_aps=available_aps,
            num_variants=num_variants_per_template,
            rng=rng,
        )
        for variant in variants
            variant["id"] = next_id
            push!(task_specific_records, variant)
            next_id += 1
        end
    end

    save_json_records(output_path, task_specific_records)
    println("Template records: ", length(template_records))
    println("Discarded templates: ", discarded_templates)
    println("Variants per template requested: ", num_variants_per_template)
    println("Task-specific records: ", length(task_specific_records))
    return task_specific_records
end

function main()
    build_task_specific_benchmark()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end