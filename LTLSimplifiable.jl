using JSON3
using OrderedCollections

const DEFAULT_INPUT_DATASET = joinpath(@__DIR__, "dataset", "DatasetWithNaturalNL.json")
const DEFAULT_OUTPUT_JSON = joinpath(@__DIR__, "dataset", "DatasetSimplifiablePairs.json")

function load_dataset(dataset_path::String = DEFAULT_INPUT_DATASET)
    if !isfile(dataset_path)
        throw(ArgumentError("Dataset file not found: $(dataset_path)"))
    end

    content = strip(read(dataset_path, String))
    isempty(content) && throw(ArgumentError("Dataset file is empty: $(dataset_path)"))

    parsed = JSON3.read(content)
    return [OrderedDict{String,Any}(String(k) => v for (k, v) in pairs(record)) for record in parsed]
end

function ensure_parent_directory(path::String)
    parent = dirname(path)
    isdir(parent) || mkpath(parent)
end

function normalize_text(value)::String
    if value === nothing
        return ""
    end
    return strip(String(value))
end

function normalize_surface_form(formula::String)::String
    s = strip(formula)
    s = replace(s, r"\s+" => "")
    s = replace(s, "(" => "", ")" => "")
    return s
end

function has_different_simplified_version(record::OrderedDict{String,Any})::Bool
    ltl = normalize_text(get(record, "LTL", ""))
    simplified = normalize_text(get(record, "simplified_formula", ""))

    isempty(ltl) && return false
    isempty(simplified) && return false

    if ltl == simplified
        return false
    end

    if normalize_surface_form(ltl) == normalize_surface_form(simplified)
        return false
    end

    return true
end

function make_base_entry(record::OrderedDict{String,Any}; variant::String, formula_value::String)
    paired_formula = variant == "original" ? normalize_text(get(record, "simplified_formula", "")) : normalize_text(get(record, "LTL", ""))

    entry = OrderedDict{String,Any}(
        "pair_id" => string(get(record, "id", "unknown"), "_", variant),
        "original_record_id" => get(record, "id", nothing),
        "variant" => variant,
        "LTL" => formula_value,
        "paired_formula" => paired_formula,
    )

    if variant == "original"
        if haskey(record, "natural_paraphrase")
            entry["natural_paraphrase"] = record["natural_paraphrase"]
        end
        if haskey(record, "paraphrase_gpt5.4-mini")
            entry["paraphrase_gpt5.4-mini"] = record["paraphrase_gpt5.4-mini"]
        end
        if haskey(record, "paraphrase_gemini-2.5-flash")
            entry["paraphrase_gemini-2.5-flash"] = record["paraphrase_gemini-2.5-flash"]
        end
        if haskey(record, "paraphrase_deepseek")
            entry["paraphrase_deepseek"] = record["paraphrase_deepseek"]
        end
        if haskey(record, "paraphrase_claude")
            entry["paraphrase_claude"] = record["paraphrase_claude"]
        end
    end

    return entry
end

function collect_simplifiable_pairs(dataset_path::String = DEFAULT_INPUT_DATASET)
    dataset = load_dataset(dataset_path)
    output_entries = OrderedDict{String,Any}[]

    for record in dataset
        has_different_simplified_version(record) || continue

        original_ltl = normalize_text(get(record, "LTL", ""))
        simplified_ltl = normalize_text(get(record, "simplified_formula", ""))

        push!(output_entries, make_base_entry(record; variant="original", formula_value=original_ltl))
        push!(output_entries, make_base_entry(record; variant="simplified", formula_value=simplified_ltl))
    end

    return output_entries
end

function next_available_id(dataset)::Int
    max_id = 0
    for record in dataset
        if haskey(record, "id")
            value = try
                Int(record["id"])
            catch
                continue
            end
            max_id = max(max_id, value)
        end
    end
    return max_id + 1
end

function simplified_entry_for_main_dataset(record::OrderedDict{String,Any}, new_id::Int)
    simplified_ltl = normalize_text(get(record, "simplified_formula", ""))
    entry = OrderedDict{String,Any}(
        "id" => new_id,
        "LTL" => simplified_ltl,
        "simplified_formula" => simplified_ltl,
        "origin" => "simplified_from_existing",
        "source_record_id" => get(record, "id", nothing),
    )
    return entry
end

function add_simplified_formulas_to_main_dataset(
    dataset_path::String = DEFAULT_INPUT_DATASET;
    output_path::String = joinpath(@__DIR__, "dataset", "DatasetWithNaturalNL_plus_simplified.json"),
)
    dataset = load_dataset(dataset_path)
    updated_dataset = OrderedDict{String,Any}[copy(record) for record in dataset]
    next_id = next_available_id(updated_dataset)
    added_count = 0

    for record in dataset
        has_different_simplified_version(record) || continue
        push!(updated_dataset, simplified_entry_for_main_dataset(record, next_id))
        next_id += 1
        added_count += 1
    end

    ensure_parent_directory(output_path)
    open(output_path, "w") do io
        JSON3.pretty(io, updated_dataset)
        write(io, "\n")
    end

    println("Added ", added_count, " simplified formulas to the dataset.")
    println("Saved updated dataset to: ", output_path)
end

function save_simplifiable_pairs(
    output_path::String = DEFAULT_OUTPUT_JSON;
    dataset_path::String = DEFAULT_INPUT_DATASET,
)
    entries = collect_simplifiable_pairs(dataset_path)
    ensure_parent_directory(output_path)

    open(output_path, "w") do io
        JSON3.pretty(io, entries)
        write(io, "\n")
    end

    println("Saved ", length(entries), " entries to: ", output_path)
    println("This corresponds to ", length(entries) ÷ 2, " original/simplified pairs.")
end

function preview_simplifiable_pairs(n::Int = 5; dataset_path::String = DEFAULT_INPUT_DATASET)
    entries = collect_simplifiable_pairs(dataset_path)
    limit = min(n * 2, length(entries))

    for i in 1:limit
        entry = entries[i]
        println("------------------------------------------------------------")
        println("pair_id: ", entry["pair_id"])
        println("original_record_id: ", entry["original_record_id"])
        println("variant: ", entry["variant"])
        println("LTL: ", entry["LTL"])
        println("paired_formula: ", entry["paired_formula"])
        if haskey(entry, "natural_paraphrase")
            println("natural_paraphrase: ", entry["natural_paraphrase"])
        end
    end
end

function main()
    println("Loaded LTLSimplifiable.jl")
    println("Run `preview_simplifiable_pairs()` to inspect a few examples.")
    println("Run `save_simplifiable_pairs()` to write dataset/DatasetSimplifiablePairs.json.")
    println("Run `add_simplified_formulas_to_main_dataset()` to append simplified formulas with new ids. Then run `update_dataset_with_backtranslations()` separately.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end