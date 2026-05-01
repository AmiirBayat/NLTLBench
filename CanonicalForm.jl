# CanonicalForm.jl
#
# Compute a Spot-based canonical form of LTL formulas using `ltlfilt`, and update the dataset
# by writing that canonical form into each record's `simplified_formula` field.
#
# Example usage in the Julia REPL:
#     include("CanonicalForm.jl")
#     canonical_form("G(prop_4 | prop_2)")
#     preview_canonical_forms()
#     update_dataset_with_canonical_simplified_formulas()

using JSON3
using OrderedCollections

const DEFAULT_DATASET_PATH = joinpath(@__DIR__, "dataset", "DatasetWithNaturalNL.json")
const DEFAULT_INPUT_FIELD = "LTL"
const DEFAULT_OUTPUT_FIELD = "simplified_formula"

function require_ltlfilt()
    isnothing(Sys.which("ltlfilt")) && throw(ArgumentError(
        "Spot's `ltlfilt` was not found in PATH. Install Spot and make sure `ltlfilt` is available."
    ))
end

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

function convert_spot_props_to_benchmark_names(formula::AbstractString)::String
    return replace(String(formula), r"\bp(\d+)\b" => (m -> begin
        matched = String(m)
        idx = parse(Int, matched[2:end]) + 1
        return "prop_$(idx)"
    end))
end

"""
    canonical_form(formula; full_parentheses=false, relabel=false, simplify=true)

Return a Spot-based canonical form of an LTL formula as a string.

Options:
- `simplify=true` applies Spot rewriting/simplification (`-r`)
- `relabel=false` keeps the original proposition names by default
- `full_parentheses=false` keeps Spot's more readable default formatting

The canonical form is also forced to avoid the operators `W`, `R`, and `M`
by passing `--unabbreviate=MRW` to Spot.
"""
function canonical_form(
    formula::AbstractString;
    full_parentheses::Bool = false,
    relabel::Bool = false,
    simplify::Bool = true,
)::String
    require_ltlfilt()
    ltlfilt_path = Sys.which("ltlfilt")

    args = String[ltlfilt_path, "-f", String(formula)]

    if simplify
        push!(args, "-r")
    end
    push!(args, "--unabbreviate=MRW")
    if relabel
        push!(args, "--relabel=pnn")
    end
    if full_parentheses
        push!(args, "-p")
    end

    output = read(Cmd(args), String)
    return convert_spot_props_to_benchmark_names(strip(output))
end

function preview_canonical_forms(
    dataset_path::String = DEFAULT_DATASET_PATH;
    input_field::String = DEFAULT_INPUT_FIELD,
    n::Int = 5,
)
    records = load_dataset(dataset_path)
    shown = 0

    for record in records
        haskey(record, input_field) || continue
        original = String(record[input_field])
        simplified = canonical_form(original)

        println("ID: ", get(record, "id", "?"))
        println("LTL: ", original)
        println("simplified_formula: ", simplified)
        println()

        shown += 1
        shown >= n && break
    end
end

function update_dataset_with_canonical_simplified_formulas(
    dataset_path::String = DEFAULT_DATASET_PATH;
    input_field::String = DEFAULT_INPUT_FIELD,
    output_field::String = DEFAULT_OUTPUT_FIELD,
)
    records = load_dataset(dataset_path)
    updated_count = 0

    for record in records
        haskey(record, input_field) || continue
        original = String(record[input_field])
        simplified = canonical_form(original)
        record[output_field] = simplified
        updated_count += 1
    end

    save_dataset(records, dataset_path)

    println("Updated $(updated_count) dataset entries with canonical simplified formulas.")
    println("Dataset path: $(dataset_path)")
end

function print_canonical_form(formula::AbstractString)
    println("Input formula: ", formula)
    println("Canonical form: ", canonical_form(formula))
end

function main()
    println("Loaded CanonicalForm.jl")
    println("Run `preview_canonical_forms()` to inspect a few examples.")
    println("Run `update_dataset_with_canonical_simplified_formulas()` to write Spot canonical forms into simplified_formula.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end