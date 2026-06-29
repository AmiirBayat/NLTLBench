include("GenerateLTL.jl")
include("Filter.jl")

using JSON3
using OrderedCollections

const DEFAULT_DATASET_PATH = joinpath(@__DIR__, "dataset", "DatasetWithNaturalNL_plus_simplified.json")
const DEFAULT_OUTPUT_FIELD = "back_translation"

function load_dataset(dataset_path::String = DEFAULT_DATASET_PATH)
    if !isfile(dataset_path)
        throw(ArgumentError("Dataset file not found: $(dataset_path)"))
    end

    content = strip(read(dataset_path, String))
    isempty(content) && throw(ArgumentError("Dataset file is empty: $(dataset_path)"))

    parsed = JSON3.read(content)
    return [OrderedDict{String,Any}(String(k) => v for (k, v) in pairs(record)) for record in parsed]
end

function save_dataset(records, dataset_path::String = DEFAULT_DATASET_PATH)
    open(dataset_path, "w") do io
        JSON3.pretty(io, records)
        write(io, "\n")
    end
end

function ap_to_nl(name::AbstractString)
    return "$(name) is true"
end

function formula_to_nl(formula::AP)::String
    if formula.name == "true"
        return "true"
    elseif formula.name == "false"
        return "false"
    else
        return ap_to_nl(formula.name)
    end
end

function formula_to_nl(formula::UnaryLTL)::String
    child_nl = formula_to_nl(formula.child)

    if formula.op == :!
        if formula.child isa AP && !(formula.child.name in ("true", "false"))
            return "$(formula.child.name) is false"
        else
            return "it is not the case that ($(child_nl))"
        end
    elseif formula.op == :X
        return "in the next step, ($(child_nl))"
    elseif formula.op == :F
        return "eventually, ($(child_nl))"
    elseif formula.op == :G
        return "always, ($(child_nl))"
    else
        return "$(formula.op)($(child_nl))"
    end
end

function formula_to_nl(formula::BinaryLTL)::String
    left_nl = formula_to_nl(formula.left)
    right_nl = formula_to_nl(formula.right)

    if formula.op == :&
        return "($(left_nl)) and ($(right_nl))"
    elseif formula.op == :|
        return "($(left_nl)) or ($(right_nl))"
    elseif formula.op == :->
        return "if ($(left_nl)), then ($(right_nl))"
    elseif formula.op == Symbol("<->")
        return "($(left_nl)) if and only if ($(right_nl))"
    elseif formula.op == :U
        return "($(formula_to_nl(formula.left))) holds until ($(formula_to_nl(formula.right)))"
    elseif formula.op == :R
        return "($(formula_to_nl(formula.right))) must hold until ($(formula_to_nl(formula.left))) holds, and if it never does, then ($(formula_to_nl(formula.right))) must keep holding forever"
    elseif formula.op == :W
        return "($(formula_to_nl(formula.left))) holds until ($(formula_to_nl(formula.right))) holds, or else ($(formula_to_nl(formula.left))) keeps holding forever"
    elseif formula.op == :M
        return "($(formula_to_nl(formula.right))) must hold until ($(formula_to_nl(formula.left))) holds, and ($(formula_to_nl(formula.left))) must eventually hold"
    else
        return "($(left_nl)) $(formula.op) ($(right_nl))"
    end
end

function ltl_string_to_nl(formula_str::AbstractString)
    normalized = strip(String(formula_str))
    normalized == "1" && (normalized = "true")
    normalized == "0" && (normalized = "false")
    isempty(normalized) && throw(ArgumentError("Empty LTL formula string."))
    ast = parse_ltl_formula_string(normalized)
    return formula_to_nl(ast)
end

function update_dataset_with_backtranslations(
    dataset_path::String = DEFAULT_DATASET_PATH;
    output_field::String = DEFAULT_OUTPUT_FIELD,
    overwrite::Bool = false,
)
    records = load_dataset(dataset_path)
    updated_count = 0
    skipped_count = 0
    error_count = 0

    for record in records
        haskey(record, "LTL") || continue

        already_has_output = haskey(record, output_field) && !isempty(strip(String(record[output_field])))
        if !overwrite && already_has_output
            skipped_count += 1
            continue
        end

        ltl_value = strip(String(record["LTL"]))
        try
            record[output_field] = ltl_string_to_nl(ltl_value)
            updated_count += 1
        catch err
            error_count += 1
            record_id = haskey(record, "id") ? string(record["id"]) : "unknown"
            println("Skipping back-translation for record id ", record_id, " because its LTL could not be parsed: ", ltl_value)
            println("  Error: ", sprint(showerror, err))
        end
    end

    save_dataset(records, dataset_path)

    println("Updated $(updated_count) dataset entries with algorithmic back-translations.")
    println("Skipped $(skipped_count) existing entries.")
    println("Encountered $(error_count) errors.")
    println("Dataset path: $(dataset_path)")
end

function preview_backtranslations(dataset_path::String = DEFAULT_DATASET_PATH; n::Int = 5)
    records = load_dataset(dataset_path)
    shown = 0

    for record in records
        haskey(record, "LTL") || continue
        ltl = String(record["LTL"])

        println("LTL: ", ltl)
        println("NL : ", ltl_string_to_nl(ltl))
        println()

        shown += 1
        shown >= n && break
    end
end

function main()
    println("Loaded BackTranslation.jl")
    println("Run `preview_backtranslations()` to inspect translations for dataset/ltl_dataset.json.")
    println("Run `update_dataset_with_backtranslations(overwrite=false)` to write them into dataset/ltl_dataset.json.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end