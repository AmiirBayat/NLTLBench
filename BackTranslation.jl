

include("GenerateLTL.jl")
include("Filter.jl")

using JSON3
using OrderedCollections

const DEFAULT_DATASET_PATH = joinpath(@__DIR__, "dataset", "ltl_dataset.json")
const DEFAULT_OUTPUT_FIELD = "back_translation"

"""
    load_dataset(dataset_path=DEFAULT_DATASET_PATH)

Load the dataset JSON file and return it as a Julia vector of records.
"""
function load_dataset(dataset_path::String = DEFAULT_DATASET_PATH)
    if !isfile(dataset_path)
        throw(ArgumentError("Dataset file not found: $(dataset_path)"))
    end

    data = JSON3.read(read(dataset_path, String))
    return collect(data)
end

"""
    save_dataset(records, dataset_path=DEFAULT_DATASET_PATH)

Write the updated dataset back to disk with pretty JSON formatting.
"""
function save_dataset(records, dataset_path::String = DEFAULT_DATASET_PATH)
    open(dataset_path, "w") do io
        JSON3.pretty(io, records)
        write(io, "\n")
    end
end

"""
    ap_to_nl(name)

Translate an atomic proposition name into a readable phrase.
"""
function ap_to_nl(name::AbstractString)
    return "$(name) is true"
end

"""
    formula_to_nl(formula)

Translate an LTL formula AST into a deterministic natural-language description.
This is a rule-based, algorithmic back-translation intended for dataset construction.
"""
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

"""
    ltl_string_to_nl(formula_str)

Parse an LTL formula string and translate it to natural language.
"""
function ltl_string_to_nl(formula_str::AbstractString)
    ast = parse_ltl_formula_string(String(formula_str))
    return formula_to_nl(ast)
end

"""
    update_dataset_with_backtranslations(dataset_path=DEFAULT_DATASET_PATH; output_field=DEFAULT_OUTPUT_FIELD, overwrite=true)

For each dataset entry, translate the saved `LTL` formula to natural language and store it
under `output_field`. The dataset is then written back in place.
"""
function update_dataset_with_backtranslations(
    dataset_path::String = DEFAULT_DATASET_PATH;
    output_field::String = DEFAULT_OUTPUT_FIELD,
    overwrite::Bool = true,
)
    records = load_dataset(dataset_path)
    updated_records = OrderedDict[]
    updated_count = 0

    for record in records
        dict_record = OrderedDict{String,Any}()
        for (k, v) in pairs(record)
            dict_record[String(k)] = v
        end

        if !haskey(dict_record, "LTL")
            push!(updated_records, dict_record)
            continue
        end

        if overwrite || !haskey(dict_record, output_field)
            dict_record[output_field] = ltl_string_to_nl(String(dict_record["LTL"]))
            updated_count += 1
        end

        push!(updated_records, dict_record)
    end

    save_dataset(updated_records, dataset_path)

    println("Updated $(updated_count) dataset entries with algorithmic back-translations.")
    println("Dataset path: $(dataset_path)")
end

function preview_backtranslations(dataset_path::String = DEFAULT_DATASET_PATH; n::Int = 5)
    records = load_dataset(dataset_path)
    shown = 0

    for record in records
        ltl = haskey(record, :LTL) ? String(record[:LTL]) : (haskey(record, "LTL") ? String(record["LTL"]) : nothing)
        isnothing(ltl) && continue

        println("LTL: ", ltl)
        println("NL : ", ltl_string_to_nl(ltl))
        println()

        shown += 1
        shown >= n && break
    end
end

function main()
    update_dataset_with_backtranslations()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
else
    println("Loaded BackTranslation.jl. Run `preview_backtranslations()` to inspect translations or `update_dataset_with_backtranslations()` to write them into the dataset.")
end