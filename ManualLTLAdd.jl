

include("GenerateLTL.jl")
include("Filter.jl")
include("Satisfiability.jl")

using JSON3
using OrderedCollections

const DEFAULT_DATASET_PATH = joinpath(@__DIR__, "dataset", "ltl_dataset.json")
const DEFAULT_CONFORMAL_PATH = joinpath(@__DIR__, "final_calib_normalized_fixed.json")

# ----------------------------------------------------------------------------------------------
# Dataset I/O
# ----------------------------------------------------------------------------------------------

function load_dataset(dataset_path::String = DEFAULT_DATASET_PATH)
    if !isfile(dataset_path)
        return OrderedDict[]
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
# Exact equivalence checks against dataset
# ----------------------------------------------------------------------------------------------

function formula_strings_are_equivalent(formula_a::AbstractString, formula_b::AbstractString)
    require_ltlfilt()
    ltlfilt_path = Sys.which("ltlfilt")

    cmd = `$(ltlfilt_path) -f $(String(formula_a)) --equivalent-to $(String(formula_b)) -q`
    process = run(cmd; wait=false)
    wait(process)

    if process.exitcode == 0
        return true
    elseif process.exitcode == 1
        return false
    else
        throw(ErrorException("`ltlfilt` failed while checking equivalence between `$(formula_a)` and `$(formula_b)` (exit code $(process.exitcode))."))
    end
end

function find_duplicate_in_dataset(ltl::AbstractString, dataset_records)
    for record in dataset_records
        haskey(record, "LTL") || continue
        existing_ltl = String(record["LTL"])
        if formula_strings_are_equivalent(ltl, existing_ltl)
            return record
        end
    end
    return nothing
end

# ----------------------------------------------------------------------------------------------
# Record construction
# ----------------------------------------------------------------------------------------------

function operator_counts_ordered(formula::LTLFormula)
    return OrderedDict(string(k) => v for (k, v) in sort!(collect(operator_counts(formula)); by=x -> string(x[1])))
end

function temporal_behavior_from_ast(formula::LTLFormula)
    return classify_temporal_behavior(formula)
end

function build_manual_record(id::Int, generated_formula_str::AbstractString)
    parsed_formula = parse_ltl_formula_string(String(generated_formula_str))

    # Exact satisfiability check
    if !is_satisfiable_exact(parsed_formula)
        return nothing, "The formula is unsatisfiable and cannot be added to the dataset."
    end

    # Local simplification traceability
    simplified_formula_str = formula_to_string(simplify_formula_local(parsed_formula))

    # Final saved representation used in the benchmark
    selected_formula_str = final_selected_formula_string(parsed_formula)
    selected_formula_ast = final_selected_formula_ast(parsed_formula)

    # Enforce temporal benchmark policy on the final saved formula
    if !has_temporal_operator(selected_formula_ast)
        return nothing, "The final simplified form is non-temporal, so it is rejected by the benchmark policy."
    end

    num_atomic_props = length(count_atomic_props(selected_formula_ast))

    record = OrderedDict(
        "id" => id,
        "LTL" => selected_formula_str,
        "generated_formula" => String(generated_formula_str),
        "simplified_formula" => simplified_formula_str,
        "ast_size" => ast_size(selected_formula_ast),
        "ast_depth" => ast_depth(selected_formula_ast),
        "temporal_depth" => temporal_depth(selected_formula_ast),
        "num_atomic_props" => num_atomic_props,
        "operator_counts" => operator_counts_ordered(selected_formula_ast),
        "temporal_behavior" => temporal_behavior_from_ast(selected_formula_ast),
    )

    return record, nothing
end

# ----------------------------------------------------------------------------------------------
# Add one or many manual formulas
# ----------------------------------------------------------------------------------------------

"""
    add_manual_ltl_formula(formula_str; dataset_path=DEFAULT_DATASET_PATH, verbose=true)

Take a manually provided LTL formula string, process it through the benchmark pipeline,
check for duplicates against the existing dataset, and add it if everything is valid.

Returns `(added::Bool, message::String, record_or_duplicate)`.
"""
function add_manual_ltl_formula(
    formula_str::AbstractString;
    dataset_path::String = DEFAULT_DATASET_PATH,
    verbose::Bool = true,
)
    dataset_records = load_dataset(dataset_path)

    record, err = build_manual_record(length(dataset_records) + 1, formula_str)
    if !isnothing(err)
        verbose && println(err)
        return false, err, nothing
    end

    duplicate = find_duplicate_in_dataset(String(record["LTL"]), dataset_records)
    if !isnothing(duplicate)
        msg = "Duplicate found in dataset. The formula was not added."
        verbose && begin
            println(msg)
            println("Existing ID: ", get(duplicate, "id", "?"))
            println("Existing LTL: ", get(duplicate, "LTL", "?"))
        end
        return false, msg, duplicate
    end

    push!(dataset_records, record)

    for (i, rec) in enumerate(dataset_records)
        rec["id"] = i
    end

    save_dataset(dataset_records, dataset_path)

    msg = "Formula added successfully."
    verbose && begin
        println(msg)
        println("Saved as LTL: ", record["LTL"])
        println("Temporal depth: ", record["temporal_depth"])
        println("AST size: ", record["ast_size"])
        println("Temporal behavior: ", record["temporal_behavior"])
    end
    return true, msg, record
end

"""
    add_manual_ltl_formulas(formulas; dataset_path=DEFAULT_DATASET_PATH, verbose=true)

Add multiple manual LTL formulas one by one.
Returns a vector of result tuples.
"""
function add_manual_ltl_formulas(
    formulas::Vector{<:AbstractString};
    dataset_path::String = DEFAULT_DATASET_PATH,
    verbose::Bool = true,
)
    results = []
    for formula_str in formulas
        push!(results, add_manual_ltl_formula(formula_str; dataset_path=dataset_path, verbose=verbose))
        verbose && println()
    end
    return results
end

# ----------------------------------------------------------------------------------------------
# Import formulas from final_calib_normalized_fixed.json
# ----------------------------------------------------------------------------------------------

function load_json_array(path::String)
    if !isfile(path)
        throw(ArgumentError("File not found: $(path)"))
    end
    data = JSON3.read(read(path, String))
    return [OrderedDict{String,Any}(String(k) => v for (k, v) in pairs(record)) for record in data]
end

function collect_ltlequ_normalized_formulas(records)
    formulas = String[]
    seen = Set{String}()

    for record in records
        haskey(record, "ltlequ_normalized") || continue
        for item in record["ltlequ_normalized"]
            formula = strip(String(item))
            isempty(formula) && continue
            if !(formula in seen)
                push!(formulas, formula)
                push!(seen, formula)
            end
        end
    end

    return formulas
end

"""
    add_all_ltlequ_normalized_from_conformal_file(; input_path=DEFAULT_CONFORMAL_PATH,
                                                    dataset_path=DEFAULT_DATASET_PATH,
                                                    verbose=true)

Load `ltlequ_normalized` formulas from `final_calib_normalized_fixed.json` and add them to
`ltl_dataset.json` using `add_manual_ltl_formulas(...)`, so every formula still goes through
parsing, satisfiability checks, simplification, temporal-policy checks, and duplicate detection.
"""
function add_all_ltlequ_normalized_from_conformal_file(
    ;
    input_path::String = DEFAULT_CONFORMAL_PATH,
    dataset_path::String = DEFAULT_DATASET_PATH,
    verbose::Bool = true,
)
    records = load_json_array(input_path)
    formulas = collect_ltlequ_normalized_formulas(records)

    verbose && begin
        println("Loaded ", length(records), " records from: ", input_path)
        println("Collected ", length(formulas), " unique formulas from `ltlequ_normalized`.")
        println("Adding them through `add_manual_ltl_formulas(...)`...")
        println()
    end

    return add_manual_ltl_formulas(formulas; dataset_path=dataset_path, verbose=verbose)
end

# ----------------------------------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------------------------------

function main()
    println("Loaded ManualLTLAdd.jl.")
    println("Example usage:")
    println("  add_manual_ltl_formula(\"F(prop_1)\")")
    println("  add_manual_ltl_formulas([\"F(prop_1)\", \"G(F(prop_1))\"])")
    println("  add_all_ltlequ_normalized_from_conformal_file()")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end