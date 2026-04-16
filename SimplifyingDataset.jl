include("GenerateLTL.jl")
include("Filter.jl")
include("Satisfiability.jl")

using JSON3

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

const DEFAULT_DATASET_PATH = joinpath(@__DIR__, "dataset", "ltl_dataset.json")

"""
    load_dataset_formulas(dataset_path=DEFAULT_DATASET_PATH)

Load the saved dataset JSON and return it as a vector of records.
Each record is expected to contain at least an `"LTL"` field.
"""
function load_dataset_formulas(dataset_path::String = DEFAULT_DATASET_PATH)
    if !isfile(dataset_path)
        throw(ArgumentError("Dataset file not found: $(dataset_path)"))
    end

    data = JSON3.read(read(dataset_path, String))
    return collect(data)
end

"""
    spot_simplified_formula_string(formula_str; level=3, full_parentheses=true)

Call Spot's simplifier directly on a formula string.
"""
function spot_simplified_formula_string(
    formula_str::AbstractString;
    level::Int = 3,
    full_parentheses::Bool = true,
)
    require_ltlfilt()
    level in 1:3 || throw(ArgumentError("`level` must be 1, 2, or 3."))

    ltlfilt_path = Sys.which("ltlfilt")
    args = String[ltlfilt_path, "-f", String(formula_str), "--simplify=$(level)"]
    if full_parentheses
        push!(args, "-p")
    end

    output = read(Cmd(args), String)
    return strip(output)
end

"""
    is_simplifiable_formula_string(formula_str; level=3)

Return `(is_simplifiable, simplified_str, equivalent)` where:
- `is_simplifiable` is true iff Spot returns a different string
- `simplified_str` is the Spot-simplified string
- `equivalent` is true iff the simplified result is semantically equivalent to the original
"""
function is_simplifiable_formula_string(formula_str::AbstractString; level::Int = 3)
    original = strip(String(formula_str))
    simplified = spot_simplified_formula_string(original; level=level, full_parentheses=true)
    equivalent = formula_strings_are_equivalent(original, simplified)
    return original != simplified, simplified, equivalent
end

"""
    print_simplifiable_formulas(dataset_path=DEFAULT_DATASET_PATH; level=3)

Load formulas from the dataset and print the ones that Spot simplifies to a different form.
"""
function print_simplifiable_formulas(
    dataset_path::String = DEFAULT_DATASET_PATH;
    level::Int = 3,
)
    records = load_dataset_formulas(dataset_path)

    println("Dataset: ", dataset_path)
    println("Checking $(length(records)) formulas for Spot-based simplification...\n")

    simplifiable_count = 0

    for record in records
        haskey(record, :LTL) || haskey(record, "LTL") || continue
        id_value = haskey(record, :id) ? record[:id] : get(record, "id", "?")
        formula_str = haskey(record, :LTL) ? String(record[:LTL]) : String(record["LTL"])

        is_simplifiable, simplified_str, equivalent = is_simplifiable_formula_string(formula_str; level=level)

        if is_simplifiable
            simplifiable_count += 1
            println("ID: ", id_value)
            println("  Original:   ", formula_str)
            println("  Simplified: ", simplified_str)
            println("  Equivalent: ", equivalent)
            println()
        end
    end

    if simplifiable_count == 0
        println("No formulas in the dataset were changed by Spot's simplifier at level $(level).")
    else
        println("Total simplifiable formulas found: ", simplifiable_count)
    end
end


function main()
    print_simplifiable_formulas()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
else
    println("Loaded SimplifyingDataset.jl. Run `print_simplifiable_formulas()` or `main()` to inspect Spot simplifications.")
end
