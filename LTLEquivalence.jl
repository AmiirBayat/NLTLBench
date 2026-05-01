

"""
LTLEquivalence.jl

Check whether two LTL formulas are semantically equivalent using Spot's `ltlfilt`.

Example usage in the Julia REPL:
    include("LTLEquivalence.jl")
    are_equivalent("F(prop_1)", "true U prop_1")

Or run this file directly after editing `formula1` and `formula2` in `main()`.
"""

function require_ltlfilt()
    isnothing(Sys.which("ltlfilt")) && throw(ArgumentError(
        "Spot's `ltlfilt` was not found in PATH. Install Spot and make sure `ltlfilt` is available."
    ))
end

"""
    are_equivalent(formula1, formula2) -> Bool

Return `true` iff the two LTL formulas are semantically equivalent.
This uses Spot's exact equivalence check:
    ltlfilt -f <formula1> --equivalent-to <formula2> -q
"""
function are_equivalent(formula1::AbstractString, formula2::AbstractString)::Bool
    require_ltlfilt()
    ltlfilt_path = Sys.which("ltlfilt")

    cmd = `$(ltlfilt_path) -f $(String(formula1)) --equivalent-to $(String(formula2)) -q`
    process = run(cmd; wait=false)
    wait(process)

    if process.exitcode == 0
        return true
    elseif process.exitcode == 1
        return false
    else
        throw(ErrorException(
            "`ltlfilt` failed while checking equivalence between `$(formula1)` and `$(formula2)` (exit code $(process.exitcode))."
        ))
    end
end

"""
    print_equivalence_result(formula1, formula2)

Print a readable equivalence result for two formulas.
"""
function print_equivalence_result(formula1::AbstractString, formula2::AbstractString)
    equivalent = are_equivalent(formula1, formula2)

    println("Formula 1: ", formula1)
    println("Formula 2: ", formula2)
    println()

    if equivalent
        println("Result: The two LTL formulas are semantically equivalent.")
    else
        println("Result: The two LTL formulas are NOT semantically equivalent.")
    end
end

function main()
    # Edit these two formulas and run the file.
    formula1 = "F(prop_1) <-> (prop_2 <-> (prop_3 | prop_2))"
    formula2 = "F(prop_1) <-> (prop_2 <-> (prop_3))"

    print_equivalence_result(formula1, formula2)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end