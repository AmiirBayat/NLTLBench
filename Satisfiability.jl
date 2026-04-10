################################################################################################
# Satisfiability utilities for LTL formulas.                                                   #
#                                                                                              #
# Design notes                                                                                 #
# ------------                                                                                 #
# - This file provides an exact satisfiability check through Spot's `ltlfilt` executable.      #
# - A formula is unsatisfiable iff it is equivalent to `false`.                                #
# - Therefore, satisfiability is decided here by checking semantic equivalence to `false`.     #
#                                                                                              #
# Literature notes                                                                             #
# ----------------                                                                             #
# - The automata-theoretic view of LTL semantics and decision procedures goes back to          #
#   Vardi & Wolper (1986), and related on-the-fly tableau/automata methods were developed      #
#   by Gerth, Peled, Vardi, and Wolper (1995).                                                 #
# - Operationally, this file uses Spot's `ltlfilt --equivalent-to` interface as the backend.   #
################################################################################################

# This file assumes that GenerateLTL.jl has already been included, so the following are
# already defined in scope:
#   - abstract type LTLFormula
#   - structs AP, UnaryLTL, BinaryLTL
#   - formula_to_string(...)

# ----------------------------------------------------------------------------------------------
# Backend availability
# ----------------------------------------------------------------------------------------------

"""
    ltlfilt_available()

Return `true` if Spot's `ltlfilt` executable is available on the current machine.
"""
function ltlfilt_available()
    return Sys.which("ltlfilt") !== nothing
end

"""
    require_ltlfilt()

Throw an informative error if Spot's `ltlfilt` executable is not available.
"""
function require_ltlfilt()
    if !ltlfilt_available()
        throw(ArgumentError("Spot's `ltlfilt` executable was not found in PATH. Install Spot and ensure `ltlfilt` is available from the command line."))
    end
end

# ----------------------------------------------------------------------------------------------
# Exact semantic checks via Spot
# ----------------------------------------------------------------------------------------------

"""
    semantically_equivalent_to_constant(formula, constant_name)

Return `true` iff `formula` is semantically equivalent to the constant formula named by
`constant_name`, which must be either `"true"` or `"false"`.
"""
function semantically_equivalent_to_constant(formula::LTLFormula, constant_name::String)
    constant_name in ("true", "false") || throw(ArgumentError("`constant_name` must be either \"true\" or \"false\"."))

    require_ltlfilt()

    ltlfilt_path = Sys.which("ltlfilt")
    formula_str = formula_to_string(formula)

    cmd = `$(ltlfilt_path) -f $(formula_str) --equivalent-to $(constant_name) -q`
    process = run(cmd; wait=false)
    wait(process)

    if process.exitcode == 0
        return true
    elseif process.exitcode == 1
        return false
    else
        throw(ErrorException("`ltlfilt` failed while checking whether `$(formula_str)` is equivalent to `$(constant_name)` (exit code $(process.exitcode))."))
    end
end

"""
    is_unsatisfiable_exact(formula)

Return `true` iff `formula` is unsatisfiable.

This is an exact semantic check delegated to Spot: an LTL formula is unsatisfiable iff it is
semantically equivalent to `false`.
"""
function is_unsatisfiable_exact(formula::LTLFormula)
    return semantically_equivalent_to_constant(formula, "false")
end

"""
    is_satisfiable_exact(formula)

Return `true` iff `formula` is satisfiable.

This is an exact semantic check delegated to Spot.
"""
function is_satisfiable_exact(formula::LTLFormula)
    return !is_unsatisfiable_exact(formula)
end

"""
    is_tautology_exact(formula)

Return `true` iff `formula` is valid, i.e. equivalent to `true`.
"""
function is_tautology_exact(formula::LTLFormula)
    return semantically_equivalent_to_constant(formula, "true")
end

"""
    satisfiability_status(formula)

Return one of:
- `:satisfiable`
- `:unsatisfiable`
"""
function satisfiability_status(formula::LTLFormula)
    return is_unsatisfiable_exact(formula) ? :unsatisfiable : :satisfiable
end

# ----------------------------------------------------------------------------------------------
# Batch helpers
# ----------------------------------------------------------------------------------------------

"""
    classify_satisfiability(formulas)

Return a vector of `(formula, status)` tuples, where `status` is `:satisfiable` or
`:unsatisfiable`.
"""
function classify_satisfiability(formulas::Vector{<:LTLFormula})
    return [(formula, satisfiability_status(formula)) for formula in formulas]
end

"""
    split_by_satisfiability(formulas)

Return `(satisfiable, unsatisfiable)` where both outputs are vectors of formulas.
"""
function split_by_satisfiability(formulas::Vector{<:LTLFormula})
    satisfiable = LTLFormula[]
    unsatisfiable = LTLFormula[]

    for formula in formulas
        if is_satisfiable_exact(formula)
            push!(satisfiable, formula)
        else
            push!(unsatisfiable, formula)
        end
    end

    return satisfiable, unsatisfiable
end

# ----------------------------------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------------------------------

if abspath(PROGRAM_FILE) == @__FILE__
    println("Satisfiability.jl loaded. Include GenerateLTL.jl first, then call:")
    println("  is_satisfiable_exact(formula)")
    println("  is_unsatisfiable_exact(formula)")
    println("  is_tautology_exact(formula)")
    println("  satisfiable, unsatisfiable = split_by_satisfiability(formulas)")
end
