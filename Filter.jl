

################################################################################################
# Filtering utilities for generated LTL formulas.                                              #
#                                                                                              #
# This file implements three practical filters:                                                #
#   1) trivial formulas, detected via rewrite-based simplification                             #
#   2) obviously unsatisfiable formulas                                                        #
#   3) redundant formulas with respect to an existing dataset                                  #
#      (exact, normalized, and optional semantic redundancy)                                   #
#                                                                                              #
# Design notes                                                                                 #
# ------------                                                                                 #
# - The redundancy filter uses exact or normalized canonical keys as cheap prefilters.         #
# - The normalized key implements canonical AP renaming, so formulas that differ only by       #
#   proposition names are treated as duplicates.                                               #
# - Optional semantic redundancy is checked by language equivalence against already accepted    #
#   formulas, using an external LTL backend when available.                                    #
# - The triviality filter is rewrite-based: we first simplify the formula using sound          #
#   equivalence-preserving rewrite rules, then classify the result.                            #
# - The unsatisfiability filter implemented here is conservative and syntactic. It catches     #
#   obvious contradictions, but it is NOT a complete LTL satisfiability solver.                #
#                                                                                              #
# Literature notes                                                                             #
# ----------------                                                                             #
# - The rewrite-based design is inspired by normalization work for LTL, especially:            #
#     Esparza, Rubio, Sickert (2022),                                                          #
#     "A Simple Rewrite System for the Normalization of Linear Temporal Logic".                #
# - For exact LTL satisfiability and equivalence, the standard literature is automata/tableau  #
#   based, e.g. Vardi & Wolper (1986) and Gerth, Peled, Vardi & Wolper (1995).                 #
# - The semantic redundancy check implemented below uses equivalence of omega-languages,        #
#   operationalized through an external backend such as Spot's `ltlfilt --equivalent-to`.      #
################################################################################################
#
# ----------------------------------------------------------------------------------------------
# Semantic redundancy checks
# ----------------------------------------------------------------------------------------------

"""
    ltlfilt_available()

Return `true` if Spot's `ltlfilt` executable is available on the current machine.
"""
function ltlfilt_available()
    return Sys.which("ltlfilt") !== nothing
end

"""
    semantically_equivalent_spot(formula_a, formula_b)

Check semantic equivalence using Spot's `ltlfilt --equivalent-to`.
Returns `true` iff the two formulas define the same omega-language.
Throws an error if `ltlfilt` is not available or the command fails unexpectedly.
"""
function semantically_equivalent_spot(formula_a::LTLFormula, formula_b::LTLFormula)
    ltlfilt_path = Sys.which("ltlfilt")
    isnothing(ltlfilt_path) && throw(ArgumentError("Spot's `ltlfilt` executable was not found in PATH."))

    formula_a_str = formula_to_string(formula_a)
    formula_b_str = formula_to_string(formula_b)

    cmd = `$(ltlfilt_path) -f $(formula_a_str) --equivalent-to $(formula_b_str) -q`
    process = run(cmd; wait=false)
    wait(process)

    if process.exitcode == 0
        return true
    elseif process.exitcode == 1
        return false
    else
        throw(ErrorException("`ltlfilt` failed while checking semantic equivalence between `$(formula_a_str)` and `$(formula_b_str)` with exit code $(process.exitcode)."))
    end
end

function formula_preference_tuple(formula::LTLFormula)
    counts = operator_counts(formula)
    total_ops = isempty(counts) ? 0 : sum(values(counts))
    return (
        ast_size(formula),
        temporal_depth(formula),
        total_ops,
        length(formula_to_string(formula)),
        formula_to_string(formula),
    )
end

function preferred_formula(a::LTLFormula, b::LTLFormula)
    return formula_preference_tuple(a) <= formula_preference_tuple(b) ? a : b
end

"""
    find_semantic_redundancy(formula, accepted_formulas; backend=:spot)

Return the index of an accepted formula that is semantically equivalent to `formula`, or
`nothing` if none is found.
"""
function find_semantic_redundancy(
    formula::LTLFormula,
    accepted_formulas::Vector{<:LTLFormula};
    backend::Symbol = :spot,
)
    if backend == :none
        return nothing
    elseif backend != :spot
        throw(ArgumentError("Unsupported semantic redundancy backend: $(backend). Use :spot or :none."))
    end

    for (idx, accepted_formula) in enumerate(accepted_formulas)
        if semantically_equivalent_spot(formula, accepted_formula)
            return idx
        end
    end

    return nothing
end

# This file assumes that GenerateLTL.jl has already been included, so the following are
# already defined in scope:
#   - abstract type LTLFormula
#   - structs AP, UnaryLTL, BinaryLTL
#   - formula_to_string(...)
#   - ast_size(...)
#   - temporal_depth(...)

# ----------------------------------------------------------------------------------------------
# Formula normalization for redundancy checking
# ----------------------------------------------------------------------------------------------

function _normalize_formula_local(formula::AP, mapping::Dict{String,String}, counter::Base.RefValue{Int})
    if formula.name in ("true", "false")
        return AP(formula.name)
    end
    if !haskey(mapping, formula.name)
        mapping[formula.name] = "prop_$(counter[])"
        counter[] += 1
    end
    return AP(mapping[formula.name])
end

function _normalize_formula_local(formula::UnaryLTL, mapping::Dict{String,String}, counter::Base.RefValue{Int})
    return UnaryLTL(formula.op, _normalize_formula_local(formula.child, mapping, counter))
end

function _normalize_formula_local(formula::BinaryLTL, mapping::Dict{String,String}, counter::Base.RefValue{Int})
    return BinaryLTL(
        formula.op,
        _normalize_formula_local(formula.left, mapping, counter),
        _normalize_formula_local(formula.right, mapping, counter),
    )
end

function normalize_formula_local(formula::LTLFormula)
    mapping = Dict{String,String}()
    counter = Ref(1)
    return _normalize_formula_local(formula, mapping, counter)
end

function normalized_formula_string_local(formula::LTLFormula)
    return formula_to_string(normalize_formula_local(formula))
end

function redundancy_key(formula::LTLFormula; mode::Symbol = :normalized)
    if mode == :exact
        return formula_to_string(formula)
    elseif mode == :normalized
        return normalized_formula_string_local(formula)
    else
        throw(ArgumentError("Unsupported redundancy mode: $(mode). Use :exact or :normalized."))
    end
end

# ----------------------------------------------------------------------------------------------
# Structural helpers
# ----------------------------------------------------------------------------------------------

is_true(formula::AP) = formula.name == "true"
is_true(formula::LTLFormula) = false

is_false(formula::AP) = formula.name == "false"
is_false(formula::LTLFormula) = false

is_negation_of(a::LTLFormula, b::LTLFormula) = b isa UnaryLTL && b.op == :! && formula_to_string(a) == formula_to_string(b.child)
is_negation_pair(a::LTLFormula, b::LTLFormula) = is_negation_of(a, b) || is_negation_of(b, a)

same_formula(a::LTLFormula, b::LTLFormula) = formula_to_string(a) == formula_to_string(b)

function has_temporal_operator(formula::AP)
    return false
end

function has_temporal_operator(formula::UnaryLTL)
    return (formula.op in (:X, :F, :G)) || has_temporal_operator(formula.child)
end

function has_temporal_operator(formula::BinaryLTL)
    return (formula.op in (:U, :R, :W)) || has_temporal_operator(formula.left) || has_temporal_operator(formula.right)
end

# ----------------------------------------------------------------------------------------------
# Rewrite-based simplification and triviality checks
# design inspired by Esparza, Rubio, Sickert (2022), "A Simple Rewrite System for the Normalization of Linear Temporal Logic"
# ----------------------------------------------------------------------------------------------

const LTL_TRUE = AP("true")
const LTL_FALSE = AP("false")

function formula_complexity_tuple(formula::LTLFormula)
    return (
        ast_size(formula),
        temporal_depth(formula),
        length(formula_to_string(formula)),
        formula_to_string(formula),
    )
end

function formula_reduction_tuple(formula::LTLFormula)
    return (
        ast_size(formula),
        temporal_depth(formula),
        length(formula_to_string(formula)),
    )
end

function simpler_formula(a::LTLFormula, b::LTLFormula)
    return formula_complexity_tuple(a) <= formula_complexity_tuple(b) ? a : b
end

function rewrite_once(formula::AP)
    return formula
end

function rewrite_once(formula::UnaryLTL)
    child = simplify_formula(formula.child)

    if formula.op == :!
        if is_true(child)
            return LTL_FALSE
        elseif is_false(child)
            return LTL_TRUE
        elseif child isa UnaryLTL && child.op == :!
            return child.child
        end
    elseif formula.op == :G
        if is_true(child)
            return LTL_TRUE
        elseif is_false(child)
            return LTL_FALSE
        end
    elseif formula.op == :F
        if is_true(child)
            return LTL_TRUE
        elseif is_false(child)
            return LTL_FALSE
        end
    elseif formula.op == :X
        if is_true(child)
            return LTL_TRUE
        elseif is_false(child)
            return LTL_FALSE
        end
    end

    return UnaryLTL(formula.op, child)
end

function rewrite_once(formula::BinaryLTL)
    left = simplify_formula(formula.left)
    right = simplify_formula(formula.right)
    op = formula.op

    if op == :&
        if is_false(left) || is_false(right)
            return LTL_FALSE
        elseif is_true(left)
            return right
        elseif is_true(right)
            return left
        elseif same_formula(left, right)
            return left
        elseif is_negation_pair(left, right)
            return LTL_FALSE
        end
    elseif op == :|
        if is_true(left) || is_true(right)
            return LTL_TRUE
        elseif is_false(left)
            return right
        elseif is_false(right)
            return left
        elseif same_formula(left, right)
            return left
        elseif is_negation_pair(left, right)
            return LTL_TRUE
        end
    elseif op == :->
        if is_false(left) || is_true(right)
            return LTL_TRUE
        elseif is_true(left)
            return right
        elseif is_false(right)
            return simplify_formula(UnaryLTL(:!, left))
        elseif same_formula(left, right)
            return LTL_TRUE
        end
    elseif op == Symbol("<->")
        if same_formula(left, right)
            return LTL_TRUE
        elseif is_true(left)
            return right
        elseif is_true(right)
            return left
        elseif is_false(left)
            return simplify_formula(UnaryLTL(:!, right))
        elseif is_false(right)
            return simplify_formula(UnaryLTL(:!, left))
        end
    elseif op == :U
        if is_true(right)
            return LTL_TRUE
        elseif is_false(right)
            return LTL_FALSE
        elseif is_false(left)
            return right
        elseif is_true(left)
            return UnaryLTL(:F, right)
        end
    elseif op == :R
        if is_true(left)
            return right
        elseif is_false(left)
            return UnaryLTL(:G, right)
        elseif is_true(right)
            return LTL_TRUE
        elseif is_false(right)
            return LTL_FALSE
        end
    elseif op == :W
        if is_true(left) || is_true(right)
            return LTL_TRUE
        elseif is_false(left)
            return right
        elseif is_false(right)
            return UnaryLTL(:G, left)
        end
    end

    if op in (:&, :|, Symbol("<->"))
        ordered_left = simpler_formula(left, right)
        ordered_right = same_formula(ordered_left, left) ? right : left
        return BinaryLTL(op, ordered_left, ordered_right)
    end

    return BinaryLTL(op, left, right)
end

function simplify_formula(formula::LTLFormula)
    current = formula

    while true
        rewritten = rewrite_once(current)
        if formula_to_string(rewritten) == formula_to_string(current)
            return rewritten
        end
        current = rewritten
    end
end

function is_constant_formula(formula::LTLFormula)
    return is_true(formula) || is_false(formula)
end

"""
    is_trivial(formula)

Return `true` if the formula simplifies, under sound rewrite rules, to a constant or to a
strictly smaller formula according to a reduction measure based on structural complexity.
Pure canonical reordering does not count as triviality.
"""
function is_trivial(formula::LTLFormula)
    simplified = simplify_formula(formula)

    if is_constant_formula(simplified)
        return true
    end

    return formula_reduction_tuple(simplified) < formula_reduction_tuple(formula)
end

# ----------------------------------------------------------------------------------------------
# Conservative unsatisfiability checks
# ----------------------------------------------------------------------------------------------

"""
    is_unsatisfiable(formula)

Conservative syntactic unsatisfiability detector.

Important:
This is NOT a complete LTL satisfiability algorithm. It only catches obvious contradictions.
For a complete check, use an automata/tableau-based backend such as Spot, BLACK, or another full LTL satisfiability solver.
"""
function is_unsatisfiable(formula::LTLFormula)
    return _is_unsatisfiable(formula)
end

function _is_unsatisfiable(formula::AP)
    return is_false(formula)
end

function _is_unsatisfiable(formula::UnaryLTL)
    child = formula.child

    if formula.op == :G && is_false(child)
        return true
    end

    if formula.op == :F && is_false(child)
        return true
    end

    if formula.op == :X && is_false(child)
        return true
    end

    return _is_unsatisfiable(child)
end

function _is_unsatisfiable(formula::BinaryLTL)
    left = formula.left
    right = formula.right
    op = formula.op

    if op == :&
        if is_negation_pair(left, right)
            return true
        end
        if _is_unsatisfiable(left) || _is_unsatisfiable(right)
            return true
        end
    end

    if op == :| && _is_unsatisfiable(left) && _is_unsatisfiable(right)
        return true
    end

    # G(p) & G(!p) shows up as a conjunction of two satisfiable conjuncts whose combination is not.
    if op == :& && left isa UnaryLTL && right isa UnaryLTL && left.op == :G && right.op == :G && is_negation_pair(left.child, right.child)
        return true
    end

    return _is_unsatisfiable(left) || _is_unsatisfiable(right)
end

# ----------------------------------------------------------------------------------------------
# Redundancy checks
# ----------------------------------------------------------------------------------------------

"""
    is_redundant(formula, existing_keys; mode=:normalized)

Check whether `formula` is redundant with respect to a set of already accepted redundancy keys.
Use `mode=:exact` for literal string identity, or `mode=:normalized` to ignore AP renaming.
"""
function is_redundant(formula::LTLFormula, existing_keys::Set{String}; mode::Symbol = :normalized)
    return redundancy_key(formula; mode=mode) in existing_keys
end

function add_redundancy_key!(existing_keys::Set{String}, formula::LTLFormula; mode::Symbol = :normalized)
    push!(existing_keys, redundancy_key(formula; mode=mode))
    return existing_keys
end

# ----------------------------------------------------------------------------------------------
# High-level filtering interface
# ----------------------------------------------------------------------------------------------

"""
    filter_reasons(formula; existing_keys=Set{String}(), redundancy_mode=:normalized)

Return a vector of rejection reasons among:
- `:trivial`
- `:unsatisfiable`
- `:redundant`

Semantic redundancy is handled in `filter_formulas(...)`, because it must compare a candidate
against already accepted formulas rather than only precomputed key sets.
"""
function filter_reasons(
    formula::LTLFormula;
    existing_keys::Set{String} = Set{String}(),
    redundancy_mode::Symbol = :normalized,
)
    reasons = Symbol[]

    if is_trivial(formula)
        push!(reasons, :trivial)
    end

    if is_unsatisfiable(formula)
        push!(reasons, :unsatisfiable)
    end

    if !isempty(existing_keys) && is_redundant(formula, existing_keys; mode=redundancy_mode)
        push!(reasons, :redundant)
    end

    return reasons
end


function passes_filters(
    formula::LTLFormula;
    existing_keys::Set{String} = Set{String}(),
    redundancy_mode::Symbol = :normalized,
)
    return isempty(filter_reasons(formula; existing_keys=existing_keys, redundancy_mode=redundancy_mode))
end

# Attempt to salvage a simplified version of the formula if it passes all filters
function salvage_simplified_formula(
    formula::LTLFormula;
    existing_keys::Set{String} = Set{String}(),
    redundancy_mode::Symbol = :normalized,
)
    simplified = simplify_formula(formula)

    # If no actual simplification happened, nothing to salvage
    if formula_to_string(simplified) == formula_to_string(formula)
        return nothing
    end

    reasons = filter_reasons(simplified; existing_keys=existing_keys, redundancy_mode=redundancy_mode)
    if isempty(reasons)
        return simplified
    end

    return nothing
end

"""
    filter_formulas(formulas; existing_keys=Set{String}(), redundancy_mode=:normalized,
                    mutate_keys=true, semantic_redundancy=false,
                    semantic_backend=:spot, semantic_existing_formulas=LTLFormula[])

Filter a batch of formulas and return `(accepted, rejected)` where:
- `accepted` is a vector of formulas that passed all filters
- `rejected` is a vector of `(formula, reasons)` tuples

If a formula is rejected only because it is `:trivial`, the function attempts to salvage its
simplified version. If the simplified formula passes all filters, it is accepted instead.

If `semantic_redundancy=true`, the function also checks equivalence against already accepted
formulas and `semantic_existing_formulas`. If two formulas are semantically equivalent, the
minimal representative according to `formula_preference_tuple` is kept.

If `mutate_keys=true`, accepted formulas are added to `existing_keys` as they are accepted,
so exact/normalized redundancy is checked both against prior dataset entries and within the
current batch.
"""
function filter_formulas(
    formulas::Vector{<:LTLFormula};
    existing_keys::Set{String} = Set{String}(),
    redundancy_mode::Symbol = :normalized,
    mutate_keys::Bool = true,
    semantic_redundancy::Bool = false,
    semantic_backend::Symbol = :spot,
    semantic_existing_formulas::Vector{<:LTLFormula} = LTLFormula[],
)
    accepted = LTLFormula[]
    rejected = Vector{Tuple{LTLFormula, Vector{Symbol}}}()

    if semantic_redundancy && semantic_backend == :spot && !ltlfilt_available()
        throw(ArgumentError("Semantic redundancy with backend :spot was requested, but Spot's `ltlfilt` executable was not found in PATH."))
    end

    for formula in semantic_existing_formulas
        push!(accepted, formula)
        if mutate_keys
            add_redundancy_key!(existing_keys, formula; mode=redundancy_mode)
        end
    end

    newly_accepted = LTLFormula[]

    for formula in formulas
        candidate = formula
        reasons = filter_reasons(candidate; existing_keys=existing_keys, redundancy_mode=redundancy_mode)

        if reasons == [:trivial]
            simplified = salvage_simplified_formula(
                candidate;
                existing_keys=existing_keys,
                redundancy_mode=redundancy_mode,
            )
            if !isnothing(simplified)
                candidate = simplified
                reasons = filter_reasons(candidate; existing_keys=existing_keys, redundancy_mode=redundancy_mode)
            end
        end

        if !isempty(reasons)
            push!(rejected, (formula, reasons))
            continue
        end

        if semantic_redundancy
            semantic_pool = vcat(semantic_existing_formulas, newly_accepted)
            equivalent_idx = find_semantic_redundancy(candidate, semantic_pool; backend=semantic_backend)

            if !isnothing(equivalent_idx)
                existing_formula = semantic_pool[equivalent_idx]
                winner = preferred_formula(candidate, existing_formula)

                if formula_to_string(winner) == formula_to_string(existing_formula)
                    push!(rejected, (formula, [:semantic_redundant]))
                    continue
                else
                    if equivalent_idx <= length(semantic_existing_formulas)
                        push!(rejected, (existing_formula, [:semantic_redundant_replaced]))
                    else
                        local_idx = equivalent_idx - length(semantic_existing_formulas)
                        replaced_formula = newly_accepted[local_idx]
                        newly_accepted[local_idx] = candidate
                        if mutate_keys
                            add_redundancy_key!(existing_keys, candidate; mode=redundancy_mode)
                        end
                        push!(rejected, (replaced_formula, [:semantic_redundant_replaced]))
                        continue
                    end
                end
            end
        end

        push!(newly_accepted, candidate)
        if mutate_keys
            add_redundancy_key!(existing_keys, candidate; mode=redundancy_mode)
        end
    end

    return vcat(semantic_existing_formulas, newly_accepted), rejected
end

# ----------------------------------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------------------------------

if abspath(PROGRAM_FILE) == @__FILE__
    println("Filter.jl loaded. Include GenerateLTL.jl first, then call:")
    println("  accepted, rejected = filter_formulas(formulas; redundancy_mode=:normalized)")
    println("  accepted, rejected = filter_formulas(formulas; redundancy_mode=:normalized, semantic_redundancy=true)")
    println("  simplified = simplify_formula(formulas[1])")
end