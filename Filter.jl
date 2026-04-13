
################################################################################################
# Filtering utilities for generated LTL formulas.                                              #
#                                                                                              #
# This file implements three practical filters:                                                #
#   1) trivial formulas, detected via algorithmic simplification / normalization               #
#   2) obviously unsatisfiable formulas                                                        #
#   3) redundant formulas with respect to an existing dataset                                  #
#      (exact, normalized, and optional semantic redundancy)                                   #
#                                                                                              #
# Design notes                                                                                 #
# ------------                                                                                 #
# - The redundancy filter uses exact or normalized canonical keys as cheap prefilters.         #
# - The normalized key implements canonical AP renaming, so formulas that differ only by       #
#   proposition names are treated as duplicates.                                               #
# - A local canonical form is also available for saving formulas after simplification + AP      #
#   renumbering in order of first appearance.                                                  #
# - Optional semantic redundancy is checked by language equivalence against already accepted    #
#   formulas, using an external LTL backend when available.                                    #
# - The triviality filter first tries Spot-backed simplification / normalization, then falls   #
#   back to a small local rewrite system if Spot is unavailable.                               #
# - No benchmark-style filter is applied to nested temporal operators by default.              #
# - The unsatisfiability filter implemented here is conservative and syntactic unless the      #
#   caller separately uses the exact Spot-based satisfiability utilities.                      #
#                                                                                              #
# Literature notes                                                                             #
# ----------------                                                                             #
# - The rewrite-based normalization direction is inspired by normalization work for LTL,       #
#   especially Esparza, Rubio, Sickert (2022),                                                 #
#   "A Simple Rewrite System for the Normalization of Linear Temporal Logic".                  #
# - For exact LTL satisfiability and equivalence, the standard literature is automata/tableau  #
#   based, e.g. Vardi & Wolper (1986) and Gerth, Peled, Vardi & Wolper (1995).                 #
# - Operationally, this file can use Spot's `ltlfilt` as a backend for simplification and      #
#   semantic equivalence checks.                                                               #
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
    require_ltlfilt()

Throw an informative error if Spot's `ltlfilt` executable is not available.
"""
function require_ltlfilt()
    if !ltlfilt_available()
        throw(ArgumentError("Spot's `ltlfilt` executable was not found in PATH. Install Spot and ensure `ltlfilt` is available from the command line."))
    end
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

"""
    canonicalize_formula_local(formula)

Return a locally canonicalized AST obtained by first applying the local simplifier and then
renumbering atomic propositions in order of first appearance as `prop_1`, `prop_2`, ...

Examples:
- `F(prop_4)` becomes `F(prop_1)`
- `(prop_3 U prop_7)` becomes `(prop_1 U prop_2)`
"""
function canonicalize_formula_local(formula::LTLFormula)
    simplified = simplify_formula_local(formula)
    return normalize_formula_local(simplified)
end

"""
    canonical_formula_string_local(formula)

Return the string form of `canonicalize_formula_local(formula)`.
This is intended for dataset export and canonical saved representations.
"""
function canonical_formula_string_local(formula::LTLFormula)
    return formula_to_string(canonicalize_formula_local(formula))
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

function is_eventually_of(inner::LTLFormula, outer::LTLFormula)
    return outer isa UnaryLTL && outer.op == :F && same_formula(inner, outer.child)
end

function is_globally_of(inner::LTLFormula, outer::LTLFormula)
    return outer isa UnaryLTL && outer.op == :G && same_formula(inner, outer.child)
end

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
# Spot-backed simplification / normalization
# ----------------------------------------------------------------------------------------------

"""
    simplify_formula_spot(formula)

Simplify an LTL formula using Spot's `ltlfilt -r` rewrite/simplification backend and return the
simplified formula as a string.
"""
function simplify_formula_spot(formula::LTLFormula)
    require_ltlfilt()

    formula_str = formula_to_string(formula)
    ltlfilt_path = Sys.which("ltlfilt")

    output = read(`$(ltlfilt_path) -f $(formula_str) -r`, String)
    return strip(output)
end

"""
    render_formula_spot(formula; simplify=true, full_parentheses=true)

Render an LTL formula using Spot. This is intended for human-readable display and canonical
string export. By default it applies Spot simplification (`-r`) and requests full parentheses
(`-p`) for readability.
"""
function render_formula_spot(
    formula::LTLFormula;
    simplify::Bool = true,
    full_parentheses::Bool = true,
)
    require_ltlfilt()

    formula_str = formula_to_string(formula)
    ltlfilt_path = Sys.which("ltlfilt")

    cmd = Cmd([
        ltlfilt_path,
        "-f", formula_str,
        simplify ? "-r" : "",
        full_parentheses ? "-p" : "",
    ])
    output = read(cmd, String)
    return strip(output)
end

"""
    normalize_formula_string(formula; prefer_spot=true)

Return a normalized string representation of `formula`. When `prefer_spot=true` and Spot is
available, use Spot's simplification backend first; otherwise fall back to the local rewrite
system.
"""
function normalize_formula_string(formula::LTLFormula; prefer_spot::Bool = true)
    if prefer_spot && ltlfilt_available()
        return render_formula_spot(formula; simplify=true, full_parentheses=true)
    end
    return formula_to_string(simplify_formula_local(formula))
end

# ----------------------------------------------------------------------------------------------
# Benchmark policy filters
# ----------------------------------------------------------------------------------------------

function is_non_temporal(formula::LTLFormula)
    return !has_temporal_operator(formula)
end

# ----------------------------------------------------------------------------------------------
# Local fallback simplification and triviality checks
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

function rewrite_once_local(formula::AP)
    return formula
end

function rewrite_once_local(formula::UnaryLTL)
    child = simplify_formula_local(formula.child)

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
        elseif child isa UnaryLTL && child.op == :G
            return child
        end
    elseif formula.op == :F
        if is_true(child)
            return LTL_TRUE
        elseif is_false(child)
            return LTL_FALSE
        elseif child isa UnaryLTL && child.op == :F
            return child
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

function rewrite_once_local(formula::BinaryLTL)
    left = simplify_formula_local(formula.left)
    right = simplify_formula_local(formula.right)
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
        elseif is_eventually_of(left, right)
            return left
        elseif is_eventually_of(right, left)
            return right
        elseif is_globally_of(left, right)
            return right
        elseif is_globally_of(right, left)
            return left
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
        elseif is_eventually_of(left, right)
            return right
        elseif is_eventually_of(right, left)
            return left
        elseif is_globally_of(left, right)
            return left
        elseif is_globally_of(right, left)
            return right
        end
    elseif op == :->
        if is_false(left) || is_true(right)
            return LTL_TRUE
        elseif is_true(left)
            return right
        elseif is_false(right)
            return simplify_formula_local(UnaryLTL(:!, left))
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
            return simplify_formula_local(UnaryLTL(:!, right))
        elseif is_false(right)
            return simplify_formula_local(UnaryLTL(:!, left))
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
        elseif same_formula(left, right)
            return left
        elseif right isa UnaryLTL && right.op == :F
            return right
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

function simplify_formula_local(formula::LTLFormula)
    current = formula

    while true
        rewritten = rewrite_once_local(current)
        if formula_to_string(rewritten) == formula_to_string(current)
            return rewritten
        end
        current = rewritten
    end
end

"""
    simplify_formula(formula; prefer_spot=true)

Return a simplified string representation of `formula`. When Spot is available and
`prefer_spot=true`, use Spot-backed simplification; otherwise use the local fallback rewrite
system and return its rendered form.
"""
function simplify_formula(formula::LTLFormula; prefer_spot::Bool = true)
    return normalize_formula_string(formula; prefer_spot=prefer_spot)
end

"""
    simplifies_to_constant(formula; prefer_spot=true)

Return `true` if simplification reduces `formula` to `true` or `false`.
"""
function simplifies_to_constant(formula::LTLFormula; prefer_spot::Bool = true)
    simplified_str = simplify_formula(formula; prefer_spot=prefer_spot)
    return simplified_str in ("true", "false")
end


function is_constant_formula(formula::LTLFormula)
    return is_true(formula) || is_false(formula)
end

function is_constant_string(formula_str::AbstractString)
    return formula_str in ("true", "false")
end

function is_trivial(formula::LTLFormula)
    simplified_local = simplify_formula_local(formula)
    simplified_str = simplify_formula(formula)

    if is_constant_string(simplified_str)
        return true
    end

    return formula_reduction_tuple(simplified_local) < formula_reduction_tuple(formula)
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
- `:non_temporal`
- `:unsatisfiable`
- `:redundant`

Semantic redundancy is handled in `filter_formulas(...)`, because it must compare a candidate
against already accepted formulas rather than only precomputed key sets.
"""
function filter_reasons(
    formula::LTLFormula;
    existing_keys::Set{String} = Set{String}(),
    redundancy_mode::Symbol = :normalized,
    require_temporal_operator::Bool = true,
)
    reasons = Symbol[]

    if is_trivial(formula)
        push!(reasons, :trivial)
    end

    if require_temporal_operator && is_non_temporal(formula)
        push!(reasons, :non_temporal)
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
    require_temporal_operator::Bool = true,
)
    return isempty(filter_reasons(
        formula;
        existing_keys=existing_keys,
        redundancy_mode=redundancy_mode,
        require_temporal_operator=require_temporal_operator,
    ))
end

# Attempt to salvage a simplified version of the formula if it passes all filters
function salvage_simplified_formula(
    formula::LTLFormula;
    existing_keys::Set{String} = Set{String}(),
    redundancy_mode::Symbol = :normalized,
    require_temporal_operator::Bool = true,
)
    simplified = simplify_formula_local(formula)

    if formula_to_string(simplified) == formula_to_string(formula)
        return nothing
    end

    reasons = filter_reasons(
        simplified;
        existing_keys=existing_keys,
        redundancy_mode=redundancy_mode,
        require_temporal_operator=require_temporal_operator,
    )
    if isempty(reasons)
        return simplified
    end

    return nothing
end

"""
    filter_formulas(formulas; existing_keys=Set{String}(), redundancy_mode=:normalized,
                    mutate_keys=true, semantic_redundancy=false,
                    semantic_backend=:spot, semantic_existing_formulas=LTLFormula[],
                    require_temporal_operator=true)

Filter a batch of formulas and return `(accepted, rejected)` where:
- `accepted` is a vector of formulas that passed all filters
- `rejected` is a vector of `(formula, reasons)` tuples

If a formula is rejected only because it is `:trivial`, the function attempts to salvage its
simplified version. If the simplified formula passes all filters, it is accepted instead.

If `semantic_redundancy=true`, the function also checks equivalence against already accepted
formulas and `semantic_existing_formulas`. If two formulas are semantically equivalent, the
minimal representative according to `formula_preference_tuple` is kept.

If `require_temporal_operator=true`, purely propositional formulas are rejected so the benchmark
focuses on temporal reasoning.

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
    require_temporal_operator::Bool = true,
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
        reasons = filter_reasons(
            candidate;
            existing_keys=existing_keys,
            redundancy_mode=redundancy_mode,
            require_temporal_operator=require_temporal_operator,
        )

        if reasons == [:trivial]
            simplified = salvage_simplified_formula(
                candidate;
                existing_keys=existing_keys,
                redundancy_mode=redundancy_mode,
                require_temporal_operator=require_temporal_operator,
            )
            if !isnothing(simplified)
                candidate = simplified
                reasons = filter_reasons(
                    candidate;
                    existing_keys=existing_keys,
                    redundancy_mode=redundancy_mode,
                    require_temporal_operator=require_temporal_operator,
                )
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
    println("  render_formula_spot(formulas[1]; simplify=true, full_parentheses=true)")
    println("  simplified_local = formula_to_string(simplify_formula_local(formulas[1]))")
end