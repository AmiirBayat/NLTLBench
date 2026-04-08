################################################################################################
# This code algorithmically generates syntactically valid LTL formulas from a set of atomic   #
# propositions and a selectable set of temporal operators.                                     #
################################################################################################

using Random

# ----------------------------------------------------------------------------------------------
# LTL AST
# ----------------------------------------------------------------------------------------------

abstract type LTLFormula end

struct AP <: LTLFormula
    name::String
end

struct UnaryLTL <: LTLFormula
    op::Symbol
    child::LTLFormula
end

struct BinaryLTL <: LTLFormula
    op::Symbol
    left::LTLFormula
    right::LTLFormula
end

# ----------------------------------------------------------------------------------------------
# Generator configuration
# ----------------------------------------------------------------------------------------------

Base.@kwdef struct LTLGeneratorConfig
    max_depth::Int = 4
    p_atom_at_max_depth::Float64 = 1.0
    p_atom_before_max_depth::Float64 = 0.25
    unary_weight::Float64 = 0.35
    binary_weight::Float64 = 0.40
    allow_boolean_constants::Bool = false
    boolean_constants::Vector{String} = ["true", "false"]
    boolean_unary_ops::Vector{Symbol} = [:!]
    boolean_binary_ops::Vector{Symbol} = [:&, :|, :->]
end

# ----------------------------------------------------------------------------------------------
# Utility predicates
# ----------------------------------------------------------------------------------------------

is_temporal_unary(op::Symbol) = op in (:X, :F, :G)
is_temporal_binary(op::Symbol) = op in (:U, :R, :W)
is_boolean_unary(op::Symbol) = op == :!
is_boolean_binary(op::Symbol) = op in (:&, :|, :->)
is_unary(op::Symbol) = is_temporal_unary(op) || is_boolean_unary(op)
is_binary(op::Symbol) = is_temporal_binary(op) || is_boolean_binary(op)

function split_temporal_operators(temporal_ops::Vector{Symbol})
    unary_ops = Symbol[]
    binary_ops = Symbol[]

    for op in temporal_ops
        if is_temporal_unary(op)
            push!(unary_ops, op)
        elseif is_temporal_binary(op)
            push!(binary_ops, op)
        else
            throw(ArgumentError("Unsupported temporal operator: $(op). Supported unary: X, F, G. Supported binary: U, R, W."))
        end
    end

    return unary_ops, binary_ops
end

# ----------------------------------------------------------------------------------------------
# Random formula generation
# ----------------------------------------------------------------------------------------------

"""
    random_atom(atomic_props, cfg, rng)

Return a random atomic proposition, or a Boolean constant if enabled.
"""
function random_atom(atomic_props::Vector{String}, cfg::LTLGeneratorConfig, rng::AbstractRNG)
    choices = copy(atomic_props)
    if cfg.allow_boolean_constants
        append!(choices, cfg.boolean_constants)
    end

    isempty(choices) && throw(ArgumentError("At least one atomic proposition or Boolean constant must be available."))
    return AP(rand(rng, choices))
end

"""
    random_ltl_formula(atomic_props, temporal_ops; cfg=LTLGeneratorConfig(), rng=Random.default_rng())

Generate a random syntactically valid LTL formula from:
- `atomic_props`: e.g. ["prop_1", "prop_2", "prop_3"]
- `temporal_ops`: subset of [:X, :F, :G, :U, :R, :W]

Boolean operators are always available through the generator config.
"""
function random_ltl_formula(
    atomic_props::Vector{String},
    temporal_ops::Vector{Symbol};
    cfg::LTLGeneratorConfig = LTLGeneratorConfig(),
    rng::AbstractRNG = Random.default_rng(),
)
    temporal_unary_ops, temporal_binary_ops = split_temporal_operators(temporal_ops)

    available_unary_ops = vcat(cfg.boolean_unary_ops, temporal_unary_ops)
    available_binary_ops = vcat(cfg.boolean_binary_ops, temporal_binary_ops)

    if isempty(atomic_props) && !cfg.allow_boolean_constants
        throw(ArgumentError("`atomic_props` cannot be empty unless `allow_boolean_constants=true`."))
    end

    return _random_ltl_formula(atomic_props, available_unary_ops, available_binary_ops, cfg, 0, rng)
end

function _random_ltl_formula(
    atomic_props::Vector{String},
    unary_ops::Vector{Symbol},
    binary_ops::Vector{Symbol},
    cfg::LTLGeneratorConfig,
    current_depth::Int,
    rng::AbstractRNG,
)::LTLFormula
    if current_depth >= cfg.max_depth
        return random_atom(atomic_props, cfg, rng)
    end

    p_atom = current_depth == cfg.max_depth ? cfg.p_atom_at_max_depth : cfg.p_atom_before_max_depth
    if rand(rng) < p_atom
        return random_atom(atomic_props, cfg, rng)
    end

    atom_weight = 1.0
    unary_weight = isempty(unary_ops) ? 0.0 : cfg.unary_weight
    binary_weight = isempty(binary_ops) ? 0.0 : cfg.binary_weight

    total_weight = atom_weight + unary_weight + binary_weight
    total_weight <= 0 && return random_atom(atomic_props, cfg, rng)

    draw = rand(rng) * total_weight

    if draw < atom_weight
        return random_atom(atomic_props, cfg, rng)
    elseif draw < atom_weight + unary_weight
        op = rand(rng, unary_ops)
        child = _random_ltl_formula(atomic_props, unary_ops, binary_ops, cfg, current_depth + 1, rng)
        return UnaryLTL(op, child)
    else
        op = rand(rng, binary_ops)
        left = _random_ltl_formula(atomic_props, unary_ops, binary_ops, cfg, current_depth + 1, rng)
        right = _random_ltl_formula(atomic_props, unary_ops, binary_ops, cfg, current_depth + 1, rng)
        return BinaryLTL(op, left, right)
    end
end

# ----------------------------------------------------------------------------------------------
# Formula rendering
# ----------------------------------------------------------------------------------------------

function formula_to_string(formula::AP)::String
    return formula.name
end

function formula_to_string(formula::UnaryLTL)::String
    child_str = formula_to_string(formula.child)
    if formula.op == :!
        return "!($(child_str))"
    else
        return "$(formula.op)($(child_str))"
    end
end

function formula_to_string(formula::BinaryLTL)::String
    left_str = formula_to_string(formula.left)
    right_str = formula_to_string(formula.right)
    return "($(left_str) $(formula.op) $(right_str))"
end

Base.string(formula::LTLFormula) = formula_to_string(formula)

# ----------------------------------------------------------------------------------------------
# Metadata helpers
# ----------------------------------------------------------------------------------------------

function ast_size(formula::AP)::Int
    return 1
end

function ast_size(formula::UnaryLTL)::Int
    return 1 + ast_size(formula.child)
end

function ast_size(formula::BinaryLTL)::Int
    return 1 + ast_size(formula.left) + ast_size(formula.right)
end

function ast_depth(formula::AP)::Int
    return 0
end

function ast_depth(formula::UnaryLTL)::Int
    return 1 + ast_depth(formula.child)
end

function ast_depth(formula::BinaryLTL)::Int
    return 1 + max(ast_depth(formula.left), ast_depth(formula.right))
end

function temporal_depth(formula::AP)::Int
    return 0
end

function temporal_depth(formula::UnaryLTL)::Int
    child_depth = temporal_depth(formula.child)
    return is_temporal_unary(formula.op) ? 1 + child_depth : child_depth
end

function temporal_depth(formula::BinaryLTL)::Int
    left_depth = temporal_depth(formula.left)
    right_depth = temporal_depth(formula.right)
    child_depth = max(left_depth, right_depth)
    return is_temporal_binary(formula.op) ? 1 + child_depth : child_depth
end

function operator_counts(formula::LTLFormula)
    counts = Dict{Symbol, Int}()
    _collect_operator_counts!(counts, formula)
    return counts
end

function _collect_operator_counts!(counts::Dict{Symbol, Int}, formula::AP)
    return counts
end

function _collect_operator_counts!(counts::Dict{Symbol, Int}, formula::UnaryLTL)
    counts[formula.op] = get(counts, formula.op, 0) + 1
    _collect_operator_counts!(counts, formula.child)
    return counts
end

function _collect_operator_counts!(counts::Dict{Symbol, Int}, formula::BinaryLTL)
    counts[formula.op] = get(counts, formula.op, 0) + 1
    _collect_operator_counts!(counts, formula.left)
    _collect_operator_counts!(counts, formula.right)
    return counts
end

# ----------------------------------------------------------------------------------------------
# Batch generation
# ----------------------------------------------------------------------------------------------

"""
    synthesize_formulas(n, atomic_props, temporal_ops; cfg=LTLGeneratorConfig(), rng=Random.default_rng())

Generate `n` random LTL formulas and return them as AST objects.
"""
function synthesize_formulas(
    n::Int,
    atomic_props::Vector{String},
    temporal_ops::Vector{Symbol};
    cfg::LTLGeneratorConfig = LTLGeneratorConfig(),
    rng::AbstractRNG = Random.default_rng(),
)
    n < 1 && throw(ArgumentError("`n` must be at least 1."))
    return [random_ltl_formula(atomic_props, temporal_ops; cfg=cfg, rng=rng) for _ in 1:n]
end

# ----------------------------------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------------------------------

if abspath(PROGRAM_FILE) == @__FILE__
    atomic_props = ["prop_1", "prop_2", "prop_3", "prop_4"]
    temporal_ops = [:X, :F, :G, :U]

    cfg = LTLGeneratorConfig(
        max_depth = 4,
        p_atom_before_max_depth = 0.20,
        unary_weight = 0.35,
        binary_weight = 0.45,
        allow_boolean_constants = false,
    )

    formulas = synthesize_formulas(5, atomic_props, temporal_ops; cfg=cfg)

    for (i, formula) in enumerate(formulas)
        println("Formula $(i): ", formula_to_string(formula))
        println("  AST size: ", ast_size(formula))
        println("  AST depth: ", ast_depth(formula))
        println("  Temporal depth: ", temporal_depth(formula))
        println("  Operator counts: ", operator_counts(formula))
        println()
    end
end


atomic_props = ["prop_1", "prop_2", "prop_3", "prop_4"]
temporal_ops = [:X, :F, :G, :U]

cfg = LTLGeneratorConfig(
    max_depth = 4,
    p_atom_before_max_depth = 0.20,
    unary_weight = 0.35,
    binary_weight = 0.45,
)

formulas = synthesize_formulas(10, atomic_props, temporal_ops; cfg=cfg)
for f in formulas
    println(formula_to_string(f))
end