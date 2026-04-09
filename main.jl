include("GenerateLTL.jl")
include("Filter.jl")
#using GenerateLTL


using Random

"""
    main()

Entry-point script for generating a batch of random LTL formulas.
Edit the parameters in the configuration section below as needed.
"""
function main()
    # --------------------------------------------------------------------------
    # User configuration
    # --------------------------------------------------------------------------
    atomic_props = ["prop_1", "prop_2", "prop_3", "prop_4"]
    temporal_ops = [:X, :F, :G, :U]

    n_formulas = 5
    max_depth = 4
    max_ast_size = 12

    p_atom_at_max_depth = 1.0
    p_atom_before_max_depth = 0.20
    unary_weight = 1.00 # 0.35
    binary_weight = 1.00 # 0.55

    allow_boolean_constants = false
    boolean_constants = ["true", "false"]
    boolean_unary_ops = [:!]
    boolean_binary_ops = [:&, :|, :->, Symbol("<->")]

    max_attempts_per_formula = 100
    enforce_unique_formulas = true
    uniqueness_mode = :normalized
    redundancy_mode = :normalized
    semantic_redundancy = true
    semantic_backend = :spot
    semantic_existing_formulas = LTLFormula[]
    rng = MersenneTwister()

    # --------------------------------------------------------------------------
    # Formula generation
    # --------------------------------------------------------------------------
    formulas = generate_ltl_formulas(
        atomic_props = atomic_props,
        temporal_ops = temporal_ops,
        n = n_formulas,
        max_depth = max_depth,
        max_ast_size = max_ast_size,
        p_atom_at_max_depth = p_atom_at_max_depth,
        p_atom_before_max_depth = p_atom_before_max_depth,
        unary_weight = unary_weight,
        binary_weight = binary_weight,
        allow_boolean_constants = allow_boolean_constants,
        boolean_constants = boolean_constants,
        boolean_unary_ops = boolean_unary_ops,
        boolean_binary_ops = boolean_binary_ops,
        max_attempts_per_formula = max_attempts_per_formula,
        enforce_unique_formulas = enforce_unique_formulas,
        uniqueness_mode = uniqueness_mode,
        rng = rng,
    )

    # --------------------------------------------------------------------------
    # Filtering
    # --------------------------------------------------------------------------
    existing_keys = Set{String}()
    accepted, rejected = filter_formulas(
        formulas;
        existing_keys = existing_keys,
        redundancy_mode = redundancy_mode,
        mutate_keys = true,
        semantic_redundancy = semantic_redundancy,
        semantic_backend = semantic_backend,
        semantic_existing_formulas = semantic_existing_formulas,
    )

    # --------------------------------------------------------------------------
    # Display results
    # --------------------------------------------------------------------------
    println("Generated $(length(formulas)) candidate LTL formulas.\n")
    println("Accepted $(length(accepted)) formulas after filtering.")
    println("Rejected $(length(rejected)) formulas after filtering.\n")
    println("Filtering configuration:")
    println("  Redundancy mode: ", redundancy_mode)
    println("  Semantic redundancy enabled: ", semantic_redundancy)
    println("  Semantic backend: ", semantic_backend, "\n")

    println("Accepted formulas:\n")
    for (i, formula) in enumerate(accepted)
        println("Formula $(i): ", formula_to_string(formula))
        println("  Simplified form: ", formula_to_string(simplify_formula(formula)))
        println("  AST size: ", ast_size(formula))
        println("  AST depth: ", ast_depth(formula))
        println("  Temporal depth: ", temporal_depth(formula))
        println("  Operator counts: ", operator_counts(formula))
        println()
    end

    if !isempty(rejected)
        println("Rejected formulas:\n")
        for (i, item) in enumerate(rejected)
            formula, reasons = item
            println("Rejected $(i): ", formula_to_string(formula))
            println("  Simplified form: ", formula_to_string(simplify_formula(formula)))
            println("  Reasons: ", reasons)
            println()
        end
    end
end

main()