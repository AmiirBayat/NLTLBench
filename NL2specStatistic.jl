

include("Filter.jl")

# -----------------------------------------------------------------------------
# Formulas to analyze
# -----------------------------------------------------------------------------
const RAW_FORMULAS = [
    "G(a -> F e)",
    "G(!(a & b))",
    "G(a -> X (X (X b)))",
    "e U (G (F d))",
    "(F b) -> (!b U (a & !b))",
    "G(a -> b)",
    "G (a && b)",
    "G a && G(b -> !c)",
    "G(a -> F b) -> G F c",
    "G F a -> G F b",
    "G F a || G F b",
    "F G ! a",
    "G (!(a && b) -> F c)",
    "G(!(a &&b)) && G(a || b)",
    "G ((a <-> b) -> (c <-> d))",
    "(! a) U b",
    "G (a -> X G ! b)",
    "(b U (b & ! a)) | G b",
    "G !(a & b)",
    "G (a && X b -> X X c)",
    "G (a -> X F b)",
    "a && G (a -> X ! a && X X ! a && X X X ! a && X X X X ! a && X X X X X a)",
    "G F a || X b",
    "G (a)",
    "G (a -> (b | X b))",
    "G (a | b | c)",
    "G (a -> F b)",
    "! G (! (a && X a))",
    "! G (! (a && X a))",
    "G ( a -> (X !a | XX !a | XXX !a))",
    "G ( a -> X b)",
    "F (a && b)",
    "F a && F b",
    "G (a <-> X b)",
    "b -> X ((c U a) || G c)",
    "(a U b) || G a",
    "finally ( not prop 1)",
    "globally ( not prop 1)",
    "next prop 1",
    "prop 1 until prop 2",
    "finally (prop 1 and prop 2)",
    "globally (prop 1 and prop 2)",
    "finally (prop 1 or prop 2)",
]

# -----------------------------------------------------------------------------
# Normalization helpers
# -----------------------------------------------------------------------------
function collapse_multi_x(s::String)::String
    prev = ""
    cur = s
    while cur != prev
        prev = cur
        cur = replace(cur, r"\bX\s+X\b" => "XX")
        cur = replace(cur, r"\bX\s+X\s+X\b" => "XXX")
        cur = replace(cur, r"\bX\s+X\s+X\s+X\b" => "XXXX")
        cur = replace(cur, r"\bX\s+X\s+X\s+X\s+X\b" => "XXXXX")
    end
    return cur
end

function expand_multi_x(s::String)::String
    prev = ""
    cur = s
    while cur != prev
        prev = cur
        cur = replace(cur, r"\bXXXXX\s+" => "X X X X X ")
        cur = replace(cur, r"\bXXXX\s+" => "X X X X ")
        cur = replace(cur, r"\bXXX\s+" => "X X X ")
        cur = replace(cur, r"\bXX\s+" => "X X ")
    end
    return cur
end

function normalize_formula(raw::AbstractString)::String
    s = strip(String(raw))

    # textual operators
    s = replace(s, r"\bfinally\b"i => "F")
    s = replace(s, r"\beventually\b"i => "F")
    s = replace(s, r"\bglobally\b"i => "G")
    s = replace(s, r"\balways\b"i => "G")
    s = replace(s, r"\bnext\b"i => "X")
    s = replace(s, r"\buntil\b"i => "U")
    s = replace(s, r"\bnot\b"i => "!")
    s = replace(s, r"\band\b"i => "&")
    s = replace(s, r"\bor\b"i => "|")

    # proposition naming: "prop 1" -> "prop_1"
    s = replace(s, r"\bprop\s+(\d+)\b"i => s"prop_\1")

    # common ASCII alternatives
    s = replace(s, "&&" => "&")
    s = replace(s, "||" => "|")

    # add spaces around punctuation/operators for robust token rewriting
    s = replace(s, r"([()!&|])" => s" \1 ")
    s = replace(s, r"(<->|->|U|R|W|M|F|G|X)" => s" \1 ")
    s = collapse_multi_x(s)
    s = expand_multi_x(s)
    s = replace(s, r"\s+" => " ")
    s = strip(s)

    # remove unwanted spaces after unary temporal operators and negation
    prev = ""
    cur = s
    while cur != prev
        prev = cur
        cur = replace(cur, r"\bG\s+\(" => "G(")
        cur = replace(cur, r"\bF\s+\(" => "F(")
        cur = replace(cur, r"\bX\s+\(" => "X(")
        cur = replace(cur, r"!\s+\(" => "!(")
        cur = replace(cur, r"\bG\s+(prop_[A-Za-z0-9_]+|[A-Za-z][A-Za-z0-9_]*)\b" => s"G(\1)")
        cur = replace(cur, r"\bF\s+(prop_[A-Za-z0-9_]+|[A-Za-z][A-Za-z0-9_]*)\b" => s"F(\1)")
        cur = replace(cur, r"\bX\s+(prop_[A-Za-z0-9_]+|[A-Za-z][A-Za-z0-9_]*)\b" => s"X(\1)")
        cur = replace(cur, r"!\s+(prop_[A-Za-z0-9_]+|[A-Za-z][A-Za-z0-9_]*)\b" => s"!(\1)")
        cur = replace(cur, r"\(\s+" => "(")
        cur = replace(cur, r"\s+\)" => ")")
        cur = replace(cur, r"\s+" => " ")
        cur = strip(cur)
    end

    return cur
end

# -----------------------------------------------------------------------------
# Analysis helpers
# -----------------------------------------------------------------------------
function is_formula_satisfiable(formula_str::String)::Bool
    ltlfilt_path = Sys.which("ltlfilt")
    isnothing(ltlfilt_path) && throw(ArgumentError("Spot's `ltlfilt` was not found in PATH."))

    # Returns the formula iff satisfiable; empty output means unsat.
    cmd = `$(ltlfilt_path) -f $(formula_str) --satisfiable`
    output = read(cmd, String)
    return !isempty(strip(output))
end

function is_formula_non_trivial(ast)::Bool
    if ast isa AP
        return false
    end

    if ast isa UnaryLTL && ast.op == :! && ast.child isa AP
        return false
    end

    return true
end

function count_unique_atomic_props(ast)::Int
    props = Set{String}()

    function visit(node)
        if node isa AP
            name = String(node.name)
            if startswith(name, "prop_") || occursin(r"^[A-Za-z][A-Za-z0-9_]*$", name)
                push!(props, name)
            end
        elseif node isa UnaryLTL
            visit(node.child)
        elseif node isa BinaryLTL
            visit(node.left)
            visit(node.right)
        end
    end

    visit(ast)
    return length(props)
end

function analyze_formulas(raw_formulas::Vector{String} = RAW_FORMULAS)
    normalized = [normalize_formula(f) for f in raw_formulas]
    unique_normalized = unique(normalized)

    satisfiable_nontrivial = String[]
    max_formula_size = 0
    max_atomic_props = 0
    max_temporal_depth = 0
    max_ast_depth = 0

    parsed_ok = 0
    parse_failures = 0

    for formula in unique_normalized
        try
            ast = parse_ltl_formula_string(formula)
            parsed_ok += 1

            structure = formula_structure_statistics(formula)
            max_formula_size = max(max_formula_size, Int(structure["ast_size"]))
            max_temporal_depth = max(max_temporal_depth, Int(structure["temporal_depth"]))
            max_ast_depth = max(max_ast_depth, Int(structure["ast_depth"]))
            max_atomic_props = max(max_atomic_props, count_unique_atomic_props(ast))

            if is_formula_satisfiable(formula) && is_formula_non_trivial(ast)
                push!(satisfiable_nontrivial, formula)
            end
        catch err
            parse_failures += 1
            println("Warning: failed to analyze formula: ", formula)
            println("  Error: ", sprint(showerror, err))
        end
    end

    println("Total formulas: ", length(raw_formulas))
    println("Unique normalized formulas: ", length(unique_normalized))
    println("Unique satisfiable, non-trivial formulas: ", length(unique(satisfiable_nontrivial)))
    println("Maximum formula size: ", max_formula_size)
    println("Maximum number of atomic propositions: ", max_atomic_props)
    println("Maximum temporal depth: ", max_temporal_depth)
    println("Maximum AST depth: ", max_ast_depth)
    println("Parsed successfully: ", parsed_ok)
    println("Parse failures: ", parse_failures)

    return OrderedDict(
        "total_formulas" => length(raw_formulas),
        "unique_normalized_formulas" => length(unique_normalized),
        "unique_satisfiable_non_trivial_formulas" => length(unique(satisfiable_nontrivial)),
        "maximum_formula_size" => max_formula_size,
        "maximum_number_of_atomic_propositions" => max_atomic_props,
        "maximum_temporal_depth" => max_temporal_depth,
        "maximum_ast_depth" => max_ast_depth,
        "parsed_successfully" => parsed_ok,
        "parse_failures" => parse_failures,
        "normalized_formulas" => normalized,
        "unique_satisfiable_non_trivial_formulas_list" => unique(satisfiable_nontrivial),
    )
end

function main()
    println("Analyzing formulas...")
    analyze_formulas()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end