using JSON3
using OrderedCollections

const DEFAULT_INPUT_PATH = joinpath(@__DIR__, "results", "t5_lletter_results.json")
const DEFAULT_OUTPUT_PATH = joinpath(@__DIR__, "results", "t5_lletter_results_normalized.json")

const UNARY_OPS = Set(["!", "X", "F", "G"])
const BINARY_OPS = Set(["&", "|", "U", "R", "W", "M", "i", "e"])
const RESERVED_ATOMS = Set(["true", "false", "t", "f"])
const INFIX_BINARY_OPS = Set(["&", "|", "U", "R", "W", "M", "->", "<->"])

struct PrefixNode
    op::String
    args::Vector{Any}
end

function normalized_success_rate_by_input_field(
    input_field::String = "natural_paraphrase";
    results_path::String = DEFAULT_OUTPUT_PATH,
)
    records = load_json_records(results_path)
    total = 0
    equivalent = 0
    valid_normalized = 0

    for record_any in records
        record = record_any::OrderedDict{String,Any}

        haskey(record, "input_field") || continue
        String(record["input_field"]) == input_field || continue

        total += 1

        if haskey(record, "normalized_prediction")
            pred = record["normalized_prediction"]
            if !(pred === nothing) && !isempty(strip(String(pred)))
                valid_normalized += 1
            end
        end

        if haskey(record, "equivalent") && record["equivalent"] === true
            equivalent += 1
        end
    end

    total == 0 && begin
        println("No records found for input_field=", input_field, " in: ", results_path)
        return 0.0
    end

    success_rate = equivalent / total

    println("Results path: ", results_path)
    println("Input field: ", input_field)
    println("Total records: ", total)
    println("Valid normalized predictions: ", valid_normalized)
    println("Equivalent predictions: ", equivalent)
    println("Success rate: ", round(success_rate * 100; digits=2), "%")

    return success_rate
end
function to_mutable(x)
    if x isa AbstractVector
        return Any[to_mutable(v) for v in x]
    elseif x isa AbstractDict || x isa JSON3.Object
        d = OrderedDict{String,Any}()
        for (k, v) in pairs(x)
            d[String(k)] = to_mutable(v)
        end
        return d
    else
        return x
    end
end

function load_json_records(path::String)
    isfile(path) || throw(ArgumentError("Input file not found: $(path)"))
    parsed = JSON3.read(read(path, String))
    mutable = to_mutable(parsed)
    mutable isa AbstractVector || throw(ArgumentError("Expected a JSON array in $(path)"))
    return mutable
end

function save_json_records(path::String, records)
    open(path, "w") do io
        JSON3.pretty(io, records)
    end
    println("Saved normalized results to: ", path)
end

function ensure_ltlfilt()
    path = Sys.which("ltlfilt")
    isnothing(path) && error("`ltlfilt` was not found in PATH.")
    return path
end

function validate_ltl_syntax(formula::AbstractString)
    ltlfilt_path = ensure_ltlfilt()
    cmd = `$(ltlfilt_path) -f $(String(formula))`
    proc = run(pipeline(ignorestatus(cmd), stdout=devnull, stderr=devnull))
    return proc.exitcode == 0
end

function are_equivalent_ltl(formula_a::AbstractString, formula_b::AbstractString)
    ltlfilt_path = ensure_ltlfilt()
    io = IOBuffer()
    cmd = `$(ltlfilt_path) -f $(String(formula_a)) --equivalent-to $(String(formula_b)) --count`
    proc = run(pipeline(ignorestatus(cmd), stdout=io, stderr=devnull))
    proc.exitcode == 0 || return false
    output = strip(String(take!(io)))
    return output == "1"
end

function cleanup_raw_output(raw::AbstractString)
    s = strip(String(raw))
    s = replace(s, '\n' => ' ')
    s = replace(s, '\r' => ' ')
    s = replace(s, r"^\s*LTL\s*[:\-]?\s*"i => "")
    s = replace(s, r"^\s*Output\s*[:\-]?\s*"i => "")
    s = replace(s, r"\s+" => " ")
    return strip(s)
end

function tokenize_prefixish(s::AbstractString)
    tokens = String[]
    i = firstindex(s)
    while i <= lastindex(s)
        c = s[i]
        if isspace(c)
            i = nextind(s, i)
        elseif c in ('(', ')')
            push!(tokens, string(c))
            i = nextind(s, i)
        elseif c in ('!', '&', '|')
            push!(tokens, string(c))
            i = nextind(s, i)
        elseif c in ('X', 'F', 'G', 'U', 'R', 'W', 'M', 'i', 'e')
            push!(tokens, string(c))
            i = nextind(s, i)
        elseif isletter(c) || c == '_' || isdigit(c)
            j = i
            while j <= lastindex(s)
                cj = s[j]
                if isletter(cj) || isdigit(cj) || cj == '_' || cj == '.'
                    j = nextind(s, j)
                else
                    break
                end
            end
            push!(tokens, s[i:prevind(s, j)])
            i = j
        else
            i = nextind(s, i)
        end
    end
    return tokens
end

function parse_prefix_tokens(tokens::Vector{String}, idx::Int=1)
    idx > length(tokens) && error("Unexpected end of tokens")
    tok = tokens[idx]

    if tok == "(" || tok == ")"
        error("Parenthesis token not expected in prefix parse")
    elseif tok in UNARY_OPS
        child, next_idx = parse_prefix_tokens(tokens, idx + 1)
        return PrefixNode(tok, Any[child]), next_idx
    elseif tok in BINARY_OPS
        left, next_idx = parse_prefix_tokens(tokens, idx + 1)
        right, next_idx2 = parse_prefix_tokens(tokens, next_idx)
        return PrefixNode(tok, Any[left, right]), next_idx2
    else
        return tok, idx + 1
    end
end

function parse_best_prefix(raw::AbstractString)
    cleaned = cleanup_raw_output(raw)
    tokens = tokenize_prefixish(cleaned)
    isempty(tokens) && return nothing, 0

    best_ast = nothing
    best_consumed = 0

    for start_idx in eachindex(tokens)
        try
            ast, next_idx = parse_prefix_tokens(tokens, start_idx)
            consumed = next_idx - start_idx
            if consumed > best_consumed
                best_ast = ast
                best_consumed = consumed
            end
        catch
        end
    end

    return best_ast, best_consumed
end

function ast_to_infix(ast)
    if ast isa String
        return ast
    end

    op = ast.op
    args = ast.args

    if op == "!"
        return "(!" * ast_to_infix(args[1]) * ")"
    elseif op in ["X", "F", "G"]
        return op * " (" * ast_to_infix(args[1]) * ")"
    elseif op == "i"
        return "(" * ast_to_infix(args[1]) * " -> " * ast_to_infix(args[2]) * ")"
    elseif op == "e"
        return "(" * ast_to_infix(args[1]) * " <-> " * ast_to_infix(args[2]) * ")"
    else
        return "(" * ast_to_infix(args[1]) * " " * op * " " * ast_to_infix(args[2]) * ")"
    end
end

function tokenize_infix_ltl(s::AbstractString)
    tokens = String[]
    i = firstindex(s)
    while i <= lastindex(s)
        c = s[i]
        if isspace(c)
            i = nextind(s, i)
        elseif c in ('(', ')', '!', '&', '|')
            push!(tokens, string(c))
            i = nextind(s, i)
        elseif c == '-'
            j = nextind(s, i)
            j <= lastindex(s) && s[j] == '>' || error("Invalid token in infix LTL: expected '->'")
            push!(tokens, "->")
            i = nextind(s, j)
        elseif c == '<'
            j = nextind(s, i)
            j <= lastindex(s) && s[j] == '-' || error("Invalid token in infix LTL: expected '<->'")
            k = nextind(s, j)
            k <= lastindex(s) && s[k] == '>' || error("Invalid token in infix LTL: expected '<->'")
            push!(tokens, "<->")
            i = nextind(s, k)
        elseif c in ('X', 'F', 'G', 'U', 'R', 'W', 'M')
            push!(tokens, string(c))
            i = nextind(s, i)
        elseif isletter(c) || c == '_' || isdigit(c)
            j = i
            while j <= lastindex(s)
                cj = s[j]
                if isletter(cj) || isdigit(cj) || cj == '_' || cj == '.'
                    j = nextind(s, j)
                else
                    break
                end
            end
            push!(tokens, s[i:prevind(s, j)])
            i = j
        else
            error("Unsupported character in infix LTL: $(c)")
        end
    end
    return tokens
end

function parse_infix_tokens(tokens::Vector{String})
    pos = Ref(1)

    function parse_primary()
        pos[] <= length(tokens) || error("Unexpected end of tokens in infix parse")
        tok = tokens[pos[]]

        if tok == "!"
            pos[] += 1
            return PrefixNode("!", Any[parse_primary()])
        elseif tok in ["X", "F", "G"]
            pos[] += 1
            return PrefixNode(tok, Any[parse_primary()])
        elseif tok == "("
            pos[] += 1
            expr = parse_implication()
            pos[] <= length(tokens) && tokens[pos[]] == ")" || error("Missing closing parenthesis in infix parse")
            pos[] += 1
            return expr
        elseif tok == ")"
            error("Unexpected closing parenthesis in infix parse")
        else
            pos[] += 1
            return tok
        end
    end

    function parse_until_family()
        left = parse_primary()
        while pos[] <= length(tokens) && tokens[pos[]] in ["U", "R", "W", "M"]
            op = tokens[pos[]]
            pos[] += 1
            right = parse_primary()
            left = PrefixNode(op, Any[left, right])
        end
        return left
    end

    function parse_and()
        left = parse_until_family()
        while pos[] <= length(tokens) && tokens[pos[]] == "&"
            pos[] += 1
            right = parse_until_family()
            left = PrefixNode("&", Any[left, right])
        end
        return left
    end

    function parse_or()
        left = parse_and()
        while pos[] <= length(tokens) && tokens[pos[]] == "|"
            pos[] += 1
            right = parse_and()
            left = PrefixNode("|", Any[left, right])
        end
        return left
    end

    function parse_implication()
        left = parse_or()
        while pos[] <= length(tokens) && tokens[pos[]] in ["->", "<->"]
            op = tokens[pos[]] == "->" ? "i" : "e"
            pos[] += 1
            right = parse_or()
            left = PrefixNode(op, Any[left, right])
        end
        return left
    end

    ast = parse_implication()
    pos[] > length(tokens) || error("Unexpected trailing tokens in infix parse")
    return ast
end

function parse_infix_formula(raw::AbstractString)
    tokens = tokenize_infix_ltl(raw)
    isempty(tokens) && return nothing
    return parse_infix_tokens(tokens)
end

function try_restructure_raw_output(raw::AbstractString)
    cleaned = cleanup_raw_output(raw)
    isempty(cleaned) && return nothing, "empty_output"

    if validate_ltl_syntax(cleaned)
        return cleaned, "already_valid"
    end

    ast = nothing
    status = "parse_failed"

    try
        ast = parse_infix_formula(cleaned)
        status = "restructured_from_infix"
    catch
    end

    if ast === nothing
        best_ast, _ = parse_best_prefix(cleaned)
        if best_ast !== nothing
            ast = best_ast
            status = "restructured_from_prefix"
        end
    end

    ast === nothing && return nothing, "parse_failed"

    normalized = ast_to_infix(ast)
    validate_ltl_syntax(normalized) || return nothing, "normalized_but_invalid"
    return normalized, status
end

function normalize_single_prediction(raw_output::AbstractString, gold_formula::AbstractString)
    normalized, status = try_restructure_raw_output(raw_output)

    normalized === nothing && return OrderedDict(
        "normalized_prediction" => nothing,
        "normalization_status" => status,
        "normalization_note" => "Could not restructure the raw model output into the target LTL syntax.",
    )

    equivalent = are_equivalent_ltl(normalized, gold_formula)
    note = status == "already_valid" ?
        "The raw output was already valid in the target syntax." :
        "The raw output was restructured into the target LTL syntax without changing atomic propositions."

    return OrderedDict(
        "normalized_prediction" => normalized,
        "normalization_status" => status,
        "normalization_note" => note,
        "equivalent_after_normalization" => equivalent,
    )
end

function find_gold_formula(record::OrderedDict{String,Any})
    for key in ["original_ltl", "gold_ltl", "target_ltl", "reference_ltl", "ltl", "LTL"]
        if haskey(record, key)
            value = strip(String(record[key]))
            !isempty(value) && return value
        end
    end
    return nothing
end

function find_raw_output(record::OrderedDict{String,Any})
    for key in ["raw_model_output", "model_output", "prediction", "generated_ltl"]
        if haskey(record, key)
            value = strip(String(record[key]))
            !isempty(value) && return value
        end
    end
    return nothing
end

function normalize_t5_lletter_results(
    input_path::String = DEFAULT_INPUT_PATH;
    output_path::String = DEFAULT_OUTPUT_PATH,
    overwrite_existing::Bool = true,
)
    records = load_json_records(input_path)
    normalized_count = 0
    skipped_count = 0

    for (idx, record_any) in enumerate(records)
        record = record_any::OrderedDict{String,Any}

        if !overwrite_existing && haskey(record, "normalized_prediction")
            existing = record["normalized_prediction"]
            if !(existing === nothing) && !isempty(strip(String(existing)))
                skipped_count += 1
                continue
            end
        end

        gold_formula = find_gold_formula(record)
        raw_output = find_raw_output(record)

        if isnothing(gold_formula)
            record["normalization_status"] = "missing_gold_formula"
            record["normalization_note"] = "No gold/reference LTL formula field was found."
            skipped_count += 1
            continue
        end

        if isnothing(raw_output)
            record["normalization_status"] = "missing_raw_output"
            record["normalization_note"] = "No raw model output field was found."
            skipped_count += 1
            continue
        end

        result = normalize_single_prediction(raw_output, gold_formula)
        for (k, v) in pairs(result)
            record[String(k)] = v
        end

        if haskey(result, "equivalent_after_normalization")
            record["equivalent"] = result["equivalent_after_normalization"]
        end

        normalized_count += 1
        if idx % 25 == 0
            println("Processed $(idx) / $(length(records)) records")
        end
    end

    save_json_records(output_path, records)
    println("Normalization complete.")
    println("Normalized or attempted: ", normalized_count)
    println("Skipped: ", skipped_count)
    return records
end

function normalized_success_rate(results_path::String = DEFAULT_OUTPUT_PATH)
    records = load_json_records(results_path)
    total = 0
    equivalent = 0
    valid_normalized = 0

    for record_any in records
        record = record_any::OrderedDict{String,Any}
        total += 1

        if haskey(record, "normalized_prediction")
            pred = record["normalized_prediction"]
            if !(pred === nothing) && !isempty(strip(String(pred)))
                valid_normalized += 1
            end
        end

        if haskey(record, "equivalent") && record["equivalent"] === true
            equivalent += 1
        end
    end

    total == 0 && begin
        println("No records found in: ", results_path)
        return 0.0
    end

    success_rate = equivalent / total

    println("Results path: ", results_path)
    println("Total records: ", total)
    println("Valid normalized predictions: ", valid_normalized)
    println("Equivalent predictions: ", equivalent)
    println("Success rate: ", round(success_rate * 100; digits=2), "%")

    return success_rate
end

function main()
    normalize_t5_lletter_results()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end