include("Filter.jl")
include("GenerateLTL.jl")
include("Statistics.jl")
using JSON3
using OrderedCollections

const INPUT_PATH = joinpath(@__DIR__, "final_calib.json")
const OUTPUT_PATH = joinpath(@__DIR__, "final_calib_normalized_fixed.json")

const RESERVED_TOKENS = Set([
    "U", "W", "R", "M", "X", "F", "G",
    "true", "false",
    "/",
])

function load_json_array(path::String = INPUT_PATH)
    if !isfile(path)
        throw(ArgumentError("File not found: $(path)"))
    end

    content = strip(read(path, String))
    isempty(content) && throw(ArgumentError("File is empty: $(path)"))

    parsed = JSON3.read(content)
    return [OrderedDict{String,Any}(String(k) => v for (k, v) in pairs(item)) for item in parsed]
end

function save_json_array(records, path::String = OUTPUT_PATH)
    open(path, "w") do io
        JSON3.pretty(io, records)
        write(io, "\n")
    end
end

function is_operator_letter(c::Char)::Bool
    return c in ('U', 'W', 'R', 'M', 'X', 'F', 'G')
end

function starts_with_at(s::AbstractString, i::Int, pat::String)::Bool
    j = i
    for ch in pat
        if j > lastindex(s) || s[j] != ch
            return false
        end
        j = nextind(s, j)
    end
    return true
end

function next_operator_length(s::AbstractString, i::Int)::Int
    if starts_with_at(s, i, "<->")
        return 3
    elseif starts_with_at(s, i, "->") || starts_with_at(s, i, "&&") || starts_with_at(s, i, "||") || starts_with_at(s, i, "[]") || starts_with_at(s, i, "<>")
        return 2
    elseif s[i] in ('(', ')', '!', '&', '|') || is_operator_letter(s[i])
        return 1
    else
        return 0
    end
end

function consume_n_chars(s::AbstractString, i::Int, n::Int)::Int
    j = i
    for _ in 1:n
        j = nextind(s, j)
    end
    return j
end

function tokenize_ltl_formula_string(formula_str::AbstractString)::Vector{String}
    s = strip(String(formula_str))
    tokens = String[]
    isempty(s) && return tokens

    i = firstindex(s)
    while i <= lastindex(s)
        c = s[i]

        if isspace(c)
            i = nextind(s, i)
            continue
        end

        if starts_with_at(s, i, "<->")
            push!(tokens, "<->")
            i = consume_n_chars(s, i, 3)
            continue
        elseif starts_with_at(s, i, "->")
            push!(tokens, "->")
            i = consume_n_chars(s, i, 2)
            continue
        elseif starts_with_at(s, i, "&&")
            push!(tokens, "&&")
            i = consume_n_chars(s, i, 2)
            continue
        elseif starts_with_at(s, i, "||")
            push!(tokens, "||")
            i = consume_n_chars(s, i, 2)
            continue
        elseif starts_with_at(s, i, "[]")
            push!(tokens, "[]")
            i = consume_n_chars(s, i, 2)
            continue
        elseif starts_with_at(s, i, "<>")
            push!(tokens, "<>")
            i = consume_n_chars(s, i, 2)
            continue
        elseif c in ('(', ')', '!', '&', '|')
            push!(tokens, string(c))
            i = nextind(s, i)
            continue
        elseif c == '/'
            i = nextind(s, i)
            continue
        elseif is_operator_letter(c)
            push!(tokens, string(c))
            i = nextind(s, i)
            continue
        end

        start = i
        j = i
        while j <= lastindex(s)
            cj = s[j]

            if isspace(cj) || cj == '/'
                break
            end

            oplen = next_operator_length(s, j)
            if oplen > 0 && j == start
                break
            elseif oplen > 0
                break
            end

            if is_operator_letter(cj)
                break
            end

            j = nextind(s, j)
        end

        token = strip(s[start:prevind(s, j)])
        if !isempty(token)
            push!(tokens, token)
        end
        i = j
    end

    return tokens
end

function standardize_tokens(tokens::Vector{String})::Vector{String}
    standardized = String[]
    for tok in tokens
        if tok == "[]"
            push!(standardized, "G")
        elseif tok == "<>"
            push!(standardized, "F")
        elseif tok == "&&"
            push!(standardized, "&")
        elseif tok == "||"
            push!(standardized, "|")
        else
            push!(standardized, tok)
        end
    end
    return standardized
end

function token_is_atomic_prop(tok::String)::Bool
    return !(tok in RESERVED_TOKENS) && !(tok in Set(["(", ")", "!", "&", "|", "->", "<->"]))
end

function normalize_ltl_formula(raw_formula::AbstractString)
    tokens = standardize_tokens(tokenize_ltl_formula_string(raw_formula))
    mapping = OrderedDict{String,String}()
    normalized_tokens = String[]

    for tok in tokens
        if token_is_atomic_prop(tok)
            if !haskey(mapping, tok)
                mapping[tok] = "prop_$(length(mapping) + 1)"
            end
            push!(normalized_tokens, mapping[tok])
        else
            push!(normalized_tokens, tok)
        end
    end

    return join(normalized_tokens, " "), mapping
end

function make_parser_compatible_formula(formula::AbstractString)::String
    tokens = standardize_tokens(tokenize_ltl_formula_string(formula))
    out = String[]
    i = 1

    while i <= length(tokens)
        tok = tokens[i]

        # Canonicalize an already parenthesized negated atom: ( ! p ) -> (!p)
        if tok == "(" && i + 3 <= length(tokens)
            if tokens[i + 1] == "!" && token_is_atomic_prop(tokens[i + 2]) && tokens[i + 3] == ")"
                push!(out, "(!" * tokens[i + 2] * ")")
                i += 4
                continue
            end
        end

        # Canonicalize a bare negated atom: ! p -> (!p)
        if tok == "!" && i < length(tokens)
            nxt = tokens[i + 1]
            if token_is_atomic_prop(nxt)
                push!(out, "(!" * nxt * ")")
                i += 2
                continue
            end
        end

        push!(out, tok)
        i += 1
    end

    s = join(out, " ")
    s = replace(s, r"\(\s+" => "(")
    s = replace(s, r"\s+\)" => ")")
    s = replace(s, r"\s+" => " ")
    return strip(s)
end

function operator_token_set()::Set{String}
    return Set(["(", ")", "!", "&", "|", "->", "<->", "U", "W", "R", "M", "X", "F", "G"])
end

function atomic_prop_names_from_tokens(tokens::Vector{String})::Vector{String}
    names = String[]
    seen = Set{String}()
    for tok in tokens
        if token_is_atomic_prop(tok) && !(tok in seen)
            push!(names, tok)
            push!(seen, tok)
        end
    end
    return names
end

function token_formula_size(formula::AbstractString)::Int
    tokens = standardize_tokens(tokenize_ltl_formula_string(formula))
    return count(tok -> tok != "(" && tok != ")", tokens)
end

function token_temporal_depth(formula::AbstractString)::Int
    tokens = standardize_tokens(tokenize_ltl_formula_string(formula))
    max_depth = 0

    function parse_unary_depth(i::Int)
        i > length(tokens) && return 0, i
        tok = tokens[i]

        if tok == "!"
            return parse_unary_depth(i + 1)
        elseif tok in ("X", "F", "G")
            child_depth, next_i = parse_unary_depth(i + 1)
            return 1 + child_depth, next_i
        elseif tok == "("
            inner_depth, next_i = parse_binary_depth(i + 1)
            if next_i <= length(tokens) && tokens[next_i] == ")"
                return inner_depth, next_i + 1
            end
            return inner_depth, next_i
        else
            return 0, i + 1
        end
    end

    function parse_binary_depth(i::Int)
        left_depth, i = parse_unary_depth(i)
        best = left_depth

        while i <= length(tokens)
            tok = tokens[i]
            if tok in (")",)
                break
            elseif tok in ("&", "|", "->", "<->")
                right_depth, next_i = parse_unary_depth(i + 1)
                best = max(best, right_depth)
                i = next_i
            elseif tok in ("U", "W", "R", "M")
                right_depth, next_i = parse_unary_depth(i + 1)
                best = max(best, left_depth + 1, right_depth + 1)
                left_depth = max(left_depth, right_depth) + 1
                i = next_i
            else
                break
            end
        end

        return max(best, left_depth), i
    end

    depth, _ = parse_binary_depth(1)
    max_depth = max(max_depth, depth)
    return max_depth
end

function fix_normalized_formulas!(records)
    fixed_count = 0

    for record in records
        haskey(record, "ltlequ") || continue
        raw_values = record["ltlequ"]
        fixed_list = String[]
        mapping_list = OrderedDict{String,String}[]

        for item in raw_values
            formula = strip(String(item))
            isempty(formula) && continue
            normalized, mapping = normalize_ltl_formula(formula)
            normalized = make_parser_compatible_formula(normalized)
            push!(fixed_list, normalized)
            push!(mapping_list, mapping)
        end

        record["ltlequ_normalized"] = fixed_list
        record["ap_mapping"] = [OrderedDict{String,Any}(k => v for (k, v) in pairs(m)) for m in mapping_list]
        fixed_count += 1
    end

    return fixed_count
end

function run_ltlfilt_equivalence(formula_a::String, formula_b::String)::Bool
    ltlfilt_path = Sys.which("ltlfilt")
    isnothing(ltlfilt_path) && throw(ArgumentError("`ltlfilt` was not found in PATH. Please install Spot and make sure `ltlfilt` is available."))

    cmd = `$(ltlfilt_path) -f $(formula_a) --equivalent-to $(formula_b) -q`
    process = run(ignorestatus(cmd))

    if process.exitcode == 0
        return true
    elseif process.exitcode == 1
        return false
    else
        throw(ArgumentError("`ltlfilt` failed while checking equivalence between `$(formula_a)` and `$(formula_b)` (exit code $(process.exitcode))."))
    end
end

function compute_formula_summary_stats(formulas::Vector{String})
    syntactically_unique = unique(formulas)
    unique_non_trivial = 0
    max_formula_size = 0
    max_num_aps = 0
    max_temporal_depth = 0

    for formula in syntactically_unique
        tokens = standardize_tokens(tokenize_ltl_formula_string(formula))
        max_formula_size = max(max_formula_size, token_formula_size(formula))
        max_temporal_depth = max(max_temporal_depth, token_temporal_depth(formula))
        max_num_aps = max(max_num_aps, length(atomic_prop_names_from_tokens(tokens)))

        if is_non_trivial_formula_string(formula)
            unique_non_trivial += 1
        end
    end

    return OrderedDict(
        "total_formulas" => length(formulas),
        "unique_non_trivial_formulas" => unique_non_trivial,
        "maximum_formula_size" => max_formula_size,
        "maximum_number_of_atomic_propositions" => max_num_aps,
        "maximum_temporal_depth" => max_temporal_depth,
    )
end

function is_non_trivial_formula_string(formula::String)::Bool
    tokens = standardize_tokens(tokenize_ltl_formula_string(formula))
    core = [tok for tok in tokens if tok != "(" && tok != ")"]

    if length(core) == 1 && token_is_atomic_prop(core[1])
        return false
    end

    if length(core) == 2 && core[1] == "!" && token_is_atomic_prop(core[2])
        return false
    end

    return true
end

function compute_formula_summary_stats(formulas::Vector{String})
    syntactically_unique = unique(formulas)
    unique_non_trivial = 0
    max_formula_size = 0
    max_num_aps = 0
    max_temporal_depth = 0

    for formula in syntactically_unique
        tokens = standardize_tokens(tokenize_ltl_formula_string(formula))
        max_formula_size = max(max_formula_size, token_formula_size(formula))
        max_temporal_depth = max(max_temporal_depth, token_temporal_depth(formula))
        max_num_aps = max(max_num_aps, length(atomic_prop_names_from_tokens(tokens)))

        if is_non_trivial_formula_string(formula)
            unique_non_trivial += 1
        end
    end

    return OrderedDict(
        "total_formulas" => length(formulas),
        "unique_non_trivial_formulas" => unique_non_trivial,
        "maximum_formula_size" => max_formula_size,
        "maximum_number_of_atomic_propositions" => max_num_aps,
        "maximum_temporal_depth" => max_temporal_depth,
    )
end

function collect_normalized_formulas(records)::Vector{String}
    formulas = String[]

    for record in records
        haskey(record, "ltlequ_normalized") || continue
        values = record["ltlequ_normalized"]

        for item in values
            formula = make_parser_compatible_formula(strip(String(item)))
            isempty(formula) && continue
            push!(formulas, formula)
        end
    end

    return formulas
end

function semantically_unique_formulas(formulas::Vector{String})
    syntactically_unique = unique(formulas)
    representatives = String[]
    class_sizes = OrderedDict{String,Int}()

    for formula in syntactically_unique
        matched = false
        for rep in representatives
            if run_ltlfilt_equivalence(formula, rep)
                class_sizes[rep] += 1
                matched = true
                break
            end
        end

        if !matched
            push!(representatives, formula)
            class_sizes[formula] = 1
        end
    end

    return representatives, class_sizes, syntactically_unique
end

function fix_and_analyze_ltlequ_normalized(input_path::String = INPUT_PATH; output_path::String = OUTPUT_PATH)
    records = load_json_array(input_path)
    fixed_count = fix_normalized_formulas!(records)
    save_json_array(records, output_path)

    formulas = collect_normalized_formulas(records)
    summary_stats = compute_formula_summary_stats(formulas)
    representatives, class_sizes, syntactically_unique = semantically_unique_formulas(formulas)

    println("Input file: ", input_path)
    println("Output file: ", output_path)
    println("Entries whose `ltlequ_normalized` were rebuilt from `ltlequ`: ", fixed_count)
    println("Total entries: ", length(records))
    println("Total formulas in `ltlequ_normalized`: ", length(formulas))
    println("Unique formulas by string: ", length(syntactically_unique))
    println("Semantically unique LTL formulas: ", length(representatives))

    println("\nAdditional summary statistics:")
    println("Total formulas ", summary_stats["total_formulas"])
    println("Unique non-trivial formulas ", summary_stats["unique_non_trivial_formulas"])
    println("Maximum formula size ", summary_stats["maximum_formula_size"])
    println("Maximum number of AP ", summary_stats["maximum_number_of_atomic_propositions"])
    println("Maximum temporal depth ", summary_stats["maximum_temporal_depth"])

    println("\nTop 20 semantic representatives:")
    sorted_reps = sort(collect(representatives); by = rep -> -class_sizes[rep])
    for (i, rep) in enumerate(sorted_reps[1:min(20, length(sorted_reps))])
        println(lpad(string(i), 3), ". [", class_sizes[rep], "] ", rep)
    end

    return OrderedDict(
        "input_path" => input_path,
        "output_path" => output_path,
        "fixed_entries" => fixed_count,
        "total_entries" => length(records),
        "total_formulas" => length(formulas),
        "unique_by_string" => length(syntactically_unique),
        "semantically_unique" => length(representatives),
        "representatives" => representatives,
        "class_sizes" => class_sizes,
        "unique_non_trivial_formulas" => summary_stats["unique_non_trivial_formulas"],
        "maximum_formula_size" => summary_stats["maximum_formula_size"],
        "maximum_number_of_atomic_propositions" => summary_stats["maximum_number_of_atomic_propositions"],
        "maximum_temporal_depth" => summary_stats["maximum_temporal_depth"],
    )
end

function main()
    fix_and_analyze_ltlequ_normalized()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end