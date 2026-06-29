using JSON3
using Dates

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Input JSON produced by the Colab T5 inference script.
# Update this path if needed.
const INPUT_PATH = "results/lang2ltl.json"

# Output JSON with equivalence results.
const OUTPUT_PATH = "results/lang2ltl_checked.json"

# Spot binaries. Make sure `ltlfilt` is available in PATH.
const LTLFILT = "ltlfilt"
const UNARY_OPS = Set(["F", "G", "X", "!"])
const BINARY_OPS = Set(["&", "|", "U", "->"])

# -----------------------------------------------------------------------------
# Atomic proposition normalization
# -----------------------------------------------------------------------------

"""
    normalize_props(formula::AbstractString)

Normalize atomic propositions to the form `prop_1`, `prop_2`, ... .
This is useful because Lang2LTL outputs formulas over letters such as `a`, `b`, ... .
The mapping is created in order of first appearance.

Example:
`F(a) & G(!b)` -> `F(prop_1) & G(!prop_2)`.
"""
function normalize_props(formula::AbstractString)
    # Operators and constants that should not be treated as APs.
    reserved = Set([
        "F", "G", "X", "U", "R", "W", "M",
        "true", "false", "tt", "ff",
    ])

    # Match identifiers such as a, b, prop_1, blue_room.
    token_re = r"\b[A-Za-z_][A-Za-z0-9_]*\b"

    mapping = Dict{String,String}()
    counter = 1

    function repl(m)
        tok = String(m)
        if tok in reserved
            return tok
        end
        if !haskey(mapping, tok)
            mapping[tok] = "prop_$(counter)"
            counter += 1
        end
        return mapping[tok]
    end

    normalized = replace(formula, token_re => repl)
    return normalized, mapping
end

"""
    normalize_pair(gold, pred)

Normalize APs in the gold and predicted formulas using a shared mapping.
If one formula uses `prop_i` and the other uses letters, both are mapped to a common
`prop_1`, `prop_2`, ... representation according to first appearance in `gold`, then `pred`.
"""
function normalize_pair(gold::AbstractString, pred::AbstractString)
    reserved = Set([
        "F", "G", "X", "U", "R", "W", "M",
        "true", "false", "tt", "ff",
    ])

    token_re = r"\b[A-Za-z_][A-Za-z0-9_]*\b"
    mapping = Dict{String,String}()
    counter = 1

    function normalize_formula(formula::AbstractString)
        function repl(m)
            tok = String(m)
            if tok in reserved
                return tok
            end
            if !haskey(mapping, tok)
                mapping[tok] = "prop_$(counter)"
                counter += 1
            end
            return mapping[tok]
        end
        return replace(formula, token_re => repl)
    end

    return normalize_formula(gold), normalize_formula(pred), mapping
end

# -----------------------------------------------------------------------------
# Lang2LTL prefix-output parsing and AP recovery
# -----------------------------------------------------------------------------

"""
    prop_order_from_text(text)

Return propositions `prop_i` appearing in `text`, ordered by their index.
This reconstructs the same mapping used in the Colab inference script:
`prop_1 -> a`, `prop_2 -> b`, ... .
"""
function prop_order_from_text(text::AbstractString)
    props = unique(collect(eachmatch(r"prop_\d+", text)) .|> m -> String(m.match))
    sort!(props, by = p -> parse(Int, split(p, "_")[2]))
    return props
end

function letter_to_prop_mapping(source_text::AbstractString, gold_ltl::AbstractString)
    props = prop_order_from_text(source_text)
    if isempty(props)
        props = prop_order_from_text(gold_ltl)
    end

    mapping = Dict{String,String}()
    for (i, p) in enumerate(props)
        if i <= 26
            mapping[string(Char(Int('a') + i - 1))] = p
        end
    end
    return mapping
end

function replace_letters_with_props(formula::AbstractString, mapping::Dict{String,String})
    out = String(formula)
    for (letter, prop) in mapping
        out = replace(out, Regex("\\b" * letter * "\\b") => prop)
    end
    return out
end

function tokenize_lang2ltl_prefix(raw::AbstractString)
    s = replace(String(raw), "LTL:" => "")
    s = replace(s, "(" => " ( ")
    s = replace(s, ")" => " ) ")
    s = replace(s, "->" => " -> ")
    s = replace(s, "&" => " & ")
    s = replace(s, "|" => " | ")
    s = replace(s, "U!" => " U ! ")
    s = replace(s, "G!" => " G ! ")
    s = replace(s, "F!" => " F ! ")
    s = replace(s, "X!" => " X ! ")
    s = replace(s, "!" => " ! ")
    s = strip(replace(s, r"\s+" => " "))
    return isempty(s) ? String[] : split(s)
end

function parse_prefix_tokens!(tokens::Vector{String})
    isempty(tokens) && error("empty token list")
    tok = popfirst!(tokens)

    if tok in UNARY_OPS
        child = parse_prefix_tokens!(tokens)
        return tok == "!" ? "!($(child))" : "$(tok)($(child))"
    elseif tok in BINARY_OPS
        left = parse_prefix_tokens!(tokens)
        right = parse_prefix_tokens!(tokens)
        return "($(left) $(tok) $(right))"
    elseif tok == "(" || tok == ")"
        return parse_prefix_tokens!(tokens)
    else
        return tok
    end
end

function prefix_to_infix(raw::AbstractString)
    tokens = collect(tokenize_lang2ltl_prefix(raw))
    infix = parse_prefix_tokens!(tokens)
    if !isempty(tokens)
        error("unused tokens after prefix parse: $(join(tokens, " "))")
    end
    return infix
end

function recover_prediction(d::Dict{String,Any})
    gold = String(get(d, "original_ltl", ""))
    source_text = String(get(d, "source_text", ""))
    raw = get(d, "raw_model_output", nothing)
    pred = get(d, "predicted_ltl", nothing)

    mapping = letter_to_prop_mapping(source_text, gold)

    if raw !== nothing && raw !== missing
        try
            parsed = prefix_to_infix(String(raw))
            return replace_letters_with_props(parsed, mapping), "parsed_from_raw", mapping
        catch e
            # If the raw Lang2LTL output is malformed, do not fall back to the
            # previously parsed prediction. The parsed prediction may be the result
            # of an incorrect parser and can hide syntax errors in the model output.
            return nothing, "raw_parse_error: $(sprint(showerror, e))", mapping
        end
    elseif pred !== nothing && pred !== missing
        return String(pred), "used_predicted_ltl", mapping
    else
        return nothing, "missing_formula", mapping
    end
end

# -----------------------------------------------------------------------------
# Spot equivalence checking
# -----------------------------------------------------------------------------

"""
    spot_equivalent(phi, psi)

Return `(equivalent, status)` using Spot's `ltlfilt --equivalent-to`.
"""
function spot_equivalent(phi::AbstractString, psi::AbstractString)
    try
        # ltlfilt prints the input formula if it is equivalent to the target formula.
        cmd = pipeline(
            `echo $psi`,
            ignorestatus(`$LTLFILT --equivalent-to=$phi`)
        )
        out = read(cmd, String)
        return !isempty(strip(out)), "ok"
    catch e
        return false, "spot_error: $(sprint(showerror, e))"
    end
end

# -----------------------------------------------------------------------------
# Main evaluation
# -----------------------------------------------------------------------------

function evaluate_results(input_path::AbstractString, output_path::AbstractString)
    records = JSON3.read(read(input_path, String))
    checked = Vector{Any}()

    total = length(records)
    println("Loaded $total records")

    for (i, r) in enumerate(records)
        d = Dict{String,Any}()
        for (k, v) in pairs(r)
            d[String(k)] = v
        end

        gold = get(d, "original_ltl", nothing)
        if gold === nothing || gold === missing
            d["equivalent"] = false
            d["status"] = "missing_original_ltl"
            d["checked_timestamp"] = Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS.sss")
            push!(checked, d)
            continue
        end

        recovered_pred, recovery_status, letter_mapping = recover_prediction(d)

        if recovered_pred === nothing
            d["equivalent"] = false
            d["status"] = recovery_status
            d["letter_mapping"] = letter_mapping
            d["checked_timestamp"] = Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS.sss")
            push!(checked, d)
            continue
        end

        gold_s = String(gold)
        pred_s = String(recovered_pred)

        gold_norm, pred_norm, prop_mapping = normalize_pair(gold_s, pred_s)
        equiv, spot_status = spot_equivalent(gold_norm, pred_norm)

        d["predicted_ltl_recovered"] = pred_s
        d["original_ltl_normalized"] = gold_norm
        d["predicted_ltl_normalized"] = pred_norm
        d["letter_mapping"] = letter_mapping
        d["prop_mapping"] = prop_mapping
        d["equivalent"] = equiv
        d["status"] = spot_status == "ok" ? "$(recovery_status); spot_ok" : spot_status

        d["checked_timestamp"] = Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS.sss")
        push!(checked, d)

        if i % 50 == 0 || i == total
            acc = count(x -> get(x, "equivalent", false) == true, checked) / length(checked)
            println("Checked $i / $total | current accuracy = $(round(acc * 100; digits=2))%")

            # Save intermediate results to avoid losing progress.
            open(output_path, "w") do io
                JSON3.pretty(io, checked)
            end
        end
    end

    open(output_path, "w") do io
        JSON3.pretty(io, checked)
    end

    n_ok = count(x -> occursin("spot_ok", get(x, "status", "")), checked)
    n_equiv = count(x -> get(x, "equivalent", false) == true, checked)
    println("Saved: $output_path")
    println("Total: $(length(checked))")
    println("Spot OK: $n_ok")
    println("Equivalent: $n_equiv")
    println("Accuracy: $(round(100 * n_equiv / length(checked); digits=2))%")

    return checked
end

# Run evaluation when this file is executed directly.
evaluate_results(INPUT_PATH, OUTPUT_PATH)