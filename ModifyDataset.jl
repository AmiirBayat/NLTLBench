using JSON3
using OrderedCollections

const DEFAULT_DATASET_PATH = joinpath(@__DIR__, "dataset", "DatasetWithNaturalNL_plus_simplified.json")
const DEFAULT_SOURCE_DATASET_PATH = joinpath(@__DIR__, "dataset", "ltl_dataset.json")
const DEFAULT_TARGET_DATASET_PATH = DEFAULT_DATASET_PATH
function next_available_id(records)::Int
    max_id = 0
    for record in records
        if haskey(record, "id")
            value = try
                Int(record["id"])
            catch
                continue
            end
            max_id = max(max_id, value)
        end
    end
    return max_id + 1
end

function build_appended_record(source_record::OrderedDict{String,Any}, new_id::Int)
    entry = OrderedDict{String,Any}(
        "id" => new_id,
        "LTL" => source_record["LTL"],
    )

    for key in [
        "temporal_behavior",
        "ast_size",
        "ast_depth",
        "temporal_depth",
        "num_atomic_props",
        "operator_counts",
    ]
        if haskey(source_record, key)
            entry[key] = source_record[key]
        end
    end

    return entry
end

function existing_ltl_set(records)
    ltl_set = Set{String}()
    for record in records
        if haskey(record, "LTL")
            push!(ltl_set, strip(String(record["LTL"])))
        end
    end
    return ltl_set
end

function append_new_ltl_dataset_entries_to_main_dataset(
    source_dataset_path::String = DEFAULT_SOURCE_DATASET_PATH;
    target_dataset_path::String = DEFAULT_TARGET_DATASET_PATH,
    min_source_id::Int = 525,
)
    source_records = load_dataset(source_dataset_path)
    target_records = load_dataset(target_dataset_path)

    next_id = next_available_id(target_records)
    existing_ltls = existing_ltl_set(target_records)
    added_count = 0

    for source_record in source_records
        haskey(source_record, "id") || continue
        source_id = try
            Int(source_record["id"])
        catch
            continue
        end
        if source_id < min_source_id
            continue
        end
        haskey(source_record, "LTL") || continue

        ltl = strip(String(source_record["LTL"]))
        if ltl in existing_ltls
            continue
        end

        push!(target_records, build_appended_record(source_record, next_id))
        push!(existing_ltls, ltl)
        next_id += 1
        added_count += 1
    end

    save_dataset(target_records, target_dataset_path)

    println("Appended ", added_count, " entries from ", source_dataset_path, " to: ", target_dataset_path)
    println("Only source records with id >= ", min_source_id, " and new LTL strings were added.")
end

function remove_appended_metadata_fields(
    dataset_path::String = DEFAULT_TARGET_DATASET_PATH;
    min_appended_id::Int = 525,
)
    records = load_dataset(dataset_path)
    changed_count = 0

    for record in records
        haskey(record, "id") || continue
        record_id = try
            Int(record["id"])
        catch
            continue
        end
        if record_id < min_appended_id
            continue
        end

        changed = false
        if haskey(record, "origin")
            delete!(record, "origin")
            changed = true
        end
        if haskey(record, "source_record_id")
            delete!(record, "source_record_id")
            changed = true
        end

        if changed
            changed_count += 1
        end
    end

    save_dataset(records, dataset_path)

    println("Removed `origin` and `source_record_id` from ", changed_count, " records with id >= ", min_appended_id, " in: ", dataset_path)
end

function load_dataset(dataset_path::String = DEFAULT_DATASET_PATH)
    if !isfile(dataset_path)
        throw(ArgumentError("Dataset file not found: $(dataset_path)"))
    end

    content = strip(read(dataset_path, String))
    isempty(content) && throw(ArgumentError("Dataset file is empty: $(dataset_path)"))

    parsed = JSON3.read(content)
    return [OrderedDict{String,Any}(String(k) => v for (k, v) in pairs(record)) for record in parsed]
end


function save_dataset(records, dataset_path::String = DEFAULT_DATASET_PATH)
    open(dataset_path, "w") do io
        JSON3.pretty(io, records)
        write(io, "\n")
    end
end

function get_ltl2tgba_path()
    path = Sys.which("ltl2tgba")
    isnothing(path) && throw(ArgumentError(
        "`ltl2tgba` was not found in PATH. Please install Spot and make sure `ltl2tgba` is available."
    ))
    return path
end

function automaton_num_states_and_transitions(formula::AbstractString)
    ltl2tgba_path = get_ltl2tgba_path()
    cmd = `$(ltl2tgba_path) -f $(String(formula))`
    output = read(cmd, String)

    states = nothing
    transitions = 0
    in_body = false

    for raw_line in split(output, '\n')
        line = strip(raw_line)
        isempty(line) && continue

        if startswith(line, "States:")
            value_str = strip(split(line, ":"; limit=2)[2])
            states = try
                parse(Int, value_str)
            catch
                throw(ArgumentError("Could not parse automaton state count from ltl2tgba output line: $(repr(line))"))
            end
        elseif line == "--BODY--"
            in_body = true
        elseif line == "--END--"
            in_body = false
        elseif in_body && startswith(line, "[")
            transitions += 1
        end
    end

    isnothing(states) && throw(ArgumentError(
        "Could not find `States:` in ltl2tgba output for formula $(repr(String(formula)))."
    ))

    return states, transitions
end

function tokenize_ltl_metadata(formula::AbstractString)
    s = strip(String(formula))
    tokens = String[]
    i = firstindex(s)

    while i <= lastindex(s)
        c = s[i]

        if isspace(c)
            i = nextind(s, i)
        elseif c in ('(', ')', '&', '|', '!')
            push!(tokens, string(c))
            i = nextind(s, i)
        elseif c == '<'
            j = nextind(s, i)
            k = j <= lastindex(s) ? nextind(s, j) : j
            if j <= lastindex(s) && k <= lastindex(s) && s[j] == '-' && s[k] == '>'
                push!(tokens, "<->")
                i = nextind(s, k)
            else
                push!(tokens, string(c))
                i = nextind(s, i)
            end
        elseif c == '-'
            j = nextind(s, i)
            if j <= lastindex(s) && s[j] == '>'
                push!(tokens, "->")
                i = nextind(s, j)
            else
                push!(tokens, string(c))
                i = nextind(s, i)
            end
        else
            j = i
            while j <= lastindex(s)
                cj = s[j]
                if isspace(cj) || cj in ('(', ')', '&', '|', '!', '<', '-')
                    break
                end
                j = nextind(s, j)
            end
            push!(tokens, s[i:prevind(s, j)])
            i = j
        end
    end

    return tokens
end

function is_unary_temporal_token(tok::AbstractString)
    return tok in ("G", "F", "X")
end

function is_binary_temporal_token(tok::AbstractString)
    return tok in ("U", "R", "W", "M")
end

function is_logical_operator_token(tok::AbstractString)
    return tok in ("!", "&", "|", "->", "<->")
end

function is_proposition_token(tok::AbstractString)
    return startswith(tok, "prop_")
end

function is_nnf_formula(formula::AbstractString)
    tokens = tokenize_ltl_metadata(formula)
    isempty(tokens) && return false

    function parse_formula(i::Int)
        i > length(tokens) && error("Unexpected end of formula while checking NNF")
        tok = tokens[i]

        if tok == "("
            inner_node, j = parse_formula(i + 1)
            j > length(tokens) && error("Missing closing parenthesis while checking NNF")

            if tokens[j] == ")"
                return inner_node, j + 1
            end

            op = tokens[j]
            op in ("&", "|", "U", "R", "W", "M", "->", "<->") || error("Unexpected binary operator $(op) while checking NNF")
            right_node, k = parse_formula(j + 1)
            k > length(tokens) && error("Missing closing parenthesis while checking NNF")
            tokens[k] == ")" || error("Missing closing parenthesis while checking NNF")
            return (kind=:binary, op=op, left=inner_node, right=right_node), k + 1
        elseif tok == "!"
            child_node, j = parse_formula(i + 1)
            return (kind=:not, child=child_node), j
        elseif tok in ("X", "F", "G")
            child_node, j = parse_formula(i + 1)
            return (kind=:unary, op=tok, child=child_node), j
        else
            return (kind=:atom, value=tok), i + 1
        end
    end

    function nnf_ok(node)
        kind = node.kind

        if kind == :atom
            return is_proposition_token(String(node.value))
        elseif kind == :not
            return node.child.kind == :atom && is_proposition_token(String(node.child.value))
        elseif kind == :unary
            node.op in ("X", "F", "G") || return false
            return nnf_ok(node.child)
        elseif kind == :binary
            node.op in ("->", "<->", "R", "W", "M") && return false
            node.op in ("&", "|", "U") || return false
            return nnf_ok(node.left) && nnf_ok(node.right)
        else
            return false
        end
    end

    try
        node, next_i = parse_formula(1)
        next_i == length(tokens) + 1 || return false
        return nnf_ok(node)
    catch
        return false
    end
end

function compute_formula_metadata(formula::AbstractString)
    tokens = tokenize_ltl_metadata(formula)

    formula_size = count(tok -> tok != "(" && tok != ")", tokens)
    proposition_set = Set{String}(tok for tok in tokens if is_proposition_token(tok))
    proposition_count = length(proposition_set)

    operators = String[]
    temporal_ops = String[]
    for tok in tokens
        if is_unary_temporal_token(tok) || is_binary_temporal_token(tok) || is_logical_operator_token(tok)
            push!(operators, tok)
        end
        if is_unary_temporal_token(tok) || is_binary_temporal_token(tok)
            push!(temporal_ops, tok)
        end
    end

    max_paren_depth = 0
    current_paren_depth = 0
    for tok in tokens
        if tok == "("
            current_paren_depth += 1
            max_paren_depth = max(max_paren_depth, current_paren_depth)
        elseif tok == ")"
            current_paren_depth = max(0, current_paren_depth - 1)
        end
    end

    temporal_nesting_depth = 0
    current_temporal_depth = 0
    stack = String[]
    for tok in tokens
        if is_unary_temporal_token(tok)
            current_temporal_depth += 1
            temporal_nesting_depth = max(temporal_nesting_depth, current_temporal_depth)
            push!(stack, tok)
        elseif tok == "("
            continue
        elseif tok == ")"
            if !isempty(stack)
                pop!(stack)
                current_temporal_depth = max(0, current_temporal_depth - 1)
            end
        elseif is_binary_temporal_token(tok)
            temporal_nesting_depth = max(temporal_nesting_depth, max(1, current_temporal_depth))
        end
    end

    temporal_behavior = if isempty(temporal_ops)
        "non_temporal"
    elseif all(op -> op == "G", temporal_ops)
        "safety"
    elseif all(op -> op == "F", temporal_ops)
        "eventuality"
    elseif any(op -> op in ("U", "R", "W", "M"), temporal_ops)
        "until_release"
    elseif any(op -> op == "X", temporal_ops)
        "next_step"
    else
        "mixed_temporal"
    end

    return OrderedDict(
        "formula_size" => formula_size,
        "nesting_depth" => max(max_paren_depth, temporal_nesting_depth),
        "operators" => unique(operators),
        "proposition_count" => proposition_count,
        "temporal_behavior" => temporal_behavior,
    )
end

function rename_paraphrase_fields!(record::OrderedDict{String,Any})
    updated = OrderedDict{String,Any}()

    for (k, v) in pairs(record)
        key = String(k)
        if key == "paraphrase"
            updated["paraphrase_gpt5.4-mini"] = v
        elseif key == "paraphrase_model"
            continue
        else
            updated[key] = v
        end
    end

    empty!(record)
    for (k, v) in pairs(updated)
        record[k] = v
    end

    return record
end


function update_dataset_paraphrase_keys(dataset_path::String = DEFAULT_DATASET_PATH)
    records = load_dataset(dataset_path)
    changed_count = 0

    for record in records
        had_paraphrase = haskey(record, "paraphrase")
        had_paraphrase_model = haskey(record, "paraphrase_model")

        if had_paraphrase || had_paraphrase_model
            rename_paraphrase_fields!(record)
            changed_count += 1
        end
    end

    save_dataset(records, dataset_path)

    println("Updated ", changed_count, " records in: ", dataset_path)
    println("Renamed `paraphrase` to `paraphrase_gpt5.4-mini` and removed `paraphrase_model`.")
end

function update_new_records_with_formula_metadata(
    dataset_path::String = DEFAULT_DATASET_PATH;
    min_id::Int = 525,
)
    records = load_dataset(dataset_path)
    changed_count = 0

    for record in records
        haskey(record, "id") || continue
        record_id = try
            Int(record["id"])
        catch
            continue
        end
        if record_id < min_id
            continue
        end
        haskey(record, "LTL") || continue

        metadata = compute_formula_metadata(String(record["LTL"]))
        for (k, v) in pairs(metadata)
            record[k] = v
        end
        changed_count += 1
    end

    save_dataset(records, dataset_path)

    println("Updated formula metadata for ", changed_count, " records in: ", dataset_path)
    println("Added/updated fields: formula_size, nesting_depth, operators, proposition_count, temporal_behavior")
end

function update_existing_records_with_formula_size(
    dataset_path::String = DEFAULT_DATASET_PATH;
    max_id::Int = 524,
)
    records = load_dataset(dataset_path)
    changed_count = 0

    for record in records
        haskey(record, "id") || continue
        record_id = try
            Int(record["id"])
        catch
            continue
        end
        if record_id > max_id
            continue
        end
        haskey(record, "LTL") || continue

        metadata = compute_formula_metadata(String(record["LTL"]))
        record["formula_size"] = metadata["formula_size"]
        changed_count += 1
    end

    save_dataset(records, dataset_path)

    println("Updated formula_size for ", changed_count, " records with id <= ", max_id, " in: ", dataset_path)
end

function update_dataset_with_automaton_size_information(
    dataset_path::String = DEFAULT_DATASET_PATH;
    min_id::Int = 1,
    max_id::Int = typemax(Int),
    overwrite::Bool = false,
)
    records = load_dataset(dataset_path)
    changed_count = 0
    skipped_count = 0

    for record in records
        haskey(record, "id") || continue
        record_id = try
            Int(record["id"])
        catch
            continue
        end
        if record_id < min_id || record_id > max_id
            continue
        end
        haskey(record, "LTL") || continue

        already_has_all = haskey(record, "automaton_num_states") &&
                          haskey(record, "automaton_num_transitions") &&
                          haskey(record, "automaton_size")
        if !overwrite && already_has_all
            skipped_count += 1
            continue
        end

        num_states, num_transitions = automaton_num_states_and_transitions(String(record["LTL"]))
        record["automaton_num_states"] = num_states
        record["automaton_num_transitions"] = num_transitions
        record["automaton_size"] = num_states + num_transitions
        changed_count += 1
    end

    save_dataset(records, dataset_path)

    println("Updated automaton size information for ", changed_count, " records in: ", dataset_path)
    println("Skipped ", skipped_count, " records that already had automaton_num_states, automaton_num_transitions, and automaton_size.")
    println("Processed record ids in [", min_id, ", ", max_id, "].")
end

function update_dataset_with_nnf_information(
    dataset_path::String = DEFAULT_DATASET_PATH;
    min_id::Int = 1,
    max_id::Int = typemax(Int),
)
    records = load_dataset(dataset_path)
    changed_count = 0

    for record in records
        haskey(record, "id") || continue
        record_id = try
            Int(record["id"])
        catch
            continue
        end
        if record_id < min_id || record_id > max_id
            continue
        end
        haskey(record, "LTL") || continue

        record["is_nnf"] = is_nnf_formula(String(record["LTL"]))
        changed_count += 1
    end

    save_dataset(records, dataset_path)

    println("Updated NNF information for ", changed_count, " records in: ", dataset_path)
    println("Added/updated field: is_nnf")
    println("Processed record ids in [", min_id, ", ", max_id, "].")
end

function main()
    println("Loaded ModifyDataset.jl")
    println("Run `update_dataset_paraphrase_keys()` to rename `paraphrase` to `paraphrase_gpt5.4-mini` and remove `paraphrase_model`.")
    println("Run `update_new_records_with_formula_metadata(min_id=525)` to add formula_size, nesting_depth, operators, proposition_count, and temporal_behavior to the newly added records.")
    println("Run `update_existing_records_with_formula_size(max_id=524)` to add only formula_size to the earlier records.")
    println("Run `append_new_ltl_dataset_entries_to_main_dataset(min_source_id=525)` to append new entries from dataset/ltl_dataset.json to DatasetWithNaturalNL_plus_simplified.json.")
    println("Run `remove_appended_metadata_fields(min_appended_id=525)` to remove `origin` and `source_record_id` from the appended records.")
    println("Run `update_dataset_with_automaton_size_information(min_id=1)` to compute automaton_num_states, automaton_num_transitions, and automaton_size for each LTL formula.")
    println("Run `update_dataset_with_nnf_information(min_id=1)` to add `is_nnf`, where NNF means negations appear only in front of atomic propositions and only X, F, G, and U are kept as temporal operators, with implication/equivalence excluded.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end