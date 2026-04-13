include("GenerateLTL.jl")
include("Filter.jl")
include("Satisfiability.jl")
#using GenerateLTL


using Random
using Dates
using OrderedCollections


function ensure_directory(path::String)
    isdir(path) || mkpath(path)
    return path
end


function timestamp_string()
    return Dates.format(now(), "yyyy-mm-dd_HHMMSS")
end

function json_escape_string(value::AbstractString)
    text = replace(String(value), '\\' => "\\\\")
    text = replace(text, '"' => "\\\"")
    text = replace(text, '\n' => "\\n")
    text = replace(text, '\r' => "\\r")
    text = replace(text, '\t' => "\\t")
    return text
end


function json_indent(level::Int)
    return repeat("  ", level)
end

function to_json_pretty(value; level::Int = 0)::String
    if value === nothing
        return "null"
    elseif value isa Bool
        return value ? "true" : "false"
    elseif value isa Integer || value isa AbstractFloat
        return string(value)
    elseif value isa AbstractString
        return "\"$(json_escape_string(value))\""
    elseif value isa Symbol
        return to_json_pretty(string(value); level=level)
    elseif value isa AbstractVector
        if isempty(value)
            return "[]"
        end
        items = [json_indent(level + 1) * to_json_pretty(item; level=level + 1) for item in value]
        return "[\n" * join(items, ",\n") * "\n" * json_indent(level) * "]"
    elseif value isa AbstractDict
        if isempty(value)
            return "{}"
        end
        parts = String[]
        for (k, v) in value
            push!(parts, json_indent(level + 1) * "\"$(json_escape_string(string(k)))\": " * to_json_pretty(v; level=level + 1))
        end
        return "{\n" * join(parts, ",\n") * "\n" * json_indent(level) * "}"
    else
        return to_json_pretty(string(value); level=level)
    end
end

function accepted_formula_record(id::Int, formula::LTLFormula, sat_status, exact_tautology_check::Bool)
    original_formula = formula_to_string(formula)
    simplified_formula = formula_to_string(simplify_formula_local(formula))
    selected_formula = canonical_formula_string_local(formula)

    # compute number of distinct atomic propositions
    function count_atomic_props(f::LTLFormula, seen=Set{String}())
        if f isa AP
            push!(seen, f.name)
        elseif f isa UnaryLTL
            count_atomic_props(f.child, seen)
        elseif f isa BinaryLTL
            count_atomic_props(f.left, seen)
            count_atomic_props(f.right, seen)
        end
        return seen
    end

    num_atomic_props = length(count_atomic_props(formula))

    return OrderedDict(
        "id" => id,
        "LTL" => selected_formula,
        "generated_formula" => original_formula,
        "simplified_formula" => simplified_formula,
        "ast_size" => ast_size(formula),
        "ast_depth" => ast_depth(formula),
        "temporal_depth" => temporal_depth(formula),
        "num_atomic_props" => num_atomic_props,
        "operator_counts" => OrderedDict(string(k) => v for (k, v) in sort!(collect(operator_counts(formula)); by=x -> string(x[1]))),
        "temporal_behavior" => classify_temporal_behavior(formula),
    )
end
function classify_temporal_behavior(formula::LTLFormula)
    text = formula_to_string(simplify_formula_local(formula))
    tags = String[]

    if occursin("G(", text) && !occursin("F(", text)
        push!(tags, "safety")
    end
    if occursin("F(", text) && !occursin("G(", text)
        push!(tags, "liveness")
    end
    if occursin("G(F(", text)
        push!(tags, "recurrence")
    end
    if occursin("F(G(", text)
        push!(tags, "persistence")
    end
    if occursin("U", text)
        push!(tags, "until")
    end
    if occursin("W", text)
        push!(tags, "weak_until")
    end
    if occursin("R", text)
        push!(tags, "release")
    end
    if occursin("X(", text)
        push!(tags, "next_step")
    end
    if occursin("->", text) || occursin("<->", text)
        push!(tags, "conditional")
    end
    if isempty(tags)
        push!(tags, "other")
    end

    return tags
end

function build_accepted_records(accepted_statuses, exact_tautology_check::Bool)
    return [
        accepted_formula_record(i, item[1], item[2], exact_tautology_check)
        for (i, item) in enumerate(accepted_statuses)
    ]
end

function merge_records_into_dataset_json(
    dataset_path::String,
    new_records,
    semantic_redundancy::Bool,
)
    temp_new_path = dataset_path * ".new.tmp.json"
    open(temp_new_path, "w") do io
        write(io, to_json_pretty(new_records))
        write(io, "\n")
    end

    python_code = raw"""
import json
import os
import subprocess
import sys

DATASET_PATH = sys.argv[1]
NEW_PATH = sys.argv[2]
SEMANTIC = sys.argv[3].lower() == 'true'

with open(NEW_PATH, 'r', encoding='utf-8') as f:
    new_records = json.load(f)

if os.path.exists(DATASET_PATH):
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        existing_records = json.load(f)
else:
    existing_records = []

def selected_formula(record):
    return record.get('LTL', record.get('selected_formula', record.get('simplified_formula', record.get('formula', ''))))

def preference_tuple(record):
    operator_counts = record.get('operator_counts', {}) or {}
    total_ops = sum(operator_counts.values()) if isinstance(operator_counts, dict) else 0
    selected = selected_formula(record)
    return (
        record.get('ast_size', 10**9),
        record.get('temporal_depth', 10**9),
        total_ops,
        len(selected),
        selected,
    )

def semantically_equivalent(a, b):
    if a == b:
        return True
    proc = subprocess.run(['ltlfilt', '-f', a, '--equivalent-to', b, '-q'])
    if proc.returncode == 0:
        return True
    if proc.returncode == 1:
        return False
    raise RuntimeError(f'ltlfilt failed while comparing {a!r} and {b!r} with exit code {proc.returncode}')

def deduplicate_records(records):
    deduped = []
    for record in records:
        matched_index = None
        for i, existing_record in enumerate(deduped):
            if semantically_equivalent(selected_formula(record), selected_formula(existing_record)):
                matched_index = i
                break

        if matched_index is None:
            deduped.append(record)
        else:
            if preference_tuple(record) < preference_tuple(deduped[matched_index]):
                deduped[matched_index] = record
    return deduped

existing_records = deduplicate_records(existing_records)

for new_record in new_records:
    matched_index = None
    for i, existing_record in enumerate(existing_records):
        if semantically_equivalent(selected_formula(new_record), selected_formula(existing_record)):
            matched_index = i
            break

    if matched_index is None:
        existing_records.append(new_record)
    else:
        if preference_tuple(new_record) < preference_tuple(existing_records[matched_index]):
            existing_records[matched_index] = new_record

for idx, record in enumerate(existing_records, start=1):
    record['id'] = idx

with open(DATASET_PATH, 'w', encoding='utf-8') as f:
    json.dump(existing_records, f, indent=2, ensure_ascii=False)
    f.write('\n')
"""

    run(`python3 -c $(python_code) $(dataset_path) $(temp_new_path) $(string(semantic_redundancy))`)
    rm(temp_new_path; force=true)
end

function save_run_json(
    filepath::String,
    accepted_statuses,
    exact_tautology_check::Bool,
    semantic_redundancy::Bool,
)
    accepted_records = build_accepted_records(accepted_statuses, exact_tautology_check)
    merge_records_into_dataset_json(filepath, accepted_records, semantic_redundancy)
end


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
    temporal_ops = [:X, :F, :G]
    output_dir = ensure_directory("dataset")
    dataset_name = "ltl_dataset"
    dataset_json_path = joinpath(output_dir, "$(dataset_name).json")
    save_json = true
    run_tag = timestamp_string()

    n_formulas = 50
    max_depth = 3
    max_ast_size = 5

    p_atom_at_max_depth = 1.0
    p_atom_before_max_depth = 0.35
    unary_weight = 0.50
    binary_weight = 0.25

    allow_boolean_constants = false
    boolean_constants = ["true", "false"]
    boolean_unary_ops = [:!]
    boolean_binary_ops = [:&, :|, :->, Symbol("<->")]

    max_attempts_per_formula = 500
    enforce_unique_formulas = true
    uniqueness_mode = :normalized
    redundancy_mode = :normalized
    semantic_redundancy = true
    semantic_backend = :spot
    semantic_existing_formulas = LTLFormula[]
    require_temporal_operator = true
    exact_satisfiability_check = true
    exact_tautology_check = true
    require_spot_backend = semantic_redundancy || exact_satisfiability_check || exact_tautology_check
    rng = MersenneTwister()

    # --------------------------------------------------------------------------
    # Backend checks
    # --------------------------------------------------------------------------
    if require_spot_backend
        require_ltlfilt()
    end

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
        require_temporal_operator = require_temporal_operator,
    )

    # --------------------------------------------------------------------------
    # Exact satisfiability / tautology analysis
    # --------------------------------------------------------------------------
    accepted_statuses = exact_satisfiability_check ? classify_satisfiability(accepted) : [(formula, :unknown) for formula in accepted]
    rejected_statuses = exact_satisfiability_check ? classify_satisfiability([item[1] for item in rejected]) : [(item[1], :unknown) for item in rejected]


    # --------------------------------------------------------------------------
    # Save results
    # --------------------------------------------------------------------------
    json_output_path = dataset_json_path

    if save_json
        save_run_json(
            json_output_path,
            accepted_statuses,
            exact_tautology_check,
            semantic_redundancy,
        )
    end

    # --------------------------------------------------------------------------
    # Display results
    # --------------------------------------------------------------------------
    println("Generated $(length(formulas)) candidate LTL formulas.\n")
    println("Note: this simple config has a small unique formula space, so `n_formulas` is kept modest to avoid exhausting the generator.\n")
    println("Accepted $(length(accepted)) formulas after filtering.")
    println("Rejected $(length(rejected)) formulas after filtering.\n")
    println("Filtering configuration:")
    println("  Redundancy mode: ", redundancy_mode)
    println("  Semantic redundancy enabled: ", semantic_redundancy)
    println("  Semantic backend: ", semantic_backend, "\n")
    println("  Require temporal operator: ", require_temporal_operator)
    println("  Exact satisfiability check: ", exact_satisfiability_check)
    println("  Exact tautology check: ", exact_tautology_check, "\n")
    println("Saved files:")
    if save_json
        println("  JSON dataset (updated in place): ", json_output_path)
    end
    println()

    println("Accepted formulas:\n")
    for (i, item) in enumerate(accepted_statuses)
        formula, sat_status = item
        println("Formula $(i): ", formula_to_string(formula))
        simplified_local_str = formula_to_string(simplify_formula_local(formula))
        if simplified_local_str != formula_to_string(formula)
            println("  Simplified form: ", simplified_local_str)
        end
        canonical_local_str = canonical_formula_string_local(formula)
        if canonical_local_str != simplified_local_str
            println("  Selected formula: ", canonical_local_str)
        end
        println("  Satisfiability: ", sat_status)
        println("  Temporal behavior: ", classify_temporal_behavior(formula))
        if exact_tautology_check
            println("  Tautology: ", is_tautology_exact(formula))
        end
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
            sat_status = rejected_statuses[i][2]
            println("Rejected $(i): ", formula_to_string(formula))
            simplified_local_str = formula_to_string(simplify_formula_local(formula))
            if simplified_local_str != formula_to_string(formula)
                println("  Simplified form: ", simplified_local_str)
            end
            canonical_local_str = canonical_formula_string_local(formula)
            if canonical_local_str != simplified_local_str
                println("  Selected formula: ", canonical_local_str)
            end
            println("  Reasons: ", reasons)
            println("  Temporal behavior: ", classify_temporal_behavior(formula))
            println("  Satisfiability: ", sat_status)
            if exact_tautology_check
                println("  Tautology: ", is_tautology_exact(formula))
            end
            println()
        end
    end
end

main()