

using JSON3
using OrderedCollections
using Dates

const DEFAULT_RESULTS_PATH = joinpath(@__DIR__, "results", "Mistral.json")

function load_results(results_path::String = DEFAULT_RESULTS_PATH)
    if !isfile(results_path)
        throw(ArgumentError("Results file not found: $(results_path)"))
    end

    content = strip(read(results_path, String))
    isempty(content) && throw(ArgumentError("Results file is empty: $(results_path)"))

    parsed = JSON3.read(content)
    results_obj = OrderedDict{String,Any}(String(k) => v for (k, v) in pairs(parsed))

    if haskey(results_obj, "results")
        materialized = OrderedDict{String,Any}[]
        for entry in results_obj["results"]
            push!(materialized, OrderedDict{String,Any}(String(k) => v for (k, v) in pairs(entry)))
        end
        results_obj["results"] = materialized
    else
        throw(ArgumentError("Results file does not contain a `results` field: $(results_path)"))
    end

    return results_obj
end

function save_results(results_obj::OrderedDict{String,Any}, results_path::String = DEFAULT_RESULTS_PATH)
    open(results_path, "w") do io
        JSON3.pretty(io, results_obj)
        write(io, "\n")
    end
end

function result_key(entry::OrderedDict{String,Any})
    record_id = get(entry, "record_id", nothing)
    input_field = String(get(entry, "input_field", ""))
    original_ltl = String(get(entry, "original_ltl", ""))
    source_model = String(get(entry, "source_model", ""))
    return "$(record_id)||$(input_field)||$(original_ltl)||$(source_model)"
end

function is_true_result(entry::OrderedDict{String,Any})
    return get(entry, "equivalent", nothing) === true
end

function is_ok_result(entry::OrderedDict{String,Any})
    return String(get(entry, "status", "")) == "ok"
end

function choose_better_entry(current::OrderedDict{String,Any}, candidate::OrderedDict{String,Any})
    # Highest priority: keep the one whose equivalence is true.
    if is_true_result(candidate) && !is_true_result(current)
        return candidate
    elseif is_true_result(current) && !is_true_result(candidate)
        return current
    end

    # Next priority: prefer a successful parse/run over an error.
    if is_ok_result(candidate) && !is_ok_result(current)
        return candidate
    elseif is_ok_result(current) && !is_ok_result(candidate)
        return current
    end

    # If both have the same quality level, keep the later one encountered.
    return candidate
end

function deduplicate_mistral_results(results_path::String = DEFAULT_RESULTS_PATH)
    results_obj = load_results(results_path)
    entries = results_obj["results"]

    kept = Dict{String,OrderedDict{String,Any}}()
    original_count = length(entries)

    for entry in entries
        key = result_key(entry)
        if haskey(kept, key)
            kept[key] = choose_better_entry(kept[key], entry)
        else
            kept[key] = entry
        end
    end

    deduplicated = OrderedDict{String,Any}[]
    seen_keys = Set{String}()

    # Preserve stable order of first appearance of each deduplicated key.
    for entry in entries
        key = result_key(entry)
        if !(key in seen_keys)
            push!(deduplicated, kept[key])
            push!(seen_keys, key)
        end
    end

    results_obj["results"] = deduplicated
    if haskey(results_obj, "updated_at")
        results_obj["updated_at"] = string(Dates.now())
    end

    save_results(results_obj, results_path)

    println("Original result count: ", original_count)
    println("Deduplicated result count: ", length(deduplicated))
    println("Removed duplicates: ", original_count - length(deduplicated))
    println("Saved cleaned results to: ", results_path)
end

function main()
    println("Loaded Duplicate_remove.jl")
    println("Run `deduplicate_mistral_results()` to clean duplicate entries in results/Mistral.json.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end