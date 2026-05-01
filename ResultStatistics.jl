

using JSON3
using OrderedCollections
using Plots

const DEFAULT_DATASET_PATH = joinpath(@__DIR__, "dataset", "DatasetWithNaturalNL.json")
const DEFAULT_RESULTS_DIR = joinpath(@__DIR__, "results")
const DEFAULT_OUTPUT_DIR = joinpath(@__DIR__, "results")

# =================================================================================================
# I/O helpers
# =================================================================================================

function ensure_directory(path::String)
    isdir(path) || mkpath(path)
end

function load_json_array(path::String)
    if !isfile(path)
        throw(ArgumentError("JSON file not found: $(path)"))
    end
    parsed = JSON3.read(read(path, String))
    return collect(parsed)
end

function load_json_object(path::String)
    if !isfile(path)
        throw(ArgumentError("JSON file not found: $(path)"))
    end
    content = strip(read(path, String))
    isempty(content) && throw(ArgumentError("JSON file is empty: $(path)"))
    parsed = JSON3.read(content)
    return OrderedDict{String,Any}(String(k) => v for (k, v) in pairs(parsed))
end

function available_result_files(results_dir::String = DEFAULT_RESULTS_DIR)
    if !isdir(results_dir)
        throw(ArgumentError("Results directory not found: $(results_dir)"))
    end
    files = sort(filter(f -> endswith(lowercase(f), ".json") && isfile(f) && filesize(f) > 0, readdir(results_dir; join=true)))
    isempty(files) && throw(ArgumentError("No non-empty JSON result files found in $(results_dir)"))
    return files
end

# =================================================================================================
# Dataset indexing
# =================================================================================================

function load_dataset_index(dataset_path::String = DEFAULT_DATASET_PATH)
    dataset = load_json_array(dataset_path)
    index = Dict{Int,OrderedDict{String,Any}}()

    for record in dataset
        dict_record = OrderedDict{String,Any}(String(k) => v for (k, v) in pairs(record))
        haskey(dict_record, "id") || continue
        index[Int(dict_record["id"])] = dict_record
    end

    return index
end

function get_record_ast_size(record::OrderedDict{String,Any})
    if haskey(record, "ast_size")
        return Int(record["ast_size"])
    else
        throw(ArgumentError("Dataset record ID $(get(record, "id", "?")) does not contain `ast_size`."))
    end
end

# =================================================================================================
# Result parsing
# =================================================================================================

function load_result_entries(results_path::String)
    results_obj = load_json_object(results_path)
    haskey(results_obj, "results") || throw(ArgumentError("Result file $(results_path) does not contain a `results` field."))

    entries = OrderedDict{String,Any}[]
    for entry in results_obj["results"]
        push!(entries, OrderedDict{String,Any}(String(k) => v for (k, v) in pairs(entry)))
    end
    return results_obj, entries
end

function result_model_name(results_obj::OrderedDict{String,Any}, results_path::String)
    if haskey(results_obj, "translation_model")
        return String(results_obj["translation_model"])
    end
    return splitext(basename(results_path))[1]
end

function result_provider_name(results_obj::OrderedDict{String,Any})
    if haskey(results_obj, "translation_provider")
        return String(results_obj["translation_provider"])
    end
    return "unknown"
end

function source_label_from_field(field::String)
    if field == "natural_paraphrase"
        return "natural"
    elseif field == "paraphrase_gpt5.4-mini"
        return "gpt-5.4-mini"
    elseif field == "paraphrase_gemini-2.5-flash"
        return "gemini-2.5-flash"
    elseif field == "paraphrase_deepseek"
        return "deepseek"
    elseif field == "paraphrase_claude"
        return "claude"
    else
        return field
    end
end

function success_rate(entries::Vector{OrderedDict{String,Any}})
    total = length(entries)
    total == 0 && return 0.0

    success_count = 0
    for entry in entries
        status_ok = haskey(entry, "status") && String(entry["status"]) == "ok"
        equivalent_true = haskey(entry, "equivalent") && entry["equivalent"] === true
        success_count += (status_ok && equivalent_true) ? 1 : 0
    end

    return success_count / total
end

function fieldwise_statistics(entries::Vector{OrderedDict{String,Any}})
    stats = OrderedDict{String,OrderedDict{String,Int}}()

    for entry in entries
        field = haskey(entry, "input_field") ? String(entry["input_field"]) : "unknown"
        if !haskey(stats, field)
            stats[field] = OrderedDict(
                "total" => 0,
                "ok" => 0,
                "equivalent" => 0,
                "success" => 0,
                "error" => 0,
            )
        end

        stats[field]["total"] += 1

        status_ok = haskey(entry, "status") && String(entry["status"]) == "ok"
        equivalent_true = haskey(entry, "equivalent") && entry["equivalent"] === true

        if status_ok
            stats[field]["ok"] += 1
        else
            stats[field]["error"] += 1
        end

        if equivalent_true
            stats[field]["equivalent"] += 1
        end

        if status_ok && equivalent_true
            stats[field]["success"] += 1
        end
    end

    return stats
end

function size_bucket_statistics(entries::Vector{OrderedDict{String,Any}}, dataset_index::Dict{Int,OrderedDict{String,Any}})
    bucket_counts = OrderedDict{Int,OrderedDict{String,Int}}()

    for entry in entries
        haskey(entry, "record_id") || continue
        record_id = Int(entry["record_id"])
        haskey(dataset_index, record_id) || continue
        size_value = get_record_ast_size(dataset_index[record_id])

        if !haskey(bucket_counts, size_value)
            bucket_counts[size_value] = OrderedDict(
                "total" => 0,
                "success" => 0,
            )
        end

        bucket_counts[size_value]["total"] += 1

        status_ok = haskey(entry, "status") && String(entry["status"]) == "ok"
        equivalent_true = haskey(entry, "equivalent") && entry["equivalent"] === true
        if status_ok && equivalent_true
            bucket_counts[size_value]["success"] += 1
        end
    end

    return bucket_counts
end


function size_bucket_success_rates(entries::Vector{OrderedDict{String,Any}}, dataset_index::Dict{Int,OrderedDict{String,Any}})
    bucket_stats = size_bucket_statistics(entries, dataset_index)
    sizes = sort(collect(keys(bucket_stats)))
    rates = Float64[]
    totals = Int[]

    for size_value in sizes
        total = bucket_stats[size_value]["total"]
        success = bucket_stats[size_value]["success"]
        push!(totals, total)
        push!(rates, total == 0 ? 0.0 : success / total)
    end

    return sizes, rates, totals
end

function overall_success_statistics(entries::Vector{OrderedDict{String,Any}})
    total = length(entries)
    success_count = count(entry -> (haskey(entry, "status") && String(entry["status"]) == "ok") && (haskey(entry, "equivalent") && entry["equivalent"] === true), entries)
    ok_count = count(entry -> haskey(entry, "status") && String(entry["status"]) == "ok", entries)
    equivalent_count = count(entry -> haskey(entry, "equivalent") && entry["equivalent"] === true, entries)
    error_count = total - ok_count
    success_rate_value = total == 0 ? 0.0 : success_count / total

    return OrderedDict(
        "total" => total,
        "ok" => ok_count,
        "equivalent" => equivalent_count,
        "success" => success_count,
        "error" => error_count,
        "success_rate" => success_rate_value,
    )
end

function aggregate_result_file_statistics(results_path::String)
    results_obj, entries = load_result_entries(results_path)
    return OrderedDict(
        "results_path" => results_path,
        "provider" => result_provider_name(results_obj),
        "model" => result_model_name(results_obj, results_path),
        "entries" => entries,
        "overall" => overall_success_statistics(entries),
        "fieldwise" => fieldwise_statistics(entries),
    )
end

function aggregate_all_result_files(results_dir::String = DEFAULT_RESULTS_DIR)
    files = available_result_files(results_dir)
    aggregated = OrderedDict{String,Any}[]

    for path in files
        try
            push!(aggregated, aggregate_result_file_statistics(path))
        catch err
            println("Warning: skipping result file `", path, "` because it could not be parsed: ", err)
        end
    end

    isempty(aggregated) && throw(ArgumentError("No valid result JSON files could be loaded from $(results_dir)."))
    return aggregated
end

# =================================================================================================
# Printing and plotting
# =================================================================================================

function print_model_comparison_table(aggregated_stats::Vector{OrderedDict{String,Any}})
    println("================================================================================")
    println("Overall success rate comparison")
    println("================================================================================")

    sorted_stats = sort(aggregated_stats; by = x -> -Float64(x["overall"]["success_rate"]))
    for stats in sorted_stats
        overall = stats["overall"]
        println("Model: ", stats["model"], " (provider: ", stats["provider"], ")")
        println("  total: ", overall["total"])
        println("  success: ", overall["success"])
        println("  error: ", overall["error"])
        println("  success_rate: ", round(Float64(overall["success_rate"]); digits=4))
        println()
    end
end

function field_success_rate_matrix(aggregated_stats::Vector{OrderedDict{String,Any}})
    model_labels = [String(stats["model"]) for stats in aggregated_stats]
    field_order = [
        "natural_paraphrase",
        "paraphrase_gpt5.4-mini",
        "paraphrase_gemini-2.5-flash",
        "paraphrase_deepseek",
        "paraphrase_claude",
    ]
    present_fields = String[]

    for field in field_order
        for stats in aggregated_stats
            if haskey(stats["fieldwise"], field)
                push!(present_fields, field)
                break
            end
        end
    end

    matrix = fill(NaN, length(present_fields), length(aggregated_stats))

    for (j, stats) in enumerate(aggregated_stats)
        fieldwise = stats["fieldwise"]
        for (i, field) in enumerate(present_fields)
            if haskey(fieldwise, field)
                total = fieldwise[field]["total"]
                success = fieldwise[field]["success"]
                matrix[i, j] = total == 0 ? NaN : success / total
            end
        end
    end

    field_labels = [source_label_from_field(field) for field in present_fields]
    return model_labels, field_labels, matrix
end

function plot_llm_paraphrase_heatmap(
    aggregated_stats::Vector{OrderedDict{String,Any}};
    output_dir::String = DEFAULT_OUTPUT_DIR,
)
    ensure_directory(output_dir)
    model_labels, field_labels, matrix = field_success_rate_matrix(aggregated_stats)
    isempty(model_labels) && throw(ArgumentError("No result files available for heatmap plotting."))
    isempty(field_labels) && throw(ArgumentError("No paraphrase fields available for heatmap plotting."))

    output_path = joinpath(output_dir, "llm_vs_paraphrase_success_heatmap.png")

    p = heatmap(
        1:length(model_labels),
        1:length(field_labels),
        matrix;
        xticks=(1:length(model_labels), model_labels),
        yticks=(1:length(field_labels), field_labels),
        xrotation=20,
        xlabel="Translation model",
        ylabel="Paraphrase source",
        title="Success rate by translation model and paraphrase source",
        clims=(0.0, 1.0),
        colorbar_title="Success rate",
        aspect_ratio=:equal,
        size=(1600, 1000),
        bottom_margin=35Plots.mm,
        left_margin=22Plots.mm,
        right_margin=16Plots.mm,
        guidefontsize=18,
        tickfontsize=14,
        titlefontsize=20,
        colorbar_tickfontsize=12,
        colorbar_titlefontsize=14,
    )

    for i in 1:size(matrix, 1)
        for j in 1:size(matrix, 2)
            value = matrix[i, j]
            if !isnan(value)
                annotate!(p, j, i, text(string(round(value; digits=2)), 12, :white, :center))
            end
        end
    end

    savefig(p, output_path)
    display(p)
    println("Saved heatmap: ", output_path)
    return output_path
end

function print_summary(results_path::String, entries::Vector{OrderedDict{String,Any}})
    total = length(entries)
    ok_count = count(entry -> haskey(entry, "status") && String(entry["status"]) == "ok", entries)
    equivalent_count = count(entry -> haskey(entry, "equivalent") && entry["equivalent"] === true, entries)
    success_count = count(entry -> (haskey(entry, "status") && String(entry["status"]) == "ok") && (haskey(entry, "equivalent") && entry["equivalent"] === true), entries)
    error_count = total - ok_count

    println("Results file: ", results_path)
    println("Total evaluations: ", total)
    println("Successful translations (status = ok): ", ok_count)
    println("Semantically equivalent outputs: ", equivalent_count)
    println("Success count (ok AND equivalent): ", success_count)
    println("Error count: ", error_count)
    println("Success rate: ", round(total == 0 ? 0.0 : success_count / total; digits=4))
    println()

    field_stats = fieldwise_statistics(entries)
    println("Field-wise statistics:")
    for (field, stats) in field_stats
        rate = stats["total"] == 0 ? 0.0 : stats["success"] / stats["total"]
        println("  Field: ", field)
        println("    total: ", stats["total"])
        println("    ok: ", stats["ok"])
        println("    equivalent: ", stats["equivalent"])
        println("    success: ", stats["success"])
        println("    error: ", stats["error"])
        println("    success_rate: ", round(rate; digits=4))
    end
    println()
end

function sanitize_basename(path::String)
    base = splitext(basename(path))[1]
    return replace(base, r"[^A-Za-z0-9._-]" => "_")
end

function plot_success_rate_vs_formula_size(
    entries::Vector{OrderedDict{String,Any}},
    dataset_index::Dict{Int,OrderedDict{String,Any}};
    results_path::String,
    output_dir::String = DEFAULT_OUTPUT_DIR,
)
    sizes, rates, totals = size_bucket_success_rates(entries, dataset_index)
    isempty(sizes) && throw(ArgumentError("No size statistics could be computed for $(results_path)."))

    ensure_directory(output_dir)
    output_path = joinpath(output_dir, sanitize_basename(results_path) * "_success_rate_vs_formula_size.png")

    p = plot(
        sizes,
        rates;
        seriestype=:bar,
        xlabel="LTL formula size (ast_size)",
        ylabel="Success rate",
        title="Success rate vs formula size: " * splitext(basename(results_path))[1],
        label="success rate",
        legend=:topright,
        ylim=(0.0, 1.0),
    )

    annotations = [(sizes[i], min(rates[i] + 0.03, 0.98), text(string(totals[i]), 8, :center)) for i in eachindex(sizes)]
    annotate!(p, annotations)

    savefig(p, output_path)
    println("Saved figure: ", output_path)
    return output_path
end


function compare_all_models_and_plot_heatmap(
    results_dir::String = DEFAULT_RESULTS_DIR;
    output_dir::String = DEFAULT_OUTPUT_DIR,
)
    aggregated_stats = aggregate_all_result_files(results_dir)
    print_model_comparison_table(aggregated_stats)
    heatmap_path = plot_llm_paraphrase_heatmap(aggregated_stats; output_dir=output_dir)
    println("Heatmap written to: ", heatmap_path)
    return nothing
end

function study_result_file(
    results_path::String;
    dataset_path::String = DEFAULT_DATASET_PATH,
    output_dir::String = DEFAULT_OUTPUT_DIR,
)
    dataset_index = load_dataset_index(dataset_path)
    _, entries = load_result_entries(results_path)
    print_summary(results_path, entries)
    plot_success_rate_vs_formula_size(entries, dataset_index; results_path=results_path, output_dir=output_dir)
end

function study_all_result_files(
    results_dir::String = DEFAULT_RESULTS_DIR;
    dataset_path::String = DEFAULT_DATASET_PATH,
    output_dir::String = DEFAULT_OUTPUT_DIR,
)
    files = available_result_files(results_dir)
    for path in files
        println("================================================================================")
        study_result_file(path; dataset_path=dataset_path, output_dir=output_dir)
    end
end

function main()
    println("Loaded ResultStatistics.jl")
    println("Run `compare_all_models_and_plot_heatmap()` to compare overall success rates and plot the heatmap.")
    println("Run `study_result_file(joinpath(@__DIR__, \"results\", \"GPT54.json\"))` for one detailed file report.")
    println("Run `study_all_result_files()` to analyze all JSON files in the results folder one by one.")
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end


