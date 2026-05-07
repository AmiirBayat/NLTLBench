

using JSON3
using OrderedCollections
using Plots

const DEFAULT_DATASET_PATH = joinpath(@__DIR__, "dataset", "DatasetWithNaturalNL_plus_simplified.json")
const DEFAULT_RESULTS_DIR = joinpath(@__DIR__, "results")

const DEFAULT_OUTPUT_DIR = joinpath(@__DIR__, "results")

const USE_PAPER_MODEL_SUBSET = true
const PAPER_MODEL_SUBSET = Set([
    "mistral-medium-latest",
    "claude-opus-4-7",
    "gpt-5.4",
    "gpt-5.5",
])

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


function tokenize_formula_size_fallback(formula::AbstractString)
    s = strip(String(formula))
    isempty(s) && return String[]

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

function get_record_formula_size(record::OrderedDict{String,Any})
    if haskey(record, "formula_size")
        return Int(record["formula_size"])
    elseif haskey(record, "ast_size")
        return Int(record["ast_size"])
    elseif haskey(record, "LTL")
        tokens = tokenize_formula_size_fallback(String(record["LTL"]))
        return count(tok -> tok != "(" && tok != ")", tokens)
    else
        throw(ArgumentError("Dataset record ID $(get(record, "id", "?")) does not contain `formula_size`, `ast_size`, or `LTL`."))
    end
end

function get_record_temporal_depth(record::OrderedDict{String,Any})
    if haskey(record, "temporal_depth")
        return Int(record["temporal_depth"])
    elseif haskey(record, "LTL")
        structure = formula_structure_statistics(String(record["LTL"]))
        return Int(structure["temporal_depth"])
    else
        throw(ArgumentError("Dataset record ID $(get(record, "id", "?")) does not contain `temporal_depth` or `LTL`."))
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
    model_name = if haskey(results_obj, "translation_model")
        String(results_obj["translation_model"])
    else
        splitext(basename(results_path))[1]
    end

    lowercase_model = lowercase(model_name)
    if occursin("deepseek", lowercase_model)
        return "deepseek-v4-flash"
    end

    return model_name
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
        return "deepseek-v4-flash"
    elseif field == "paraphrase_claude"
        return "claude"
    else
        return field
    end
end

function ignore_errors_in_success_rate(results_obj::OrderedDict{String,Any}, results_path::String)
    model_name = result_model_name(results_obj, results_path)
    return model_name == "finetuned-t5" || splitext(basename(results_path))[1] == "NL2TL"
end

function success_rate(entries::Vector{OrderedDict{String,Any}}; ignore_errors::Bool = false)
    success_count = 0
    denominator = 0

    for entry in entries
        status_ok = haskey(entry, "status") && String(entry["status"]) == "ok"
        equivalent_true = haskey(entry, "equivalent") && entry["equivalent"] === true
        success_count += (status_ok && equivalent_true) ? 1 : 0
        if ignore_errors
            denominator += status_ok ? 1 : 0
        else
            denominator += 1
        end
    end

    denominator == 0 && return 0.0
    return success_count / denominator
end

function fieldwise_statistics(entries::Vector{OrderedDict{String,Any}}; ignore_errors::Bool = false)
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
                "effective_total" => 0,
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

        if ignore_errors
            stats[field]["effective_total"] += status_ok ? 1 : 0
        else
            stats[field]["effective_total"] += 1
        end
    end

    return stats
end

function size_bucket_statistics(entries::Vector{OrderedDict{String,Any}}, dataset_index::Dict{Int,OrderedDict{String,Any}}; ignore_errors::Bool = false)
    bucket_counts = OrderedDict{Int,OrderedDict{String,Int}}()

    for entry in entries
        haskey(entry, "record_id") || continue
        record_id = Int(entry["record_id"])
        haskey(dataset_index, record_id) || continue
        size_value = get_record_formula_size(dataset_index[record_id])

        if !haskey(bucket_counts, size_value)
            bucket_counts[size_value] = OrderedDict(
                "total" => 0,
                "success" => 0,
            )
        end

        status_ok = haskey(entry, "status") && String(entry["status"]) == "ok"
        equivalent_true = haskey(entry, "equivalent") && entry["equivalent"] === true

        if ignore_errors
            bucket_counts[size_value]["total"] += status_ok ? 1 : 0
        else
            bucket_counts[size_value]["total"] += 1
        end

        if status_ok && equivalent_true
            bucket_counts[size_value]["success"] += 1
        end
    end

    return bucket_counts
end

function temporal_depth_bucket_statistics(entries::Vector{OrderedDict{String,Any}}, dataset_index::Dict{Int,OrderedDict{String,Any}}; ignore_errors::Bool = false)
    bucket_counts = OrderedDict{Int,OrderedDict{String,Int}}()

    for entry in entries
        haskey(entry, "record_id") || continue
        record_id = Int(entry["record_id"])
        haskey(dataset_index, record_id) || continue
        depth_value = get_record_temporal_depth(dataset_index[record_id])

        if !haskey(bucket_counts, depth_value)
            bucket_counts[depth_value] = OrderedDict(
                "total" => 0,
                "success" => 0,
            )
        end

        status_ok = haskey(entry, "status") && String(entry["status"]) == "ok"
        equivalent_true = haskey(entry, "equivalent") && entry["equivalent"] === true

        if ignore_errors
            bucket_counts[depth_value]["total"] += status_ok ? 1 : 0
        else
            bucket_counts[depth_value]["total"] += 1
        end

        if status_ok && equivalent_true
            bucket_counts[depth_value]["success"] += 1
        end
    end

    return bucket_counts
end

function temporal_depth_bucket_success_rates(entries::Vector{OrderedDict{String,Any}}, dataset_index::Dict{Int,OrderedDict{String,Any}}; ignore_errors::Bool = false)
    bucket_stats = temporal_depth_bucket_statistics(entries, dataset_index; ignore_errors=ignore_errors)
    depths = sort(collect(keys(bucket_stats)))
    rates = Float64[]
    totals = Int[]

    for depth_value in depths
        total = bucket_stats[depth_value]["total"]
        success = bucket_stats[depth_value]["success"]
        push!(totals, total)
        push!(rates, total == 0 ? 0.0 : success / total)
    end

    return depths, rates, totals
end


function size_bucket_success_rates(entries::Vector{OrderedDict{String,Any}}, dataset_index::Dict{Int,OrderedDict{String,Any}}; ignore_errors::Bool = false)
    bucket_stats = size_bucket_statistics(entries, dataset_index; ignore_errors=ignore_errors)
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

function overall_success_statistics(entries::Vector{OrderedDict{String,Any}}; ignore_errors::Bool = false)
    total = length(entries)
    success_count = count(entry -> (haskey(entry, "status") && String(entry["status"]) == "ok") && (haskey(entry, "equivalent") && entry["equivalent"] === true), entries)
    ok_count = count(entry -> haskey(entry, "status") && String(entry["status"]) == "ok", entries)
    equivalent_count = count(entry -> haskey(entry, "equivalent") && entry["equivalent"] === true, entries)
    error_count = total - ok_count
    effective_total = ignore_errors ? ok_count : total
    success_rate_value = effective_total == 0 ? 0.0 : success_count / effective_total

    return OrderedDict(
        "total" => total,
        "effective_total" => effective_total,
        "ok" => ok_count,
        "equivalent" => equivalent_count,
        "success" => success_count,
        "error" => error_count,
        "success_rate" => success_rate_value,
    )
end

function aggregate_result_file_statistics(results_path::String)
    results_obj, entries = load_result_entries(results_path)
    ignore_errors = ignore_errors_in_success_rate(results_obj, results_path)
    return OrderedDict(
        "results_path" => results_path,
        "provider" => result_provider_name(results_obj),
        "model" => result_model_name(results_obj, results_path),
        "ignore_errors_in_success_rate" => ignore_errors,
        "entries" => entries,
        "overall" => overall_success_statistics(entries; ignore_errors=ignore_errors),
        "fieldwise" => fieldwise_statistics(entries; ignore_errors=ignore_errors),
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


function maybe_filter_to_paper_models(aggregated_stats::Vector{OrderedDict{String,Any}}; enabled::Bool = USE_PAPER_MODEL_SUBSET)
    if !enabled
        return aggregated_stats
    end

    filtered = OrderedDict{String,Any}[]
    for stats in aggregated_stats
        model_name = String(stats["model"])
        if model_name in PAPER_MODEL_SUBSET
            push!(filtered, stats)
        end
    end

    isempty(filtered) && throw(ArgumentError("Paper-model filter is enabled, but none of the result files matched the selected models."))
    return filtered
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
        println("  effective_total: ", overall["effective_total"])
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
        xrotation=25,
        xlabel="Translation model",
        ylabel="Paraphrase source",
        title="Success rate by translation model and paraphrase source",
        clims=(0.0, 1.0),
        colorbar_title="Success rate",
        aspect_ratio=:equal,
        size=(1900, 1200),
        bottom_margin=55Plots.mm,
        left_margin=35Plots.mm,
        right_margin=28Plots.mm,
        top_margin=20Plots.mm,
        guidefontsize=20,
        tickfontsize=15,
        titlefontsize=22,
        colorbar_tickfontsize=13,
        colorbar_titlefontsize=15,
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
    results_obj = load_json_object(results_path)
    ignore_errors = ignore_errors_in_success_rate(results_obj, results_path)
    overall = overall_success_statistics(entries; ignore_errors=ignore_errors)

    println("Results file: ", results_path)
    println("Total evaluations: ", overall["total"])
    println("Effective total for success rate: ", overall["effective_total"])
    println("Successful translations (status = ok): ", overall["ok"])
    println("Semantically equivalent outputs: ", overall["equivalent"])
    println("Success count (ok AND equivalent): ", overall["success"])
    println("Error count: ", overall["error"])
    println("Success rate: ", round(Float64(overall["success_rate"]); digits=4))
    println()

    field_stats = fieldwise_statistics(entries; ignore_errors=ignore_errors)
    println("Field-wise statistics:")
    for (field, stats) in field_stats
        denom = stats["effective_total"]
        rate = denom == 0 ? 0.0 : stats["success"] / denom
        println("  Field: ", field)
        println("    total: ", stats["total"])
        println("    effective_total: ", stats["effective_total"])
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
    results_obj = load_json_object(results_path)
    ignore_errors = ignore_errors_in_success_rate(results_obj, results_path)
    sizes, rates, totals = size_bucket_success_rates(entries, dataset_index; ignore_errors=ignore_errors)
    isempty(sizes) && throw(ArgumentError("No size statistics could be computed for $(results_path)."))

    ensure_directory(output_dir)
    output_path = joinpath(output_dir, sanitize_basename(results_path) * "_success_rate_vs_formula_size.png")

    p = plot(
        sizes,
        rates;
        seriestype=:scatter,
        xlabel="LTL formula size",
        ylabel="Success rate",
        title="",
        label="Success rate",
        legend=:topright,
        ylim=(0.0, 1.0),
        xlims=(minimum(sizes) - 0.5, maximum(sizes) + 0.5),
        marker=:circle,
        markersize=6,
        markerstrokewidth=0,
        linewidth=2.5,
        linecolor=:black,
        framestyle=:box,
        grid=true,
        size=(1200, 750),
        left_margin=14Plots.mm,
        right_margin=10Plots.mm,
        bottom_margin=12Plots.mm,
        top_margin=6Plots.mm,
        guidefontsize=16,
        tickfontsize=12,
        legendfontsize=11,
    )
    plot!(p, sizes, rates; label="", linewidth=2.5, linecolor=:black)

    annotations = [(sizes[i], min(rates[i] + 0.04, 0.98), text(string(totals[i]), 9, :center)) for i in eachindex(sizes)]
    annotate!(p, annotations)

    savefig(p, output_path)
    println("Saved figure: ", output_path)
    return output_path
end

function plot_success_rate_vs_temporal_depth(
    entries::Vector{OrderedDict{String,Any}},
    dataset_index::Dict{Int,OrderedDict{String,Any}};
    results_path::String,
    output_dir::String = DEFAULT_OUTPUT_DIR,
)
    results_obj = load_json_object(results_path)
    ignore_errors = ignore_errors_in_success_rate(results_obj, results_path)
    depths, rates, totals = temporal_depth_bucket_success_rates(entries, dataset_index; ignore_errors=ignore_errors)
    isempty(depths) && throw(ArgumentError("No temporal-depth statistics could be computed for $(results_path)."))

    ensure_directory(output_dir)
    output_path = joinpath(output_dir, sanitize_basename(results_path) * "_success_rate_vs_temporal_depth.png")

    p = plot(
        depths,
        rates;
        seriestype=:scatter,
        xlabel="Temporal depth",
        ylabel="Success rate",
        title="",
        label="Success rate",
        legend=:topright,
        ylim=(0.0, 1.0),
        xlims=(minimum(depths) - 0.5, maximum(depths) + 0.5),
        marker=:circle,
        markersize=6,
        markerstrokewidth=0,
        linewidth=2.5,
        linecolor=:black,
        framestyle=:box,
        grid=true,
        size=(1200, 750),
        left_margin=14Plots.mm,
        right_margin=10Plots.mm,
        bottom_margin=12Plots.mm,
        top_margin=6Plots.mm,
        guidefontsize=16,
        tickfontsize=12,
        legendfontsize=11,
    )
    plot!(p, depths, rates; label="", linewidth=2.5, linecolor=:black)

    annotations = [(depths[i], min(rates[i] + 0.04, 0.98), text(string(totals[i]), 9, :center)) for i in eachindex(depths)]
    annotate!(p, annotations)

    savefig(p, output_path)
    println("Saved figure: ", output_path)
    return output_path
end



function plot_all_models_success_rate_vs_formula_size(
    results_dir::String = DEFAULT_RESULTS_DIR;
    dataset_path::String = DEFAULT_DATASET_PATH,
    output_dir::String = DEFAULT_OUTPUT_DIR,
)
    ensure_directory(output_dir)
    dataset_index = load_dataset_index(dataset_path)
    aggregated_stats = maybe_filter_to_paper_models(aggregate_all_result_files(results_dir))

    series_data = Vector{Tuple{String,Vector{Int},Vector{Float64},Vector{Int},Float64}}()
    all_sizes = Int[]

    for stats in aggregated_stats
        model_name = String(stats["model"])
        entries = Vector{OrderedDict{String,Any}}(stats["entries"])
        ignore_errors = haskey(stats, "ignore_errors_in_success_rate") ? Bool(stats["ignore_errors_in_success_rate"]) : false
        sizes, rates, totals = size_bucket_success_rates(entries, dataset_index; ignore_errors=ignore_errors)
        overall_rate = haskey(stats, "overall") && haskey(stats["overall"], "success_rate") ? Float64(stats["overall"]["success_rate"]) : 0.0
        push!(series_data, (model_name, sizes, rates, totals, overall_rate))
        append!(all_sizes, sizes)
    end

    isempty(all_sizes) && throw(ArgumentError("No size statistics could be computed from the available result files."))

    sorted_series = sort(series_data; by = item -> -item[5])
    x_values = sort(unique(all_sizes))
    xtick_step = maximum(x_values) <= 20 ? 1 : (maximum(x_values) <= 40 ? 2 : 5)
    xtick_values = collect(minimum(x_values):xtick_step:maximum(x_values))
    marker_shapes = [:circle, :rect, :diamond, :utriangle, :dtriangle, :star5, :hexagon, :xcross]
    line_styles = [:solid, :dash, :dot, :dashdot, :dashdotdot]

    if USE_PAPER_MODEL_SUBSET
        p = plot(
            xlabel="LTL formula size",
            ylabel="Success rate",
            title="",
            legend=:outerright,
            ylim=(0.0, 1.0),
            xlims=(minimum(x_values) - 0.5, maximum(x_values) + 0.5),
            xticks=xtick_values,
            framestyle=:box,
            grid=true,
            minorgrid=false,
            size=(1600, 850),
            left_margin=16Plots.mm,
            right_margin=26Plots.mm,
            bottom_margin=12Plots.mm,
            top_margin=6Plots.mm,
            guidefontsize=18,
            tickfontsize=13,
            legendfontsize=11,
        )

        for (idx, (model_name, sizes, rates, totals, overall_rate)) in enumerate(sorted_series)
            marker_shape = marker_shapes[mod1(idx, length(marker_shapes))]
            line_style = line_styles[mod1(idx, length(line_styles))]
            plot!(
                p,
                sizes,
                rates;
                seriestype=:scatter,
                marker=marker_shape,
                markersize=6,
                markerstrokewidth=0.5,
                linewidth=2.6,
                linestyle=line_style,
                label=model_name,
            )
            plot!(
                p,
                sizes,
                rates;
                label="",
                linewidth=2.6,
                linestyle=line_style,
            )
        end
    else
        n_models = length(sorted_series)
        ncols = 2
        nrows = ceil(Int, n_models / ncols)

        plots_list = Plots.Plot[]

        for (idx, (model_name, sizes, rates, totals, overall_rate)) in enumerate(sorted_series)
            marker_shape = marker_shapes[mod1(idx, length(marker_shapes))]

            local_p = plot(
                sizes,
                rates;
                seriestype=:scatter,
                xlabel=(idx > n_models - ncols ? "LTL formula size" : ""),
                ylabel=(mod1(idx, ncols) == 1 ? "Success rate" : ""),
                title=model_name,
                label="",
                ylim=(0.0, 1.0),
                xlims=(minimum(x_values) - 0.5, maximum(x_values) + 0.5),
                xticks=xtick_values,
                marker=marker_shape,
                markersize=5,
                markerstrokewidth=0.4,
                linewidth=2.3,
                linecolor=:black,
                framestyle=:box,
                grid=true,
                guidefontsize=13,
                tickfontsize=10,
                titlefontsize=13,
                left_margin=8Plots.mm,
                right_margin=4Plots.mm,
                bottom_margin=6Plots.mm,
                top_margin=4Plots.mm,
            )
            plot!(local_p, sizes, rates; label="", linewidth=2.3, linecolor=:black)
            push!(plots_list, local_p)
        end

        p = plot(
            plots_list...;
            layout=(nrows, ncols),
            size=(1600, 420 * nrows),
        )
    end

    output_path = joinpath(output_dir, "all_models_success_rate_vs_formula_size.png")
    savefig(p, output_path)
    display(p)
    println("Saved combined success-rate figure: ", output_path)
    return output_path
end

function plot_all_models_success_rate_vs_temporal_depth(
    results_dir::String = DEFAULT_RESULTS_DIR;
    dataset_path::String = DEFAULT_DATASET_PATH,
    output_dir::String = DEFAULT_OUTPUT_DIR,
)
    ensure_directory(output_dir)
    dataset_index = load_dataset_index(dataset_path)
    aggregated_stats = maybe_filter_to_paper_models(aggregate_all_result_files(results_dir))

    series_data = OrderedDict{String,Tuple{Vector{Int},Vector{Float64},Vector{Int}}}()
    all_depths = Int[]

    for stats in aggregated_stats
        model_name = String(stats["model"])
        entries = Vector{OrderedDict{String,Any}}(stats["entries"])
        ignore_errors = haskey(stats, "ignore_errors_in_success_rate") ? Bool(stats["ignore_errors_in_success_rate"]) : false
        depths, rates, totals = temporal_depth_bucket_success_rates(entries, dataset_index; ignore_errors=ignore_errors)
        series_data[model_name] = (depths, rates, totals)
        append!(all_depths, depths)
    end

    isempty(all_depths) && throw(ArgumentError("No temporal-depth statistics could be computed from the available result files."))

    x_values = sort(unique(all_depths))
    p = plot(
        xlabel="Temporal depth",
        ylabel="Success rate",
        title="",
        legend=:topright,
        ylim=(0.0, 1.0),
        xlims=(minimum(x_values) - 0.5, maximum(x_values) + 0.5),
        framestyle=:box,
        grid=true,
        size=(1450, 850),
        left_margin=16Plots.mm,
        right_margin=12Plots.mm,
        bottom_margin=12Plots.mm,
        top_margin=6Plots.mm,
        guidefontsize=18,
        tickfontsize=13,
        legendfontsize=12,
    )

    for (model_name, (depths, rates, totals)) in series_data
        rate_map = Dict{Int,Float64}(depths[i] => rates[i] for i in eachindex(depths))
        y_values = [get(rate_map, x, NaN) for x in x_values]
        plot!(p, x_values, y_values; marker=:circle, markersize=5, markerstrokewidth=0, linewidth=2.5, label=model_name)
    end

    output_path = joinpath(output_dir, "all_models_success_rate_vs_temporal_depth.png")
    savefig(p, output_path)
    display(p)
    println("Saved combined temporal-depth figure: ", output_path)
    return output_path
end

function compare_all_models_and_plot_heatmap(
    results_dir::String = DEFAULT_RESULTS_DIR;
    output_dir::String = DEFAULT_OUTPUT_DIR,
)
    aggregated_stats = maybe_filter_to_paper_models(aggregate_all_result_files(results_dir))
    print_model_comparison_table(aggregated_stats)
    heatmap_path = plot_llm_paraphrase_heatmap(aggregated_stats; output_dir=output_dir)
    println("Heatmap written to: ", heatmap_path)
    size_plot_path = plot_all_models_success_rate_vs_formula_size(results_dir; output_dir=output_dir)
    println("Combined size plot written to: ", size_plot_path)
    temporal_depth_plot_path = plot_all_models_success_rate_vs_temporal_depth(results_dir; output_dir=output_dir)
    println("Combined temporal-depth plot written to: ", temporal_depth_plot_path)
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
    plot_success_rate_vs_temporal_depth(entries, dataset_index; results_path=results_path, output_dir=output_dir)
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
    println("Run `compare_all_models_and_plot_heatmap()` to compare overall success rates, plot the heatmap, and plot success rate vs both LTL size and temporal depth across models.")
    println("Run `study_result_file(joinpath(@__DIR__, \"results\", \"GPT54.json\"))` for one detailed file report.")
    println("Run `study_all_result_files()` to analyze all JSON files in the results folder one by one.")
    println("Set `USE_PAPER_MODEL_SUBSET = true` to plot only mistral-medium-latest, claude-opus-4-7, gpt-5.4, and gpt-5.5.")
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end


