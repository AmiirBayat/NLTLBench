include("GenerateLTL.jl")
include("Filter.jl")

using JSON3
using OrderedCollections
using Plots

const DEFAULT_DATASET_PATH = joinpath(@__DIR__, "dataset", "DatasetWithNaturalNL_plus_simplified.json")
const DEFAULT_FIELDS = ["LTL", "simplified_formula"]
const DEFAULT_SIZE_HISTOGRAM_PATH = joinpath(@__DIR__, "dataset", "formula_size_histogram.png")
const DEFAULT_AUTOMATON_SIZE_HISTOGRAM_PATH = joinpath(@__DIR__, "dataset", "automaton_size_histogram.png")
const DEFAULT_MIN_EQUIV_SIZE_HISTOGRAM_PATH = joinpath(@__DIR__, "dataset", "min_equiv_formula_size_histogram.png")
function plot_automaton_size_distribution(
    records;
    output_path::String = DEFAULT_AUTOMATON_SIZE_HISTOGRAM_PATH,
)
    automaton_sizes = Int[]

    for record in records
        if haskey(record, "automaton_size")
            push!(automaton_sizes, Int(record["automaton_size"]))
        elseif haskey(record, Symbol("automaton_size"))
            push!(automaton_sizes, Int(record[Symbol("automaton_size")]))
        end
    end

    isempty(automaton_sizes) && throw(ArgumentError("No `automaton_size` values are available for plotting."))

    max_size = 100 #maximum(automaton_sizes)
    bins = 0.5:1.0:(max_size + 0.5)

    p = histogram(
        automaton_sizes;
        bins=bins,
        normalize=false,
        alpha=0.9,
        label="LTL formulas",
        xlabel="Automaton size",
        ylabel="Number of LTL formulas",
        title="",
        legend=:topright,
        xlims=(0.5, 100.5),
        framestyle=:box,
        grid=true,
        size=(1100, 700),
        left_margin=12Plots.mm,
        right_margin=8Plots.mm,
        bottom_margin=10Plots.mm,
        top_margin=6Plots.mm,
        guidefontsize=16,
        tickfontsize=12,
        legendfontsize=11,
        linewidth=0.8,
    )
    hline!(p, [5]; linestyle=:dash, linewidth=2.0, label="Number of Büchi size = 5")
    savefig(p, output_path)
    println("Saved automaton-size distribution histogram to: ", output_path)
    return p
end


function load_dataset(dataset_path::String = DEFAULT_DATASET_PATH)
    if !isfile(dataset_path)
        throw(ArgumentError("Dataset file not found: $(dataset_path)"))
    end
    data = JSON3.read(read(dataset_path, String))
    return collect(data)
end

function get_ltlfilt_path()
    path = Sys.which("ltlfilt")
    isnothing(path) && throw(ArgumentError(
        "`ltlfilt` was not found in PATH. Please install Spot and make sure `ltlfilt` is available."
    ))
    return path
end

function formulas_from_field(records, field::String)
    formulas = String[]
    for record in records
        if !(haskey(record, field) || haskey(record, Symbol(field)))
            continue
        end
        value = haskey(record, field) ? record[field] : record[Symbol(field)]
        formula_str = strip(String(value))
        isempty(formula_str) && continue
        push!(formulas, formula_str)
    end
    return formulas
end

function ltlfilt_equivalent(formula1::AbstractString, formula2::AbstractString)::Bool
    ltlfilt_path = get_ltlfilt_path()
    cmd = `$(ltlfilt_path) -f $(String(formula1)) --equivalent-to $(String(formula2)) -q`
    process = run(ignorestatus(cmd))

    if process.exitcode == 0
        return true
    elseif process.exitcode == 1
        return false
    else
        throw(ArgumentError(
            "`ltlfilt` failed while checking equivalence between $(repr(String(formula1))) and $(repr(String(formula2))) (exit code $(process.exitcode))."
        ))
    end
end

function semantic_equivalence_groups(formulas::Vector{String})
    representatives = String[]
    groups = Vector{Vector{String}}()

    for formula in formulas
        placed = false
        for i in eachindex(representatives)
            if ltlfilt_equivalent(formula, representatives[i])
                push!(groups[i], formula)
                placed = true
                break
            end
        end

        if !placed
            push!(representatives, formula)
            push!(groups, [formula])
        end
    end

    return groups
end

function semantic_group_statistics(
    dataset_path::String = DEFAULT_DATASET_PATH;
    field::String = "LTL",
    unique_first::Bool = true,
    print_examples::Bool = true,
)
    records = load_dataset(dataset_path)
    formulas = formulas_from_field(records, field)
    unique_first && (formulas = unique(formulas))

    println("Dataset path: ", dataset_path)
    println("Field: ", field)
    println("Total formulas considered: ", length(formulas))

    groups = semantic_equivalence_groups(formulas)
    group_sizes = sort([length(g) for g in groups]; rev=true)

    println("Number of semantic-equivalence groups: ", length(groups))
    println("Largest group size: ", isempty(group_sizes) ? 0 : first(group_sizes))

    if print_examples
        println()
        println("Top 10 group sizes:")
        for (idx, size_value) in enumerate(group_sizes[1:min(10, length(group_sizes))])
            println("  Group ", idx, ": ", size_value)
        end
    end

    return OrderedDict(
        "dataset_path" => dataset_path,
        "field" => field,
        "total_formulas_considered" => length(formulas),
        "num_semantic_equivalence_groups" => length(groups),
        "group_sizes_descending" => group_sizes,
        "groups" => groups,
    )
end

function empty_operator_counter()
    return OrderedDict(
        "X" => 0,
        "F" => 0,
        "G" => 0,
        "U" => 0,
        "W" => 0,
        "R" => 0,
        "M" => 0,
        "!" => 0,
        "&" => 0,
        "|" => 0,
        "->" => 0,
        "<->" => 0,
    )
end


function empty_atomic_proposition_counter()
    return OrderedDict{String,Int}()
end

function empty_histogram()
    return OrderedDict{Int,Int}()
end

function count_formula_operators(formula_str::AbstractString)
    ast = parse_ltl_formula_string(String(formula_str))
    raw_counts = operator_counts(ast)
    counts = empty_operator_counter()

    for (op, value) in raw_counts
        key = string(op)
        if haskey(counts, key)
            counts[key] += value
        else
            counts[key] = value
        end
    end

    return counts
end

function count_atomic_propositions(formula_str::AbstractString)
    ast = parse_ltl_formula_string(String(formula_str))
    counts = empty_atomic_proposition_counter()

    function visit(node)
        if node isa AP
            name = String(node.name)
            if startswith(name, "prop_")
                counts[name] = get(counts, name, 0) + 1
            end
        elseif node isa UnaryLTL
            visit(node.child)
        elseif node isa BinaryLTL
            visit(node.left)
            visit(node.right)
        end
    end

    visit(ast)
    return counts
end

function merge_counts!(dest::OrderedDict{String,Int}, src::OrderedDict{String,Int})
    for (k, v) in src
        if !haskey(dest, k)
            dest[k] = 0
        end
        dest[k] += v
    end
    return dest
end

function increment_histogram!(hist::OrderedDict{Int,Int}, key::Int)
    hist[key] = get(hist, key, 0) + 1
    return hist
end

function merge_histograms!(dest::OrderedDict{Int,Int}, src::OrderedDict{Int,Int})
    for (k, v) in src
        dest[k] = get(dest, k, 0) + v
    end
    return dest
end

function formula_structure_statistics(formula_str::AbstractString)
    ast = parse_ltl_formula_string(String(formula_str))
    return OrderedDict(
        "ast_size" => ast_size(ast),
        "ast_depth" => ast_depth(ast),
        "temporal_depth" => temporal_depth(ast),
    )
end

function plot_min_equiv_size_distribution(
    records;
    output_path::String = DEFAULT_MIN_EQUIV_SIZE_HISTOGRAM_PATH,
)
    size_by_id = Dict{Int,Int}()

    for record in records
        if !(haskey(record, "id") || haskey(record, Symbol("id")))
            continue
        end
        record_id = Int(haskey(record, "id") ? record["id"] : record[Symbol("id")])

        if !(haskey(record, "LTL") || haskey(record, Symbol("LTL")))
            continue
        end
        formula_value = haskey(record, "LTL") ? record["LTL"] : record[Symbol("LTL")]
        formula_str = strip(String(formula_value))
        isempty(formula_str) && continue

        try
            structure_stats = formula_structure_statistics(formula_str)
            size_by_id[record_id] = structure_stats["ast_size"]
        catch err
            println("Warning: failed to parse LTL for record ID $(record_id) while building minimum-equivalent-size histogram: ", err)
        end
    end

    min_sizes = Int[]
    seen_source_ids = Set{Int}()

    for record in records
        if !(haskey(record, "id") || haskey(record, Symbol("id")))
            continue
        end
        record_id = Int(haskey(record, "id") ? record["id"] : record[Symbol("id")])
        haskey(size_by_id, record_id) || continue

        if haskey(record, "source_record_id") || haskey(record, Symbol("source_record_id"))
            source_value = haskey(record, "source_record_id") ? record["source_record_id"] : record[Symbol("source_record_id")]
            source_id = try
                Int(source_value)
            catch
                nothing
            end
            isnothing(source_id) && continue
            source_id in seen_source_ids && continue

            current_size = size_by_id[record_id]
            source_size = get(size_by_id, source_id, current_size)
            push!(min_sizes, min(current_size, source_size))
            push!(seen_source_ids, source_id)
        else
            push!(min_sizes, size_by_id[record_id])
        end
    end

    isempty(min_sizes) && throw(ArgumentError("No minimum-equivalent LTL sizes are available for plotting."))

    max_size = min(maximum(min_sizes), 35)
    bins = 0.5:1.0:(max_size + 0.5)

    p = histogram(
        min_sizes;
        bins=bins,
        normalize=false,
        alpha=0.9,
        label="LTL formulas",
        xlabel="Minimum AST size among equivalent LTLs in benchmark",
        ylabel="Number of LTL formulas",
        title="",
        legend=:topright,
        xlims=(0.5, 35.5),
        framestyle=:box,
        grid=true,
        size=(1100, 700),
        left_margin=12Plots.mm,
        right_margin=8Plots.mm,
        bottom_margin=10Plots.mm,
        top_margin=6Plots.mm,
        guidefontsize=16,
        tickfontsize=12,
        legendfontsize=11,
        linewidth=0.8,
    )

    hline!(p, [5]; linestyle=:dash, linewidth=2.0, label="Number of LTL = 5")
    savefig(p, output_path)
    println("Saved minimum-equivalent-size distribution histogram to: ", output_path)
    return p
end

function field_statistics(records, field::String)
    total_counts = empty_operator_counter()
    atomic_prop_counts = empty_atomic_proposition_counter()
    present_count = 0
    parse_failures = 0

    ast_size_hist = empty_histogram()
    ast_depth_hist = empty_histogram()
    temporal_depth_hist = empty_histogram()

    ast_size_values = Int[]
    ast_depth_values = Int[]
    temporal_depth_values = Int[]

    for record in records
        if !(haskey(record, field) || haskey(record, Symbol(field)))
            continue
        end

        value = haskey(record, field) ? record[field] : record[Symbol(field)]
        formula_str = strip(String(value))
        isempty(formula_str) && continue

        present_count += 1

        try
            merge_counts!(total_counts, count_formula_operators(formula_str))
            merge_counts!(atomic_prop_counts, count_atomic_propositions(formula_str))
            structure_stats = formula_structure_statistics(formula_str)
            increment_histogram!(ast_size_hist, structure_stats["ast_size"])
            increment_histogram!(ast_depth_hist, structure_stats["ast_depth"])
            increment_histogram!(temporal_depth_hist, structure_stats["temporal_depth"])
            push!(ast_size_values, structure_stats["ast_size"])
            push!(ast_depth_values, structure_stats["ast_depth"])
            push!(temporal_depth_values, structure_stats["temporal_depth"])
        catch err
            parse_failures += 1
            println("Warning: failed to parse field `$(field)` for record ID $(get(record, "id", "?")): ", err)
        end
    end

    ast_size_min = isempty(ast_size_values) ? nothing : minimum(ast_size_values)
    ast_size_max = isempty(ast_size_values) ? nothing : maximum(ast_size_values)
    ast_depth_min = isempty(ast_depth_values) ? nothing : minimum(ast_depth_values)
    ast_depth_max = isempty(ast_depth_values) ? nothing : maximum(ast_depth_values)
    temporal_depth_min = isempty(temporal_depth_values) ? nothing : minimum(temporal_depth_values)
    temporal_depth_max = isempty(temporal_depth_values) ? nothing : maximum(temporal_depth_values)

    return OrderedDict(
        "field" => field,
        "records_with_field" => present_count,
        "parse_failures" => parse_failures,
        "operator_counts" => total_counts,
        "atomic_proposition_counts" => atomic_prop_counts,
        "ast_size_distribution" => ast_size_hist,
        "ast_size_min" => ast_size_min,
        "ast_size_max" => ast_size_max,
        "ast_depth_distribution" => ast_depth_hist,
        "ast_depth_min" => ast_depth_min,
        "ast_depth_max" => ast_depth_max,
        "temporal_depth_distribution" => temporal_depth_hist,
        "temporal_depth_min" => temporal_depth_min,
        "temporal_depth_max" => temporal_depth_max,
    )
end

function print_field_statistics(stats::OrderedDict{String,Any})
    println("Field: ", stats["field"])
    println("  Records with field: ", stats["records_with_field"])
    println("  Parse failures: ", stats["parse_failures"])

    ast_size_min = stats["ast_size_min"]
    ast_size_max = stats["ast_size_max"]
    ast_depth_min = stats["ast_depth_min"]
    ast_depth_max = stats["ast_depth_max"]
    temporal_depth_min = stats["temporal_depth_min"]
    temporal_depth_max = stats["temporal_depth_max"]

    println("  AST size range: ", isnothing(ast_size_min) ? "n/a" : string(ast_size_min), " to ", isnothing(ast_size_max) ? "n/a" : string(ast_size_max))
    println("  AST depth range: ", isnothing(ast_depth_min) ? "n/a" : string(ast_depth_min), " to ", isnothing(ast_depth_max) ? "n/a" : string(ast_depth_max))
    println("  Temporal depth range: ", isnothing(temporal_depth_min) ? "n/a" : string(temporal_depth_min), " to ", isnothing(temporal_depth_max) ? "n/a" : string(temporal_depth_max))
    println("  Maximum nesting level: ", isnothing(ast_depth_max) ? "n/a" : string(ast_depth_max))
    println("  Maximum temporal nesting level: ", isnothing(temporal_depth_max) ? "n/a" : string(temporal_depth_max))

    println("  Operator counts:")

    counts = stats["operator_counts"]
    op_hist = OrderedDict{Int,Int}()
    op_labels = String[]
    idx = 1
    for (op, value) in counts
        push!(op_labels, op)
        op_hist[idx] = value
        idx += 1
    end

    if isempty(op_hist)
        println("    (empty)")
    else
        max_count = maximum(values(op_hist))
        for i in 1:length(op_labels)
            value = op_hist[i]
            bar_len = max_count == 0 ? 0 : max(1, round(Int, 40 * value / max_count))
            bar = repeat("█", bar_len)
            println("    ", rpad(op_labels[i], 4), "| ", rpad(bar, 40), " ", value)
        end
    end

    println()

    println("  Atomic proposition counts:")
    ap_counts = stats["atomic_proposition_counts"]

    if isempty(ap_counts)
        println("    (empty)")
    else
        sorted_ap_keys = sort(collect(keys(ap_counts)))
        max_ap_count = maximum(values(ap_counts))
        for ap in sorted_ap_keys
            value = ap_counts[ap]
            bar_len = max_ap_count == 0 ? 0 : max(1, round(Int, 40 * value / max_ap_count))
            bar = repeat("█", bar_len)
            println("    ", rpad(ap, 8), "| ", rpad(bar, 40), " ", value)
        end
    end

    println()
    println("  LTL AST size distribution is saved in the histogram figure when `LTL` statistics are available.")
    println("  Returned statistics also include the full AST size, AST depth, and temporal depth histograms.")
    println()
end

function histogram_to_vectors(hist::OrderedDict{Int,Int})
    values_vec = Int[]
    for key in sort(collect(keys(hist)))
        append!(values_vec, fill(key, hist[key]))
    end
    return values_vec
end

function plot_size_distributions(
    stats_list::Vector{<:OrderedDict};
    output_path::String = DEFAULT_SIZE_HISTOGRAM_PATH,
)
    ltl_stats = nothing

    for stats in stats_list
        if stats["field"] == "LTL"
            ltl_stats = stats
            break
        end
    end

    if isnothing(ltl_stats)
        throw(ArgumentError("`LTL` statistics are required to plot the size distribution."))
    end

    ltl_sizes = histogram_to_vectors(ltl_stats["ast_size_distribution"])
    isempty(ltl_sizes) && throw(ArgumentError("No LTL sizes available for plotting."))

    max_size = maximum(ltl_sizes)
    bins = 0.5:1.0:(max_size + 0.5)

    p = histogram(
        ltl_sizes;
        bins=bins,
        normalize=false,
        alpha=0.9,
        label="LTL formulas",
        xlabel="LTL formula size",
        ylabel="Number of LTL formulas",
        title="",
        legend=:topright,
        framestyle=:box,
        grid=true,
        size=(1100, 700),
        left_margin=12Plots.mm,
        right_margin=8Plots.mm,
        bottom_margin=10Plots.mm,
        top_margin=6Plots.mm,
        guidefontsize=16,
        tickfontsize=12,
        legendfontsize=11,
        linewidth=0.8,
    )

    hline!(p, [5]; linestyle=:dash, linewidth=2.0, label="Number of LTL = 5")
    savefig(p, output_path)
    println("Saved size distribution histogram to: ", output_path)
    return p
end

function dataset_statistics(
    dataset_path::String = DEFAULT_DATASET_PATH;
    fields::Vector{String} = DEFAULT_FIELDS,
)
    records = load_dataset(dataset_path)

    println("Dataset path: ", dataset_path)
    println("Total records: ", length(records))
    println()

    all_stats = OrderedDict[]
    for field in fields
        stats = field_statistics(records, field)
        push!(all_stats, stats)
        print_field_statistics(stats)
    end

    plot_size_distributions(all_stats)
    try
        plot_automaton_size_distribution(records)
    catch err
        println("Warning: could not plot automaton-size distribution: ", err)
    end
    try
        plot_min_equiv_size_distribution(records)
    catch err
        println("Warning: could not plot minimum-equivalent-size distribution: ", err)
    end
    return all_stats
end

function main()
    dataset_statistics()
    println("Run `semantic_group_statistics(field=\"LTL\")` to count semantic-equivalence groups using Spot.")
    println("Run `plot_min_equiv_size_distribution(load_dataset())` to plot the dataset distribution over minimum AST size, using the smaller size when a record has `source_record_id`.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
