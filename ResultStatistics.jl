

using JSON3
using OrderedCollections
using Plots
using Statistics
using LinearAlgebra
using SpecialFunctions: erfc

const DEFAULT_DATASET_PATH = joinpath(@__DIR__, "dataset", "DatasetWithNaturalNL_plus_simplified.json")
const DEFAULT_RESULTS_DIR = joinpath(@__DIR__, "results")

const DEFAULT_OUTPUT_DIR = joinpath(@__DIR__, "results")

const USE_PAPER_MODEL_SUBSET = false

const PAPER_MODEL_SUBSET = Set([
    "mistral-medium-latest",
    "claude-opus-4-7",
    "gpt-5.4",
    "gpt-5.5",
])

const USE_ZERO_VS_FEWSHOT_SUBPLOTS = true
const ZERO_VS_FEWSHOT_MODELS = [
    "gpt-5.4",
    "gpt-5.5",
    "claude-opus-4-7",
    "deepseek-v4-flash",
    "mistral-medium-latest",
]
const ZERO_VS_FEWSHOT_FEWSHOT_FILES = Set([
    "Claude_fewshot",
    "GPT55_fewshot",
    "GPT54_fewshot",
    "DeepSeek_fewshot",
    "Mistral_fewshot",
])
const MAX_PLOTTED_FORMULA_SIZE = 34

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

function get_record_automaton_size(record::OrderedDict{String,Any})
    if haskey(record, "automaton_size")
        return Int(record["automaton_size"])
    elseif haskey(record, "buchi_size")
        return Int(record["buchi_size"])
    elseif haskey(record, "minimized_buchi_size")
        return Int(record["minimized_buchi_size"])
    else
        throw(ArgumentError("Dataset record ID $(get(record, "id", "?")) does not contain `automaton_size`, `buchi_size`, or `minimized_buchi_size`."))
    end
end

function get_entry_input_field(entry::OrderedDict{String,Any})
    for key in ("input_field", "field", "source_field")
        if haskey(entry, key)
            return String(entry[key])
        end
    end
    return nothing
end

function nl_phrase_word_count(text::AbstractString)
    s = strip(String(text))
    isempty(s) && return 0
    return length(split(s))
end

function get_entry_nl_phrase_length(entry::OrderedDict{String,Any}, dataset_index::Dict{Int,OrderedDict{String,Any}})
    haskey(entry, "record_id") || throw(ArgumentError("Entry does not contain `record_id`."))
    record_id = Int(entry["record_id"])
    haskey(dataset_index, record_id) || throw(ArgumentError("Dataset index does not contain record ID $(record_id)."))

    input_field = get_entry_input_field(entry)
    isnothing(input_field) && throw(ArgumentError("Entry for record ID $(record_id) does not contain `input_field`."))
    haskey(dataset_index[record_id], input_field) || throw(ArgumentError("Dataset record ID $(record_id) does not contain field `$(input_field)`."))

    text_value = String(dataset_index[record_id][input_field])
    return nl_phrase_word_count(text_value)
end

function build_source_record_min_formula_size_map(dataset_path::String = DEFAULT_DATASET_PATH)
    dataset_index = load_dataset_index(dataset_path)
    min_size_map = Dict{Int,Int}()

    for (record_id, record) in dataset_index
        haskey(record, "LTL") || continue
        current_size = get_record_formula_size(record)

        if haskey(record, "source_record_id")
            source_id = try
                Int(record["source_record_id"])
            catch
                nothing
            end

            if !isnothing(source_id) && haskey(dataset_index, source_id) && haskey(dataset_index[source_id], "LTL")
                source_size = get_record_formula_size(dataset_index[source_id])
                min_size_map[record_id] = min(current_size, source_size)
            else
                min_size_map[record_id] = current_size
            end
        else
            min_size_map[record_id] = current_size
        end
    end

    return min_size_map
end

function average_tied_ranks(values::Vector{Float64})
    n = length(values)
    n == 0 && return Float64[]

    perm = sortperm(values)
    ranks = zeros(Float64, n)
    i = 1
    while i <= n
        j = i
        while j < n && values[perm[j + 1]] == values[perm[i]]
            j += 1
        end
        avg_rank = (i + j) / 2
        for k in i:j
            ranks[perm[k]] = avg_rank
        end
        i = j + 1
    end

    return ranks
end

function pearson_corr(x::Vector{Float64}, y::Vector{Float64})
    n = length(x)
    n == length(y) || throw(ArgumentError("Vectors must have the same length."))
    n < 2 && return NaN

    mx = mean(x)
    my = mean(y)
    x_centered = x .- mx
    y_centered = y .- my
    denom = sqrt(sum(abs2, x_centered) * sum(abs2, y_centered))
    denom == 0 && return NaN
    return sum(x_centered .* y_centered) / denom
end

function spearman_rank_correlation(x::Vector{Float64}, y::Vector{Float64})
    length(x) == length(y) || throw(ArgumentError("Vectors must have the same length."))
    length(x) < 2 && return NaN
    rx = average_tied_ranks(x)
    ry = average_tied_ranks(y)
    return pearson_corr(rx, ry)
end

function normal_two_sided_pvalue_from_z(z::Float64)
    return erfc(abs(z) / sqrt(2))
end

function approximate_spearman_pvalue(rho::Float64, n::Int)
    if isnan(rho) || n < 4 || abs(rho) >= 1.0
        return isnan(rho) ? NaN : 0.0
    end
    z = atanh(rho) * sqrt((n - 3) / 1.06)
    return normal_two_sided_pvalue_from_z(z)
end

function success_indicator(entry::OrderedDict{String,Any})
    status_ok = haskey(entry, "status") && String(entry["status"]) == "ok"
    equivalent_true = haskey(entry, "equivalent") && entry["equivalent"] === true
    return (status_ok && equivalent_true) ? 1.0 : 0.0
end

function measure_vectors_for_correlation(
    entries::Vector{OrderedDict{String,Any}},
    dataset_index::Dict{Int,OrderedDict{String,Any}};
    dataset_path::String = DEFAULT_DATASET_PATH,
)
    min_size_map = build_source_record_min_formula_size_map(dataset_path)

    measures = OrderedDict(
        "NL length" => Float64[],
        "AST size" => Float64[],
        "Minimum AST size in equivalent class" => Float64[],
        "Nesting" => Float64[],
        "Minimized Buchi size" => Float64[],
    )
    success = Float64[]

    for entry in entries
        haskey(entry, "record_id") || continue
        record_id = Int(entry["record_id"])
        haskey(dataset_index, record_id) || continue
        record = dataset_index[record_id]

        try
            nl_length = Float64(get_entry_nl_phrase_length(entry, dataset_index))
            ast_size = Float64(get_record_formula_size(record))
            min_ast_size = Float64(get(min_size_map, record_id, Int(round(ast_size))))
            nesting = Float64(get_record_temporal_depth(record))
            buchi_size = Float64(get_record_automaton_size(record))
            succ = success_indicator(entry)

            push!(measures["NL length"], nl_length)
            push!(measures["AST size"], ast_size)
            push!(measures["Minimum AST size in equivalent class"], min_ast_size)
            push!(measures["Nesting"], nesting)
            push!(measures["Minimized Buchi size"], buchi_size)
            push!(success, succ)
        catch err
            println("Warning: skipping record ID $(record_id) in correlation computation: ", err)
        end
    end

    return measures, success
end

function spearman_correlation_statistics(
    entries::Vector{OrderedDict{String,Any}},
    dataset_index::Dict{Int,OrderedDict{String,Any}};
    dataset_path::String = DEFAULT_DATASET_PATH,
)
    measures, success = measure_vectors_for_correlation(entries, dataset_index; dataset_path=dataset_path)
    stats = OrderedDict{String,OrderedDict{String,Any}}()

    for (measure_name, values) in measures
        rho = spearman_rank_correlation(values, success)
        n = length(values)
        p_value = approximate_spearman_pvalue(rho, n)
        stats[measure_name] = OrderedDict(
            "n" => n,
            "spearman_rho" => rho,
            "p_value" => p_value,
        )
    end

    return stats
end

function print_spearman_correlation_statistics(
    entries::Vector{OrderedDict{String,Any}},
    dataset_index::Dict{Int,OrderedDict{String,Any}};
    dataset_path::String = DEFAULT_DATASET_PATH,
)
    stats = spearman_correlation_statistics(entries, dataset_index; dataset_path=dataset_path)
    println("Spearman rank correlation with success indicator")
    println("------------------------------------------------------------")
    for (measure_name, vals) in stats
        println(measure_name, ":")
        println("  n = ", vals["n"])
        println("  rho = ", round(Float64(vals["spearman_rho"]); digits=4))
        println("  p_value = ", round(Float64(vals["p_value"]); digits=6))
    end
    println()
    return stats
end


# =================================================================================================
# Logistic regression analysis
# =================================================================================================

function logistic_sigmoid(x::Float64)
    if x >= 0
        z = exp(-x)
        return 1.0 / (1.0 + z)
    else
        z = exp(x)
        return z / (1.0 + z)
    end
end

function standardize_vector(x::Vector{Float64})
    isempty(x) && return Float64[], NaN, NaN
    μ = mean(x)
    σ = std(x)
    if isnan(σ) || σ == 0.0
        return fill(0.0, length(x)), μ, σ
    end
    return (x .- μ) ./ σ, μ, σ
end

function logistic_regression_fit(X::Matrix{Float64}, y::Vector{Float64}; max_iter::Int = 100, tol::Float64 = 1e-8)
    n, p = size(X)
    length(y) == n || throw(ArgumentError("X and y must have compatible sizes."))

    β = zeros(Float64, p)

    for _ in 1:max_iter
        η = X * β
        μ = [clamp(logistic_sigmoid(v), 1e-9, 1 - 1e-9) for v in η]
        w = μ .* (1 .- μ)
        z = η + (y .- μ) ./ w

        WX = X .* w
        hessian = transpose(X) * WX
        gradient_rhs = transpose(X) * (w .* z)

        β_new = try
            hessian \ gradient_rhs
        catch
            pinv(hessian) * gradient_rhs
        end

        if norm(β_new - β) < tol
            β = β_new
            break
        end
        β = β_new
    end

    η = X * β
    μ = [clamp(logistic_sigmoid(v), 1e-9, 1 - 1e-9) for v in η]
    w = μ .* (1 .- μ)
    WX = X .* w
    hessian = transpose(X) * WX
    covβ = try
        inv(hessian)
    catch
        pinv(hessian)
    end
    se = sqrt.(max.(diag(covβ), 0.0))

    return β, se, μ
end

function logistic_regression_statistics(
    entries::Vector{OrderedDict{String,Any}},
    dataset_index::Dict{Int,OrderedDict{String,Any}};
    dataset_path::String = DEFAULT_DATASET_PATH,
)
    measures, success = measure_vectors_for_correlation(entries, dataset_index; dataset_path=dataset_path)

    ordered_names = [
        "NL length",
        "AST size",
        "Minimum AST size in equivalent class",
        "Nesting",
        "Minimized Buchi size",
    ]

    standardized_columns = Vector{Vector{Float64}}()
    means = Dict{String,Float64}()
    stds = Dict{String,Float64}()

    for name in ordered_names
        x_std, μ, σ = standardize_vector(measures[name])
        push!(standardized_columns, x_std)
        means[name] = μ
        stds[name] = σ
    end

    n = length(success)
    p = 1 + length(ordered_names)
    X = ones(Float64, n, p)
    for (j, col) in enumerate(standardized_columns)
        X[:, j + 1] = col
    end
    y = Float64.(success)

    β, se, fitted = logistic_regression_fit(X, y)

    stats = OrderedDict{String,OrderedDict{String,Any}}()
    intercept_z = se[1] == 0.0 ? NaN : β[1] / se[1]
    intercept_p = isnan(intercept_z) ? NaN : normal_two_sided_pvalue_from_z(intercept_z)
    stats["Intercept"] = OrderedDict(
        "coefficient" => β[1],
        "std_error" => se[1],
        "z_value" => intercept_z,
        "p_value" => intercept_p,
    )

    for (j, name) in enumerate(ordered_names)
        z_value = se[j + 1] == 0.0 ? NaN : β[j + 1] / se[j + 1]
        p_value = isnan(z_value) ? NaN : normal_two_sided_pvalue_from_z(z_value)
        odds_ratio = exp(β[j + 1])
        stats[name] = OrderedDict(
            "coefficient" => β[j + 1],
            "std_error" => se[j + 1],
            "z_value" => z_value,
            "p_value" => p_value,
            "odds_ratio" => odds_ratio,
            "mean" => means[name],
            "std" => stds[name],
        )
    end

    loglik = sum(y .* log.(fitted) .+ (1 .- y) .* log.(1 .- fitted))
    stats["Model"] = OrderedDict(
        "n" => n,
        "log_likelihood" => loglik,
    )

    return stats
end

function print_logistic_regression_statistics(
    entries::Vector{OrderedDict{String,Any}},
    dataset_index::Dict{Int,OrderedDict{String,Any}};
    dataset_path::String = DEFAULT_DATASET_PATH,
)
    stats = logistic_regression_statistics(entries, dataset_index; dataset_path=dataset_path)
    println("Logistic regression with success indicator")
    println("------------------------------------------------------------")
    println("Predictors are standardized before fitting.")
    println("Model:")
    println("  n = ", stats["Model"]["n"])
    println("  log_likelihood = ", round(Float64(stats["Model"]["log_likelihood"]); digits=4))
    println("Intercept:")
    println("  coefficient = ", round(Float64(stats["Intercept"]["coefficient"]); digits=4))
    println("  std_error = ", round(Float64(stats["Intercept"]["std_error"]); digits=4))
    println("  z_value = ", round(Float64(stats["Intercept"]["z_value"]); digits=4))
    println("  p_value = ", round(Float64(stats["Intercept"]["p_value"]); digits=6))

    for measure_name in (
        "NL length",
        "AST size",
        "Minimum AST size in equivalent class",
        "Nesting",
        "Minimized Buchi size",
    )
        vals = stats[measure_name]
        println(measure_name, ":")
        println("  coefficient = ", round(Float64(vals["coefficient"]); digits=4))
        println("  std_error = ", round(Float64(vals["std_error"]); digits=4))
        println("  z_value = ", round(Float64(vals["z_value"]); digits=4))
        println("  p_value = ", round(Float64(vals["p_value"]); digits=6))
        println("  odds_ratio = ", round(Float64(vals["odds_ratio"]); digits=4))
    end
    println()
    return stats
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

    basename_name = splitext(basename(results_path))[1]
    if basename_name == "GPT54_fewshot"
        return "gpt-5.4"
    elseif basename_name == "GPT55_fewshot"
        return "gpt-5.5"
    elseif basename_name == "Claude_fewshot"
        return "claude-opus-4-7"
    elseif basename_name == "DeepSeek_fewshot"
        return "deepseek-v4-flash"
    elseif basename_name == "Mistral_fewshot"
        return "mistral-medium-latest"
    end

    lowercase_model = lowercase(model_name)
    if occursin("deepseek", lowercase_model)
        return "deepseek-v4-flash"
    elseif occursin("claude", lowercase_model)
        return "claude-opus-4-7"
    elseif occursin("mistral", lowercase_model)
        return "mistral-medium-latest"
    elseif lowercase_model == "gpt54" || lowercase_model == "gpt-5.4" || occursin("gpt54_fewshot", lowercase_model)
        return "gpt-5.4"
    elseif lowercase_model == "gpt55" || lowercase_model == "gpt-5.5" || occursin("gpt55_fewshot", lowercase_model)
        return "gpt-5.5"
    end

    return model_name
end


function result_provider_name(results_obj::OrderedDict{String,Any})
    if haskey(results_obj, "translation_provider")
        return String(results_obj["translation_provider"])
    end
    return "unknown"
end

function result_prompt_setting(results_obj::OrderedDict{String,Any}, results_path::String)
    if haskey(results_obj, "prompt_setting")
        return lowercase(String(results_obj["prompt_setting"]))
    end

    basename_lower = lowercase(basename(results_path))
    if occursin("fewshot", basename_lower) || occursin("few_shot", basename_lower)
        return "fewshot"
    end

    return "zeroshot"
end

function use_zero_vs_fewshot_result_file(stats::OrderedDict{String,Any})
    prompt_setting = haskey(stats, "prompt_setting") ? String(stats["prompt_setting"]) : "zeroshot"
    if prompt_setting == "fewshot"
        result_base = splitext(basename(String(stats["results_path"])))[1]
        return result_base in ZERO_VS_FEWSHOT_FEWSHOT_FILES
    end
    return String(stats["model"]) in ZERO_VS_FEWSHOT_MODELS
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
        "prompt_setting" => result_prompt_setting(results_obj, results_path),
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

    dataset_index = load_dataset_index(DEFAULT_DATASET_PATH)
    print_spearman_correlation_statistics(entries, dataset_index; dataset_path=DEFAULT_DATASET_PATH)
    print_logistic_regression_statistics(entries, dataset_index; dataset_path=DEFAULT_DATASET_PATH)
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
    aggregated_stats_raw = aggregate_all_result_files(results_dir)
    aggregated_stats = USE_ZERO_VS_FEWSHOT_SUBPLOTS ? aggregated_stats_raw : maybe_filter_to_paper_models(aggregated_stats_raw)

    if USE_ZERO_VS_FEWSHOT_SUBPLOTS
        filtered_stats = OrderedDict{String,Any}[]
        for stats in aggregated_stats
            if use_zero_vs_fewshot_result_file(stats)
                push!(filtered_stats, stats)
            end
        end

        isempty(filtered_stats) && throw(ArgumentError("Zero-vs-few-shot subplot mode is enabled, but no matching result files were found for the selected models."))

        series_data = Vector{Tuple{String,String,Vector{Int},Vector{Float64},Vector{Int}}}()
        all_sizes = Int[]

        for stats in filtered_stats
            model_name = String(stats["model"])
            prompt_setting = haskey(stats, "prompt_setting") ? String(stats["prompt_setting"]) : "zeroshot"
            entries = Vector{OrderedDict{String,Any}}(stats["entries"])
            ignore_errors = haskey(stats, "ignore_errors_in_success_rate") ? Bool(stats["ignore_errors_in_success_rate"]) : false
            sizes, rates, totals = size_bucket_success_rates(entries, dataset_index; ignore_errors=ignore_errors)
            keep_idx = [i for i in eachindex(sizes) if 1 <= sizes[i] <= MAX_PLOTTED_FORMULA_SIZE]
            isempty(keep_idx) && continue
            sizes = sizes[keep_idx]
            rates = rates[keep_idx]
            totals = totals[keep_idx]
            push!(series_data, (model_name, prompt_setting, sizes, rates, totals))
            append!(all_sizes, sizes)
        end

        isempty(all_sizes) && throw(ArgumentError("No size statistics could be computed from the available zero-shot/few-shot result files."))

        x_values = sort(unique(all_sizes))
        xtick_step = maximum(x_values) <= 20 ? 1 : (maximum(x_values) <= 40 ? 2 : 5)
        xtick_values = collect(minimum(x_values):xtick_step:maximum(x_values))
        marker_shapes = Dict(
            "gpt-5.4" => :circle,
            "gpt-5.5" => :rect,
            "claude-opus-4-7" => :diamond,
            "deepseek-v4-flash" => :utriangle,
            "mistral-medium-latest" => :dtriangle,
        )

        subplot_list = Plots.Plot[]
        for prompt_setting in ["zeroshot", "fewshot"]
            prompt_series = [(model_name, sizes, rates, totals) for (model_name, ps, sizes, rates, totals) in series_data if ps == prompt_setting]

            local_p = plot(
                xlabel=(prompt_setting == "fewshot" ? "LTL formula size" : ""),
                ylabel=(prompt_setting == "zeroshot" ? "Zero-shot success rate" : "Few-shot success rate"),
                title="",
                legend=(prompt_setting == "zeroshot" ? :topright : false),
                ylim=(0.0, 1.0),
                xlims=(minimum(x_values) - 0.5, maximum(x_values) + 0.5),
                xticks=xtick_values,
                framestyle=:box,
                grid=true,
                guidefontsize=25,
                tickfontsize=15,
                titlefontsize=18,
                legendfontsize=14,
                guidefontweight=:bold,
                left_margin=14Plots.mm,
                right_margin=10Plots.mm,
                bottom_margin=10Plots.mm,
                top_margin=6Plots.mm,
            )

            for model_name in ZERO_VS_FEWSHOT_MODELS
                matching = [item for item in prompt_series if item[1] == model_name]
                isempty(matching) && continue
                _, sizes, rates, totals = matching[1]
                marker_shape = get(marker_shapes, model_name, :circle)
                plot!(
                    local_p,
                    sizes,
                    rates;
                    marker=marker_shape,
                    markersize=5,
                    markerstrokewidth=0.4,
                    linewidth=2.3,
                    label=(prompt_setting == "zeroshot" ? model_name : ""),
                )
            end

            push!(subplot_list, local_p)
        end

        p = plot(
            subplot_list...;
            layout=(2, 1),
            size=(1700, 1250),
        )
    else
        series_data = Vector{Tuple{String,Vector{Int},Vector{Float64},Vector{Int},Float64}}()
        all_sizes = Int[]

        for stats in aggregated_stats
            model_name = String(stats["model"])
            entries = Vector{OrderedDict{String,Any}}(stats["entries"])
            ignore_errors = haskey(stats, "ignore_errors_in_success_rate") ? Bool(stats["ignore_errors_in_success_rate"]) : false
            sizes, rates, totals = size_bucket_success_rates(entries, dataset_index; ignore_errors=ignore_errors)
            keep_idx = [i for i in eachindex(sizes) if 1 <= sizes[i] <= MAX_PLOTTED_FORMULA_SIZE]
            isempty(keep_idx) && continue
            sizes = sizes[keep_idx]
            rates = rates[keep_idx]
            totals = totals[keep_idx]
            overall_rate = haskey(stats, "overall") && haskey(stats["overall"], "success_rate") ? Float64(stats["overall"]["success_rate"]) : 0.0
            push!(series_data, (model_name, sizes, rates, totals, overall_rate))
            append!(all_sizes, sizes)
        end

        isempty(all_sizes) && throw(ArgumentError("No size statistics could be computed from the available result files."))

        sorted_series = sort(series_data; by = item -> -item[5])
        x_values = sort(unique(all_sizes))
        xtick_step = maximum(x_values) <= 20 ? 1 : (maximum(x_values) <= 40 ? 2 : 5)
        xtick_values = collect(minimum(x_values):xtick_step:maximum(x_values))

        n_models = length(sorted_series)
        ncols = 2
        nrows = ceil(Int, n_models / ncols)

        plots_list = Plots.Plot[]
        marker_shapes = [:circle, :rect, :diamond, :utriangle, :dtriangle, :star5, :hexagon, :xcross]

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
    println("Run `print_spearman_correlation_statistics(entries, load_dataset_index())` to print Spearman rho and approximate p-values for NL length, AST size, minimum equivalent-class AST size, nesting, and minimized Buchi size.")
    println("Run `print_logistic_regression_statistics(entries, load_dataset_index())` to fit a logistic regression over the same five predictors and inspect which measures remain significant jointly across all models.")
    println("Set `USE_PAPER_MODEL_SUBSET = true` to plot only mistral-medium-latest, claude-opus-4-7, gpt-5.4, and gpt-5.5.")
    println("Set `USE_ZERO_VS_FEWSHOT_SUBPLOTS = true` to plot zero-shot (top) and few-shot (bottom) success-rate-vs-size subplots for gpt-5.4, gpt-5.5, claude-opus-4-7, deepseek-v4-flash, and mistral-medium-latest.")
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end


