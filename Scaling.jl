

using CSV
using DataFrames
using GLM
using Statistics
using Printf
using Plots

"""
    fit_scaling_law(df; model_col=:model, size_col=:ast_size, success_col=:success,
                    setting_col=nothing, setting_value=nothing)

Fit a logistic scaling law of the form

    P(success | size) = 1 / (1 + exp(-(β₀ + β₁ * size)))

on instance-level benchmark data.

Arguments
- `df`: a DataFrame with one row per benchmark instance.
- `model_col`: column containing the model name.
- `size_col`: column containing the LTL formula size (e.g. AST size).
- `success_col`: binary column (0/1 or false/true) indicating whether the
  produced formula is logically equivalent to the ground-truth formula.
- `setting_col`: optional column for selecting a subset, e.g. `:setting`.
- `setting_value`: optional value to filter `setting_col`, e.g. `"zero-shot"`.

Returns
- A DataFrame with one row per model and columns:
  `:model`, `:beta0`, `:beta1`, `:intercept`, `:slope`, `:n`, `:formula_size_50`

Notes
- `formula_size_50` is the size at which the fitted success probability crosses
  0.5, i.e. `-β₀ / β₁` when `β₁ != 0`.
- A negative slope indicates that success decreases as formula complexity grows.
"""
function fit_scaling_law(df::DataFrame;
    model_col::Symbol = :model,
    size_col::Symbol = :ast_size,
    success_col::Symbol = :success,
    setting_col::Union{Nothing,Symbol} = nothing,
    setting_value = nothing)

    work_df = copy(df)

    if setting_col !== nothing && setting_value !== nothing
        work_df = filter(row -> row[setting_col] == setting_value, work_df)
    end

    work_df[!, success_col] = Int.(work_df[!, success_col])

    out = DataFrame(
        model = String[],
        beta0 = Float64[],
        beta1 = Float64[],
        intercept = Float64[],
        slope = Float64[],
        n = Int[],
        formula_size_50 = Float64[],
    )

    for model_name in sort(unique(work_df[!, model_col]))
        sub = filter(row -> row[model_col] == model_name, work_df)
        nrow(sub) == 0 && continue

        fit = glm(@formula(success ~ ast_size),
                  rename!(select(sub, [success_col, size_col]),
                          success_col => :success, size_col => :ast_size),
                  Binomial(), LogitLink())

        coefs = coef(fit)
        β0 = coefs[1]
        β1 = coefs[2]
        size50 = iszero(β1) ? NaN : -β0 / β1

        push!(out, (
            string(model_name),
            β0,
            β1,
            β0,
            β1,
            nrow(sub),
            size50,
        ))
    end

    return out
end

"""
    add_predicted_success!(results, sizes)

Given the DataFrame returned by `fit_scaling_law`, add one column per size in
`sizes` containing the predicted success probability at that formula size.
"""
function add_predicted_success!(results::DataFrame, sizes)
    for s in sizes
        col = Symbol("p_size_" * string(s))
        results[!, col] = 1.0 ./ (1.0 .+ exp.(-(results.beta0 .+ results.beta1 .* s)))
    end
    return results
end

"""
    plot_scaling_law(df; model_col=:model, size_col=:ast_size, success_col=:success,
                     setting_col=nothing, setting_value=nothing,
                     output_path="scaling_law.png")

Create a plot of empirical success rates and fitted logistic scaling-law curves.
Empirical rates are computed per formula size. Fitted curves are learned from the
instance-level data.
"""
function plot_scaling_law(df::DataFrame;
    model_col::Symbol = :model,
    size_col::Symbol = :ast_size,
    success_col::Symbol = :success,
    setting_col::Union{Nothing,Symbol} = nothing,
    setting_value = nothing,
    output_path::AbstractString = "scaling_law.png")

    work_df = copy(df)
    if setting_col !== nothing && setting_value !== nothing
        work_df = filter(row -> row[setting_col] == setting_value, work_df)
    end

    work_df[!, success_col] = Int.(work_df[!, success_col])

    xmin = minimum(work_df[!, size_col])
    xmax = maximum(work_df[!, size_col])
    xgrid = collect(range(xmin, xmax; length=300))

    p = plot(
        xlabel = "LTL formula size",
        ylabel = "Success probability",
        legend = :outertopright,
        ylim = (0.0, 1.0),
        size = (1100, 700),
        guidefontsize = 16,
        tickfontsize = 12,
        legendfontsize = 10,
        grid = false,
        framestyle = :box,
    )

    for model_name in sort(unique(work_df[!, model_col]))
        sub = filter(row -> row[model_col] == model_name, work_df)
        nrow(sub) == 0 && continue

        agg = combine(groupby(sub, size_col), success_col => mean => :success_rate, nrow => :count)
        sort!(agg, size_col)

        # empirical points
        scatter!(
            p,
            agg[!, size_col],
            agg.success_rate;
            markersize = 4,
            markerstrokewidth = 0,
            label = nothing,
            alpha = 0.6,
        )

        # fitted logistic curve
        fit = glm(@formula(success ~ ast_size),
                  rename!(select(sub, [success_col, size_col]),
                          success_col => :success, size_col => :ast_size),
                  Binomial(), LogitLink())
        β0, β1 = coef(fit)
        ygrid = 1.0 ./ (1.0 .+ exp.(-(β0 .+ β1 .* xgrid)))

        plot!(
            p,
            xgrid,
            ygrid;
            linewidth = 2.5,
            label = string(model_name),
        )
    end

    savefig(p, output_path)
    return p
end

"""
    print_scaling_summary(results)

Pretty-print the fitted scaling-law coefficients.
"""
function print_scaling_summary(results::DataFrame)
    println("\nFitted logistic scaling law per model")
    println("------------------------------------")
    for row in eachrow(results)
        @printf(
            "%s\n  β₀ = %.4f\n  β₁ = %.4f\n  n = %d\n  size@p=0.5 = %.2f\n\n",
            row.model,
            row.beta0,
            row.beta1,
            row.n,
            row.formula_size_50,
        )
    end
end

"""
    load_instance_results(csv_path)

Load instance-level results from a CSV file. Expected columns:
- `model`
- `ast_size`
- `success`
Optional columns:
- `setting` (e.g. zero-shot / few-shot)

The `success` column should contain either `0/1` or `true/false` values.
"""
function load_instance_results(csv_path::AbstractString)
    df = CSV.read(csv_path, DataFrame)

    required = [:model, :ast_size, :success]
    missing_cols = setdiff(required, names(df))
    if !isempty(missing_cols)
        error("Missing required columns: $(missing_cols)")
    end

    return df
end

"""
Example usage:

    df = load_instance_results("results_instances.csv")

    zero_shot_results = fit_scaling_law(df; setting_col=:setting, setting_value="zero-shot")
    print_scaling_summary(zero_shot_results)
    add_predicted_success!(zero_shot_results, 1:36)
    CSV.write("zero_shot_scaling_law.csv", zero_shot_results)

    plot_scaling_law(df;
        setting_col=:setting,
        setting_value="zero-shot",
        output_path="zero_shot_scaling_law.png")

If your file has no `setting` column, simply call:

    results = fit_scaling_law(df)
    plot_scaling_law(df)
"""