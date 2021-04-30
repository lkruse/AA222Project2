# ******************************************************************************
# PACKAGES AND INCLUDES
# ******************************************************************************
using Plots
using ImplicitEquations

include("project2.jl")
include("helpers.jl")
include("simple.jl")

# ******************************************************************************
# PATH STORING FUNCTIONS
# ******************************************************************************
function path_hooke_jeeves(f, g, c, x, num_evals, ρ1, ρ2, γ, δ)
    x1_arr = Float64[]; x2_arr = Float64[]; y_arr = Float64[]; max_c_arr = Float64[];
    push!(x1_arr, x[1]); push!(x2_arr, x[2]); push!(y_arr, f(x))
    evals = c(x); max_c = maximum([evals[1], evals[2], 0]); push!(max_c_arr, max_c)
    α, y, n, ϵ = 1, f(x) + penalty(c, x, ρ1, ρ2, δ), length(x), 1e-9
    while α > ϵ && count(f, g, c) < (num_evals-8)
        evals = c(x); max_c = maximum([evals[1], evals[2], 0]); push!(max_c_arr, max_c)
        x, y, α = hooke_jeeves(x -> f(x) + penalty(c, x, ρ1, ρ2, δ), x, y, α, n, 0.5)
        push!(x1_arr, x[1]);push!(x2_arr, x[2]); push!(y_arr, f(x))
        ρ1 *= γ; ρ2 *= γ
    end
    return x, x1_arr, x2_arr, y_arr, max_c_arr
end

function path_mesh_adaptive(f, g, c, x, num_evals, ρ1, ρ2, γ, δ)
    x1_arr = Float64[]; x2_arr = Float64[]; y_arr = Float64[]; max_c_arr = Float64[];
    push!(x1_arr, x[1]); push!(x2_arr, x[2]); push!(y_arr, f(x))
    evals = c(x); max_c = maximum([evals[1], evals[2], 0]); push!(max_c_arr, max_c)
    α, y, n, ϵ = 1, f(x) + penalty(c, x, ρ1, ρ2, δ), length(x), 1e-9
    while α > ϵ && count(f, g, c) < (num_evals-8)
        evals = c(x); max_c = maximum([evals[1], evals[2], 0]); push!(max_c_arr, max_c)
        x, y, α = mesh_adaptive_direct_search(x -> f(x) + penalty(c, x, ρ1, ρ2, δ), x, y, α, n)
        push!(x1_arr, x[1]); push!(x2_arr, x[2]); push!(y_arr, f(x))
        ρ1 *= γ; ρ2 *= γ
    end
    return x, x1_arr, x2_arr, y_arr, max_c_arr
end

# ******************************************************************************
# PLOTTING HELPER FUNCTIONS
# ******************************************************************************
function scatter_helper(x1_arr, x2_arr, x0, x_best, color)
    scatter!([x1_arr], [x2_arr], color=color, label = "", markersize = 3, outline = color)
    scatter!([x0[1]], [x0[2]], color=color,  marker = :rect,label = "", markersize = 5)
    scatter!([x_best[1]], [x_best[2]], color=color, marker = :star4, label = "", markersize = 7)
end

function plot_simple1_contour()
    x1 = -3.0:0.01:3.0
    x2 = -3.0:0.01:3.0
    plot_f(x1, x2) = begin
        -x1 * x2 + 2.0 / (3.0 * sqrt(3.0))
    end
    X = repeat(reshape(x1, 1, :), length(x2), 1)
    Y = repeat(x2, 1, length(x1))
    Z = map(plot_f, X, Y)
    contour(x1, x2, Z, levels=31, fill=false, c=cgrad(:viridis, rev = true), linewidth=2.0, colorbar=true)

    c1(x1,x2) = -x1 - x2
    c2(x1,x2) = x1 + x2^2 - 1

    plot!((Lt(c1,0)) & (Lt(c2,0)), fill=(0.65,"#495C6F"), lims=[-3,3],
        label="Feasible Region")
end

function plot_simple2_contour()
    x1 = -3.0:0.01:3.0
    x2 = -3.0:0.01:3.0
    plot_f(x1, x2) = begin
        (1.0 - x1)^2 + 100.0 * (x2 - x1^2)^2
    end
    X = repeat(reshape(x1, 1, :), length(x2), 1)
    Y = repeat(x2, 1, length(x1))
    Z = map(plot_f, X, Y)
    h = [minimum(Z), 2.5, 10, 25, 100, 250, 500, 750,1000, 1500, 2500, 3500, 5000, 6500, 8000, 10000,maximum(Z)]
    contour(x1, x2, Z, levels=h, fill=false, c=cgrad(:viridis, rev = true), linewidth=2.0, colorbar=true)

    c1(x1,x2) = (x1-1)^3 - x2 + 1
    c2(x1,x2) = x1 + x2 - 2

    plot!((Lt(c1,0)) & (Lt(c2,0)), fill=(0.65,"#495C6F"), lims=[-3,3],
        label="Feasible Region")
end

# ******************************************************************************
# PATH PLOTTING FUNCTIONS
# ******************************************************************************
function plot_simple1_hooke_jeeves(f, g, c, n)

    plot_simple1_contour()

    x01 = [2.5, -2.5]; x02 = [2.5, 2.5]; x03 = [-2.0, 2.0]
    x1_best, x11_arr, x21_arr, _, _ = 
        path_hooke_jeeves(f, g, c, x01, n, 1.01, 0, 1.001, 0.5);
    x2_best, x12_arr, x22_arr, _, _ =
        path_hooke_jeeves(f, g, c, x02, n, 1.01, 0, 1.001, 0.5);
    x3_best, x13_arr, x23_arr, _, _ = 
        path_hooke_jeeves(f, g, c, x03, n, 1.01, 0, 1.001, 0.5);

    # Plot first path
    plot!(x11_arr,x21_arr,color=:magenta, label = "x0: (2.5, -2.5)",legend = :bottomleft,
        title = "Hooke-Jeeves Method to Minimize Simple1", titlefontsize = 12,
        xlabel = "x1", ylabel = "x2", dpi=300, colorbar = true)
    scatter_helper(x11_arr, x21_arr, x01, x1_best, :magenta)
    # Plot second path
    plot!(x12_arr,x22_arr,color=:red, label = "x0: (2.5, 2.5)")
    scatter_helper(x12_arr, x22_arr, x02, x2_best, :red)
    # Plot third path
    plot!(x13_arr,x23_arr,color=:darkorange, label = "x0: (-2.0, 2.0)")
    scatter_helper(x13_arr, x23_arr, x03, x3_best, :darkorange)

    png("simple1_hooke_jeeves.png")
end

function plot_simple1_mesh_adaptive(f, g, c, n)
    # Plot contour
    plot_simple1_contour()

    # Store paths
    x01 = [2.5, -2.5]; x02 = [2.5, 2.5]; x03 = [-2.0, 2.0]
    x1_best, x11_arr, x21_arr, _, _ = 
        path_mesh_adaptive(f, g, c, x01, n, 1.01, 0, 1.001, 0.5);
    x2_best, x12_arr, x22_arr, _, _ = 
        path_mesh_adaptive(f, g, c, x02, n, 1.01, 0, 1.001, 0.5);
    x3_best, x13_arr, x23_arr, _, _ = 
        path_mesh_adaptive(f, g, c, x03, n, 1.01, 0, 1.001, 0.5);

    # Plot first path
    plot!(x11_arr,x21_arr,color=:magenta, label = "x0: (2.5, -2.5)",legend = :bottomleft,
        title = "Mesh Adaptive Direct Search Method to Minimize Simple1", titlefontsize = 12,
        xlabel = "x1", ylabel = "x2", dpi=300, colorbar = true)
    scatter_helper(x11_arr, x21_arr, x01, x1_best, :magenta)
    # Plot second path
    plot!(x12_arr,x22_arr,color=:red, label = "x0: (2.5, 2.5)")
    scatter_helper(x12_arr, x22_arr, x02, x2_best, :red)
    # Plot third path
    plot!(x13_arr,x23_arr,color=:darkorange, label = "x0: (-2.0, 2.0)")
    scatter_helper(x13_arr, x23_arr, x03, x3_best, :darkorange)

    png("simple1_mesh_adaptive.png")
end

function plot_simple2_hooke_jeeves(f, g, c, n)
    # Plot contour
    plot_simple2_contour()

    # Store paths
    x01 = [2.5, -2.5]; x02 = [2.5, 2.5]; x03 = [-2.0, 2.0]
    x1_best, x11_arr, x21_arr, _, _ = 
        path_hooke_jeeves(f, g, c, x01, n, 100, 1, 1.1, 0.75);
    x2_best, x12_arr, x22_arr, _, _ =
        path_hooke_jeeves(f, g, c, x02, n, 100, 1, 1.1, 0.75);
    x3_best, x13_arr, x23_arr, _, _ = 
        path_hooke_jeeves(f, g, c, x03, n, 100, 1, 1.1, 0.75);

    # Plot first path
    plot!(x11_arr,x21_arr,color=:magenta, label = "x0: (2.5, -2.5)",legend = :bottomleft,
        title = "Hooke-Jeeves Method to Minimize Simple2", titlefontsize = 12,
        xlabel = "x1", ylabel = "x2", dpi=300, colorbar = true)
    scatter_helper(x11_arr, x21_arr, x01, x1_best, :magenta)
    # Plot second path
    plot!(x12_arr,x22_arr,color=:red, label = "x0: (2.5, 2.5)")
    scatter_helper(x12_arr, x22_arr, x02, x2_best, :red)
    # Plot third path
    plot!(x13_arr,x23_arr,color=:darkorange, label = "x0: (-2.0, 2.0)")
    scatter_helper(x13_arr, x23_arr, x03, x3_best, :darkorange)

    png("simple2_hooke_jeeves.png")
end

function plot_simple2_mesh_adaptive(f, g, c, n)
    # Plot contour
    plot_simple2_contour()

    # Store paths
    x01 = [0.0, -2.0]; x02 = [1.0, -1.0]; x03 = [2.0, 0.0]
    x1_best, x11_arr, x21_arr, _, _ = 
        path_mesh_adaptive(f, g, c, x01, n, 100, 1, 1.01, 1.0);
    x2_best, x12_arr, x22_arr, _, _ =
        path_mesh_adaptive(f, g, c, x02, n, 100, 1, 1.01, 1.0);
    x3_best, x13_arr, x23_arr, _, _ = 
        path_mesh_adaptive(f, g, c, x03, n, 100, 1, 1.01, 1.0);

    # Plot first path
    plot!(x11_arr,x21_arr,color=:magenta, label = "x0: (0.0, -2.0)",legend = :bottomleft,
        title = "Mesh Adaptive Direct Search Method to Minimize Simple2", titlefontsize = 12,
        xlabel = "x1", ylabel = "x2", dpi=300, colorbar = true)
    scatter_helper(x11_arr, x21_arr, x01, x1_best, :magenta)
    # Plot second path
    plot!(x12_arr,x22_arr,color=:red, label = "x0: (1.0, -1.0)")
    scatter_helper(x12_arr, x22_arr, x02, x2_best, :red)
    # Plot third path
    plot!(x13_arr,x23_arr,color=:darkorange, label = "x0: (2.0, 0.0)")
    scatter_helper(x13_arr, x23_arr, x03, x3_best, :darkorange)

    png("simple2_mesh_adaptive.png")
end

function plot_obj_fun_max_con_hooke_jeeves(f, g, c, n)

    x01 = [2.5, -2.5]; x02 = [2.5, 2.5]; x03 = [-2.0, 2.0]
    _, _, _, y1_arr, max_c1_arr = 
        path_hooke_jeeves(f, g, c, x01, n, 100, 1, 1.1, 0.75);
    _, _, _, y2_arr, max_c2_arr =
        path_hooke_jeeves(f, g, c, x02, n, 100, 1, 1.1, 0.75);
    _, _, _, y3_arr, max_c3_arr = 
        path_hooke_jeeves(f, g, c, x03, n, 100, 1, 1.1, 0.75);

    x_lim = minimum([length(y1_arr), length(y2_arr), length(y3_arr), 20]) 
    x_vals = 1:x_lim

    plot(x_vals, y1_arr[1:x_lim], lw=3, color = "#B2BfCC", label="x0: (2.5, -2.5)",legend = :topright,
        title="Objective Function vs Iteration for Hooke-Jeeves",
        xlabel="Iteration",ylabel="Maximum Constraint Violation",dpi=300)
    plot!(x_vals, y2_arr[1:x_lim], lw = 3, color = "#495C6F", label = "x0: (2.5, 2.5)")
    plot!(x_vals, y3_arr[1:x_lim], lw = 3, color = "#1E262E", label = "x0: (-2.0, 2.0)")
    
    png("objective_function_hooke_jeeves.png")

    plot(x_vals, max_c1_arr[1:x_lim], lw=3, color = "#B2BfCC", label="x0: (2.5, -2.5)",legend = :topright,
        title="Maximum Constraint Violation for Hooke-Jeeves", 
        xlabel="Iteration",ylabel="Maximum Constraint Violation",dpi=300)
    plot!(x_vals, max_c2_arr[1:x_lim], lw = 3, color = "#495C6F", label = "x0: (2.5, 2.5)")
    plot!(x_vals, max_c3_arr[1:x_lim], lw = 3, color = "#1E262E", label = "x0: (-2.0, 2.0)")

    png("constraint_violation_hooke_jeeves.png")
end

function plot_obj_fun_max_con_mesh_adaptive(f, g, c, n)

    x01 = [0.0, -2.0]; x02 = [1.0, -1.0]; x03 = [2.0, 0.0]
    _, _, _, y1_arr, max_c1_arr = 
        path_mesh_adaptive(f, g, c, x01, n, 100, 1, 1.01, 1.0);
    _, _, _, y2_arr, max_c2_arr =
        path_mesh_adaptive(f, g, c, x02, n, 100, 1, 1.01, 1.0);
    _, _, _, y3_arr, max_c3_arr = 
        path_mesh_adaptive(f, g, c, x03, n, 100, 1, 1.01, 1.0);

    x_lim = minimum([length(y1_arr), length(y2_arr), length(y3_arr), 20]) 
    x_vals = 1:x_lim

    plot(x_vals, y1_arr[1:x_lim], lw=3, color = "#B2BfCC", label="x0: (0.0, -2.0)",legend = :topright,
        title="Objective Function for Mesh Adaptive",
        xlabel="Iteration",ylabel="Maximum Constraint Violation",dpi=300)
    plot!(x_vals, y2_arr[1:x_lim], lw = 3, color = "#495C6F", label = "x0: (1.0, -1.0)")
    plot!(x_vals, y3_arr[1:x_lim], lw = 3, color = "#1E262E", label = "x0: (2.0, 0.0)")
    
    png("objective_function_mesh_adaptive.png")

    plot(x_vals, max_c1_arr[1:x_lim], lw=3, color = "#B2BfCC", label="x0: (0.0, -2.0)",legend = :topright,
        title="Maximum Constraint Violation for Mesh Adaptive", 
        xlabel="Iteration", ylabel="Maximum Constraint Violation",dpi=300)
    plot!(x_vals, max_c2_arr[1:x_lim], lw = 3, color = "#495C6F", label = "x0: (1.0, -1.0)")
    plot!(x_vals, max_c3_arr[1:x_lim], lw = 3, color = "#1E262E", label = "x0: (2.0, 0.0)")

    png("constraint_violation_mesh_adaptive.png")
end


n = 100000
# Path plots
plot_simple1_hooke_jeeves(simple1, simple1_gradient, simple1_constraints, n)
plot_simple1_mesh_adaptive(simple1, simple1_gradient, simple1_constraints, n)
plot_simple2_hooke_jeeves(simple2, simple2_gradient, simple2_constraints, n)
plot_simple2_mesh_adaptive(simple2, simple2_gradient, simple2_constraints, n)

# Objective function and maximum constraint plots
plot_obj_fun_max_con_hooke_jeeves(simple2, simple2_gradient, simple2_constraints, n)
plot_obj_fun_max_con_mesh_adaptive(simple2, simple2_gradient, simple2_constraints, n)