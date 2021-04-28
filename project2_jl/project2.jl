# *******************************************************************************
# PACKAGES
# *******************************************************************************
using LinearAlgebra
using Random
using Distributions

#include("helpers.jl")
#include("simple.jl")

# *******************************************************************************
# OPTIMIZATION FUNCTION DEFNITIONS
# *******************************************************************************
basis(i,n) = [k == i ? 1.0 : 0.0 for k in 1:n]

function hooke_jeeves(f, x, y, α, n, γ=0.5)
    improved = false
    x_best, y_best = x, y
    for i in 1 : n
        for sgn in (-1,1)
            x′ = x + sgn*α*basis(i, n)
            y′ = f(x′)
            if y′ < y_best
                x_best, y_best, improved = x′, y′, true
            end
        end
    end
    x, y = x_best, y_best

    if !improved
        α *= γ
    end
    return x, y, α
end

function rand_positive_spanning_set(α, n)
    δ = round(Int, 1 / sqrt(α))
    L = Matrix(Diagonal(δ * rand([1,-1], n)))
    for i in 1:n - 1
        for j in 1:i - 1
            L[i,j] = rand(-δ + 1:δ - 1)
        end
    end
    D = L[randperm(n),:]
    D = D[:,randperm(n)]
    D = hcat(D, -sum(D, dims=2))
    return [D[:,i] for i in 1:n + 1]
end

function mesh_adaptive_direct_search(f, x, y, α, n)
    improved = false
    for (i, d) in enumerate(rand_positive_spanning_set(α, n))
        x′ = x + α * d
        y′ = f(x′)
        if y′ < y
            x, y, improved = x′, y′, true
            x′ = x + 3 * α * d
            y′ = f(x′)
            if y′ < y
                x, y = x′, y′
            end
            break
        end
    end
    α = improved ? min(4 * α, 1) : α / 4
    return x, y, α
end

function cross_entropy_method(f,P,m=100,m_elite=10)
    #for k in 1:k_max
    samples = rand(P,m)
    order = sortperm([f(samples[:,i]) for i in 1:m])
    P = fit(typeof(P), samples[:,order[1:m_elite]])
    return P
end

function penalty(c, x, ρ1, ρ2)
    evals = c(x)
    max_arr = [max(evals[i] + 1, 0) for i = 1:length(evals)]
    greater_than_zero = [evals[i] > 0 for i = 1:length(evals)]
    max_arr = max_arr.^2
    return ρ1*sum(max_arr) + ρ2*sum(greater_than_zero)
end

function optimize_simple1(f, g, c, x, num_evals; ρ1=0.9, ρ2=1, γ=1.1)
    α, y, n = 1, f(x) + penalty(c, x, ρ1, ρ2), length(x)
    ϵ = 0.000000001
    while α > ϵ && count(f, g, c) < num_evals
        #x, y, α = mesh_adaptive_direct_search(x -> f(x) + penalty(c, x, ρ1, ρ2), x, y, α, n)
        x, y, α = hooke_jeeves(x -> f(x) + penalty(c, x, ρ1, ρ2), x, y, α, n, 0.5)
        ρ1 *= γ; ρ2 *= γ
        if penalty(c, x, ρ1, ρ2) == 0
            return x   
        end
    end
    return x
end

function optimize_simple2(f, g, c, x, num_evals; ρ1=50, ρ2=1, γ=0.9)
    α, y, n = 1, f(x) + penalty(c, x, ρ1, ρ2), length(x)
    ϵ = 0.000000001
    while α > ϵ && count(f, g, c) < num_evals
        #x, y, α = mesh_adaptive_direct_search(x -> f(x) + penalty(c, x, ρ1, ρ2), x, y, α, n)
        x, y, α = hooke_jeeves(x -> f(x) + penalty(c, x, ρ1, ρ2), x, y, α, n, 0.5)
        ρ1 *= γ; ρ2 *= γ
        if penalty(c, x, ρ1, ρ2) == 0
            return x   
        end
    end
    return x
end

function optimize_simple3(f, g, c, x, num_evals; ρ1=10, ρ2=1, γ=1.1)
    α, y, n = 1, f(x) + penalty(c, x, ρ1, ρ2), length(x)
    ϵ = 0.000000001
    while α > ϵ && count(f, g, c) < num_evals
        x, y, α = mesh_adaptive_direct_search(x -> f(x) + penalty(c, x, ρ1, ρ2), x, y, α, n)
        ρ1 *= γ; ρ2 *= γ
        if penalty(c, x, ρ1, ρ2) == 0
            return x   
        end
    end
    return x
end

#=
function optimize_secret1(f, g, c, x, num_evals; ρ1=50, ρ2=1, γ=1.1)
    α, y, n = 1, f(x) + penalty(c, x, ρ1, ρ2), length(x)
    ϵ = 0.000000001
    while α > ϵ && count(f, g, c) < num_evals
        #x, y, α = mesh_adaptive_direct_search(x -> f(x) + penalty(c, x, ρ1, ρ2), x, y, α, n)
        x, y, α = hooke_jeeves(x -> f(x) + penalty(c, x, ρ1, ρ2), x, y, α, n, 0.5)
        ρ1 *= γ; ρ2 *= γ
        if penalty(c, x, ρ1, ρ2) == 0
            return x   
        end
    end
    return x
end
=#
function optimize_secret1(f, g, c, x, num_evals; ρ1=50, ρ2=1, γ=1.1)
    α, y, n = 1, f(x) + penalty(c, x, ρ1, ρ2), length(x)
    μ = x; Σ = 10
    P = MvNormal(μ, Σ)
    m = 50
    while count(f, g, c) < num_evals - 3*m
        #x, y, α = mesh_adaptive_direct_search(x -> f(x) + penalty(c, x, ρ1, ρ2), x, y, α, n)
        P = cross_entropy_method(x -> f(x) + penalty(c, x, ρ1, ρ2), P, m, 1)
        #x, y, α = hooke_jeeves(x -> f(x) + penalty(c, x, ρ1, ρ2), x, y, α, n, 0.5)
        ρ1 *= γ; ρ2 *= γ
    end
    return P.μ
end


function optimize_secret2(f, g, c, x, num_evals; ρ1=10, ρ2=1, γ=1.1)
    α, y, n = 1, f(x) + penalty(c, x, ρ1, ρ2), length(x)
    ϵ = 0.000000001
    while α > ϵ && count(f, g, c) < num_evals
        x, y, α = mesh_adaptive_direct_search(x -> f(x) + penalty(c, x, ρ1, ρ2), x, y, α, n)
        #x, y, α = hooke_jeeves(x -> f(x) + penalty(c, x, ρ1, ρ2), x, y, α, n, 0.5)
        ρ1 *= γ; ρ2 *= γ
        if penalty(c, x, ρ1, ρ2) == 0
            return x   
        end
    end
    return x
end


# *******************************************************************************
# MAIN OPTIMIZATION FUNCTION
# *******************************************************************************
function optimize(f, g, c, x0, n, prob)
    if prob == "simple1"
        x_best = optimize_simple1(f, g, c, x0, n)
    elseif prob == "simple2"
        x_best = optimize_simple2(f, g, c, x0, n)
    elseif prob == "simple3"
        x_best = optimize_simple3(f, g, c, x0, n)
    elseif prob == "secret1"
        x_best = optimize_secret1(f, g, c, x0, n)
    elseif prob == "secret2"
        x_best = optimize_secret2(f, g, c, x0, n)
    else
        print("Invalid Problem")
    end
end
#=
        len = length(x0)
        if len == 1
            x_best = optimize_secret1(f, g, c, x0, n)
        else
            x_best = optimize_secret2(f, g, c, x0, n)
        end
    end
    return x_best
end
=#
#my_x = optimize(simple2, simple2_gradient, simple2_constraints, simple2_init(), 2000, "simple2")