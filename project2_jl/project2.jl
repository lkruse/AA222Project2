# *******************************************************************************
# PACKAGES
# *******************************************************************************
using LinearAlgebra
using Random

include("helpers.jl")
include("simple.jl")


# *******************************************************************************
# OPTIMIZATION FUNCTION DEFNITIONS
# *******************************************************************************
# Create an abstract type for first-order descent methods
abstract type FirstOrderMethod end

# *******************************************************************************
# Adam Implementation
mutable struct Adam <: FirstOrderMethod
    α; γv; γs; ϵ; k; v; s
end

adam_init!(A::Adam) = A

function adam_step!(A::Adam, ∇f, x)
    α, γv, γs, ϵ, k, s, v, g = A.α, A.γv, A.γs, A.ϵ, A.k, A.s, A.v, ∇f(x)
    v[:] = γv * v + (1 - γv) * g
    s[:] = γs * s + (1 - γs) * g .* g
    A.k = k += 1
    v̂ = v ./ (1 - γv^k)
    ŝ = s ./ (1 - γs^k)
    return x - α * v̂ ./ (sqrt.(ŝ) .+ ϵ)
end

basis(i,n) = [k == i ? 1.0 : 0.0 for k in 1:n]

function hooke_jeeves(f, x, y, α, n, γ=0.5)
    #n = length(x)
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
    # α,y,n = 1, f(x), length(x)

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

function penalty(c, x, ρ1, ρ2)
    evals = c(x)
    max_arr = [max(evals[i] + 1, 0) for i = 1:length(evals)]
    greater_than_zero = [evals[i] > 0 for i = 1:length(evals)]
    max_arr = max_arr.^2
    return ρ1*sum(max_arr) + ρ2*sum(greater_than_zero)
end


function optimize_simple1(f, g, c, x, num_evals; ρ1=0.9, ρ2=0, γ=1.1)
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

function optimize_secret1(f, g, c, x, num_evals; ρ1=10, ρ2=1, γ=1.1)
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


function optimize_secret2(f, g, c, x, num_evals; ρ1=10, ρ2=1, γ=1.1)
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


# *******************************************************************************
# MAIN OPTIMIZATION FUNCTION
# *******************************************************************************
function optimize(f, g, c, x0, n, prob)
    if prob == "simple1"
        x_best = optimize_simple1(f, g, c, x0, n)
        # x_best = optimize_simple1(f, g, c, x0, n)
    elseif prob == "simple2"
        x_best = optimize_simple2(f, g, c, x0, n)
    elseif prob == "simple3"
        x_best = optimize_simple3(f, g, c, x0, n)
    else
        len = length(x0)
        if len == 2
            x_best = optimize_secret1(f, g, c, x0, n)
        else
            x_best = optimize_secret2(f, g, c, x0, n)
        end
    end
    return x_best
end

#my_x = optimize(simple2, simple2_gradient, simple2_constraints, simple2_init(), 2000, "simple2")