# *******************************************************************************
# PACKAGES
# *******************************************************************************
using LinearAlgebra
using Random
using Distributions

# *******************************************************************************
# OPTIMIZATION FUNCTION DEFNITIONS
# *******************************************************************************
# A function for constructing a basis vector
basis(i,n) = [k==i ? 1.0 : 0.0 for k in 1:n]

# A function to perform a single step of the Hooke-Jeeves optimization method
function hooke_jeeves(f, x, y, α, n, γ = 0.5)
    improved = false
    x_best, y_best = x, y
    for i in 1:n
        for sign in (-1, 1)
            x′ = x + sign*α*basis(i,n)
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

# A function to randomly sample a positive spanning set according to mesh
# adaptive direct search
function random_positive_spanning_set(α, n)
    δ = round(Int, 1/sqrt(α))
    lower_triangular_matrix = Matrix(Diagonal(δ*rand([1,-1], n)))
    for i in 1:(n-1)
        for j in 1:(i-1) 
            lower_triangular_matrix[i,j] = rand(-δ + 1:(δ-1))
        end
    end
    D = lower_triangular_matrix[randperm(n),:]
    D = D[:, randperm(n)]
    D = hcat(D, -sum(D,dims=2))
    return [D[:,i] for i in 1:(n+1)]
end 

# A function to perform a single step of mesh adaptive direct search
function mesh_adaptive_direct_search(f, x, y, α, n)
    improved = false
    for (i,d) in enumerate(random_positive_spanning_set(α, n))
        x′ = x + α*d
        y′ = f(x′)
        if y′ < y
            x, y, improved = x′, y′, true
            x′ = x + 3*α*d
            y′ = f(x′)
            if y′ < y
                x, y = x′, y′
            end
            break
        end
    end
    α = improved ? min(2*α, 1) : α/2
    return x, y, α
end

# A function to perform a single step of the cross entropy method
function cross_entropy(f, P, m=100, m_elite=10)
    samples = rand(P,m)
    order = sortperm([f(samples[:,i]) for i in 1:m])
    P = fit(typeof(P), samples[:, order[1:m_elite]])
    return P
end

# A penalty method mixture between a count and a quadratic penalty function
function penalty(c, x, ρ1, ρ2, δ)
    evals = c(x)
    max_arr = [max(evals[i] + δ, 0) for i = 1:length(evals)]
    greater_than_zero = [evals[i] > 0 for i = 1:length(evals)]
    max_arr = max_arr.^2
    return ρ1*sum(max_arr) + ρ2*sum(greater_than_zero)
end

# HJ parameters: ρ1=1.01, ρ2=0, γ=1.001, δ=0.5; 0.1378383301
# MADS parameters: ρ1=1.01, ρ2=0, γ=1.001, δ=0.5; 0.1395356843
function optimize_simple1(f, g, c, x, num_evals; ρ1=1.01, ρ2=0, γ=1.001, δ=0.5)
    α, y, n, ϵ = 1, f(x) + penalty(c, x, ρ1, ρ2, δ), length(x), 1e-9
    while α > ϵ && count(f, g, c) < (num_evals-8)
        #x, y, α = mesh_adaptive_direct_search(x -> f(x) + penalty(c, x, ρ1, ρ2, δ), x, y, α, n)
        x, y, α = hooke_jeeves(x -> f(x) + penalty(c, x, ρ1, ρ2, δ), x, y, α, n, 0.5)
        ρ1 *= γ; ρ2 *= γ
    end
    return x
end

# HJ parameters: ρ1=100, ρ2=1, γ=1.1, δ=0.75; 1.4993756041
# MADS parameters: ρ1=100, ρ2=1, γ=1.01, δ=1.0; 1.5604945284
function optimize_simple2(f, g, c, x, num_evals; ρ1=100, ρ2=1, γ=1.1, δ=0.75)
    α, y, n, ϵ = 1, f(x) + penalty(c, x, ρ1, ρ2, δ), length(x), 1e-9
    while α > ϵ && count(f, g, c) < (num_evals-8)
        #x, y, α = mesh_adaptive_direct_search(x -> f(x) + penalty(c, x, ρ1, ρ2, δ), x, y, α, n)
        x, y, α = hooke_jeeves(x -> f(x) + penalty(c, x, ρ1, ρ2, δ), x, y, α, n, 0.5)
        ρ1 *= γ; ρ2 *= γ
    end
    return x
end

# HJ parameters: ρ1=1, ρ2=1, γ=1.01, δ=0.5; 0.0631003421
# MADS parameters: ρ1=1, ρ2=1, γ=1.01, δ=0.25; 0.0425930332
function optimize_simple3(f, g, c, x, num_evals; ρ1=1, ρ2=1, γ=1.01, δ=0.25)
    α, y, n, ϵ = 1, f(x) + penalty(c, x, ρ1, ρ2, δ), length(x), 1e-9
    while α > ϵ && count(f, g, c) < (num_evals-13)
        x, y, α = mesh_adaptive_direct_search(x -> f(x) + penalty(c, x, ρ1, ρ2, δ), x, y, α, n)
        #x, y, α = hooke_jeeves(x -> f(x) + penalty(c, x, ρ1, ρ2, δ), x, y, α, n, 0.5)
        ρ1 *= γ; ρ2 *= γ
    end
    return x
end

# ρ1=50, ρ2=1, γ=1.01, δ=0.15, m = 100; num_elite = 20, μ=x; Σ=10; -0.5935202993
function optimize_secret1(f, g, c, x, num_evals; ρ1=50, ρ2=1, γ=1.01, δ=0.15)
    μ = x; Σ = 10; P = MvNormal(μ, Σ)
    m = 100; num_elite = 20
    while count(f, g, c) < (num_evals - 2*m)
        P = cross_entropy(x -> f(x) + penalty(c, x, ρ1, ρ2, δ), P, m, num_elite)
        ρ1 *= γ; ρ2 *= γ
    end
    return P.μ
end

# HJ parameters:  ρ1=1, ρ2=1, γ=1.001, δ=0.15; 0.657090133
# MADS parameters:  ρ1=1, ρ2=1, γ=1.001, δ=0.15; 0.3410080274
function optimize_secret2(f, g, c, x, num_evals; ρ1=1, ρ2=1, γ=1.001, δ=0.15)
    α, y, n, ϵ = 0.1, f(x) + penalty(c, x, ρ1, ρ2, δ), length(x), 1e-9
    while α > ϵ && count(f, g, c) < (num_evals - 308)
        x, y, α = mesh_adaptive_direct_search(x -> f(x) + penalty(c, x, ρ1, ρ2, δ), x, y, α, n)
        #x, y, α = hooke_jeeves(x -> f(x) + penalty(c, x, ρ1, ρ2, δ), x, y, α, n, 0.5)
        ρ1 *= γ; ρ2 *= γ
    end
    return x
end

function test_call()
    print("HELLO WORLD")
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