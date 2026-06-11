using BosonSampling,LinearAlgebra,Random
using Plots,Statistics,Distributions
using ProgressMeter, Permanents
using LaTeXStrings

pyplot() # Switch to the PyPlot backend

function glynn_estimator(A::AbstractMatrix, x::AbstractVector)
    n = size(A, 1)
    if size(A, 2) != n
        throw(ArgumentError("Matrix A must be square."))
    end
    

    if length(x) != n
        throw(ArgumentError("Vector x must have the same length as the dimensions of A."))
    end

    prod_x = prod(x)

    Ax = A * x

    prod_Ax = prod(Ax)

    return prod_x * prod_Ax
end

function estimate_permanent(A::AbstractMatrix; num_samples=10^4)
    n = size(A, 1)
    if size(A, 2) != n
        throw(ArgumentError("Matrix A must be square."))
    end

    estimates = 0

    for i in 1:num_samples
        estimates += glynn_estimator(A, rand([-1, 1], n))
    end

    return estimates/num_samples
end

function U_tot(U,ϕ)
    m=size(U,1)
    U₂=cat(U,U,dims=(1,2))
    S₂ = Matrix(I,2m,2m)[vcat(m+1:2m, 1:m),:] ## swap the first m rows with the last m rows 
    Φ=cat(exp(im*diagm(ϕ)),Matrix(I,m,m),dims=(1,2))

    return U₂'*Φ*S₂*U₂
end

function CF(ϕ,U,x;Samples=10^6)
    m=size(U,1)
    G=(1-x)*Matrix(I,2m,2m)+x*ones(2m,2m)
    return estimate_permanent(U_tot(U,ϕ).*G ;num_samples=Samples)/estimate_permanent(U_tot(U,zeros(m)).*G ;num_samples=Samples)
end

function calc_n1n2(δ,U,x; num_samples=10^6)
    # Calculate the common denominator
    m=size(U,1)
    denom = 4 * δ^2
    
    # Evaluate chi for the four different sign combinations.
    # The third argument is hardcoded to 0, and args... handles the rest.
    chi_1 = -CF([δ;δ;zeros(m-2)],U,x;Samples=num_samples)
    chi_2 = +CF([-δ;δ;zeros(m-2)],U,x;Samples=num_samples)
    chi_3 = +CF([δ;-δ;zeros(m-2)],U,x;Samples=num_samples)
    chi_4 = -CF([-δ;-δ;zeros(m-2)],U,x;Samples=num_samples)

    # Combine the terms according to the formula
    numerator = chi_1 + chi_2 + chi_3 + chi_4
    
    return real(numerator / denom)
end


function compute_density_correlation(i::Integer, j::Integer, U::AbstractMatrix, S::AbstractMatrix)
    n = size(U, 2)
    
    # Ensure S has the correct dimensions to match the sum limits
    if size(S, 1) < n || size(S, 2) < n
        throw(ArgumentError("Matrix S must be at least $n x $n (where n is the number of columns in U)"))
    end

    # Type stability: infer the correct return type based on the input matrices
    T = promote_type(eltype(U), eltype(S))

    # --- Term 1: δ_ij Σ |U_ik|^2 ---
    # The Kronecker delta δ_ij means this is only non-zero if i == j
    term1 = zero(real(T))
    if i == j
        for k in 1:n
            term1 += abs2(U[i, k])
        end
    end

    # --- Term 2: Σ_{k ≠ l} (...) ---
    term2 = zero(T)
    for l in 1:n
        for k in 1:n
            if k != l
                # |U_ik|^2 |U_jl|^2
                part_a = abs2(U[i, k]) * abs2(U[j, l])
                
                # |S_kl|^2 U_ik^* U_il U_jl^* U_jk
                part_b = abs2(S[k, l]) * conj(U[i, k]) * U[i, l] * conj(U[j, l]) * U[j, k]
                
                term2 += part_a + part_b
            end
        end
    end

    return real(term1 + term2)
end

function Thermal_Gram(v)

    G=zeros(length(v),length(v))
    
    for i=1:length(v)
        for j=1:length(v)
            if v[i]==v[j]
                G[i,j]=1 
            end
        end
    end

    return G
end

function CF_thermal(ϕ,U,β;Samples=10^6)

    m=size(U,1)
    V_num=U_tot(U,ϕ)
    V_den=U_tot(U,zeros(m))

    num,den=0,0
    p = 1 - exp(-β)
    bosonic_dist = Geometric(p)

    for i=1:Samples
        states=rand(bosonic_dist, 2m)
        num += glynn_estimator(V_num.*Thermal_Gram(states), rand([-1, 1], 2m))
        den += glynn_estimator(V_den.*Thermal_Gram(states), rand([-1, 1], 2m))
    end    
    num=num/Samples
    den=den/Samples

    return num/den
end

function calc_n1n2_thermal(δ,U,β; num_samples=10^6)
   
    m=size(U,1)
    denom = 4 * δ^2
    
    chi_1 = -CF_thermal([δ;δ;zeros(m-2)],U,β;Samples=num_samples)
    chi_2 = +CF_thermal([-δ;δ;zeros(m-2)],U,β;Samples=num_samples)
    chi_3 = +CF_thermal([δ;-δ;zeros(m-2)],U,β;Samples=num_samples)
    chi_4 = -CF_thermal([-δ;-δ;zeros(m-2)],U,β;Samples=num_samples)
    
    numerator = chi_1 + chi_2 + chi_3 + chi_4
    
    
    return real(numerator / denom)
end

println("Code correctly loaded.")