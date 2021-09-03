using sparsePolyChaos
using PolyChaos

maxDeg = 10
interactions = 1

## Define the model
# μ = 0
# σ = 1
# model(x) = exp(μ + σ * x)
# model(x) = x^7 - 21 * x^5 + 105 * x^3 - 105 * x + x^4 - 6 * x^2 + 3 # He7 + He4
model(x) = x^4 - 3* x^2 
# model(x) = x^10 - 45 * x^8 + 630 * x^6 - 3150 * x^4 + 4725 * x^2 - 945
# model(x) = 0


## Build the full model
op = GaussOrthoPoly(maxDeg)


## Compute sparse PCE solution
pce, Ap, p, R², Q² = sparsePCE(op, model; pMax = maxDeg, jMax = interactions)

println("Basis polynomials: ", Ap)
println("PCE coefficients: ", pce)
# println("Max degree:: ", p)
println("R² error: ", R²)
println("Q² error: ", Q²)



## Sparse PCE with custom sampling method

println("\n\n== Sparse PCE with custom sampling method ==")
function sampleFromGaussianMixture(n::Int,μ::Vector{},σ::Vector{},w::Vector{})
    X = Float64[]
    for i in 1:n
        k = findfirst(x -> x > rand(), cumsum(w))
        push!(X, μ[k] + σ[k]*randn())
    end
    return X
end

function ρ_gauss(x,μ,σ)
    1 / sqrt(2*π*σ^2) * exp(-(x - μ)^2 / (2σ^2))
end


μ, σ, w = [2.1, 3.2], [0.3, 0.4], [0.3, 0.7]
ρ(x) = sum( w[i]*ρ_gauss(x,μ[i],σ[i]) for i in 1:length(w) )
meas = Measure("my_GaussMixture", ρ, (-Inf,Inf), false, Dict(:μ=>μ,:σ=>σ,:w=>w)) # build measure
op_gm = OrthoPoly("my_op",maxDeg,meas;Nquad=150,Nrec = 5*maxDeg, discretization=stieltjes) # construct orthogonal polynomial

sampleFun(sampleSize) = sampleFromGaussianMixture(sampleSize,μ,σ,w)
pce, Ap, p, R², Q² = sparsePCE(op_gm, model, sampleFun; pMax = maxDeg)

println("Basis polynomials: ", Ap)
println("PCE coefficients: ", pce)
# println("Max degree:: ", p)
println("R² error: ", R²)
println("Q² error: ", Q²)


#TODO: Comparison to monte carlo 
# # ------- Monte-Carlo -------
# # Evaluate model function on same set of previously drawn samples X
# y_mc = model.(X)