using sparsePolyChaos
using PolyChaos

maxDeg = 10
interactions = 1

## Define the model
# μ = 0
# σ = 1
# model(x) = exp(μ + σ * x)
# model(x) = x^7 - 21 * x^5 + 105 * x^3 - 105 * x + x^4 - 6 * x^2 + 3 # He7 + He4
# model(x) = x^4 - 3* x^2 
model(x) = x^10 - 45 * x^8 + 630 * x^6 - 3150 * x^4 + 4725 * x^2 - 945
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