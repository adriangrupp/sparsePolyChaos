using sparsePolyChaos
using PolyChaos

# Example file for subspace pursuit algorithm
K = 2 # SP hyper parameter default
maxDeg = 10 # max degree of full basis


### UNIVARIATE ###

# 0. define sample model
model(x) = x^7 - 21 * x^5 + 105 * x^3 - 105 * x + x^4 - 6 * x^2 + 3 # He7 + He4
# model(x) = x^10 - 45 * x^8 + 630 * x^6 - 3150 * x^4 + 4725 * x^2 - 945 # He10

# 1. setup basis
op = GaussOrthoPoly(maxDeg)

# 2. sample for experimental design X and model response Y
sampleSize = dim(op) * 3 # N = 3P
X = sampleMeasure(sampleSize, op)
Y = model.(X)

# 3. build full regression matrix (maybe put to SP algo)
Ψ = [evaluate(j, X[i], op) for i = 1:sampleSize, j = 0:maxDeg]

# 4- run SP algo
coefficients, regressors = subspacePursuit(Ψ, Y, K)


### MULTIVARIATE ###