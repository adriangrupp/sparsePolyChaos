using sparsePolyChaos
using PolyChaos

# Example file for subspace pursuit algorithm.


### UNIVARIATE ###

maxDeg = 10 # max degree of full basis
K = 2       # SP hyper parameter default
ρ = 3       # ED-size factor

function univariateSP()

    # 0. define sample model
    model(x) = x^7 - 21 * x^5 + 105 * x^3 - 105 * x + x^5 - 10 * x^3 + 15 * x # He7 + He5
    # model(x) = x^7 - 21 * x^5 + 105 * x^3 - 105 * x + x^4 - 6 * x^2 + 3 # He7 + He4
    # model(x) = x^10 - 45 * x^8 + 630 * x^6 - 3150 * x^4 + 4725 * x^2 - 945 # He10 - Bad conditioning!

    # 1. setup basis
    op = GaussOrthoPoly(maxDeg)

    # 2. sample for experimental design X and model response Y
    sampleSize = dim(op) * ρ # N = ρ*P
    X = sampleMeasure(sampleSize, op)
    Y = model.(X)

    # 3. build full regression matrix (maybe put to SP algo)
    Ψ = [evaluate(j, X[i], op) for i = 1:sampleSize, j = 0:maxDeg]

    # 4. run SP algo
    coefficients, regressors = subspacePursuit(Ψ, Y, K)
    return coefficients, regressors
end


### MULTIVARIATE ###

# Χ²- Distribution (quadratic model with k gaussian input uncertainties)
maxDeg = 5 # max degree of full basis
K = 20     # hyperparameter for SP
k = 3      # degrees of freedom
ρ = 3      # ED-size factor

# Model equation: Y = X^2
function model(X)
    @assert length(X) == k "Degrees of freedom is $k, but $(length(X)) inputs given"
    sum(x^2 for x in X)
end

function multivariateSP()
    # 1. Setup and compute multivariate basis
    op = GaussOrthoPoly(maxDeg, Nrec = 2 * maxDeg, addQuadrature = true)
    mop = MultiOrthoPoly([op for i in 1:k], maxDeg)
    ind = mop.ind

    # 2. sample for experimental design X and model response Y
    sampleSize = maxDeg * k * ρ # N = ρ*P
    X = sampleMeasure(sampleSize, mop)
    Y = model.(eachrow(X)) # evaluate model with row-wise elements of ED matrix

    # 3. build full regression matrix (maybe put to SP algo)
    Ψ = evaluate(X, mop)' # Transpose due to switched dimensions in PolyChaos

    # 4. run SP algo
    coeffs, regressors = subspacePursuit(Ψ, Y, K)


    ## Analysis of results
    println()

    # Analytic moments
    mean_ana = k
    std_ana = sqrt(2 * k)

    # Compare sparse moments to analytic moments
    mean_sp = mean(coeffs, mop)
    std_sp = std(coeffs, mop)
    error_mean_sp = abs(mean_ana - mean_sp)
    error_std_sp = abs(std_ana - std_sp)
    print("Expected value:\t\t$(mean_ana) = $(mean_sp)\n")
    print("\t\t\terror = $(error_mean_sp)\n")
    print("Standard deviation:\t$(std_ana) = $(std_sp)\n")
    print("\t\t\terror = $(error_std_sp)\n")
end

multivariateSP()