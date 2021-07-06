using Base: Float64
## This is an example on the usage of PolyChaos for non-intrusive PCE using
## the projection and the regression approach and a comarison of them in a multi-
## variate setting.
## Model: f = ΣX^2, X = {X_1, ..., X_k}, X_i ~ N(0, 1), Y ~ Χ²(k)

using PolyChaos
using LinearAlgebra


### Parameters ###
μ, σ = 0, 1
maxdeg = 3
Nrec = 20
k = 3 # degrees of freedom

### Model equation ###
function model(X)
    @assert length(X) == k "Degrees of freedom is $k, but $(length(X)) inputs given"
    println(X)
    sum(x^2 for x in X)
end

### Setup and compute PCE coefficients of x ###
op = GaussOrthoPoly(maxdeg, Nrec=Nrec, addQuadrature=true)
mop = MultiOrthoPoly([op for i in 1:k], maxdeg)
ind = mop.ind

# Find PCE for all X_i
L = dim(mop)
# assign2multi(): take coefficients x for univariate elements and assign then to their accoring multi index. 
# If x has 1 element, only take degree 1 basis elements, if it has 2 take the first 2 etc.
x = [ assign2multi(convert2affinePCE(μ, σ, op), i, mop.ind) for i in 1:k ]
# we can describe a gaussian RV with ~N(0,1) by PCE coefficients x_0 = 1 and x_1 = 1

# Compute tensors <ϕ_m,ϕ_m> and <ϕ_1,ϕ_2,ϕ_m>
t2 = Tensor(2, mop)
T2 = [t2.get([i,i]) for i = 0:L-1]
t3 = Tensor(3, mop)


# ------- Reference -------
### Intrusive PCE ###
println("\t == Galerking Projection ==")
# x[i][j] - i = which variable, j = coefficient for degree j
y_intr = [ sum( x[i][j1] * x[i][j2] * t3.get([j1-1, j2-1, m-1]) for i = 1:k, j1 = 1:L, j2 = 1:L ) / t2.get([m-1, m-1]) for m in 1:L ]
# y_intr = [ sum( x[i][j1] * x[i][j2] * t3.get([j1-1, j2-1, m-1]) for i = 1:k, j1 = 1:L, j2 = 1:L ) for m in 1:L ]
println(y_intr)
println()

### Analytic ###
# ...


# ------- Non-intrusive PCE -------
### Projection 
println("\t == Projection approach ==")

# Perform projection via numerical integration
function computeProjection(i::Int, index::AbstractVector{<:Int})
    g(v::AbstractVector{<:Real}) = model(v) * evaluate(index, v, mop)
    γ = t2.get([i,i]) # Normalization of coefficients (-> ortho normal) TODO: Why?
    # integ = integrate(x1 -> integrate(x2 -> integrate(x3 -> g([x1,x2,x3]), mop.uni[3]), mop.uni[2]), mop.uni[1])
    println("foo")
    v = Vector{Real}()
    integ = multi_integral(v, k, g)
    integ / γ
end

function multi_integral(v::AbstractVector{<:Real}, n::Int, g)
    if n == 1
        integrate(x -> g(push!(v,x)), mop.uni[n])
        v = Vector{Real}()
    else
        integrate(x -> multi_integral(push!(v,x), n-1, g), mop.uni[n])
    end
end

y_proj = [computeProjection(i-1, index) for (i, index) in enumerate(eachrow(ind))]
println.(y_proj)
println()

# # Comparison to intrusive solution
print("Comapre coefficients intrusive PCE <-> projection:")
println(norm(y_intr - y_proj, Inf), "\n")
println()


### Regression
println("\t == Regression approach ==")
# Draw n samples, where N > P has to hold, P = basis size
nSamples = maxdeg * k * 10
X = sampleMeasure(nSamples, mop)

# Evaluate model -> vector Y
Y = model.(X)

# Build matrix Φ with ϕ(x(i))
# Φ = Array{Float64}(undef, nSamples, maxdegree+1)
# Φ = [ evaluate(j, X[i], op) for i = 1:nSamples, j = 0:maxdeg]

# Ordinary least squares regression
# y_reg = leastSquares(Φ, Y)
# println.(y_reg)
# println()


# Validation of PCE model
# println("Comapre coefficients analytic <-> regression:")
# println(norm(y_ana - y_reg, Inf))

# genError = empError(Y, Φ, y_reg)
# println("Determination coefficient R² (normalized empicial error): ", 1 - genError)

# ϵLoo = looError(Y, Φ, y_reg)
# println("Determination coefficient Q² (leave-one-out error): ", 1- ϵLoo)
# println()



### Monte-Carlo
# Evaluate model function on same set of previously drawn samples X
# y_mc = model.(X)
#println()



# ------- Comparison of moments -------
println("\n\t Comparison of moments to analytic solution:")

# Analytic moments for y
mean_ana = k
std_ana = sqrt(2*k)
skew_ana = sqrt(8/k)

# PCE skewness
function skew(y)
    e3 = sum( y[i] * y[j] * y[k] * t3.get([i-1, j-1, k-1]) for i = 1:L, j = 1:L, k = 1:L )
    μ = y[1]
    σ = std(y, mop)
    (e3 - 3 * μ * σ^2 - μ^3) / (σ^3)
end

# Intrusive PCE moments
println("= Error Intrusive PCE vs Analytic=")
mean_intr = mean(y_intr, mop)
std_intr  = std(y_intr, mop)
skew_intr = skew(y_intr)
error_mean_intr = abs(mean_ana - mean_intr)
error_std_intr  = abs(std_ana - std_intr)
error_skew_intr = abs(skew_ana - skew_intr)
println("\t\t\t error mean: \t $(error_mean_intr)")
println("\t\t\t error std: \t $(error_std_intr)")
println("\t\t\t error skew: \t $(error_skew_intr)\n")


# # MC moments
# mean_mc, std_mc, skew_mc = mean(y_mc), std(y_mc), skewness(y_mc)
# error_mean_mc = abs(mean_ana - mean_mc)
# error_std_mc = abs(std_ana - std_mc)
# println("\t\t\t error MC, mean: \t $(error_mean_mc)")
# println("\t\t\t error MC, std: \t $(error_std_mc)")

# # Projection moments
# mean_proj = mean(y_proj, op)
# std_proj = std(y_proj, op)
# error_mean_proj = abs(mean_ana - mean_proj)
# error_std_proj = abs(std_ana - std_proj)
# println("\t\t\t error proj, mean: \t $(error_mean_proj)")
# println("\t\t\t error proj, std: \t $(error_std_proj)")

# # Regression moments
# mean_reg = mean(y_reg, op)
# std_reg = std(y_reg, op)
# error_mean_reg = abs(mean_ana - mean_reg)
# error_std_reg = abs(std_ana - std_reg)
# println("\t\t\t error reg, mean: \t $(error_mean_reg)")
# println("\t\t\t error reg, std: \t $(error_std_reg)")


# ------- Plotting of PDF -------
using Plots
Nsmpl = 10000
ξ = sampleMeasure(Nsmpl, mop)

import SpecialFunctions: gamma

### Intrusive PCE ###
samp_intr = evaluatePCE(y_intr, ξ, mop)
histogram(samp_intr; normalize=true, xlabel="t", ylabel="ρ(t)")
# Analytic comparison 
ρ(t) = 1  / (2^(0.5*k) * gamma(.5*k)) * t^(.5*k-1) * exp(-.5*t) # analytic pdf
t = range(.1; stop=maximum(samp_intr), length=100)
plot!(t, ρ.(t), w=4)
