using sparsePolyChaos
using PolyChaos

###
### Multivariate sparse PCE
###
maxDeg = 10
interactions = 3


model(x,y) = (x^4 - 6 * x^2  + 3) + (x^2 - 1) * (y^2 - 1) + (y^3 - 3 * y) # He_4(x) + He_2(x) * He_2(y) + He_3(y)
# model(x,y) = (x^2 - 1) * (y^2 - 1) # He2(x) * He2(y)
# model(x,y) = (x^2 - 1) # He2(x)


### Setup and compute PCE coefficients of x ###
k = 2
op = GaussOrthoPoly(maxDeg)
mop = MultiOrthoPoly([op for i in 1:k], maxDeg)
L = dim(mop)

pce, Ap, p, R², Q² = sparsePCE(mop, model; pMax = maxDeg, jMax = interactions)

println("Basis polynomials: ", Ap)
println("PCE coefficients: ", pce)
println("Max degree:: ", p)
println("R² error: ", R²)
println("Q² error: ", Q²)