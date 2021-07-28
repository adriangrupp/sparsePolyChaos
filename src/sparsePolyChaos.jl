module sparsePolyChaos

using PolyChaos
using LinearAlgebra: diag, cond

include("error_estimation.jl")
include("regression.jl")
include("sparse_pce.jl")
include("sparse_pce_mult.jl")

end # module
