module sparsePolyChaos

using PolyChaos
using LinearAlgebra: diag, cond, norm, pinv

include("error_estimation.jl")
include("regression.jl")
include("sparse_pce.jl")
include("sparse_pce_mult.jl")
include("subspace_pursuit.jl")

end # module
