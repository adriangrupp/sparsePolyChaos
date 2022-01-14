export subspacePursuit

"""
Subspace pursuit core algorithm
Parameters:
    * Ψ - Regression matrix
    * Y - Experimental design
    * K - Sparsity parameter (target basis size)
Returns:
    * A_old - Set of K best regressors
    * c - PCE coefficients wrt A_old
    * err - error
"""
function subspacePursuit(Ψ::AbstractMatrix{<:Real}, Y::AbstractVector{<:Real}, K::Int)
    N, P = size(Ψ)

    # Check size of K
    if K > floor(min(N / 2, P / 2))
        @warn("subspace_pursuit: the specified K = $K is too large, set to ", floor(min(N / 2, P / 2)))
        K = floor(min(N / 2, P / 2))
    end
end


subspacePursuit([1 1; 2 4], [1, 2, 3], 2)