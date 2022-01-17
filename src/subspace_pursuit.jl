export subspacePursuit

"""
Subspace pursuit core algorithm.
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
    @assert size(Ψ, 1) == length(Y) "Dimension mismatch in matrix Ψ and ED vector Y."
    N, P = size(Ψ)

    # Check size of K. Has to be: 2K < min{N,P}
    if K > floor(min(N / 2, P / 2))
        @warn("subspace_pursuit: the specified K = $K is too large, set to ", floor(min(N / 2, P / 2)))
        K = floor(Int, min(N / 2, P / 2))
    end

    # Hybrid?

    # Normalization?

    # Check for constant regressors?

    ## Initialization
    max_iterations = P
    counter = 0

    # S_old = sort(abs.(Ψ'*Y), rev=true) # Active set
    S_old = sortperm(abs.(Ψ' * Y), rev = true)  # Indices of active set (k best polynomials)
    S_old = S_old[1:K]
    c_old = leastSquares(Ψ[:, S_old], Y)         # PCE Coefficients for active basis
    r_old = Y - (Ψ[:, S_old] * c_old)          # Residual between ED and PCE approx with active Set

    ## Main loop
    while false
        counter += 1


    end
end


"""
Estimation of K by cross-validation.
"""
function estimateK()

end


subspacePursuit([1 1 -4; 2 4 -6; 0 -9 1], [1, -2, -3], 2)