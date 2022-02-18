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
        @warn("@SP: the specified K = $K is too large, set to ", floor(min(N / 2, P / 2)))
        K = floor(Int, min(N / 2, P / 2))
    end

    # Hybrid?

    # Normalization?

    # Check for constant regressors?

    ## Initialization
    max_iterations = P
    counter = 0

    S_old = sortperm(abs.(Ψ' * Y), rev = true) # Indices of active set (K best polynomials)
    S_old = S_old[1:K]
    println("@SP: Initial regressor set: $S_old")

    c_old = leastSquares(Ψ[:, S_old], Y)       # PCE Coefficients for active basis
    r_old = Y - (Ψ[:, S_old] * c_old)          # Residual between ED and PCE approx with active Set

    S_new = []  # Iteration result
    c_new = []  # Iteration result

    ## Main loop
    while true
        counter += 1
        # println("current residuals: $r_old")

        S_add = sortperm(abs.(Ψ' * r_old), rev = true) # Add K new regressors, best correlated with residual
        # add = sort(abs.(Ψ' * r_old), rev = true) # Add K new regressors, best correlated with residual
        # println("add: $add")
        # println("S_add: $S_add")
        S_add = S_add[1:K]
        c_temp = zeros(P)
        c_temp[[S_old; S_add]] = leastSquares(Ψ[:, [S_old; S_add]], Y) # PCE coefficients for 2K regressors

        S_new = sortperm(abs.(c_temp), rev = true) # New active set with K new best regressors
        # new = sort(abs.(Ψ' * r_old), rev = true) # Add K new regressors, best correlated with residual
        # println("new: $new")
        # println("S_new: $S_new")
        S_new = S_new[1:K]
        c_new = leastSquares(Ψ[:, S_new], Y) # coefficients for new active set
        r_new = Y - (Ψ[:, S_new] * c_new) # Residual between ED and PCE approx with active Set

        println("@SP: Iteration $counter. New regressor set: $S_old")

        # Check termination conditions
        if all(sort(S_new) == sort(S_old))
            println("@SP: Set of regressors converged. STOP.\n")
            break
        end
        if norm(r_new) > norm(r_old)
            println("@SP: Last iteration deteriorated the solution. Taking previous solution. STOP.\n")
            S_new = S_old
            c_new = c_old
            break
        end
        if counter == max_iterations
            println("@SP: Reached maximum number of iterations ($max_iterations). STOP.\n")
            break
        end

        # Iteration continues: update values
        S_old = S_new
        c_old = c_new
        r_old = r_new

    end

    # Postprocessing for hybrid, normalized and constant

    coeffs = zeros(P)
    coeffs[S_new] = c_new
    return coeffs, S_new
end


"""
Estimation of K by cross-validation.
"""
function estimateK()

end


# subspacePursuit([1 1 -4; 2 4 -6; 0 -9 1], [1, -2, -3], 2)