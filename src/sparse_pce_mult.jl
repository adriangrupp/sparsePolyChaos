export sparsePCE

# Compute a sparse basis for the provided pce model and parameters
# Parameters:
#   * op: The full candidate orthogonal basis
function sparsePCE(mop::MultiOrthoPoly, modelFun::Function; Q²tgt=.999, pMax=10, jMax=5)
    # Parameter specification
    pMax = min(mop.deg, pMax)   # pMax can be at most size of given full basis
    dim = length(mop.uni)       # basis dimension
    jMax = min(dim, jMax)       # Maximum interactions is basis dimension at max
    COND = 1e4                  # Maximum allowed matrix condition number (see Blatman2010)
    α = 0.001                   # Tuning parameter (see Blatman2010)
    ϵ = α * (1 - Q²tgt)         # Error threshold of coefficients

    # Initialize basis and determination coefficients
    R² = 0.0    # coefficient of determination
    Q² = 0.0    # leave-one out coefficient determination 
    ind = mop.ind # index of complete basis
    pce = []    # pce coefficients, array of arrays
    p = 0       # max degree
    Ap = []

    # 1. Build initial ED, compute Y
    sampleSize = pMax * 20 # TODO: How to determine?
    X = sampleMeasure(sampleSize, mop)
    Y = [modelFun(x...) for x in eachrow(X)] # This is the most expensive part

    restart = true
    # Outer loop: Iterate on experimental design
    while restart
        restart = false
        
        # 2. Initialize loop
        p = 0
        Ap = [ind[1,:]]    # Set of basis indices for current degree, ind[1] is zero element (array of arrays)
        Φ = [ evaluate(j, X, mop) for j in Ap ] # FIX create matrix directly
        Φ = reduce(hcat, Φ)
        pce = leastSquares(Φ, Y)
        R² = 1 - empError(Y, Φ, pce)
        Q² = 1 - looError(Y, Φ, pce)
        println("** Initialization:")
        println("R²: $R²")
        println("Q²: $Q²")

        # Main loop: Iterate max degree p
        while Q² ≤ Q²tgt && p ≤ pMax && !restart
            p += 1  # current max degree
            candidates = [el for el in eachrow(ind) if sum(el) == p] # only inices with total degree p
            
            # Iterate number of interactions j
            jM = min(p, jMax) # p limits #interactions
            j = 0   # number of allowed interactions
            while Q² ≤ Q²tgt && j < jM && !restart
                j += 1
                J = [] # Temporary store potential new basis elements
                cands = filter(el -> count(!iszero, el) == j, candidates) # only indices with interaction order j
                println("Candidates for p = $p, j = $j: ", cands)

                # Forward step: compute R² for all candidate terms and keep the relevant ones
                for a in cands
                    A = Ap ∪ [a]
                    Φ = [ evaluate(j, X, mop) for j in A ] # FIX create matrix directly
                    Φ = reduce(hcat, Φ)
                    pce = leastSquares(Φ, Y)
                    R²new = 1 - empError(Y, Φ, pce) # Only need R² error here "due to more efficiency" (Blatman 2010)
                    # println("R²new (p=$p, j=$j): ", R²new)

                    # Compute accuracy gain. If high enough, add cadidate polynomial
                    ΔR² = R²new - R²
                    # println("ΔR²: $ΔR² \t ϵ: $ϵ")
                    if ΔR² > ϵ
                        J = J ∪ [a]
                    end
                end

                # FUTURE: Sort Jp and ΦJ according to ΔR² <- why?
                # J = sort(J)
                # R = []

                # Conditioning Check: If resulting enriched basis does not yield a well-conditioned moments matrix, we have to restart
                Ap_new = Ap ∪ J
                println("Candidates after forward step (p=$p, j=$j): $Ap_new")

                Φ = [ evaluate(j, X, mop) for j in Ap_new]
                Φ = reduce(hcat, Φ)
                # Check Φ
                if cond(Φ) > COND 
                    # Increase experimental design and restart computations
                    restart = true
                    k = 3 # rescale factor according to Blatman
                    println("Moments matrix is ill-conditioned. Restart computation with new sample size: $(k * sampleSize)) (Old size: $sampleSize)")
                    sampleSize *= k   # TODO: build properly
                    X = sampleMeasure(sampleSize, mop)
                    Y = [modelFun(x...) for x in eachrow(X)] # TODO: Reuse old ED data, this part is very expensive!
                else
                #     # If conditioning is okay, update accuracy R² for backward step
                    # Φ = [ evaluate(j, X[i], op) for i = 1:sampleSize, j in Ap_new]
                    pce = leastSquares(Φ, Y)
                    R² = 1 - empError(Y, Φ, pce)
                end
                    
                
                # Backward step: Remove candidate polynomials one by one and compute the effect on Q²
                if !restart 
                    println("\n ** Backward Step - Ap_new: ", Ap_new)
                    Del = [] # polynomials to be discarded
                    for a in Ap_new
                        A = filter(e->e≠a,Ap_new)
                        # New candiadte basis -> compute new determination coefficients
                        Φ = [ evaluate(j, X, mop) for j in A]
                        Φ = reduce(hcat, Φ)
                        pce = leastSquares(Φ, Y)
                        R²new = 1 - empError(Y, Φ, pce)
                        
                        # If decrease in accuracy is too small, throw polynomial away
                        ΔR² = R² - R²new
                        if ΔR² ≤ ϵ
                            Del = Del ∪ [a]
                            println("throw away a = ", a)
                        end
                    end

                    # Update basis and compute errors for next iteration
                    Ap = filter(e->e∉Del, Ap_new)
                    Φ = [ evaluate(j, X, mop) for j in Ap]
                    Φ = reduce(hcat, Φ)
                    pce = leastSquares(Φ, Y)
                    println("pce: ", pce)
                    R² = 1 - empError(Y, Φ, pce)
                    Q² = 1 - looError(Y, Φ, pce)
                    println("Q² (p=$p, j=$j): ", Q²)
                end

            end
            println()
        end
    
    end
    
    if Q² < Q²tgt
        println("Computation reached max degree $pMax. However, accuracy is below target with Q² = $(Q²). \n")
    else
        println("Computation reached target accuracy with Q² = $(Q²) and degree $p \n")
    end

    return pce, Ap, p, R², Q²
end