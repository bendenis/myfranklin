# This file was generated, do not modify it. # hide
function linear_forward(A, W, b)
    Z = W * A .+ b
    cache = (A = A, W = W, b = b)
    return((Z = Z, cache = cache))
end