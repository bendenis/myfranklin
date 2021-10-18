# This file was generated, do not modify it. # hide
# computing the W and b gradients 
function linear_backward(∇Z, linear_cache)
    Ap, Wl, bl = linear_cache
    m = size(Ap)[2]

    ∇Wl = ∇Z*transpose(Ap) .* 1/m
    ∇bl = mean(∇Z, dims = 2)
    ∇Ap = transpose(Wl)*∇Z

    return(∇Ap, ∇Wl, ∇bl)
end