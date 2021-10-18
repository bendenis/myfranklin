# This file was generated, do not modify it. # hide
# Functions to compute dZ given dA and Z (from forward prop cache)
function sigmoid_backward(∇A, activation_cache)
    Zl = activation_cache
    Z = σ.(Zl).*(1 .- σ.(Zl))
    ∇Z = ∇A .* Z
    return(∇Z)
end

function relu_backward(∇A, activation_cache)
    Zl = activation_cache
    Z = copy(Zl)
    Z[Zl .> 0] .= 1
    Z[Zl .<= 0] .= 0

    ∇Z = ∇A .* Z

    return(∇Z)
end