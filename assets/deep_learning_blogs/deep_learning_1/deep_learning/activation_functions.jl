# This file was generated, do not modify it. # hide
function sigmoid(Z)
    A = 1 ./ (1 .+ exp.(-Z))
    return((A = A, Z = Z))
end

function relu(Z)
    A = copy(Z)
    A[A .< 0] .= 0
    return((A = A, Z = Z))
end