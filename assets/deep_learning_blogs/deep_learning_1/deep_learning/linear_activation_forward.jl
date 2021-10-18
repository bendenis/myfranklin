# This file was generated, do not modify it. # hide
function linear_activation_forward(Aprev, W, b, activation)
    if activation == "sigmoid"
        Z, linear_cache = linear_forward(Aprev, W, b)
        A, activation_cache = sigmoid(Z)
    elseif  activation == "relu"
        Z, linear_cache = linear_forward(Aprev, W, b)
        A, activation_cache = relu(Z)
    end
    
    cache = (linear_cache, activation_cache)

    return((A = A, cache = cache))
end