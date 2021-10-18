# This file was generated, do not modify it. # hide
function linear_activation_backward(∇A, cache, activation)
    linear_cache, activation_cache = cache
    if activation == "sigmoid"
        ∇Z =  sigmoid_backward(∇A, activation_cache) 
        ∇Ap, ∇W, ∇b =  linear_backward(∇Z, linear_cache)
    elseif activation == "relu"
        ∇Z =  relu_backward(∇A, activation_cache)
        ∇Ap, ∇W, ∇b =  linear_backward(∇Z, linear_cache)
    end
    return(∇Ap, ∇W, ∇b)
end