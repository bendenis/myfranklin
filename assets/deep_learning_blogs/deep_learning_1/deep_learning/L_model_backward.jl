# This file was generated, do not modify it. # hide
function L_model_backward(AP, Y, caches)
    grads = Dict()
    L = length(caches)

    # derivative of the Loss Function (Cost) w.r.t AP
    dAP = -((Y ./ AP) .- ((1 .-Y)./(1 .- AP)))
    current_cache = caches[L]

    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAP, current_cache, "sigmoid")
    grads["dA" * string(L-1)] = dA_prev_temp
    grads["dW" * string(L)] = dW_temp
    grads["db" * string(L)] = db_temp

    for l in (L-1):-1:1
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" * string(l)], 
                                                current_cache, 
                                                "relu")

        grads["dA" * string(l-1)] = dA_prev_temp
        grads["dW" * string(l)] = dW_temp
        grads["db" * string(l)] = db_temp
    end

    return(grads)
end