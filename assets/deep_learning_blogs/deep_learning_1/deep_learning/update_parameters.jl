# This file was generated, do not modify it. # hide
function update_parameters(parameters, grads, learning_rate)
    updated_parameters = copy(parameters)
    L = Int(length(updated_parameters) / 2)
    for l in 1:L
        updated_parameters["W" * string(l)] = updated_parameters["W" * string(l)] .- learning_rate*grads["dW" * string(l)]
        updated_parameters["b" * string(l)] = updated_parameters["b" * string(l)] .- learning_rate*grads["db" * string(l)]
    end
    return(updated_parameters)
end