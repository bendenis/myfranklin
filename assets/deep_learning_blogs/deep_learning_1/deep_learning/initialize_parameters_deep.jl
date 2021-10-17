# This file was generated, do not modify it. # hide
function initialize_parameters_deep(layer_dims)
    parameters = Dict()

    for l in 1:(length(layer_dims) - 1)
        parameters["W" * string(l)] = rand(layer_dims[l+1], layer_dims[l]) .* 0.01
        parameters["b" * string(l)] = rand(layer_dims[l+1], 1) .* 0.01
    end

    return(parameters)
end