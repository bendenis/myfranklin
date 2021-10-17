

# Building a NN with Julia from the ground up

\toc

## Initializing Parameters (Weights and Bias)

```julia:./deep_learning/initialize_parameters_deep
function initialize_parameters_deep(layer_dims)
    parameters = Dict()

    for l in 1:(length(layer_dims) - 1)
        parameters["W" * string(l)] = rand(layer_dims[l+1], layer_dims[l]) .* 0.01
        parameters["b" * string(l)] = rand(layer_dims[l+1], 1) .* 0.01
    end

    return(parameters)
end
```

```julia:./deep_learning/initialize_parameters_deep_example
init_parameters = initialize_parameters_deep([3,8,4,2,1])

[
    println("Parameters: " * k * " of size " * string(size(init_parameters[k]))) 
    for k in keys(init_parameters)
]
```
\output{./deep_learning/initialize_parameters_deep_example.jl}

## Forward Propagation Step

## Activation Step

## Combining Forward & Activation Steps

## Compute Fitted Values (Forward Propagation)