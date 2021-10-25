

# Building a NN with Julia from the ground up


```julia:./deep_learning/setup
using Distributions
using RDatasets
using DataFrames
using Flux
using Plots
using Random

market_data = dataset("ISLR", "Smarket")
Y = Int.(transpose(market_data.Direction .== "Up"))
X = transpose(Matrix(select(market_data, Not([:Direction,:Today,:Year]))))

@show (first(market_data,10))
println("")
@show size(X), size(Y)
```

\output{./deep_learning/setup.jl}

```julia:market_data_today
#hideall
plot(market_data.Today)
savefig(joinpath(@OUTPUT, "market_data_today.svg"))
```

\fig{market_data_today}

\toc

## Forward Propagation

### Connection to Linear Regression

If you are familiar with linear regression the formulas for the forward propagation will look familiar. There are a couple things worth pointing out though. Just as a referrence here is the formula for the linear regression

$ E(Y) = 1 β_0 + Xβ_1 $

Here $X$ is the $(n,m)$ data where each row is an observation and each column is a covariate/feature. So we have $n$ observations and $m$ variables. Consequently $β_1$ is an $(m,1)$ vector of parameters. $β_0$ is a scalar and the *1* is an $(n,1)$ column vector of 1s.

A regression model like the one above can be represented as a single layer Neural Network model with one neuron. The notation is just a little bit different.

$ Z = WA + b $

$A$ is now an $(m,n)$ matrix of data with one observation per *column* and one feature per *row*. Consequently $W$ is an $(1,m)$ vector of weights (just like $β^T_1$). $b$ is the same scalar as $β_0$ but the *1* is a $(1,n)$ row vector. 

Essentially when computing NNs we transpose everything. Coming from a statistics background this can be a bit uncomfortable at first but with time you get use to it.

### Multiple Layers and Activation

But of course the point of Neural Networks is that you can increase the coplexity of the model with many layers and many neurons per layer. And of course we introduce non-linearity by transforming the output of each layer. This transformation is called an *activation*. 

For example if we wish to have $k$ neurons in out layer we would set $W$ to be $(k,m)$. The output $Z$ will then become a $(k,n)$ matrix as aposed to a $(1,n)$ in a linear regression model. 

Next we activate $Z$ by applying an activation function and get a $(k,n)$, $A = g(Z)$. If we stop with one layer then this is the final output. But we won't. 

To add another layer with $j$ units we introduce another $(j,k)$ matrix $W2$ and $b2$. And so on..

With multiple layers the general equations for forward propagation are:

1) $ Z_{l} = W_{l}^T A_{l-1} + b_{l} $
2) $ A_{l} = g_l(Z_l) $

where l goes from $1 to L$ and L is the number of layers.

### Initializing Parameters (Weights and Bias)

Step 0 is to randomly initialize all of the parameter matrices $W_l$ and vecotrs $b_l$. This is where keeping track of the dimentions becomes extremly important. `initialize_parameters_deep` is the function that does this given an array of integers. The length of the array is the number of layers L and the integers are the number of neurons per layer.



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
number_of_input_features = size(X)[1]
model_form = [number_of_input_features,8,4,2,1]

init_parameters = initialize_parameters_deep(model_form)

[
    println("Parameters: " * k * " of size " * string(size(init_parameters[k]))) 
    for k in keys(init_parameters)
]
```
\output{./deep_learning/initialize_parameters_deep_example.jl}



### Forward Propagation Step

```julia:./deep_learning/linear_forward
function linear_forward(A, W, b)
    Z = W * A .+ b
    cache = (A = A, W = W, b = b)
    return((Z = Z, cache = cache))
end
```

```julia:./deep_learning/linear_forward_example

A = X
W = init_parameters["W1"]
b = init_parameters["b1"]
Z, cache = linear_forward(A, W, b)

@show DataFrame(Z, :auto)

```

\output{./deep_learning/linear_forward_example.jl}


### Activation Step

```julia:./deep_learning/activation_functions
function sigmoid(Z)
    A = 1 ./ (1 .+ exp.(-Z))
    return((A = A, Z = Z))
end

function relu(Z)
    A = copy(Z)
    A[A .< 0] .= 0
    return((A = A, Z = Z))
end
```

```julia:./deep_learning/activation_example
next_A, cache_Z = relu(Z)
@show DataFrame(next_A, :auto)
```

\output{./deep_learning/activation_example.jl}


### Combining Forward & Activation Steps

```julia:./deep_learning/linear_activation_forward
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
```

```julia:./deep_learning/linear_activation_forward_example

A = X
W = init_parameters["W1"]
b = init_parameters["b1"]
A, cache = linear_activation_forward(A, W, b, "relu")

@show DataFrame(A, :auto)

```

\output{./deep_learning/linear_activation_forward_example.jl}



### Compute Fitted Values (Forward Propagation)


```julia:./deep_learning/L_model_forward
function L_model_forward(X, parameters)
    A = copy(X)
    L = Int(length(parameters) / 2)
    caches = []

    for l in 1:(L-1)
        Aprev = copy(A)
        A, cache = linear_activation_forward(Aprev, 
                    parameters["W"*string(l)], 
                    parameters["b"*string(l)],
                    "relu")
        push!(caches, cache)
    end

    AP, cache = linear_activation_forward(A, 
                parameters["W"*string(L)], 
                parameters["b"*string(L)],
                "sigmoid")
    push!(caches, cache)

    return AP, caches
end
```

```julia:./deep_learning/L_model_forward_example
Ŷ, caches = L_model_forward(X, init_parameters)
@show Ŷ
```

\output{./deep_learning/L_model_forward_example.jl}

Accuracy with no training, just the dafult random parameters

```julia:./deep_learning/pred_acc_example
predictions = Ŷ .> 0.5
acc = mean(Y .== predictions)
@show  acc
```

\output{./deep_learning/pred_acc_example.jl}


## Backward  Propagation

### Computing Parameter Gradients

```julia:./deep_learning/linear_backward
# computing the W and b gradients 
function linear_backward(∇Z, linear_cache)
    Ap, Wl, bl = linear_cache
    m = size(Ap)[2]

    ∇Wl = ∇Z*transpose(Ap) .* 1/m
    ∇bl = mean(∇Z, dims = 2)
    ∇Ap = transpose(Wl)*∇Z

    return(∇Ap, ∇Wl, ∇bl)
end
```

```julia:./deep_learning/activation_derivatives

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

```

### Computing Layer Gradients

```julia:./deep_learning/linear_activation_backward
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

```

### Computing All of the Gradients

```julia:./deep_learning/L_model_backward
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
```

## Optimization

```julia:./deep_learning/L_model_forward

```

### Updating Parameters

```julia:./deep_learning/update_parameters

function update_parameters(parameters, grads, learning_rate)
    updated_parameters = copy(parameters)
    L = Int(length(updated_parameters) / 2)
    for l in 1:L
        updated_parameters["W" * string(l)] = updated_parameters["W" * string(l)] .- learning_rate*grads["dW" * string(l)]
        updated_parameters["b" * string(l)] = updated_parameters["b" * string(l)] .- learning_rate*grads["db" * string(l)]
    end
    return(updated_parameters)
end

```

### Cost Function

```julia:./deep_learning/compute_cost
function compute_cost(AP, Y)
    cost = -mean(log.(AP) .* Y + log.(1 .- AP) .* (1 .- Y))
    return cost
end
```

### Itterating

```julia:./deep_learning/prediction_function

function pred_nn_model(parameters, X)
    Ŷ, cache = L_model_forward(X, parameters)
    predictions = Ŷ .> 0.5

    return(predictions)
end

```

```julia:./deep_learning/training
Random.seed!(007)
params = [initialize_parameters_deep([size(X)[1],5,5,1])]

for i in 1:2000000
    AP, chs = L_model_forward(X, params[i])
    grads = L_model_backward(AP, Y, chs)
    up = update_parameters(params[i], grads, 0.1)
    push!(params, up)
    if i % 200000 == 0
        println(compute_cost(AP, Y))
    end
end

acc = mean(Y .== pred_nn_model(params[end], X))

@show params[begin]
@show params[end]
@show acc
```

\output{./deep_learning/training.jl}


<!--
-->