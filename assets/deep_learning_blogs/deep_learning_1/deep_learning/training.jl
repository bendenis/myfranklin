# This file was generated, do not modify it. # hide
Random.seed!(007)
params = [initialize_parameters_deep([size(X)[1],5,5,1])]

for i in 1:1000000
    AP, caches = L_model_forward(X, params[i])
    grads = L_model_backward(AP, Y, caches)
    up = update_parameters(params[i], grads, 0.1)
    push!(params, up)
    if i % 100000 == 0
        println(compute_cost(AP, Y))
    end
end

acc = mean(Y .== pred_nn_model(params[end], X))

@show params[begin]
@show params[end]
@show acc