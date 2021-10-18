# This file was generated, do not modify it. # hide
function pred_nn_model(parameters, X)
    Ŷ, cache = L_model_forward(X, parameters)
    predictions = Ŷ .> 0.5

    return(predictions)
end