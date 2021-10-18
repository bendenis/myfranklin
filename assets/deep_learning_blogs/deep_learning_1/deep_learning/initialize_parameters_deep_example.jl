# This file was generated, do not modify it. # hide
number_of_input_features = size(X)[1]
model_form = [number_of_input_features,8,4,2,1]

init_parameters = initialize_parameters_deep(model_form)

[
    println("Parameters: " * k * " of size " * string(size(init_parameters[k]))) 
    for k in keys(init_parameters)
]