# This file was generated, do not modify it. # hide
init_parameters = initialize_parameters_deep([3,8,4,2,1])

[
    println("Parameters: " * k * " of size " * string(size(init_parameters[k]))) 
    for k in keys(init_parameters)
]