# This file was generated, do not modify it. # hide
A = X
W = init_parameters["W1"]
b = init_parameters["b1"]
A, cache = linear_activation_forward(A, W, b, "relu")

@show DataFrame(A, :auto)