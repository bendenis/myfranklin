# This file was generated, do not modify it. # hide
A = X
W = init_parameters["W1"]
b = init_parameters["b1"]
Z, cache = linear_forward(A, W, b)

@show DataFrame(Z, :auto)