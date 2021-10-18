# This file was generated, do not modify it. # hide
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