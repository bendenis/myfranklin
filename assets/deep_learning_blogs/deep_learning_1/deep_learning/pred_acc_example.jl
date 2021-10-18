# This file was generated, do not modify it. # hide
predictions = Ŷ .> 0.5
acc = mean(Y .== predictions)
@show  acc