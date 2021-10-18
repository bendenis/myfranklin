# This file was generated, do not modify it. # hide
function compute_cost(AP, Y)
    cost = -mean(log.(AP) .* Y + log.(1 .- AP) .* (1 .- Y))
    return cost
end