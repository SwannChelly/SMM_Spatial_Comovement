
using DataFrames
using Plots
using CSV







best_params_first_loop = CSV.read("parameters.csv",DataFrame)
params_second_loop = CSV.read("parameters_2.csv",DataFrame)
sort!(params_second_loop, :score_index)
params_second_loop

columns = names(params_second_loop)[1:6]

plots = Any[]
for i in 1:6
    tmp = Symbol(columns[i])
    param = params_second_loop[(400*(i-1)+1):(400*(i)), [tmp, :score]]
    param[:,tmp] = (best_params_first_loop[:1,tmp].-param[:,tmp])./best_params_first_loop[:1,tmp]
    p1 = plot(param[:,tmp],param[:,:score],title = columns[i])
    push!(plots,p1)
end
plot(plots..., layout=(2,3), size=(800,800))

