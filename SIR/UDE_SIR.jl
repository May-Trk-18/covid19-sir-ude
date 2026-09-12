using DifferentialEquations
using Plots
using Lux
using DiffEqFlux
using Optimization
using OptimizationOptimisers
using OptimizationOptimJL
using Optim
using Random
using ComponentArrays
using NNlib: softplus
using SciMLSensitivity
using Statistics



include("build_neural_network.jl")
include("Load.jl")

ann, p1, st1 = build_network()

country_data = load_country_data(
    "covid_features.csv",
    "Germany"
)

N = Float64(country_data.Population[1])
S0 = Float64(country_data.Susceptible[1])
I0 = Float64(country_data.Infected[1])
R0 = Float64(country_data.Removed[1])


u0 = Float64[
    S0,
    I0,
    R0  
]

t = Float64.(country_data.Day)

Infected = Float64.(country_data.Infected)
Removed  = Float64.(country_data.Removed)

tspan = (minimum(t), maximum(t))


p0_vec = ComponentArray(
    nn = p1,
    β = 0.4677637765151079,
    γ = 0.42537047157693386


)

α = p0_vec




function USIR(du, u, p, t)
    # State variables
    S, I, R = u
    
    # Parameters
    β = softplus(p.β)
    γ = softplus(p.γ)

    nn_input =[
     S / N,
     I / N,
     R / N
    ]

   y, _ = ann(nn_input, p.nn, st1)

   βeff = softplus(p.β + y[1])

   du[1] = -βeff * S * I / N
   du[2] =  βeff * S * I / N - γ * I
   du[3] =  γ * I

    
end


prob = ODEProblem(
    USIR,
    u0,
    tspan,
    α
)


function predict(θ)

    sol = solve(
        prob,
        Tsit5(),
        p = θ,
        saveat = t,
        abstol = 1e-8,
        reltol = 1e-6,
        maxiters = 1_000_000,
        sensealg = InterpolatingAdjoint(
            autojacvec = ReverseDiffVJP(true)
        )
    )

    if sol.retcode != ReturnCode.Success
        return fill(NaN, 3, length(t))
    end

    return Array(sol)
end

function loss(θ)

    prediction = predict(θ)

    if any(isnan, prediction)
        return 1e12
    end

    prediction_I = prediction[2, :]

    return mean((prediction_I .- Infected).^2)

end



iter = 0
function callback3(state, l)

    global iter
    iter += 1

    if iter % 100 == 0
        println(
            "Iter = ", iter,
            " Loss = ", l,
            " β = ", state.u.β,
            " γ = ", state.u.γ
        )
    end

    return false
end


### optimizing the neural network weights
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x,p) -> loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, α)

res1 = Optimization.solve(optprob,OptimizationOptimisers.Adam(0.001), callback = callback3, maxiters = 6000)
        
optprob2 = remake(optprob; u0 = res1.u)

@time loss(res1.u)

@time res2 = Optimization.solve( optprob2, Optim.BFGS(initial_stepnorm = 0.01),callback = callback3,maxiters = 50)

data_pred = predict(res2.u)
p_final = res2.u




println("Final loss = ", res2.objective)
println("Estimated β = ", abs(p_final.β))
println("Estimated γ = ", abs(p_final.γ))

S_pred = data_pred[1, :]
I_pred = data_pred[2, :]
R_pred = data_pred[3, :]


predicted_I = I_pred
observed_I = Infected

mse = mean((predicted_I .- observed_I).^2)
rmse = sqrt(mse)
mae = mean(abs.(predicted_I .- observed_I))

ss_res = sum((observed_I .- predicted_I).^2)
ss_tot = sum((observed_I .- mean(observed_I)).^2)
r2 = 1 - ss_res / ss_tot

println("MSE  = ", mse)
println("RMSE = ", rmse)
println("MAE  = ", mae)
println("R²   = ", r2)

#UDE Parameters
UDE_parameter = [
    begin
        y, _ = ann(
    Float64[
        S_pred[i] / N,
        I_pred[i] / N,
        R_pred[i] / N
    ],
    p_final.nn,
    st1
    )
    end
    for i in eachindex(S_pred)
]


plot(
    t,
    Infected,
    label = "Observed Infected",
    xlabel = "Day",
    ylabel = "Number of Individuals",
    linewidth = 2
)

plot!(
    t,
    I_pred,
    label = "Predicted Infected",
    linewidth = 2
)


sol = solve(
    prob,
    Tsit5(),
    p = res2.u,
    saveat = t,
    abstol = 1e-8,
    reltol = 1e-6
)



anim = @animate for i in 1:length(country_data.Day)

    plot(
        country_data.Day[1:i],
        data_pred[2, 1:i],
        label = "Predicted UDE",
        linewidth = 3,
        xlabel = "Day",
        ylabel = "Infected Population",
        title = "UDE vs Observed Data",
        legend = :topleft,
        xlims = (minimum(country_data.Day), maximum(country_data.Day)),
        ylims = (0, maximum(Infected) * 1.05)
    )

    scatter!(
        country_data.Day[1:i],
        Infected[1:i],
        label = "Observed Data",
        markersize = 3
    )

end

gif(anim, "ude_vs_data3.gif", fps = 15)

println(length(t))



    