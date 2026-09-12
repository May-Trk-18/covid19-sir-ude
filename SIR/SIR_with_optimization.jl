using ModelingToolkit
using OrdinaryDiffEq
using Plots
using CSV
using DataFrames
using DifferentialEquations
using Optimization
using OptimizationOptimJL
using ForwardDiff
using Statistics

df = CSV.read(
    "C:\\Users\\Turki\\Desktop\\SIR CLASSIC PROJECT\\SIR\\src\\covid_features.csv",
    DataFrame
)

country = "Germany"

country_data = filter(row -> row.Location == country, df)

#Population
N = country_data.Population[1]

#initial condition 
S0 = country_data.Susceptible[1]
I0 = country_data.Infected[1]
R0 = country_data.Removed[1]

u0 = [S0, I0, R0]


# Initial parameter guesses
p0 = [0.30, 0.10]

# Time span
tmin = minimum(country_data.Day)
tmax = maximum(country_data.Day)

tspan = (tmin, tmax)


# Classical SIR model
function sir!(du, u, p, t)
    S, I, R = u
    β, γ = p

    du[1] = -(β/N) * S * I 
    du[2] = (β /N) * S * I - γ * I
    du[3] = γ * I

    nothing
end

# loss function
function loss(p, _)

    prob = ODEProblem(sir!, u0, tspan, p)

    sol = solve(prob, Tsit5(), saveat = country_data.Day)

    predicted_I = Array(sol)[2, :]

    observed_I = country_data.Infected

    #MSE
    return mean((predicted_I .- observed_I).^2)

end


#optimization function
optf = OptimizationFunction(loss, Optimization.AutoForwardDiff())

#optimization problem
lb = [0.0, 0.0]      # lower bounds
ub = [2.0, 2.0]      # upper bounds

optprob = OptimizationProblem(optf, p0;
    lb = lb,
    ub = ub)

result = solve(optprob,BFGS())
println(result.retcode)

#display the resul
β_opt, γ_opt = result.u

println("Estimated β = ", β_opt)
println("Estimated γ = ", γ_opt)
println("Loss = ", result.objective)

#Simulate with the optimized parameters
prob_opt = ODEProblem(sir!, u0, tspan, result.u)
sol_opt = solve(prob_opt, Tsit5(), saveat = country_data.Day)

predicted_I = Array(sol_opt)[2, :]
observed_I = country_data.Infected

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

#Compare with the observed data
plot(
    country_data.Day,
    Array(sol_opt)[2, :],
    label = "Optimized SIR",
    linewidth = 2
)

scatter!(
    country_data.Day,
    country_data.Infected,
    label = "Observed Data",
    markersize = 3
)

xlabel!("Day")
ylabel!("Infected Population")
title!("Optimized SIR vs Observed Data")


#animate 

#=anim = @animate for i in 1:length(country_data.Day)

    plot(
        country_data.Day[1:i],
        Array(sol_opt)[2,1:i],
        label = "Optimized SIR",
        linewidth = 3,
        xlabel = "Day",
        ylabel = "Infected Population",
        title = "Optimized SIR vs Observed Data",
        legend = :topleft
    )

    scatter!(
        country_data.Day[1:i],
        country_data.Infected[1:i],
        label = "Observed Data",
        markersize = 3
    )

end

gif(anim, "sir_vs_data.gif", fps = 15)=#


