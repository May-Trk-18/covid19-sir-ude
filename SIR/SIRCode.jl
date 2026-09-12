using ModelingToolkit
using OrdinaryDiffEq
using Plots
using CSV
using DataFrames

df = CSV.read(
    "C:\\Users\\Turki\\Desktop\\SIR CLASSIC PROJECT\\SIR\\src\\covid_features.csv",
    DataFrame
)

country = "Germany"

country_data = filter(row -> row.Location == country, df)

#@parameters β γ
#@independent_variables  t
#@variables S(t) I(t) R(t)

#Dt = Differential(t)
#=eqs = [
    Dt(S) ~ -(β/N) * S * I,
    Dt(I) ~  (β/N) * S * I - γ * I,
    Dt(R) ~  γ * I
]=#

# the Population 
N = country_data.Population[1]

#initial condition 
S0 = country_data.Susceptible[1]
I0 = country_data.Infected[1]
R0 = country_data.Removed[1]

#=u0 = [
    S => S0,
    I => I0,
    R => R0
]=#

u0 = [S0, I0, R0]

#=p = [
    β => 0.30,
    γ => 0.10
]=#

# Initial parameter guesses
β = 0.30
γ = 0.10

p = [β, γ]

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

prob = ODEProblem(sir!, u0, tspan, p)

sol = solve(prob, Tsit5(),saveat = country_data.Day)

# Plot the SIR model solution
using Plots

plot(
    sol,
    xlabel = "Days",
    ylabel = "Population",
    label = ["S" "I" "R"],
    linewidth = 2
)

plot(
    country_data.Day,
    Array(sol)[2, :],
    label = "Predicted Infected",
    linewidth = 2
)

# Plot observed infected
scatter!(
    country_data.Day,
    country_data.Infected,
    label = "Observed Infected",
    markersize = 3
)

xlabel!("Day")
ylabel!("Number of Infected")
title!("Classical SIR vs Observed Data")


# Predicted infected from the SIR model
predicted_I = Array(sol)[2, :]

# Observed infected from the dataset
observed_I = country_data.Infected

println("Maximum predicted infected = ", maximum(predicted_I))
println("Maximum observed infected = ", maximum(observed_I))


