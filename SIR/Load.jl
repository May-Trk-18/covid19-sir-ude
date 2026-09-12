using CSV
using DataFrames

function load_country_data(filename, country)

    filepath = joinpath(@__DIR__, filename)

    df = CSV.read(filepath, DataFrame)

    country_data = filter(row -> row.Location == country, df)

    return country_data

end







#=df = CSV.read(
    "C:\\Users\\Turki\\Desktop\\SIR CLASSIC PROJECT\\SIR\\src\\covid_features.csv",
    DataFrame
)

country = "Germany"

country_data = filter(row -> row.Location == country, df)


#the Population
N = country_data.Population[1]

#initial condition
S0 = country_data.Susceptible[1]
I0 = country_data.Infected[1]
R0 = country_data.Removed[1]

#time span
tmin = minimum(country_data.Day)
tmax = maximum(country_data.Day)


println(N)

println(S0)
println(I0)
println(R0)


println(tmax)=#
