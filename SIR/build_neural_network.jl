using Lux
using Random

#=rng = Random.default_rng()

nn = Chain(
    Dense(3 => 10, relu),
    Dense(10 => 1)
)

ps, st = Lux.setup(rng, nn)

println(nn)
println(ps)=#

function build_network()

    rng = Random.default_rng()

    nn = Chain(
    Dense(3 => 16, tanh),
    Dense(16 => 16, tanh),
    Dense(16 => 1)
    )

    ps, st = Lux.setup(rng, nn)

    return nn, ps, st

end