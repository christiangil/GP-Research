println("Testing Stuff...")
t = @elapsed include("tests.jl")
println("done (took $t seconds).")
