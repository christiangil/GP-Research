println("Testing Stuff...")
t = @elapsed include("tests.jl")
println("testing took $t seconds")
