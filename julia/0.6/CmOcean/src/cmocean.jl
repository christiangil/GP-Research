#=
Create all the colormaps from cmocean and register them with
Matplotlib.

Example:
	include("path/to/cmocean.jl")

	imshow(Z, cmap="ice")

WARNING: This script overwrites the "gray" colormap from Matplotlib.
         The one from cmocean has a slightly longer range. If you
         don't want that, just delete "gray" below.

=#

using PyPlot
using DelimitedFiles

for name in ["algae", "amp", "balance", "curl", "deep", "delta",
			"dense", "gray", "haline", "ice", "matter", "oxy",
			"phase", "solar", "speed", "tempo", "thermal", "turbid"]
	cm = ColorMap(name, readdlm("rgb/$name-rgb.txt"))
	plt[:register_cmap](name=name,cmap=cm)
end
