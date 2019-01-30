#
# Example usage of cmocean.jl
#

using PyPlot
include("cmocean.jl")

data = ones(10) * linspace(0,1,101)'

figure(1)
clf()
subplot(3,1,1)
imshow(data, cmap="ice")

subplot(3,1,2)
imshow(data, cmap="speed")

subplot(3,1,3)
imshow(data, cmap="balance")


