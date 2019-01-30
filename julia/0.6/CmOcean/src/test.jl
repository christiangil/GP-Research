#=
Test images for cmocean colormaps.

The design of these images is described in Kovesi (2015),
and illustrated in Figure 2.

https://arxiv.org/abs/1509.03700


<summary>
The wavelength of the sine wave is 8px, with an image width of 512px.
On a monitor with a nominal pixel pitch of 0.25mm this gives a physical
wavelength of 2mm. At a viewing distance of 600mm this correspondes to
0.19 deg, or 5.2 cycles / degree. This falls within the range of spatial
frequencies (3-7 cycles / degree) at which most people have maimal contrast
sensitivity to a sine wave grating. A wavelength of 8px also provides a
reasonable discrete representation of a sine wave.

The sine wave amplitude increases with the square of the distance from
the bottom of this image.

Each row of the test image is normalised so that it spans a range of
0 to 255. This means that the underlying ramp at the top of the image
will be reduced in slope slightly to accommodate the sine wave.
</summary>

=#

using PyPlot
include("cmocean.jl")

data = zeros(51,512)

for row in 1:51
	A = 511 * 0.05 * ( (51-row)/50 )^2
	for col in 1:512
		x = col - 1
		data[row,col] = x + A * sin(x * 2pi / 8)
	end
	#
	# Renormalize row.
	#
	ymax = maximum(data[row,:])
	ymin = minimum(data[row,:])
	
	data[row,:] = (data[row,:] - ymin) * 255 / (ymax - ymin)
end

##################################################
#
#
figure(1)
clf()
plot(data[1,:], "C0-")

##################################################
#
#
figure(2, figsize=(10,10))
clf()

labels = ["algae", "amp", "balance", "curl", "deep", "delta",
		"dense", "gray", "haline", "ice", "matter", "oxy",
		"phase", "solar", "speed", "tempo", "thermal", "turbid"]

for j in 1:9
	for k in [2j-1, 2j]
		subplot(9,2,k)
		title(labels[k])
		imshow(data, cmap=labels[k])
		xticks([])
		yticks([])
	end
end

##################################################
#
# Compare some colormaps from cmocean vs matplotlib.
#
figure(3, figsize=(10,10))
clf()

labels = [
	"jet"		"phase"
	"winter"	"haline"
	"YlGn"		"speed"
	"Blues"		"ice"
	"hot"		"solar"
	"Greens"	"algae"
	"BuPu"		"dense"
	"coolwarm"	"balance"
]
N = size(labels,1)

for j in 1:N
	for k in [1,2]
		subplot(N,2,2j+1-k)
		title(labels[j,k])
		imshow(data, cmap=labels[j,k])
		xticks([])
		yticks([])
	end
end
