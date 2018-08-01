# all of the functions to simplify using PlotlyJS

# include("reusable_code.jl")
using PlotlyJS
using JLD
using Rsvg


# a default layout
layout = Layout(; title="Gaussian Processes",
    xaxis=attr(title="x (time)",
        tickcolor="rgb(127, 127, 127)",
        ticks="outside"),
    yaxis=attr(title="y (flux or something lol)",
        tickcolor="rgb(127, 127, 127)",
        ticks="outside"),
    showlegend=false)


#convert an Any type array to a trace array
function trace_list(a)
    traces = []
    for i in 1:size(a)[1]
        if i == 1
            traces = [a[1]]
        else
            append!(traces, [a[i]])
        end
    end

    return traces
end


# a wrapper to make lines with the scatter function
function line_trace(x, y; width=1, color=1)
    if color == 1
        return scatter(; x=x, y=y, mode="lines",
            line_width=width, showlegend=false)
    else
        return scatter(; x=x, y=y, mode="lines",
            line_width=width, showlegend=false, line_color=color)
    end
end


# a wrapper to make surface plots with the surface_trace function
function surface_trace(x, y, z, lenx, leny; opacity=1)
    return surface(z=reshape(z, (lenx, leny)),
        x=reshape(x, (lenx, leny)),
        y=reshape(y, (lenx, leny)),
        showscale=false, opacity=opacity, colorscale="Viridis")
end


# generic trace collection wrapper function
function traces(coords...; opacity=1, color=1, width=1)

    x = coords[1]
    y = coords[2]

    all_traces = []
    if length(coords) > 2
        z = coords[3]
        if length(size(z)) > 1
            all_traces = []
            for i in 1:size(z)[1]
                append!(all_traces, [surface_trace(x, y, z[i,:], opacity=opacity)])
            end
            all_traces = trace_list(all_traces)
        else
            all_traces = surface_trace(x, y, z)
        end
    else
        if length(size(y)) > 1
            for i in 1:size(y)[1]
                append!(all_traces, [line_trace(x, y[i,:]; width=width, color=color)])
            end
            all_traces = trace_list(all_traces)
        else
            all_traces = [line_trace(x, y; width=width, color=color)]
        end
    end

    return all_traces
end


# # this is broken for some reason
# function traces(coords...; opacity=1)
#
#     x = coords[1]
#     if length(coords) > 2
#         y = coords[2]
#         func(response) = surface_trace(x, y, response, opacity=opacity)
#     else
#         func(response) = line_trace(x, response)
#     end
#
#     all_traces = []
#     dims = length(coords)
#     responses = coords[dims]
#     if length(size(responses)) > 1
#         for i in 1:size(responses)[1]
#             append!(all_traces, [func(responses[i,:])])
#         end
#         all_traces = trace_list(all_traces)
#     else
#         all_traces = func(responses)
#     end
#
#     return all_traces
# end


# this is horrible, but the only way I've found to actually return a grid of plots
# PlotlyJS doesn't like it when you try to assemble these matrices automatically
function return_plot_matrix(data)
    try
        n = convert(Int64, sqrt(length(data)))
        if n == 1
            return [data[1]]
        elseif n == 2
            return [data[1] data[2];
                data[3] data[4]]
        elseif n == 3
            return [data[1] data[2] data[3];
                data[4] data[5] data[6];
                data[7] data[8] data[9]]
        elseif n == 5
            return [data[1] data[6] data[11] data[16] data[21];
                data[2] data[7] data[12] data[17] data[22];
                data[3] data[8] data[13] data[18] data[23];
                data[4] data[9] data[14] data[19] data[24];
                data[5] data[10] data[15] data[20] data[25]]
        elseif n == 7
            return [data[1] data[8] data[15] data[22] data[29] data[36] data[43];
                data[2] data[9] data[16] data[23] data[30] data[37] data[44];
                data[3] data[10] data[17] data[24] data[31] data[38] data[45];
                data[4] data[11] data[18] data[25] data[32] data[39] data[46];
                data[5] data[12] data[19] data[26] data[33] data[40] data[47];
                data[6] data[13] data[20] data[27] data[34] data[41] data[48];
                data[7] data[14] data[21] data[28] data[35] data[42] data[49]]
        elseif n == 10
            return [data[1] data[11] data[21] data[31] data[41] data[51] data[61] data[71] data[81] data[91];
                data[2] data[12] data[22] data[32] data[42] data[52] data[62] data[72] data[82] data[92];
                data[3] data[13] data[23] data[33] data[43] data[53] data[63] data[73] data[83] data[93];
                data[4] data[14] data[24] data[34] data[44] data[54] data[64] data[74] data[84] data[94];
                data[5] data[15] data[25] data[35] data[45] data[55] data[65] data[75] data[85] data[95];
                data[6] data[16] data[26] data[36] data[46] data[56] data[66] data[76] data[86] data[96];
                data[7] data[17] data[27] data[37] data[47] data[57] data[67] data[77] data[87] data[97];
                data[8] data[18] data[28] data[38] data[48] data[58] data[68] data[78] data[88] data[98];
                data[9] data[19] data[29] data[39] data[49] data[59] data[69] data[79] data[89] data[99];
                data[10] data[20] data[30] data[40] data[50] data[60] data[70] data[80] data[90] data[100]]
        else
            println("that dimension of plots isn't handcoded yet")
            return data
        end
    catch
        return data
    end
end
