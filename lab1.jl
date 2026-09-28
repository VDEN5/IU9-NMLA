using PyPlot

a = 2
b = 4

x = range(-10, stop=10, length=100)
y = range(-10, stop=10, length=100)

X = [i for i in x, j in y]
Y = [j for i in x, j in y]
Z = @. (X^4)/a^4 + (Y/b)^4 - (X^2 * Y^2)/(a*b)

surf(X, Y, Z, alpha=0.9, cmap="inferno")

xlabel("X")
ylabel("Y")
zlabel("Z")
title("тот самый график из 1 варианта")

savefig("my_3d_plot.png", dpi=300)

println("$(pwd())\\my_3d_plot.png")