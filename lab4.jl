using LinearAlgebra
using Plots

function jacobi(A, b, eps, maxiter)
    n = length(b)
    count = 0

    P = zeros(n, n)
    g = zeros(n)
    for i in 1:n
        for j in 1:n
            if j != i
                P[i,j] = -A[i,j] / A[i,i]
            end
        end
        g[i] = b[i] / A[i,i]
    end

    x     = zeros(n)
    x_new = zeros(n)
    traj  = [copy(x)]                  
    for _ in 1:maxiter
        count += 1
        for i in 1:n
            s = g[i]
            for j in 1:n
                s += P[i,j] * x[j]
            end
            x_new[i] = s
        end

        acc = 0.0
        for i in 1:n
            d = x_new[i] - x[i]
            acc += d * d
        end
        norm2 = sqrt(acc)

        for i in 1:n
            x[i] = x_new[i]
        end
        push!(traj, copy(x))             

        if norm2 <= eps
            break
        end
    end
    println(count)
    return x, traj
end

function seidel(A, b, eps, maxiter)
    n = length(b)
    count = 0

    A_plus = zeros(n, n)
    for i in 1:n
        for j in 1:n
            if j > i
                A_plus[i,j] = A[i,j]
            end
        end
    end

    Am = zeros(n, n)
    for i in 1:n
        for j in 1:n
            Am[i,j] = A[i,j] - A_plus[i,j]
        end
    end

    Am_inv = inv(Am)
    P = -Am_inv * A_plus
    g =  Am_inv * b

    x     = zeros(n)
    x_new = zeros(n)
    traj  = [copy(x)]                  
    for _ in 1:maxiter
        count += 1
        for i in 1:n
            s = g[i]
            for j in 1:n
                s += P[i,j] * x[j]
            end
            x_new[i] = s
        end

        acc = 0.0
        for i in 1:n
            d = x_new[i] - x[i]
            acc += d * d
        end
        norm2 = sqrt(acc)

        for i in 1:n
            x[i] = x_new[i]
        end
        push!(traj, copy(x))

        if norm2 <= eps
            break
        end
    end
    println(count)
    return x, traj
end

function plot_method(traj, name)
    m  = length(traj)
    xs = [traj[k][1] for k in 1:m]
    ys = [traj[k][2] for k in 1:m]
    zs = [traj[k][3] for k in 1:m]

    p1 = plot(title="$name: (x1, x2)   [итераций: $(m-1)]",
              xlabel="x1", ylabel="x2", legend=:topleft)
    for k in 1:m
        plot!(p1, [0.0, xs[k]], [0.0, ys[k]],
              arrow=true, color=:steelblue, alpha=0.35, label="")
    end
    plot!(p1, xs, ys, color=:orange, linewidth=2, label="траектория")
    scatter!(p1, xs, ys, color=:red, markersize=3, label="x_k")
    scatter!(p1, [xs[end]], [ys[end]], color=:green, markersize=8, label="решение")

    p2 = plot(title="$name: (x2, x3)   [итераций: $(m-1)]",
              xlabel="x2", ylabel="x3", legend=:topleft)
    for k in 1:m
        plot!(p2, [0.0, ys[k]], [0.0, zs[k]],
              arrow=true, color=:steelblue, alpha=0.35, label="")
    end
    plot!(p2, ys, zs, color=:orange, linewidth=2, label="траектория")
    scatter!(p2, ys, zs, color=:red, markersize=3, label="x_k")
    scatter!(p2, [ys[end]], [zs[end]], color=:green, markersize=8, label="решение")

    return plot(p1, p2, layout=(1,2), size=(1000, 480))
end

A = [4.0 -1.0  0.0;
    -1.0  4.0 -1.0;
     0.0 -1.0  3.0]
b = [3.0, 2.0, 1.0]

x_j, traj_j = jacobi(A, b, 1e-10, 10000)
x_s, traj_s = seidel(A, b, 1e-10, 10000)

println("Якоби:  x = ", x_j)
println("Зейдель: x = ", x_s)

n = length(b)
for (name, x) in (("Якоби", x_j), ("Зейдель", x_s))
    r = zeros(n)
    for i in 1:n
        s = 0.0
        for j in 1:n
            s += A[i,j] * x[j]
        end
        r[i] = s - b[i]
    end
    println("A*x - b ($name) = ", r)
end

display(plot_method(traj_j, "Якоби"))
display(plot_method(traj_s, "Зейдель"))