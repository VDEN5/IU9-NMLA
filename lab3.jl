using LinearAlgebra
using LinearAlgebra.LAPACK
using Printf
using Plots
using Random

function lapack_solve(A, b)
    LU, ipiv, info = LAPACK.getrf!(copy(A))
    info != 0 && error("getrf! info=$info")
    return LAPACK.getrs!('N', LU, ipiv, copy(b))
end

function lapack_svdvals(A)
    _, s, _ = LAPACK.gesdd!('N', copy(A))
    return s
end

function gauss_simple(A, b)
    n = size(A, 1); M = copy(A); rhs = copy(b)
    for k in 1:n
        abs(M[k, k]) < 1e-14 && error("нулевой ведущий элемент")
        for i in k+1:n
            f = M[i, k] / M[k, k]
            M[i, k:n] .-= f .* M[k, k:n]
            rhs[i]    -= f * rhs[k]
        end
    end
    x = zeros(n)
    for i in n:-1:1
        acc = sum(M[i, j] * x[j] for j in i+1:n; init = 0.0)
        x[i] = (rhs[i] - acc) / M[i, i]
    end
    return x
end

function gauss_row_pivot(A, b)
    n = size(A, 1); M = copy(A); rhs = copy(b)
    for k in 1:n
        p = k - 1 + argmax(abs.(M[k:n, k]))
        abs(M[p, k]) < 1e-14 && error("вырождена")
        if p != k
            M[[k, p], :] = M[[p, k], :]
            rhs[[k, p]]  = rhs[[p, k]]
        end
        for i in k+1:n
            f = M[i, k] / M[k, k]
            M[i, k:n] .-= f .* M[k, k:n]
            rhs[i]    -= f * rhs[k]
        end
    end
    x = zeros(n)
    for i in n:-1:1
        acc = sum(M[i, j] * x[j] for j in i+1:n; init = 0.0)
        x[i] = (rhs[i] - acc) / M[i, i]
    end
    return x
end

function gauss_col_pivot(A, b)
    n = size(A, 1); M = copy(A); rhs = copy(b)
    col_perm = collect(1:n)
    for k in 1:n
        p = k - 1 + argmax(abs.(M[k, k:n]))
        abs(M[k, p]) < 1e-14 && error("вырождена")
        if p != k
            M[:, [k, p]] = M[:, [p, k]]
            col_perm[k], col_perm[p] = col_perm[p], col_perm[k]
        end
        for i in k+1:n
            f = M[i, k] / M[k, k]
            M[i, k:n] .-= f .* M[k, k:n]
            rhs[i]    -= f * rhs[k]
        end
    end
    y = zeros(n)
    for i in n:-1:1
        acc = sum(M[i, j] * y[j] for j in i+1:n; init = 0.0)
        y[i] = (rhs[i] - acc) / M[i, i]
    end
    x = zeros(n)
    for i in 1:n
        x[col_perm[i]] = y[i]
    end
    return x
end

function gauss_mixed(A, b)
    n = size(A, 1); M = copy(A); rhs = copy(b)
    col_perm = collect(1:n)
    for k in 1:n
        p = k - 1 + argmax(abs.(M[k:n, k]))
        abs(M[p, k]) < 1e-14 && error("вырождена")
        if p != k
            M[[k, p], :] = M[[p, k], :]
            rhs[[k, p]]  = rhs[[p, k]]
        end
        q = k - 1 + argmax(abs.(M[k, k:n]))
        if q != k
            M[:, [k, q]] = M[:, [q, k]]
            col_perm[k], col_perm[q] = col_perm[q], col_perm[k]
        end
        for i in k+1:n
            f = M[i, k] / M[k, k]
            M[i, k:n] .-= f .* M[k, k:n]
            rhs[i]    -= f * rhs[k]
        end
    end
    y = zeros(n)
    for i in n:-1:1
        acc = sum(M[i, j] * y[j] for j in i+1:n; init = 0.0)
        y[i] = (rhs[i] - acc) / M[i, i]
    end
    x = zeros(n)
    for i in 1:n
        x[col_perm[i]] = y[i]
    end
    return x
end

function lab_dominance_vs_error(n::Int;
                                k_range  = range(0.0, 3.0, length = 80),
                                n_trials = 3,
                                seed     = 1234)
    Random.seed!(seed)
    x_exact = ones(n)

    doms   = Float64[]
    kappas = Float64[]
    names  = ("simple", "row", "col", "mixed", "lapack")
    E      = Dict(name => Float64[] for name in names)

    methods = [
        ("simple", gauss_simple),
        ("row",    gauss_row_pivot),
        ("col",    gauss_col_pivot),
        ("mixed",  gauss_mixed),
        ("lapack", lapack_solve),
    ]

    for trial in 1:n_trials
        A = randn(n, n)
        for i in 1:n
            A[i, i] = 0.0
        end
        s = [sum(abs(A[i, j]) for j in 1:n if j != i) for i in 1:n]

        for k in k_range
            A_d = copy(A)
            for i in 1:n
                A_d[i, i] = k * s[i]
            end

            d = minimum(abs(A_d[i, i]) / s[i] for i in 1:n)

            σ = lapack_svdvals(A_d)
            κ = σ[1] / σ[end]

            b = A_d * x_exact

            push!(doms, d)
            push!(kappas, κ)
            for (name, f) in methods
                try
                    xh = f(A_d, b)
                    push!(E[name], norm(xh - x_exact) / norm(x_exact))
                catch
                    push!(E[name], NaN)
                end
            end
        end
    end

    idx = sortperm(doms)
    D = doms[idx]
    K = kappas[idx]
    for name in names
        E[name] = E[name][idx]
    end

    p = scatter(D, E["simple"];
                label      = "простой Гаусс",
                xlabel     = "d(A) = min_i |a_ii| / Σ_{j≠i}|a_ij|   →   рост преобладания",
                ylabel     = "отн. погрешность ||x-x*||/||x*||",
                yscale     = :log10,
                marker     = :xcross, ms = 4,
                legend     = :topright,
                title      = "Погрешность vs диагональное преобладание (n=$n)")
    scatter!(p, D, E["row"];    label = "перест. строк",     marker = :circle,    ms = 4)
    scatter!(p, D, E["col"];    label = "перест. столбцов",  marker = :diamond,   ms = 4)
    scatter!(p, D, E["mixed"];  label = "смешанный (стр→стл)", marker = :utriangle, ms = 4)
    scatter!(p, D, E["lapack"]; label = "LAPACK",            marker = :square,    ms = 4)

    vline!(p, [1.0]; label = "d = 1 (граница преобладания)",
           linestyle = :dash, color = :green, linewidth = 2)

    println("   i |   d(A)   |       κ₂(A)  | simple       row          col          mixed        lapack")
    println("-"^100)
    for i in 1:length(D)
        @printf("%4d | %8.3f | %11.4e | %11.4e %11.4e %11.4e %11.4e %11.4e\n",
                i, D[i], K[i],
                E["simple"][i], E["row"][i], E["col"][i],
                E["mixed"][i],  E["lapack"][i])
    end

    return p, D, K, E
end

p, D, K, E = lab_dominance_vs_error(100;
                                    k_range  = range(0.0, 3.0, length = 80),
                                    n_trials = 3)
display(p)