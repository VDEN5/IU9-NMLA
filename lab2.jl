using LinearAlgebra
using LinearAlgebra.LAPACK
using Plots
plotlyjs()

function lapack_solve(A, b)
    LU, ipiv, info = LAPACK.getrf!(copy(A))
    info != 0 && error("getrf! info = $info")
    return LAPACK.getrs!('N', LU, ipiv, copy(b))
end

function lapack_inv(A)
    LU, ipiv, info = LAPACK.getrf!(copy(A))
    info != 0 && error("getrf! info = $info")
    return LAPACK.getri!(LU, ipiv)
end

function lapack_svdvals(A)
    U, s, Vt = LAPACK.gesdd!('N', copy(A))
    return s
end


function generate_well_conditioned(n::Int)
    return Matrix(Diagonal(collect(1.0:n)))
end

function generate_ill_conditioned(n::Int)
    return [1.0 / (i + j - 1) for i in 1:n, j in 1:n]
end

function report_conditioning(A; name::String = "A")
    σ  = lapack_svdvals(A)
    κ2 = σ[1] / σ[end]
    println(name, ":  n = ", size(A, 1),
            ",  σ_min = ", round(σ[end], sigdigits = 4),
            ",  σ_max = ", round(σ[1],   sigdigits = 4),
            ",  cond₂ = ", round(κ2,     sigdigits = 4))
    return κ2
end

function perturbation_plots(A, b;
                            grid = range(0.0, 1.0, length = 30))
    n = size(A, 1)
    @assert size(A) == (n, n)
    @assert length(b) == n

    println("n = ", n, ",  сетка: ", length(grid), "×", length(grid))

    x    = lapack_solve(A, b)
    Ainv = lapack_inv(A)

    println("x = ", x)
    println("‖A·A⁻¹ - I‖₂ = ", norm(A * Ainv - I, 2))

    α = collect(grid)
    β = collect(grid)
    N = length(α)

    X = [ α[i] for i in 1:N, j in 1:N ]
    Y = [ β[j] for i in 1:N, j in 1:N ]

    function compute(p)
        Z_form = zeros(N, N)
        Z_real = zeros(N, N)

        if p == 2
            σA    = lapack_svdvals(A)
            σAinv = lapack_svdvals(Ainv)
            κ = σAinv[1] * σA[1]

            println("\np = 2:  σ_max(A) = ", σA[1],
                    ",  σ_max(A⁻¹) = ", σAinv[1],
                    ",  κ₂(A) = ", κ)

            for i in 1:N, j in 1:N
                dA = α[i] .* Matrix(I, n, n)
                db = β[j] .* ones(n)

                Z_form[i, j] = κ * ( norm(db, 2) / norm(b, 2) +
                                     lapack_svdvals(dA)[1] / σA[1] )

                xp = lapack_solve(A + dA, b + db)
                dx = xp - x
                Z_real[i, j] = norm(dx, 2) / norm(x, 2)
            end

        elseif p == Inf
            m∞(M) = maximum(sum(abs, M, dims = 2))
            v∞(v) = maximum(abs, v)

            κ = m∞(Ainv) * m∞(A)

            println("\np = ∞:  ‖A‖∞ = ", m∞(A),
                    ",  ‖A⁻¹‖∞ = ", m∞(Ainv),
                    ",  κ∞(A) = ", κ)

            for i in 1:N, j in 1:N
                dA = α[i] .* Matrix(I, n, n)
                db = β[j] .* ones(n)

                Z_form[i, j] = κ * ( v∞(db) / v∞(b) + m∞(dA) / m∞(A) )

                xp = lapack_solve(A + dA, b + db)
                dx = xp - x
                Z_real[i, j] = v∞(dx) / v∞(x)
            end
        else
            error("p должно быть 2 или Inf")
        end

        println("  Z_form: [", minimum(Z_form), ", ", maximum(Z_form), "]")
        println("  Z_real: [", minimum(Z_real), ", ", maximum(Z_real), "]")

        return Z_form, Z_real
    end

    Zform2, Zreal2 = compute(2)
    ZformI, ZrealI = compute(Inf)

    s1 = surface(X, Y, Zform2;
                 label = "δX формула", cmap = :viridis, alpha = 0.85,
                 xlabel = "dA", ylabel = "db", zlabel = "δX",
                 title = "p = 2")
    surface!(s1, X, Y, Zreal2;
                 label = "δX = ‖dx‖/‖x‖", cmap = :plasma, alpha = 0.85)

    s2 = surface(X, Y, ZformI;
                 label = "δX формула", cmap = :viridis, alpha = 0.85,
                 xlabel = "dA", ylabel = "db", zlabel = "δX",
                 title = "p = ∞")
    surface!(s2, X, Y, ZrealI;
                 label = "δX = ‖dx‖/‖x‖", cmap = :plasma, alpha = 0.85)

    return plot(s1, s2, layout = (1, 2), size = (1100, 500))
end


n = 50
b = ones(n)

A_good = generate_well_conditioned(n)
A_bad  = generate_ill_conditioned(n)


println("### Хорошо обусловленная матрица ###")
report_conditioning(A_good; name = "A_good")

println("\n### Плохо обусловленная матрица ###")
report_conditioning(A_bad; name = "A_bad")

p_good = perturbation_plots(A_good, b)
p_bad  = perturbation_plots(A_bad,  b)

display(p_good)
display(p_bad)