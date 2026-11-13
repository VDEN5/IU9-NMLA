using LinearAlgebra
using Polynomials
using Random
using Printf

function krylov_method(A::Matrix{Float64}; y0::Union{Nothing,Vector{Float64}}=nothing)
    n = size(A, 1)
    if y0 === nothing
        y0 = zeros(n); y0[1] = 1.0
    end

    y = Vector{Vector{Float64}}(undef, n + 1)
    y[1] = copy(y0)
    for k in 1:n
        y[k+1] = A * y[k]
    end

    #y_n = p1*y_{n-1} + p2*y_{n-2} + ... + pn*y_0, затем делим на y_n
    Y = hcat([y[n - k] for k in 0:n-1]...)
    p = Y \ y[n+1]   
    coeffs_inc = vcat(-reverse(p), 1.0)#оно 1 равно
    poly = Polynomial(coeffs_inc)
    lambdas = sort(real.(roots(poly)))#через либу решение

    #Горнер
    vectors = Vector{Vector{Float64}}()
    for λ in lambdas
        q = zeros(n - 1)
        q_prev = 1.0
        for j in 1:(n-1)
            q[j] = λ * q_prev - p[j]
            q_prev = q[j]
        end
        x = copy(y[n])
        for j in 1:(n-1)
            x .+= q[j] .* y[n - j]
        end
        push!(vectors, x ./ norm(x))#собственные векторы
    end

    return lambdas, vectors, p, poly
end

function danilevsky_method(A::Matrix{Float64})
    n = size(A, 1)
    D = copy(A)
    B_total = Matrix{Float64}(I, n, n)

    for k in 1:(n-1)
        r = n - k#что вставляем
        s = n - k + 1#куда вставляем
        p = D[s, r]

        if abs(p) < 1e-12
            error("Danilevsky: нулевой опорный элемент на шаге k=$k (D[$s,$r]=$p). Нужна перестановка.")
        end

        B_k = Matrix{Float64}(I, n, n)
        B_k_inv = Matrix{Float64}(I, n, n)

        for j in 1:n#тупо формулы
            if j == r
                B_k[r, j] = 1.0 / p
            else
                B_k[r, j] = -D[s, j] / p
            end
            B_k_inv[r, j] = D[s, j] 
        end

        D = B_k_inv * D * B_k
        B_total = B_total * B_k
    end

    P = D
    p_coeffs = P[1, :] 

    coeffs_inc = vcat(-reverse(p_coeffs), 1.0)
    poly = Polynomial(coeffs_inc)#тот самый многочлен
    lambdas = sort(real.(roots(poly)))

    vectors = Vector{Vector{Float64}}()
    for λ in lambdas
        y_vec = [λ^(n - k) for k in 1:n]
        x = B_total * y_vec
        push!(vectors, x ./ norm(x))
    end

    return lambdas, vectors, B_total, P, poly
end

function random_symmetric_3x3()
    n = 3
    M = rand(-5:5, n, n)
    A = Float64.(M + M')
    for i in 1:n
        A[i, i] = sum(abs(A[i, j]) for j in 1:n if j != i) + rand(1:3)
    end
    return A
end

Random.seed!(42)
A = random_symmetric_3x3()

println("A")
display(A)
println("МЕТОД КРЫЛОВА")
lk, vk, pk, polyk = krylov_method(A)
println("Собственные значения: ", lk)
for (i, v) in enumerate(vk)
    @printf("  x_%d = [% .6f, % .6f, % .6f]\n", i, v[1], v[2], v[3])
end

println("МЕТОД ДАНИЛЕВСКОГО")
ld, vd, B, P, polyd = danilevsky_method(A)
println("Собственные значения: ", ld)
for (i, v) in enumerate(vd)
    @printf("  x_%d = [% .6f, % .6f, % .6f]\n", i, v[1], v[2], v[3])
end

for i in 1:length(vk)
    for j in (i+1):length(vk)
        @printf("  x_%d · x_%d  (Крылов) = % .3e   (Данилевский) = % .3e\n",
                i, j, dot(vk[i], vk[j]), dot(vd[i], vd[j]))
    end
end