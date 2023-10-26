"""
    x = xvec(λ,N)

Create `x` values with periodicity for box specified by length `λ`, using `N` points.
"""
xvec(L,N) = LinRange(-L/2,L/2,N+1)[2:end] |> collect

"""
    x = xvec0(λ,N)

Create `x` values with periodicity for box specified by length `λ`, using `N` points. Range from 0 to L-L/N.
"""
xvec0(L,N) = LinRange(0,L,N+1)[1:end-1] |> collect

"""
    k = kvec(L,N)

Create `k` values with correct periodicity for box specified by length `λ` for number of points `N`.
"""
kvec(L,N) = fftfreq(N)*N*2*π/L |> Vector


"""
    X = xvecs(L,N)

Create a tuple containing the spatial coordinate array for each spatial dimension.
"""
function xvecs(L,N)
    X = []
    for (λ,ν) in zip(L,N)
        x = xvec(λ,ν)
        push!(X,x)
    end
    return X |> Tuple
end

"""
    X = xvecs0(L,N)

Create a tuple containing the spatial coordinate array for each spatial dimension using xvec0 instead of xvec.
"""
function xvecs0(L,N)
    X = []
    for (λ,ν) in zip(L,N)
        x = xvec0(λ,ν)
        push!(X,x)
    end
    return X |> Tuple
end

"""
    K = kvecs(L,N)

Create a tuple containing the spatial coordinate array for each spatial dimension.
"""
function kvecs(L,N)
    K = []
    for (λ,ν) in zip(L,N)
        k = kvec(λ,ν)
        push!(K,k)
    end
    return K |> Tuple
end

"""
    k² = k2(K)

Create the kinetic energy array `k²` on the `k`-grid defined by the tuple `K`.
"""
function k2(K)
    kind = Iterators.product(K...)
    return map(k-> sum(abs2.(k)),kind)
end

"""
    X,K,dX,dK = xk_arrays(L,N)

Create all `x` and `k` arrays for box specified by tuples `L=(Lx,...)` and `N=(Nx,...)`.
For convenience, differentials `dX`, `dK` are also reaturned. `L` and `N` must be tuples of equal length.
"""
function xk_arrays(L,N)
    @assert length(L) == length(N)
    X = xvecs(L,N)
    K = kvecs(L,N)
    dX = Float64[]; dK = Float64[]
    for j ∈ eachindex(X)
        x = X[j]; k = K[j]
        dx = x[2]-x[1]; dk = k[2]-k[1]
        push!(dX,dx)
        push!(dK,dk)
    end
    dX = dX |> Tuple
    dK = dK |> Tuple
    return X,K,dX,dK
end

"""
    X,K,dX,dK = xk_arrays0(L,N)

Create all `x` and `k` arrays for box specified by tuples `L=(Lx,...)` and `N=(Nx,...)` using xvecs0 rather than xvecs.
For convenience, differentials `dX`, `dK` are also reaturned. `L` and `N` must be tuples of equal length.
"""
function xk_arrays0(L,N)
    @assert length(L) == length(N)
    X = xvecs0(L,N)
    K = kvecs(L,N)
    dX = Float64[]; dK = Float64[]
    for j ∈ eachindex(X)
        x = X[j]; k = K[j]
        dx = x[2]-x[1]; dk = k[2]-k[1]
        push!(dX,dx)
        push!(dK,dk)
    end
    dX = dX |> Tuple
    dK = dK |> Tuple
    return X,K,dX,dK
end

"""
    Dx,Dk = dfft(x,k)

Measures that make `fft`, `ifft` 2-norm preserving.
Correct measures for mapping between `x`- and `k`-space.
"""
function dfft(x,k)
    dx = x[2]-x[1]; dk = k[2]-k[1]
    Dx = dx/sqrt(2*pi)
    Dk = length(k)*dk/sqrt(2*pi)
    return Dx, Dk
end

"""
    DX,DK = fft_differentials(X,K)

Evalutes tuple of measures that make `fft`, `ifft` 2-norm preserving for each
`x` or `k` dimension.
"""
function fft_differentials(X,K)
    M = length(X)
    DX = zeros(M); DK = zeros(M)
    for i ∈ eachindex(X)
        DX[i],DK[i] = dfft(X[i],K[i])
    end
    return DX,DK
end

"""
    DΣ = correlation_measure(X,K)

Evalutes measure for auto_correlate, cross_correlate and convolve
"""
function correlation_measure(X,K)
    M = length(X)
    DX = zeros(M); DK = zeros(M)
    for i ∈ eachindex(X)
        DX[i],DK[i] = dfft(X[i],K[i])
    end
    DΣ = prod(DX)^2*prod(DK)*(2*pi)^(M/2)
    return DΣ
end

"""
    P = fft_planner(X,K,f,wisdom=nothing)

Evalutes tuple of planners for FFTs for dimensions 1, 2, 3.
f can equal "e" (uses FFTW.ESTIMATE), "m" (uses FFTW.MEASURE), or "p" (uses FFTW.PATIENT)
It is strongly recommended that you create a wisdom file in your home directory (say, ~/fftw_wisdom)
if one does not already exist and pass it to fft_planner as wisdom = "path_to_wisdom".
If the wisdom file does not contain an FFT plan for the array sizes you are using, it is recommended
that you select f = "m" or "p" and then run FFTW.export_wisdom(path_to_wisdom).
N.B. Do NOT select f = "p" without passing a valid wisdom file path and do NOT do so if FFTW.PATIENT
plans already exist for the arrays under consideration; otherwise you will waste a LOT of time.
"""
function fft_planner(X,K,f,wisdom=nothing)
           M = length(X)
           if M == 1
               planstate[i,j,k] = exp.(im*K[1][2]*X[1]);
           elseif M == 2
               @tullio planstate[i,j] := exp.(im*K[1][2]*X[1])[i]*exp.(im*K[2][2]*X[2])[j];
           elseif M == 3
               @tullio planstate[i,j,k] := exp.(im*K[1][2]*X[1])[i]*exp.(im*K[2][2]*X[2])[j]*exp.(im*K[3][2]*X[3])[k];
           end
           if !isnothing(wisdom)
               FFTW.import_wisdom(wisdom)
           end
           if f == "e"
               Pall = FFTW.plan_fft(copy(planstate), flags=FFTW.ESTIMATE);
               Pbig = FFTW.plan_fft(zeropad(planstate), flags=FFTW.ESTIMATE);
           elseif f == "m"
               Pall = FFTW.plan_fft(copy(planstate), flags=FFTW.MEASURE);
               Pbig = FFTW.plan_fft(zeropad(planstate), flags=FFTW.MEASURE);
           elseif f == "p"
               Pall = FFTW.plan_fft(copy(planstate), flags=FFTW.PATIENT);
               Pbig = FFTW.plan_fft(zeropad(planstate), flags=FFTW.PATIENT);
               if !isnothing(wisdom)
                   FFTW.import_wisdom(wisdom)
               end
           end
       return (Pall, Pbig)
end
