module QuantumFluidSpectra

using LinearAlgebra
using Tullio
using Hwloc
using FFTW 
using SpecialFunctions
using PaddedViews
using UnPack
using ThreadsX
using Referenceables

# fallback since fast_hypot is 2 argument only
@fastmath hypot(x::Float64, y::Float64, z::Float64)=sqrt(x^2+y^2+z^2)
export hypot 

abstract type Field end
struct Psi{D} <: Field
    ψ::Array{Complex{Float64},D}
    X::NTuple{D}
    K::NTuple{D}
end

struct Psi_qper2{D} <: Field
    ψ::Array{Complex{Float64},D}
    X::NTuple{D}
    K::NTuple{D}
    Γ::NTuple{1}
    s::NTuple{1}
end

struct Psi_qper3{D} <: Field
    ψ::Array{Complex{Float64},D}
    X::NTuple{D}
    K::NTuple{D}
    Γ::NTuple{3}
    s::NTuple{3}
end

include("arrays.jl")
include("analysis.jl")

export Psi, Psi_qper2, Psi_qper3, xvecs, kvecs
export auto_correlate, cross_correlate, auto_correlate_batch, auto_correlate_batch_nopad
export bessel_reduce, sinc_reduce, sinc_reduce_real, sinc_reduce_complex, sinc_reduce_real_nopad, sinc_reduce_complex_nopad, gv, gv3
export log10range, convolve

export xk_arrays, xk_arrays0, fft_differentials, correlation_measure, fft_planner
export gradient, gradient_qper, velocity, velocity_qper, weightedvelocity, weightedvelocity_qper, current, current_qper
export energydecomp, energydecomp_qper, helmholtz, kinetic_density, full_spectrum, full_current_spectrum
export incompressible_spectrum, compressible_spectrum, decomposed_spectra, qpressure_spectrum, decomposed_spectra_nopad, qpressure_spectrum_nopad
export incompressible_current_spectrum, compressible_current_spectrum, decomposed_current_spectra, decomposed_current_spectra_nopad
export incompressible_density, incompressible_density_qper, compressible_density, compressible_density_qper, qpressure_density
export ic_density, ic_density_qper, iq_density, iq_density_qper, cq_density, cq_density_qper
export density_spectrum, trap_spectrum
export kdensity, wave_action

end
