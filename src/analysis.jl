"""
	gradient(psi::Psi{D})

Compute the `D` vector gradient components of a wavefunction `Psi` of spatial dimension `D`.
The `D` gradient components returned are `D`-dimensional arrays.
"""
function gradient(psi::Psi{1})
	@unpack ψ,K = psi; kx = K[1] 
    	ϕ = fft(ψ)
	ψx = ifft(im*kx.*ϕ)
    return ψx
end

function gradient(psi::Psi{2})
	@unpack ψ,K = psi; kx,ky = K 
	ϕ = fft(ψ)
	ψx = ifft(im*kx.*ϕ)
	ψy = ifft(im*ky'.*ϕ)
	return ψx,ψy
end


function gradient(psi::Psi{3})
	@unpack ψ,K = psi; kx,ky,kz = K 
	ϕ = fft(ψ)
	ψx = ifft(im*kx.*ϕ)
	ψy = ifft(im*ky'.*ϕ)
	ψz = ifft(im*reshape(kz,1,1,length(kz)).*ϕ)
	return ψx,ψy,ψz
end

function gradient(Pall, psi::Psi{1})
	@unpack ψ,K = psi; kx = K[1] 
    	ϕ = Pall*ψ
	ψx = inv(Pall)*(im*kx.*ϕ)
    return ψx
end

function gradient(Pall, psi::Psi{2})
	@unpack ψ,K = psi; kx,ky = K 
	ϕ = Pall*ψ
	ψx = inv(Pall)*(im*kx.*ϕ)
	ψy = inv(Pall)*(im*ky'.*ϕ)
	return ψx,ψy
end

function gradient(Pall, psi::Psi{3})
	@unpack ψ,K = psi; kx,ky,kz = K 
	ϕ = Pall*ψ
	ψx = inv(Pall)*(im*kx.*ϕ)
	ψy = inv(Pall)*(im*ky'.*ϕ)
	ψz = inv(Pall)*(im*reshape(kz,1,1,length(kz)).*ϕ)
	return ψx,ψy,ψz
end

"""
	gradient_qper(psi::Psi{D})

Compute the `D` vector gradient components of a wavefunction `Psi` of spatial dimension `D`.
The `D` gradient components returned are `D`-dimensional arrays.
Uses quasiperiodic boundary conditions as determined by the circulation tuple `Γ` and the gauge transformation tuple `s`.
This is not well-defined when D = 1.
"""
function gradient_qper(psi::Psi_qper2{2})
	@unpack ψ,X,K,Γ,s = psi; kx,ky = K; x,y = X
	ϕ1 = fft(exp.(-2im*π*x.*(Γ[1]*y' .- s[1])).*ψ,[1])
	ϕ2 = fft(exp.(2im*π*s[1]*x).*ψ,[2])
	ψx = exp.(-2im*π*x.*(Γ[1]*y' .- s[1])).*ifft(im*(kx .+ 2π*(Γ[1]*y' .- s[1])).*ϕ1,[1])
	ψy = exp.(2im*π*s[1]*x).*ifft(im*(ky' .- 2π*Γ[1]*x).*ϕ2,[2])
	return ψx,ψy
end

function gradient_qper(psi::Psi_qper3{3})
	@unpack ψ,X,K,Γ,s = psi; kx,ky,kz = K; x,y,z = X
	ϕ1 = fft(exp.(2im*π*(s[3]*reshape(z,1,1,length(z)) .- x.*(Γ[1]*y' .- s[1]))).*ψ,[1])
	ϕ2 = fft(exp.(2im*π*(s[1]*x .- y'.*(Γ[2]*reshape(z,1,1,length(z)) .- s[2]))).*ψ,[2])
	ϕ3 = fft(exp.(2im*π*(s[2]*y' .- reshape(z,1,1,length(z)).*(Γ[3]*x .- s[3]))).*ψ,[3])
	ψx = ifft(im*(kx.- 2π*(Γ[3]*reshape(z,1,1,length(z)) .- Γ[1]*y' .+ s[1])).*ϕ1,[1])
	ψx .*= exp.(-2im*π*(s[3]*reshape(z,1,1,length(z)) .- x.*(Γ[1]*y' .- s[1])))
	ψy = ifft(im*(ky' .- 2π*(Γ[1]*x .- Γ[2]*reshape(z,1,1,length(z)) .+ s[2])).*ϕ2,[2])
	ψy .*= exp.(-2im*π*(s[1]*x .- y'.*(Γ[2]*reshape(z,1,1,length(z)) .- s[2])))
	ψz = ifft(im*(reshape(kz,1,1,length(z)).- 2π*(Γ[2]*y' .- Γ[3]*x .+ s[3])).*ϕ3,[3])
	ψz .*= exp.(-2im*π*(s[2]*y' .- reshape(z,1,1,length(z)).*(Γ[3]*x .- s[3])))
	return ψx,ψy,ψz
end

"""
	current(psi::Psi{D})

Compute the `D` current components of an `Psi` of spatial dimension `D`.
The `D` cartesian components returned are `D`-dimensional arrays.
"""
function current(psi::Psi{1})
	@unpack ψ = psi 
	ψx = gradient(psi)
	jx = @. imag(conj(ψ)*ψx)
    return jx
end

function current(psi::Psi{2},Ω = 0)
	@unpack ψ,X = psi 
    	x,y = X
    	ψx,ψy = gradient(psi)
	jx = @. imag(conj(ψ)*ψx) + Ω*abs2(ψ)*y'  
	jy = @. imag(conj(ψ)*ψy) - Ω*abs2(ψ)*x 
	return jx,jy
end

function current(psi::Psi{3})
    	@unpack ψ = psi 
    	ψx,ψy,ψz = gradient(psi)
	jx = @. imag(conj(ψ)*ψx)
	jy = @. imag(conj(ψ)*ψy)
	jz = @. imag(conj(ψ)*ψz)
	return jx,jy,jz
end

function current(Pall, psi::Psi{1})
	@unpack ψ = psi 
	ψx = gradient(Pall, psi)
	jx = @. imag(conj(ψ)*ψx)
    return jx
end

function current(Pall, psi::Psi{2},Ω = 0)
	@unpack ψ,X = psi 
    	x,y = X
    	ψx,ψy = gradient(Pall, psi)
	jx = @. imag(conj(ψ)*ψx) + Ω*abs2(ψ)*y'  
	jy = @. imag(conj(ψ)*ψy) - Ω*abs2(ψ)*x 
	return jx,jy
end

function current(Pall, psi::Psi{3})
    	@unpack ψ = psi 
    	ψx,ψy,ψz = gradient(Pall, psi)
	jx = @. imag(conj(ψ)*ψx)
	jy = @. imag(conj(ψ)*ψy)
	jz = @. imag(conj(ψ)*ψz)
	return jx,jy,jz
end

"""
	current_qper(psi::Psi{D})

Compute the `D` current components of an `Psi` of spatial dimension `D`.
The `D` cartesian components returned are `D`-dimensional arrays.
Uses quasiperiodic boundary conditions and is not well-defined when D = 1.
"""
function current_qper(psi::Psi_qper2{2})
	@unpack ψ = psi 
    	ψx,ψy = gradient_qper(psi)
	jx = @. imag(conj(ψ)*ψx)
	jy = @. imag(conj(ψ)*ψy)
	return jx,jy
end

function current_qper(psi::Psi_qper3{3})
    	@unpack ψ = psi 
    	ψx,ψy,ψz = gradient_qper(psi)
	jx = @. imag(conj(ψ)*ψx)
	jy = @. imag(conj(ψ)*ψy)
	jz = @. imag(conj(ψ)*ψz)
	return jx,jy,jz
end

"""
	velocity(psi::Psi{D})

Compute the `D` velocity components of an `Psi` of spatial dimension `D`.
The `D` velocities returned are `D`-dimensional arrays.
"""
function velocity(psi::Psi{1})
	@unpack ψ = psi
    	ψx = gradient(psi)
	vx = @. imag(conj(ψ)*ψx)/abs2(ψ)
    	@. vx[isnan(vx)] = zero(vx[1])
	return vx
end

function velocity(psi::Psi{2},Ω = 0)
	@unpack ψ,X = psi
    	x,y = X
    	ψx,ψy = gradient(psi)
    	rho = abs2.(ψ)
	vx = @. imag(conj(ψ)*ψx)/rho + Ω*y'  
	vy = @. imag(conj(ψ)*ψy)/rho - Ω*x 
    	@. vx[isnan(vx)] = zero(vx[1])
    	@. vy[isnan(vy)] = zero(vy[1])
	return vx,vy
end

function velocity(psi::Psi{3})
	@unpack ψ = psi
	rho = abs2.(ψ)
    	ψx,ψy,ψz = gradient(psi)
	vx = @. imag(conj(ψ)*ψx)/rho
	vy = @. imag(conj(ψ)*ψy)/rho
	vz = @. imag(conj(ψ)*ψz)/rho
    	@. vx[isnan(vx)] = zero(vx[1])
    	@. vy[isnan(vy)] = zero(vy[1])
    	@. vz[isnan(vz)] = zero(vz[1])
	return vx,vy,vz
end

function velocity(Pall, psi::Psi{1})
	@unpack ψ = psi
    	ψx = gradient(Pall, psi)
	vx = @. imag(conj(ψ)*ψx)/abs2(ψ)
    	@. vx[isnan(vx)] = zero(vx[1])
	return vx
end

function velocity(Pall, psi::Psi{2},Ω = 0)
	@unpack ψ,X = psi
    	x,y = X
    	ψx,ψy = gradient(Pall, psi)
    	rho = abs2.(ψ)
	vx = @. imag(conj(ψ)*ψx)/rho + Ω*y'  
	vy = @. imag(conj(ψ)*ψy)/rho - Ω*x 
    	@. vx[isnan(vx)] = zero(vx[1])
    	@. vy[isnan(vy)] = zero(vy[1])
	return vx,vy
end

function velocity(Pall, psi::Psi{3})
	@unpack ψ = psi
	rho = abs2.(ψ)
    	ψx,ψy,ψz = gradient(Pall, psi)
	vx = @. imag(conj(ψ)*ψx)/rho
	vy = @. imag(conj(ψ)*ψy)/rho
	vz = @. imag(conj(ψ)*ψz)/rho
    	@. vx[isnan(vx)] = zero(vx[1])
    	@. vy[isnan(vy)] = zero(vy[1])
    	@. vz[isnan(vz)] = zero(vz[1])
	return vx,vy,vz
end

"""
	velocity_qper(psi::Psi{D})

Compute the `D` velocity components of an `Psi` of spatial dimension `D`.
The `D` velocities returned are `D`-dimensional arrays.
Uses quasiperiodic boundary conditions and is not well-defined when D = 1.
"""
function velocity_qper(psi::Psi_qper2{2})
	@unpack ψ = psi
    	ψx,ψy = gradient_qper(psi)
    	rho = abs2.(ψ)
	vx = @. imag(conj(ψ)*ψx)/rho
	vy = @. imag(conj(ψ)*ψy)/rho 
    	@. vx[isnan(vx)] = zero(vx[1])
    	@. vy[isnan(vy)] = zero(vy[1])
	return vx,vy
end

function velocity_qper(psi::Psi_qper3{3})
	@unpack ψ = psi
	rho = abs2.(ψ)
    	ψx,ψy,ψz = gradient_qper(psi)
	vx = @. imag(conj(ψ)*ψx)/rho
	vy = @. imag(conj(ψ)*ψy)/rho
	vz = @. imag(conj(ψ)*ψz)/rho
    	@. vx[isnan(vx)] = zero(vx[1])
    	@. vy[isnan(vy)] = zero(vy[1])
    	@. vz[isnan(vz)] = zero(vz[1])
	return vx,vy,vz
end

"""
	weightedvelocity(psi::Psi{D})

Compute the `D` weighted velocity, i.e. sqrt(n)*v, components of an `Psi` of spatial dimension `D`.
The `D` velocities returned are `D`-dimensional arrays.
"""
function weightedvelocity(psi::Psi{1})
	@unpack ψ = psi
    	ψx = gradient(psi)
	wx = @. imag(conj(ψ)*ψx)/abs(ψ)
    	@. wx[isnan(wx)] = zero(wx[1])
	return wx
end

function weightedvelocity(psi::Psi{2},Ω = 0)
	@unpack ψ,X = psi
    	x,y = X
    	ψx,ψy = gradient(psi)
    	rhosq = abs.(ψ)
	wx = @. imag(conj(ψ)*ψx)/rhosq + Ω*y'.*rhosq 
	wy = @. imag(conj(ψ)*ψy)/rhosq - Ω*x.*rhosq
    	@. wx[isnan(wx)] = zero(wx[1])
    	@. wy[isnan(wy)] = zero(wy[1])
	return vx,vy
end

function weightedvelocity(psi::Psi{3})
	@unpack ψ = psi
	rhosq = abs.(ψ)
    	ψx,ψy,ψz = gradient(psi)
	wx = @. imag(conj(ψ)*ψx)/rhosq
	wy = @. imag(conj(ψ)*ψy)/rhosq
	wz = @. imag(conj(ψ)*ψz)/rhosq
    	@. wx[isnan(wx)] = zero(wx[1])
    	@. wy[isnan(wy)] = zero(wy[1])
    	@. wz[isnan(wz)] = zero(wz[1])
	return wx,wy,wz
end


function weightedvelocity(Pall, psi::Psi{1})
	@unpack ψ = psi
    	ψx = gradient(Pall, psi)
	wx = @. imag(conj(ψ)*ψx)/abs(ψ)
    	@. wx[isnan(wx)] = zero(wx[1])
	return wx
end

function weightedvelocity(Pall, psi::Psi{2},Ω = 0)
	@unpack ψ,X = psi
    	x,y = X
    	ψx,ψy = gradient(Pall, psi)
    	rhosq = abs.(ψ)
	wx = @. imag(conj(ψ)*ψx)/rhosq + Ω*y'.*rhosq
	wy = @. imag(conj(ψ)*ψy)/rhosq - Ω*x.*rhosq
    	@. wx[isnan(wx)] = zero(wx[1])
    	@. wy[isnan(wy)] = zero(wy[1])
	return vx,vy
end

function weightedvelocity(Pall, psi::Psi{3})
	@unpack ψ = psi
	rhosq = abs.(ψ)
    	ψx,ψy,ψz = gradient(Pall, psi)
	wx = @. imag(conj(ψ)*ψx)/rhosq
	wy = @. imag(conj(ψ)*ψy)/rhosq
	wz = @. imag(conj(ψ)*ψz)/rhosq
    	@. wx[isnan(wx)] = zero(wx[1])
    	@. wy[isnan(wy)] = zero(wy[1])
    	@. wz[isnan(wz)] = zero(wz[1])
	return wx,wy,wz
end

"""
	weightedvelocity_qper(psi::Psi{D})

Compute the `D` weighted velocity, i.e. sqrt(n)*v, components of an `Psi` of spatial dimension `D`.
The `D` velocities returned are `D`-dimensional arrays.
Uses quasiperiodic boundary conditions and is not well-defined when D = 1.
"""
function weightedvelocity_qper(psi::Psi_qper2{2})
	@unpack ψ = psi
    	ψx,ψy = gradient_qper(psi)
    	rhosq = abs.(ψ)
	wx = @. imag(conj(ψ)*ψx)/rhosq
	wy = @. imag(conj(ψ)*ψy)/rhosq
    	@. wx[isnan(wx)] = zero(wx[1])
    	@. wy[isnan(wy)] = zero(wy[1])
	return wx,wy
end

function weightedvelocity_qper(psi::Psi_qper3{3})
	@unpack ψ = psi
	rhosq = abs.(ψ)
    	ψx,ψy,ψz = gradient_qper(psi)
	wx = @. imag(conj(ψ)*ψx)/rhosq
	wy = @. imag(conj(ψ)*ψy)/rhosq
	wz = @. imag(conj(ψ)*ψz)/rhosq
    	@. wx[isnan(wx)] = zero(wx[1])
    	@. wy[isnan(wy)] = zero(wy[1])
    	@. wz[isnan(wz)] = zero(wz[1])
	return wx,wy,wz
end

"""
	Wi,Wc = helmholtz(wx,...,kx,...)

Computes a 2 or 3 dimensional Helmholtz decomposition of the vector field with components `wx`, `wy`, or `wx`, `wy`, `wz`. 
`psi` is passed to provide requisite arrays in `k`-space.
Returned fields `Wi`, `Wc` are tuples of cartesian components of incompressible and compressible respectively.
"""
function helmholtz(wx, wy, kx, ky)
    wxk = fft(wx); wyk = fft(wy)
    @cast kw[i,j] := (kx[i] * wxk[i,j] + ky[j] * wyk[i,j])/ (kx[i]^2+ky[j]^2)
    @cast wxkc[i,j] := kw[i,j] * kx[i] 
    @cast wykc[i,j] := kw[i,j] * ky[j]
    wxkc[1] = zero(wxkc[1]); wykc[1] = zero(wykc[1])
    wxki = @. wxk - wxkc
    wyki = @. wyk - wykc
    wxc = ifft(wxkc); wyc = ifft(wykc)
  	wxi = ifft(wxki); wyi = ifft(wyki)
  	Wi = (wxi, wyi); Wc = (wxc, wyc)
    return Wi, Wc
end

function helmholtz(wx, wy, wz, kx, ky, kz)
    wxk = fft(wx); wyk = fft(wy); wzk = fft(wz)
    @cast kw[i,j,k] := (kx[i] * wxk[i,j,k] + ky[j] * wyk[i,j,k] + kz[k] * wzk[i,j,k])/ (kx[i]^2 + ky[j]^2 + kz[k]^2)
    @cast wxkc[i,j,k] := kw[i,j,k] * kx[i]  
    @cast wykc[i,j,k] := kw[i,j,k] * ky[j] 
    @cast wzkc[i,j,k] := kw[i,j,k] * kz[k]  
    wxkc[1] = zero(wxkc[1]); wykc[1] = zero(wykc[1]); wzkc[1] = zero(wzkc[1])
    wxki = @. wxk - wxkc
    wyki = @. wyk - wykc
    wzki = @. wzk - wzkc
    wxc = ifft(wxkc); wyc = ifft(wykc); wzc = ifft(wzkc)
    wxi = ifft(wxki); wyi = ifft(wyki); wzi = ifft(wzki)
  	Wi = (wxi, wyi, wzi); Wc = (wxc, wyc, wzc)
    return Wi, Wc
end

function helmholtz(Pall, wx, wy, kx, ky)
    wxk = Pall*wx; wyk = Pall*wy
    @cast kw[i,j] := (kx[i] * wxk[i,j] + ky[j] * wyk[i,j])/ (kx[i]^2+ky[j]^2)
    @cast dumx[i,j] := kw[i,j] * kx[i] 
    @cast dumy[i,j] := kw[i,j] * ky[j]
    dumx[1] = zero(dumx[1]); dumy[1] = zero(dumy[1])
	wxc = inv(Pall)*dumx; wyc = inv(Pall)*dumy;
	wxk -= dumx
	wyk -= dumy
    wxi = inv(Pall)*(wxk); wyi = inv(Pall)*wyk
  	Wi = (wxi, wyi); Wc = (wxc, wyc)
    return Wi, Wc
end

function helmholtz(Pall, wx, wy, wz, kx, ky, kz)
    wxk = Pall*wx; wyk = Pall*wy; wzk = Pall*wz
    @cast kw[i,j,k] := (kx[i] * wxk[i,j,k] + ky[j] * wyk[i,j,k] + kz[k] * wzk[i,j,k])/ (kx[i]^2 + ky[j]^2 + kz[k]^2)
    @cast dumx[i,j,k] := kw[i,j,k] * kx[i]  
    @cast dumy[i,j,k] := kw[i,j,k] * ky[j] 
    @cast dumz[i,j,k] := kw[i,j,k] * kz[k]  
    dumx[1] = zero(dumx[1]); dumy[1] = zero(dumy[1]); dumz[1] = zero(dumz[1])
    wxc = inv(Pall)*dumx; wyc = inv(Pall)*dumy; wzc = inv(Pall)*dumz
	wxk -= dumx
        wyk -= dumy
	wzk -= dumz
    wxi = inv(Pall)*wxk; wyi = inv(Pall)*wyk; wzi = inv(Pall)*wzk
	Wi = (wxi, wyi, wzi); Wc = (wxc, wyc, wzc)
    return Wi, Wc
end

# function helmholtz(W::NTuple{N,Array{Float64,N}}, psi::Psi{N}) where N
#     return helmholtz(W..., psi)
# end

"""
	et,ei,ec = energydecomp(psi::Xfield{D})

Decomposes the hydrodynamic kinetic energy of `psi`, returning the total `et`, incompressible `ei`,
and compressible `ec` energy densities in position space. `D` can be 2 or 3 dimensions.
"""
function energydecomp(psi::Psi{2})
    @unpack ψ,K = psi; kx,ky = K
    a = abs.(ψ)
    vx, vy = velocity(psi)
    wx = @. a*vx; wy = @. a*vy
    Wi, Wc = helmholtz(wx,wy,kx,ky)
    wxi, wyi = Wi; wxc, wyc = Wc
    et = @. abs2(wx) + abs2(wy); et *= 0.5
    ei = @. abs2(wxi) + abs2(wyi); ei *= 0.5
    ec = @. abs2(wxc) + abs2(wyc); ec *= 0.5
    return et, ei, ec
end

function energydecomp(psi::Psi{3})
	@unpack ψ,K = psi; kx,ky,kz = K
    a = abs.(ψ)
    vx,vy,vz = velocity(psi)
    wx = @. a*vx; wy = @. a*vy; wz = @. a*vz
    Wi, Wc = helmholtz(wx,wy,wz,kx,ky,kz)
    wxi, wyi, wzi = Wi; wxc, wyc, wzc = Wc
    et = @. abs2(wx) + abs2(wy) + abs2(wz); et *= 0.5
    ei = @. abs2(wxi) + abs2(wyi) + abs2(wzi); ei *= 0.5
    ec = @. abs2(wxc) + abs2(wyc) + abs2(wzc); ec *= 0.5
    return et, ei, ec
end

function energydecomp(P, psi::Psi{2})
    @unpack ψ,K = psi; kx,ky = K
    a = abs.(ψ)
    vx, vy = velocity(P[1],psi)
    wx = @. a*vx; wy = @. a*vy
    Wi, Wc = helmholtz(P[1],wx,wy,kx,ky)
    wxi, wyi = Wi; wxc, wyc = Wc
    et = @. abs2(wx) + abs2(wy); et *= 0.5
    ei = @. abs2(wxi) + abs2(wyi); ei *= 0.5
    ec = @. abs2(wxc) + abs2(wyc); ec *= 0.5
    return et, ei, ec
end

function energydecomp(P, psi::Psi{3})
	@unpack ψ,K = psi; kx,ky,kz = K
    a = abs.(ψ)
    vx,vy,vz = velocity(P[1],psi)
    wx = @. a*vx; wy = @. a*vy; wz = @. a*vz
    Wi, Wc = helmholtz(P[1],wx,wy,wz,kx,ky,kz)
    wxi, wyi, wzi = Wi; wxc, wyc, wzc = Wc
    et = @. abs2(wx) + abs2(wy) + abs2(wz); et *= 0.5
    ei = @. abs2(wxi) + abs2(wyi) + abs2(wzi); ei *= 0.5
    ec = @. abs2(wxc) + abs2(wyc) + abs2(wzc); ec *= 0.5
    return et, ei, ec
end

function energydecomp(psi::Psi_qper2{2})
    @unpack ψ,K = psi; kx,ky = K
    a = abs.(ψ)
    vx, vy = velocity(psi)
    wx = @. a*vx; wy = @. a*vy
    Wi, Wc = helmholtz(wx,wy,kx,ky)
    wxi, wyi = Wi; wxc, wyc = Wc
    et = @. abs2(wx) + abs2(wy); et *= 0.5
    ei = @. abs2(wxi) + abs2(wyi); ei *= 0.5
    ec = @. abs2(wxc) + abs2(wyc); ec *= 0.5
    return et, ei, ec
end

function energydecomp(psi::Psi_qper3{3})
	@unpack ψ,K = psi; kx,ky,kz = K
    a = abs.(ψ)
    vx,vy,vz = velocity(psi)
    wx = @. a*vx; wy = @. a*vy; wz = @. a*vz
    Wi, Wc = helmholtz(wx,wy,wz,kx,ky,kz)
    wxi, wyi, wzi = Wi; wxc, wyc, wzc = Wc
    et = @. abs2(wx) + abs2(wy) + abs2(wz); et *= 0.5
    ei = @. abs2(wxi) + abs2(wyi) + abs2(wzi); ei *= 0.5
    ec = @. abs2(wxc) + abs2(wyc) + abs2(wzc); ec *= 0.5
    return et, ei, ec
end

"""
	et,ei,ec = energydecomp_qper(psi::Xfield{D})

Decomposes the hydrodynamic kinetic energy of `psi`, returning the total `et`, incompressible `ei`,
and compressible `ec` energy densities in position space. `D` can be 2 or 3 dimensions.
Uses quasiperiodic boundary conditions.
"""
function energydecomp_qper(psi::Psi_qper2{2})
    @unpack ψ,K = psi; kx,ky = K
    a = abs.(ψ)
    vx, vy = velocity_qper(psi)
    wx = @. a*vx; wy = @. a*vy
    Wi, Wc = helmholtz(wx,wy,kx,ky)
    wxi, wyi = Wi; wxc, wyc = Wc
    et = @. abs2(wx) + abs2(wy); et *= 0.5
    ei = @. abs2(wxi) + abs2(wyi); ei *= 0.5
    ec = @. abs2(wxc) + abs2(wyc); ec *= 0.5
    return et, ei, ec
end

function energydecomp_qper(psi::Psi_qper3{3})
	@unpack ψ,K = psi; kx,ky,kz = K
    a = abs.(ψ)
    vx,vy,vz = velocity_qper(psi)
    wx = @. a*vx; wy = @. a*vy; wz = @. a*vz
    Wi, Wc = helmholtz(wx,wy,wz,kx,ky,kz)
    wxi, wyi, wzi = Wi; wxc, wyc, wzc = Wc
    et = @. abs2(wx) + abs2(wy) + abs2(wz); et *= 0.5
    ei = @. abs2(wxi) + abs2(wyi) + abs2(wzi); ei *= 0.5
    ec = @. abs2(wxc) + abs2(wyc) + abs2(wzc); ec *= 0.5
    return et, ei, ec
end

"""
	zeropad(A)

Zero-pad the array `A` to twice the size with the same element type as `A`.
"""
function zeropad(A)
    S = size(A)
    if any(isodd.(S))
        error("Array dims not divisible by 2")
    end
    nO = 2 .* S
    nI = S .÷ 2

    outer = []
    inner = []

    for no in nO
        push!(outer,(1:no))
    end

    for ni in nI
        push!(inner,(ni+1:ni+2*ni))
    end

    return PaddedView(zero(eltype(A)),A,Tuple(outer),Tuple(inner)) |> collect
end

"""
	log10range(a,b,n)

Create a vector that is linearly spaced in log space, containing `n` values bracketed by `a` and `b`.
"""
function log10range(a,b,n)
	@assert a>0
    x = LinRange(log10(a),log10(b),n)
    return @. 10^x
end

@doc raw"""
	A = convolve(ψ1,ψ2,X,K)

Computes the convolution of two complex fields according to

```math
A(\rho) = \int d^2r\;\psi_1^*(r+\rho)\psi_2(r)
```
using FFTW.
"""
function convolve(ψ1,ψ2,X,K)
    n = length(X)
    DΣ = correlation_measure(X,K)
	ϕ1 = zeropad(conj.(ψ1))
    ϕ2 = zeropad(ψ2)

	χ1 = fft(ϕ1)
	χ2 = fft(ϕ2)
	return ifft(χ1.*χ2)*DΣ |> fftshift
end

function convolve(ψ1,ψ2,X,K,Pbig)
    n = length(X)
    DΣ = correlation_measure(X,K)
	ϕ1 = zeropad(conj.(ψ1))
    ϕ2 = zeropad(ψ2)

	χ1 = (Pbig*ϕ1)
	χ2 = (Pbig*ϕ2)
	return (inv(Pbig)*(χ1.*χ2))*DΣ |> fftshift
end

@doc raw"""
	auto_correlate(ψ,X,K)

Return the auto-correlation integral of a complex field ``\psi``, ``A``, given by

```
A(\rho)=\int d^2r\;\psi^*(r-\rho)\psi(r)
```

defined on a cartesian grid on a cartesian grid using FFTW.

`X` and `K` are tuples of vectors `x`,`y`,`kx`, `ky`.

This method is useful for evaluating spectra from cartesian data.
"""
function auto_correlate(ψ,X,K)
    n = length(X)
    DΣ = correlation_measure(X,K)
    ϕ = zeropad(ψ)
	χ = fft(ϕ)
	return ifft(abs2.(χ))*DΣ |> fftshift
end

auto_correlate(psi::Psi{D}) where D = auto_correlate(psi.ψ,psi.X,psi.K)

function auto_correlate(ψ,X,K,Pbig)
    n = length(X)
    DΣ = correlation_measure(X,K)
    ϕ = zeropad(ψ)
	χ = (Pbig*ϕ)
	return (inv(Pbig)*abs2.(χ))*DΣ |> fftshift
end

auto_correlate(Pbig, psi::Psi{D}) where D = auto_correlate(Pbig, psi.ψ,psi.X,psi.K)

@doc raw"""
	cross_correlate(ψ,X,K)

Cross correlation of complex field ``\psi_1``, and ``\psi_2`` given by

```
A(\rho)=\int d^2r\;\psi_1^*(r-\rho)\psi_2(r)
```

evaluated on a cartesian grid using Fourier convolution.

`X` and `K` are tuples of vectors `x`,`y`,`kx`, `ky`.

This method is useful for evaluating spectra from cartesian data.
"""
function cross_correlate(ψ1,ψ2,X,K)
    n = length(X)
    DΣ = correlation_measure(X,K)
    ϕ1 = zeropad(ψ1)
    ϕ2 = zeropad(ψ2)
	χ1 = fft(ϕ1)
    χ2 = fft(ϕ2)
	return ifft(conj(χ1).*χ2)*DΣ |> fftshift
end
cross_correlate(psi1::Psi{D},psi2::Psi{D}) where D = cross_correlate(psi1.ψ,psi2.ψ,psi1.X,psi1.K)

function cross_correlate(ψ1,ψ2,X,K,Pbig)
    n = length(X)
    DΣ = correlation_measure(X,K)
    ϕ1 = zeropad(ψ1)
    ϕ2 = zeropad(ψ2)
	χ1 = (P*ϕ1)
    χ2 = (P*ϕ2)
	return (inv(Pbig)*(conj(χ1).*χ2))*DΣ |> fftshift
end

function bessel_reduce(k,x,y,C)
    dx,dy = x[2]-x[1],y[2]-y[1]
    Nx,Ny = 2*length(x),2*length(y)
    Lx = x[end] - x[begin] + dx
    Ly = y[end] - y[begin] + dy
    xp = LinRange(-Lx,Lx,Nx+1)[1:Nx]
    yq = LinRange(-Ly,Ly,Ny+1)[1:Ny]
    E = zero(k)
    @tullio E[i] = real(besselj0(k[i]*hypot(xp[p],yq[q]))*C[p,q])
    @. E *= k*dx*dy/2/pi 
    return E 
end

function sinc_reduce(k,x,y,z,C)
    dx,dy,dz = x[2]-x[1],y[2]-y[1],z[2]-z[1]
    Nx,Ny,Nz = 2*length(x),2*length(y),2*length(z)
    Lx = x[end] - x[begin] + dx
    Ly = y[end] - y[begin] + dy
    Lz = z[end] - z[begin] + dz
    xp = LinRange(-Lx,Lx,Nx+1)[1:Nx]
    yq = LinRange(-Ly,Ly,Ny+1)[1:Ny]
    zr = LinRange(-Lz,Lz,Nz+1)[1:Nz]
    E = zero(k)
    @tullio E[i] = real(π*sinc(k[i]*hypot(xp[p],yq[q],zr[r])/π)*C[p,q,r]) 
    @. E *= k^2*dx*dy*dz/2/pi^2  
    return E 
end

function sinc_reduce_real(k,x,y,z,C)
    dx,dy,dz = x[2]-x[1],y[2]-y[1],z[2]-z[1]
    Nx,Ny,Nz = 2*length(x),2*length(y),2*length(z)
    Lx = x[end] - x[begin] + dx
    Ly = y[end] - y[begin] + dy
    Lz = z[end] - z[begin] + dz
    xp = LinRange(-Lx,Lx,Nx+1)[1:Nx]
    yq = LinRange(-Ly,Ly,Ny+1)[1:Ny]
    zr = LinRange(-Lz,Lz,Nz+1)[1:Nz÷2+1]
    E = zero(k)
    hp = sqrt.(xp.^2 .+ yq'.^2 .+ permutedims(zr.*ones(Nz÷2,1,1),[3 2 1]).^2)
	cm = ones(Nx,Ny).*cat(1,fill(2,(1,1,Nz÷2-1)),1,dims=3)
    El = similar(hp)
    for i in eachindex(k)
	ThreadsX.foreach(referenceable(El), hp, cm, C) do b, gp, dm, Dr
	    b[] = π*sinc(k[i]*gp/π)*dm*real(Dr)
	end
	E[i] = @fastmath sum(El)*k[i]^2*dx*dy*dz/2/pi^2
    end
    return E 
end

function sinc_reduce_complex(k,x,y,z,C)
    dx,dy,dz = x[2]-x[1],y[2]-y[1],z[2]-z[1]
    Nx,Ny,Nz = 2*length(x),2*length(y),2*length(z)
    Lx = x[end] - x[begin] + dx
    Ly = y[end] - y[begin] + dy
    Lz = z[end] - z[begin] + dz
    xp = LinRange(-Lx,Lx,Nx+1)[1:Nx]
    yq = LinRange(-Ly,Ly,Ny+1)[1:Ny]
    zr = LinRange(-Lz,Lz,Nz+1)[1:Nz÷2+1]
    E = zero(k)
    hp = sqrt.(xp.^2 .+ yq'.^2 .+ permutedims(zr.*ones(Nz÷2+1,1,1),[3 2 1]).^2)
	cm = ones(Nx,Ny).*cat(1,fill(2,(1,1,Nz÷2-1)),1,dims=3)
    El = similar(hp)
    for i in eachindex(k)
	ThreadsX.foreach(referenceable(El), hp, cm, C) do b, gp, dm, D
	    b[] = π*sinc(k[i]*gp/π)*dm*real(D)
	end
	E[i] = @fastmath sum(El)*k[i]^2*dx*dy*dz/2/pi^2
	E[i] += @fastmath sum(π*sinc(k[i]*hp[:,:,1]/π)*cm[:,:,1]*imag(D[:,:,1]) + π*sinc(k[i]*hp[:,:,Nz÷2+1]/π)*cm[:,:,Nz÷2+1]*imag(D[:,:,Nz÷2+1]))*k[i]^2*dx*dy*dz/2/pi^2
    end
    return E 
end

"""
	kinetic_density(k,ψ,X,K)

Calculates the kinetic energy spectrum for wavefunction ``\\psi``, at the
points `k`. Arrays `X`, `K` should be computed using `makearrays`.
"""
function kinetic_density(k,psi::Psi{2})
    @unpack ψ,X,K = psi; 
    ψx,ψy = gradient(psi)
	cx = auto_correlate(ψx,X,K)
	cy = auto_correlate(ψy,X,K)
    C = @. 0.5(cx + cy)
    return bessel_reduce(k,X...,C)
end

function kinetic_density(k,psi::Psi{3})
    @unpack ψ,X,K = psi;  
    ψx,ψy,ψz = gradient(psi)
	cx = auto_correlate(ψx,X,K)
    cy = auto_correlate(ψy,X,K)
    cz = auto_correlate(ψz,X,K)
    C = @. 0.5(cx + cy + cz)
    return sinc_reduce(k,X...,C)
end

function kinetic_density(P,k,psi::Psi{2})
    @unpack ψ,X,K = psi; 
    ψx,ψy = gradient(psi)
	cx = auto_correlate(ψx,X,K,P[2])
	cy = auto_correlate(ψy,X,K,P[2])
    C = @. 0.5(cx + cy)
    return bessel_reduce(k,X...,C)
end

function kinetic_density(P,k,psi::Psi{3})
    @unpack ψ,X,K = psi;  
    ψx,ψy,ψz = gradient(psi)
	cx = auto_correlate(ψx,X,K,P[2])[:,:,1:length(X[3])+1]
    cy = auto_correlate(ψy,X,K,P[2])[:,:,1:length(X[3])+1]
    cz = auto_correlate(ψz,X,K,P[2])[:,:,1:length(X[3])+1]
    C = @. 0.5(cx + cy + cz)
    return sinc_reduce_complex(k,X...,C)
end

function kinetic_density(k,psi::Psi_qper2{2})
    @unpack ψ,X,K = psi; 
    ψx,ψy = gradient_qper(psi)
	cx = auto_correlate(ψx,X,K)
	cy = auto_correlate(ψy,X,K)
    C = @. 0.5(cx + cy)
    return bessel_reduce(k,X...,C)
end

function kinetic_density(k,psi::Psi_qper3{3})
    @unpack ψ,X,K = psi;  
    ψx,ψy,ψz = gradient_qper(psi)
	cx = auto_correlate(ψx,X,K)[:,:,1:length(X[3])+1]
    cy = auto_correlate(ψy,X,K)[:,:,1:length(X[3])+1]
    cz = auto_correlate(ψz,X,K)[:,:,1:length(X[3])+1]
    C = @. 0.5(cx + cy + cz)
    return sinc_reduce_complex(k,X...,C)
end

"""
	kdensity(k,ψ,X,K)

Calculates the angle integrated momentum density ``|\\phi(k)|^2``, at the
points `k`, with the usual radial weight in `k` space ensuring normalization under ∫dk. Units will be population per wavenumber.  Arrays `X`, `K` should be computed using `makearrays`.
"""
function kdensity(k,psi::Psi{2})  
    @unpack ψ,X,K = psi; 
	C = auto_correlate(ψ,X,K)
    return bessel_reduce(k,X...,C)
end

function kdensity(k,psi::Psi{3})  
    @unpack ψ,X,K = psi; 
	C = auto_correlate(ψ,X,K)
    return sinc_reduce(k,X...,C)
end

function kdensity(P,k,psi::Psi{2})  
    @unpack ψ,X,K = psi; 
	C = auto_correlate(ψ,X,K,P[2])
    return bessel_reduce(k,X...,C)
end

function kdensity(P,k,psi::Psi{3})  
    @unpack ψ,X,K = psi; 
	C = auto_correlate(ψ,X,K,P[2])[:,:,1:length(X[3])+1]
    return sinc_reduce_complex(k,X...,C)
end

function kdensity(k,psi::Psi_qper2{2})  
    @unpack ψ,X,K = psi; 
	C = auto_correlate(ψ,X,K)
    return bessel_reduce(k,X...,C)
end

function kdensity(k,psi::Psi_qper3{3})  
    @unpack ψ,X,K = psi; 
	C = auto_correlate(ψ,X,K)[:,:,1:length(X[3])+1]
    return sinc_reduce_complex(k,X...,C)
end

"""
	wave_action(k,ψ,X,K)
Calculates the angle integrated wave-action spectrum ``|\\phi(\\mathbf{k})|^2``, at the
points `k`, without the radial weight in `k` space ensuring normalization under ∫dk. Units will be population per wavenumber cubed. Isotropy is not assumed. Arrays `X`, `K` should be computed using `makearrays`.
"""
wave_action(k,psi::Psi{2}) = kdensity(k,psi::Psi{2}) ./k 
wave_action(k,psi::Psi{3}) = kdensity(k,psi::Psi{3})./k^2
wave_action(P,k,psi::Psi{2}) = kdensity(P,k,psi::Psi{2}) ./k 
wave_action(P,k,psi::Psi{3}) = kdensity(P,k,psi::Psi{3})./k^2
wave_action_qper(k,psi::Psi_qper2{2}) = kdensity_qper(k,psi::Psi_qper2{2}) ./k 
wave_action_qper(k,psi::Psi_qper3{3}) = kdensity_qper(k,psi::Psi_qper3{3})./k^2

"""
	full_spectrum(k,ψ)

Caculate the velocity correlation spectrum for wavefunction ``\\psi`` without any Helmholtz decomposition being applied.
Input arrays `X`, `K` must be computed using `makearrays`.
"""
function full_spectrum(k,psi::Psi{2},Ω=0.0)
    @unpack ψ,X,K = psi;  
    vx,vy = velocity(psi,Ω)
    a = abs.(ψ)
    wx = @. a*vx; wy = @. a*vy

    cx = auto_correlate(wx,X,K)
    cy = auto_correlate(wy,X,K)
    C = @. 0.5*(cx + cy)
    return bessel_reduce(k,X...,C)
end

function full_spectrum(k,psi::Psi{3})
    @unpack ψ,X,K = psi; 
    vx,vy,vz = velocity(psi)
    a = abs.(ψ)
    wx = @. a*vx; wy = @. a*vy; wz = @. a*vz

    cx = auto_correlate(wx,X,K)
    cy = auto_correlate(wy,X,K)
    cz = auto_correlate(wz,X,K)
    C = @. 0.5*(cx + cy + cz)
    return sinc_reduce(k,X...,C)
end

function full_spectrum(P,k,psi::Psi{2},Ω=0.0)
    @unpack ψ,X,K = psi;  
    wx,wy = weightedvelocity(P[1],psi)

    cx = auto_correlate(wx,X,K,P[2])
    cy = auto_correlate(wy,X,K,P[2])
    C = @. 0.5*(cx + cy)
    return bessel_reduce(k,X...,C)
end

function full_spectrum(P,k,psi::Psi{3})
    @unpack ψ,X,K = psi; 
    wx,wy,wz = weightedvelocity(P[1],psi)

    cx = auto_correlate(wx,X,K,P[2])[:,:,1:length(X[3])+1]
    cy = auto_correlate(wy,X,K,P[2])[:,:,1:length(X[3])+1]
    cz = auto_correlate(wz,X,K,P[2])[:,:,1:length(X[3])+1]
    C = @. 0.5*(cx + cy + cz)
    return sinc_reduce_real(k,X...,C)
end

function full_spectrum(k,psi::Psi_qper2{2})
    @unpack ψ,X,K = psi;  
    wx,wy = weightedvelocity_qper(psi)

    cx = auto_correlate(wx,X,K)
    cy = auto_correlate(wy,X,K)
    C = @. 0.5*(cx + cy)
    return bessel_reduce(k,X...,C)
end

function full_spectrum(k,psi::Psi_qper3{3})
    @unpack ψ,X,K = psi; 
    wx,wy,wz = weightedvelocity_qper(psi)

    cx = auto_correlate(wx,X,K)[:,:,1:length(X[3])+1]
    cy = auto_correlate(wy,X,K)[:,:,1:length(X[3])+1]
    cz = auto_correlate(wz,X,K)[:,:,1:length(X[3])+1]
    C = @. 0.5*(cx + cy + cz)
    return sinc_reduce_real(k,X...,C)
end

"""
	full_current_spectrum(k,ψ)

Caculate the current correlation spectrum for wavefunction ``\\psi`` without any Helmholtz decomposition being applied.
Input arrays `X`, `K` must be computed using `makearrays`.
"""
function full_current_spectrum(k,psi::Psi{2},Ω=0.0)
    @unpack ψ,X,K = psi;  
    jx,jy = current(psi,Ω)

    cx = auto_correlate(jx,X,K)
    cy = auto_correlate(jy,X,K)
    C = @. 0.5*(cx + cy)
    return bessel_reduce(k,X...,C)
end

function full_current_spectrum(k,psi::Psi{3})
    @unpack ψ,X,K = psi; 
    jx,jy,jz = current(psi)

    cx = auto_correlate(jx,X,K)
    cy = auto_correlate(jy,X,K)
    cz = auto_correlate(jz,X,K)
    C = @. 0.5*(cx + cy + cz)
    return sinc_reduce(k,X...,C)
end

function full_current_spectrum(P,k,psi::Psi{2},Ω=0.0)
    @unpack ψ,X,K = psi;  
    jx,jy = current(P[1],psi,Ω)

    cx = auto_correlate(jx,X,K,P[2])
    cy = auto_correlate(jy,X,K,P[2])
    C = @. 0.5*(cx + cy)
    return bessel_reduce(k,X...,C)
end

function full_current_spectrum(P,k,psi::Psi{3})
    @unpack ψ,X,K = psi; 
    jx,jy,jz = current(P[1],psi)

    cx = auto_correlate(jx,X,K,P[2])[:,:,1:length(X[3])+1]
    cy = auto_correlate(jy,X,K,P[2])[:,:,1:length(X[3])+1]
    cz = auto_correlate(jz,X,K,P[2])[:,:,1:length(X[3])+1]
    C = @. 0.5*(cx + cy + cz)
    return sinc_reduce_real(k,X...,C)
end

function full_current_spectrum(k,psi::Psi_qper2{2})
    @unpack ψ,X,K = psi;  
    jx,jy = current_qper(psi)

    cx = auto_correlate(jx,X,K)
    cy = auto_correlate(jy,X,K)
    C = @. 0.5*(cx + cy)
    return bessel_reduce(k,X...,C)
end

function full_current_spectrum(k,psi::Psi_qper3{3})
    @unpack ψ,X,K = psi; 
    jx,jy,jz = current_qper(psi)

    cx = auto_correlate(jx,X,K)[:,:,1:length(X[3])+1]
    cy = auto_correlate(jy,X,K)[:,:,1:length(X[3])+1]
    cz = auto_correlate(jz,X,K)[:,:,1:length(X[3])+1]
    C = @. 0.5*(cx + cy + cz)
    return sinc_reduce_real(k,X...,C)
end

"""
	incompressible_spectrum(k,ψ)

Caculate the incompressible velocity correlation spectrum for wavefunction ``\\psi``, via Helmholtz decomposition.
Input arrays `X`, `K` must be computed using `makearrays`.
"""
function incompressible_spectrum(k,psi::Psi{2},Ω=0.0)
    @unpack ψ,X,K = psi;  
    vx,vy = velocity(psi,Ω)
    a = abs.(ψ)
    wx = @. a*vx; wy = @. a*vy
    Wi, _ = helmholtz(wx,wy,K...)
    wx,wy = Wi

	cx = auto_correlate(wx,X,K)
	cy = auto_correlate(wy,X,K)
    C = @. 0.5*(cx + cy)
    return bessel_reduce(k,X...,C)
end

function incompressible_spectrum(k,psi::Psi{3})
    @unpack ψ,X,K = psi; 
    vx,vy,vz = velocity(psi)
    a = abs.(ψ)
    wx = @. a*vx; wy = @. a*vy; wz = @. a*vz
    Wi, _ = helmholtz(wx,wy,wz,K...)
    wx,wy,wz = Wi

	cx = auto_correlate(wx,X,K)
    cy = auto_correlate(wy,X,K)
    cz = auto_correlate(wz,X,K)
    C = @. 0.5*(cx + cy + cz)
    return sinc_reduce(k,X...,C)
end

function incompressible_spectrum(P,k,psi::Psi{2},Ω=0.0)
    @unpack ψ,X,K = psi;  
    wx,wy = weightedvelocity(P[1],psi,Ω)
    Wi, _ = helmholtz(P[1],wx,wy,K...)
    wx,wy = Wi

	cx = auto_correlate(wx,X,K,P[2])
	cy = auto_correlate(wy,X,K,P[2])
    C = @. 0.5*(cx + cy)
    return bessel_reduce(k,X...,C)
end

function incompressible_spectrum(P,k,psi::Psi{3})
    @unpack ψ,X,K = psi; 
    wx,wy,wz = weightedvelocity(P[1],psi)
    Wi, _ = helmholtz(P[1],wx,wy,wz,K...)
    wx,wy,wz = Wi

	cx = auto_correlate(wx,X,K,P[2])[:,:,1:length(X[3])+1]
    cy = auto_correlate(wy,X,K,P[2])[:,:,1:length(X[3])+1]
    cz = auto_correlate(wz,X,K,P[2])[:,:,1:length(X[3])+1]
    C = @. 0.5*(cx + cy + cz)
    return sinc_reduce_real(k,X...,C)
end

function incompressible_spectrum(k,psi::Psi_qper2{2})
    @unpack ψ,X,K = psi;  
    wx,wy = weightedvelocity_qper(psi)
    a = abs.(ψ)
    wx = @. a*vx; wy = @. a*vy
    Wi, _ = helmholtz(wx,wy,K...)
    wx,wy = Wi

	cx = auto_correlate(wx,X,K)
	cy = auto_correlate(wy,X,K)
    C = @. 0.5*(cx + cy)
    return bessel_reduce(k,X...,C)
end

function incompressible_spectrum(k,psi::Psi_qper3{3})
    @unpack ψ,X,K = psi; 
    wx,wy,wz = weightedvelocity_qper(psi)
    a = abs.(ψ)
    wx = @. a*vx; wy = @. a*vy; wz = @. a*vz
    Wi, _ = helmholtz(wx,wy,wz,K...)
    wx,wy,wz = Wi

	cx = auto_correlate(wx,X,K)[:,:,1:length(X[3])+1]
    cy = auto_correlate(wy,X,K)[:,:,1:length(X[3])+1]
    cz = auto_correlate(wz,X,K)[:,:,1:length(X[3])+1]
    C = @. 0.5*(cx + cy + cz)
    return sinc_reduce_real(k,X...,C)
end

"""
	incompressible_current_spectrum(k,ψ)

Calculate the incompressible current correlation spectrum for wavefunction ``\\psi``, via Helmholtz decomposition.
Input arrays `X`, `K` must be computed using `makearrays`.
"""
function incompressible_current_spectrum(k,psi::Psi{2},Ω=0.0)
    @unpack ψ,X,K = psi;  
    jx,jy = current(psi,Ω)
    Ji, _ = helmholtz(jx,jy,K...)
    jx,jy = Ji

	cx = auto_correlate(jx,X,K)
	cy = auto_correlate(jy,X,K)
    C = @. 0.5*(cx + cy)
    return bessel_reduce(k,X...,C)
end

function incompressible_current_spectrum(k,psi::Psi{3})
    @unpack ψ,X,K = psi; 
    jx,jy,jz = current(psi)
    Ji, _ = helmholtz(jx,jy,jz,K...)
    jx,jy,jz = Ji

	cx = auto_correlate(jx,X,K)
    cy = auto_correlate(jy,X,K)
    cz = auto_correlate(jz,X,K)
    C = @. 0.5*(cx + cy + cz)
    return sinc_reduce(k,X...,C)
end

function incompressible_current_spectrum(k,psi::Psi{2},Ω=0.0)
    @unpack ψ,X,K = psi;  
    jx,jy = current(psi,Ω)
    Ji, _ = helmholtz(P[1],jx,jy,K...)
    jx,jy = Ji

	cx = auto_correlate(jx,X,K,P[2])
	cy = auto_correlate(jy,X,K,P[2])
    C = @. 0.5*(cx + cy)
    return bessel_reduce(k,X...,C)
end

function incompressible_current_spectrum(k,psi::Psi{3})
    @unpack ψ,X,K = psi; 
    jx,jy,jz = current(psi)
    Ji, _ = helmholtz(P[1],jx,jy,jz,K...)
    jx,jy,jz = Ji

	cx = auto_correlate(jx,X,K,P[2])[:,:,1:length(X[3])+1]
    cy = auto_correlate(jy,X,K,P[2])[:,:,1:length(X[3])+1]
    cz = auto_correlate(jz,X,K,P[2])[:,:,1:length(X[3])+1]
    C = @. 0.5*(cx + cy + cz)
    return sinc_reduce_real(k,X...,C)
end

function incompressible_current_spectrum(k,psi::Psi_qper2{2})
    @unpack ψ,X,K = psi;  
    jx,jy = current_qper(psi)
    Ji, _ = helmholtz(jx,jy,K...)
    jx,jy = Ji

	cx = auto_correlate(jx,X,K)
	cy = auto_correlate(jy,X,K)
    C = @. 0.5*(cx + cy)
    return bessel_reduce(k,X...,C)
end

function incompressible_current_spectrum(k,psi::Psi_qper3{3})
    @unpack ψ,X,K = psi; 
    jx,jy,jz = current_qper(psi)
    Ji, _ = helmholtz(jx,jy,jz,K...)
    jx,jy,jz = Ji

	cx = auto_correlate(jx,X,K)[:,:,1:length(X[3])+1]
    cy = auto_correlate(jy,X,K)[:,:,1:length(X[3])+1]
    cz = auto_correlate(jz,X,K)[:,:,1:length(X[3])+1]
    C = @. 0.5*(cx + cy + cz)
    return sinc_reduce_real(k,X...,C)
end

"""
	compressible_spectrum(k,ψ,X,K)

Caculate the compressible kinetic energy spectrum for wavefunction ``\\psi``, via Helmholtz decomposition.
Input arrays `X`, `K` must be computed using `makearrays`.
"""
function compressible_spectrum(k,psi::Psi{2})
    @unpack ψ,X,K = psi 
    vx,vy = velocity(psi)
    a = abs.(ψ)
    wx = @. a*vx; wy = @. a*vy
    _, Wc = helmholtz(wx,wy,K...)
    wx,wy = Wc

	cx = auto_correlate(wx,X,K)
	cy = auto_correlate(wy,X,K)
    C = @. 0.5*(cx + cy)
    return bessel_reduce(k,X...,C)
end

function compressible_spectrum(k,psi::Psi{3})
    @unpack ψ,X,K = psi
    vx,vy,vz = velocity(psi)
    a = abs.(ψ)
    wx = @. a*vx; wy = @. a*vy; wz = @. a*vz
    _, Wc = helmholtz(wx,wy,wz,K...)
    wx,wy,wz = Wc

	cx = auto_correlate(wx,X,K)
    cy = auto_correlate(wy,X,K)
    cz = auto_correlate(wz,X,K)
    C = @. 0.5*(cx + cy + cz)
    return sinc_reduce(k,X...,C)
end

function compressible_spectrum(P,k,psi::Psi{2})
    @unpack ψ,X,K = psi 
    wx,wy = weightedvelocity(P[1],psi)
    _, Wc = helmholtz(P[1],wx,wy,K...)
    wx,wy = Wc

	cx = auto_correlate(wx,X,K,P[2])
	cy = auto_correlate(wy,X,K,P[2])
    C = @. 0.5*(cx + cy)
    return bessel_reduce(k,X...,C)
end

function compressible_spectrum(P,k,psi::Psi{3})
    @unpack ψ,X,K = psi
    wx,wy,wz = weightedvelocity(P[1],psi)
    _, Wc = helmholtz(P[1],wx,wy,wz,K...)
    wx,wy,wz = Wc

	cx = auto_correlate(wx,X,K,P[2])[:,:,1:length(X[3])+1]
    cy = auto_correlate(wy,X,K,P[2])[:,:,1:length(X[3])+1]
    cz = auto_correlate(wz,X,K,P[2])[:,:,1:length(X[3])+1]
    C = @. 0.5*(cx + cy + cz)
    return sinc_reduce_real(k,X...,C)
end

function compressible_spectrum(k,psi::Psi_qper2{2})
    @unpack ψ,X,K = psi 
    wx,wy = weightedvelocity_qper(psi)
    _, Wc = helmholtz(wx,wy,K...)
    wx,wy = Wc

	cx = auto_correlate(wx,X,K)
	cy = auto_correlate(wy,X,K)
    C = @. 0.5*(cx + cy)
    return bessel_reduce(k,X...,C)
end

function compressible_spectrum(k,psi::Psi_qper3{3})
    @unpack ψ,X,K = psi
    wx,wy,wz = weightedvelocity_qper(psi)
    _, Wc = helmholtz(wx,wy,wz,K...)
    wx,wy,wz = Wc

	cx = auto_correlate(wx,X,K)[:,:,1:length(X[3])+1]
    cy = auto_correlate(wy,X,K)[:,:,1:length(X[3])+1]
    cz = auto_correlate(wz,X,K)[:,:,1:length(X[3])+1]
    C = @. 0.5*(cx + cy + cz)
    return sinc_reduce_real(k,X...,C)
end

"""
	compressible_current_spectrum(k,ψ,X,K)

Caculate the compressible current correlation spectrum for wavefunction ``\\psi``, via Helmholtz decomposition.
Input arrays `X`, `K` must be computed using `makearrays`.
"""
function compressible_current_spectrum(k,psi::Psi{2})
    @unpack ψ,X,K = psi 
    jx,jy = current(psi)
    _, Jc = helmholtz(jx,jy,K...)
    jx,jy = Jc

	cx = auto_correlate(jx,X,K)
	cy = auto_correlate(jy,X,K)
    C = @. 0.5*(cx + cy)
    return bessel_reduce(k,X...,C)
end

function compressible_current_spectrum(k,psi::Psi{3})
    @unpack ψ,X,K = psi
    jx,jy,jz = current(psi)
    _, Jc = helmholtz(jx,jy,jz,K...)
    jx,jy,jz = Jc

	cx = auto_correlate(jx,X,K)
    cy = auto_correlate(jy,X,K)
    cz = auto_correlate(jz,X,K)
    C = @. 0.5*(cx + cy + cz)
    return sinc_reduce(k,X...,C)
end

function compressible_current_spectrum(k,psi::Psi{2})
    @unpack ψ,X,K = psi 
    jx,jy = current(P[1],psi)
    _, Jc = helmholtz(P[1],jx,jy,K...)
    jx,jy = Jc

	cx = auto_correlate(jx,X,K,P)
	cy = auto_correlate(jy,X,K,P)
    C = @. 0.5*(cx + cy)
    return bessel_reduce(k,X...,C)
end

function compressible_current_spectrum(k,psi::Psi{3})
    @unpack ψ,X,K = psi
    jx,jy,jz = current(P[1],psi)
    _, Jc = helmholtz(P[1],jx,jy,jz,K...)
    jx,jy,jz = Jc

	cx = auto_correlate(jx,X,K,P[2])[:,:,1:length(X[3])+1]
    cy = auto_correlate(jy,X,K,P[2])[:,:,1:length(X[3])+1]
    cz = auto_correlate(jz,X,K,P[2])[:,:,1:length(X[3])+1]
    C = @. 0.5*(cx + cy + cz)
    return sinc_reduce_real(k,X...,C)
end

function compressible_current_spectrum(k,psi::Psi_qper2{2})
    @unpack ψ,X,K = psi 
    jx,jy = current_qper(psi)
    _, Jc = helmholtz(jx,jy,K...)
    jx,jy = Jc

	cx = auto_correlate(jx,X,K)
	cy = auto_correlate(jy,X,K)
    C = @. 0.5*(cx + cy)
    return bessel_reduce(k,X...,C)
end

function compressible_current_spectrum(k,psi::Psi_qper3{3})
    @unpack ψ,X,K = psi
    jx,jy,jz = current_qper(psi)
    _, Jc = helmholtz(jx,jy,jz,K...)
    jx,jy,jz = Jc

	cx = auto_correlate(jx,X,K)[:,:,1:length(X[3])+1]
    cy = auto_correlate(jy,X,K)[:,:,1:length(X[3])+1]
    cz = auto_correlate(jz,X,K)[:,:,1:length(X[3])+1]
    C = @. 0.5*(cx + cy + cz)
    return sinc_reduce_real(k,X...,C)
end

"""
	decomposed_spectra(k,ψ,X,K)

Caculate both the incompressible and compressible kinetic energy spectra for wavefunction ``\\psi``, via Helmholtz decomposition.
Input arrays `X`, `K` must be computed using `makearrays`.
"""

function decomposed_spectra(P,k,psi::Psi{2})
    @unpack ψ,X,K = psi 
    wx,wy = weightedvelocity(P[1],psi)
    Wi, Wc = helmholtz(P[1],wx,wy,K...)
    wx,wy = Wi

	cx = auto_correlate(wx,X,K,P[2])
	cy = auto_correlate(wy,X,K,P[2])
    C = @. 0.5*(cx + cy)
    εki = bessel_reduce(k,X...,C)
	wx,wy = Wc

	cx = auto_correlate(wx,X,K,P[2])
	cy = auto_correlate(wy,X,K,P[2])
    C = @. 0.5*(cx + cy)
	εkc = bessel_reduce(k,X...,C)
	return εki, εkc
end

function decomposed_spectra(P,k,psi::Psi{3})
    @unpack ψ,X,K = psi
    wx,wy,wz = weightedvelocity(P[1],psi)
    Wi, Wc = helmholtz(P[1],wx,wy,wz,K...)
    wx,wy,wz = Wi

	cx = auto_correlate(wx,X,K,P[2])[:,:,1:length(X[3])+1]
    cy = auto_correlate(wy,X,K,P[2])[:,:,1:length(X[3])+1]
    cz = auto_correlate(wz,X,K,P[2])[:,:,1:length(X[3])+1]
    C = @. 0.5*(cx + cy + cz)
    εki = sinc_reduce_real(k,X...,C)
	wx,wy,wz = Wc

	cx = auto_correlate(wx,X,K,P[2])[:,:,1:length(X[3])+1]
    cy = auto_correlate(wy,X,K,P[2])[:,:,1:length(X[3])+1]
    cz = auto_correlate(wz,X,K,P[2])[:,:,1:length(X[3])+1]
    C = @. 0.5*(cx + cy + cz)
	εkc = sinc_reduce_real(k,X...,C)
	return εki, εkc
end

function decomposed_spectra(k,psi::Psi_qper2{2})
    @unpack ψ,X,K = psi 
    wx,wy = weightedvelocity_qper(psi)
    Wi, Wc = helmholtz(wx,wy,K...)
    wx,wy = Wi

	cx = auto_correlate(wx,X,K)
	cy = auto_correlate(wy,X,K)
    C = @. 0.5*(cx + cy)
    εki = bessel_reduce(k,X...,C)
	wx,wy = Wc

	cx = auto_correlate(wx,X,K)
	cy = auto_correlate(wy,X,K)
    C = @. 0.5*(cx + cy)
    εki = bessel_reduce(k,X...,C)
	return εki, εkc
end

function decomposed_spectra(k,psi::Psi_qper3{3})
    @unpack ψ,X,K = psi
    wx,wy,wz = weightedvelocity_qper(psi)
    Wi, Wc = helmholtz(wx,wy,wz,K...)
    wx,wy,wz = Wi

	cx = auto_correlate(wx,X,K)[:,:,1:length(X[3])+1]
    cy = auto_correlate(wy,X,K)[:,:,1:length(X[3])+1]
    cz = auto_correlate(wz,X,K)[:,:,1:length(X[3])+1]
    C = @. 0.5*(cx + cy + cz)
    εki = sinc_reduce_real(k,X...,C)
	wx,wy,wz = Wc

	cx = auto_correlate(wx,X,K)[:,:,1:length(X[3])+1]
    cy = auto_correlate(wy,X,K)[:,:,1:length(X[3])+1]
    cz = auto_correlate(wz,X,K)[:,:,1:length(X[3])+1]
    C = @. 0.5*(cx + cy + cz)
    εki = sinc_reduce_real(k,X...,C)
	return εki, εkc
end

"""
	decomposed_current_spectra(k,ψ,X,K)

Caculate both the incompressible and compressible current correlation spectra for wavefunction ``\\psi``, via Helmholtz decomposition.
Input arrays `X`, `K` must be computed using `makearrays`.
"""

function decomposed_current_spectra(k,psi::Psi{2})
    @unpack ψ,X,K = psi 
    jx,jy = current(P[1],psi)
    Ji, Jc = helmholtz(P[1],jx,jy,K...)
    jx,jy = Ji

	cx = auto_correlate(jx,X,K,P)
	cy = auto_correlate(jy,X,K,P)
    C = @. 0.5*(cx + cy)
    jci = bessel_reduce(k,X...,C)
	jx,jy = Jc

	cx = auto_correlate(jx,X,K,P)
	cy = auto_correlate(jy,X,K,P)
    C = @. 0.5*(cx + cy)
    jcc = bessel_reduce(k,X...,C)
	return jci, jcc
end

function decomposed_current_spectra(k,psi::Psi{3})
    @unpack ψ,X,K = psi
    jx,jy,jz = current(P[1],psi)
    Ji, Jc = helmholtz(P[1],jx,jy,jz,K...)
    jx,jy,jz = Ji

	cx = auto_correlate(jx,X,K,P[2])[:,:,1:length(X[3])+1]
    cy = auto_correlate(jy,X,K,P[2])[:,:,1:length(X[3])+1]
    cz = auto_correlate(jz,X,K,P[2])[:,:,1:length(X[3])+1]
    C = @. 0.5*(cx + cy + cz)
    jci = sinc_reduce_real(k,X...,C)
	jx,jy,jz = Jc

	cx = auto_correlate(jx,X,K,P[2])[:,:,1:length(X[3])+1]
    cy = auto_correlate(jy,X,K,P[2])[:,:,1:length(X[3])+1]
    cz = auto_correlate(jz,X,K,P[2])[:,:,1:length(X[3])+1]
    C = @. 0.5*(cx + cy + cz)
    jcc = sinc_reduce_real(k,X...,C)
	return jci, jcc
end

function decomposed_current_spectra(k,psi::Psi_qper2{2})
    @unpack ψ,X,K = psi 
    jx,jy = current_qper(psi)
    Ji, Jc = helmholtz(jx,jy,K...)
    jx,jy = Ji

	cx = auto_correlate(jx,X,K)
	cy = auto_correlate(jy,X,K)
    C = @. 0.5*(cx + cy)
    jci = bessel_reduce(k,X...,C)
	jx,jy = Jc

	cx = auto_correlate(jx,X,K)
	cy = auto_correlate(jy,X,K)
    C = @. 0.5*(cx + cy)
    jcc = bessel_reduce(k,X...,C)
	return jci, jcc
end

function decomposed_current_spectra(k,psi::Psi_qper3{3})
    @unpack ψ,X,K = psi
    jx,jy,jz = current_qper(psi)
    Ji, Jc = helmholtz(jx,jy,jz,K...)
    jx,jy,jz = Ji

	cx = auto_correlate(jx,X,K)[:,:,1:length(X[3])+1]
    cy = auto_correlate(jy,X,K)[:,:,1:length(X[3])+1]
    cz = auto_correlate(jz,X,K)[:,:,1:length(X[3])+1]
    C = @. 0.5*(cx + cy + cz)
    jci = sinc_reduce_real(k,X...,C)
	jx,jy,jz = Jc

	cx = auto_correlate(jx,X,K)[:,:,1:length(X[3])+1]
    cy = auto_correlate(jy,X,K)[:,:,1:length(X[3])+1]
    cz = auto_correlate(jz,X,K)[:,:,1:length(X[3])+1]
    C = @. 0.5*(cx + cy + cz)
    jcc = sinc_reduce_real(k,X...,C)
	return jci, jcc
end

"""
	qpressure_spectrum(k,psi::Psi{D})

Caculate the quantum pressure correlation spectrum for wavefunction ``\\psi``.
Input arrays `X`, `K` must be computed using `makearrays`.
"""
function qpressure_spectrum(k,psi::Psi{2})
    @unpack ψ,X,K = psi
    psia = Psi(abs.(ψ) |> complex,X,K)
    wx,wy = gradient(psia)

	cx = auto_correlate(wx,X,K)
	cy = auto_correlate(wy,X,K)
    C = @. 0.5*(cx + cy)
    return bessel_reduce(k,X...,C)
end

function qpressure_spectrum(k,psi::Psi{3})
    @unpack ψ,X,K = psi
    psia = Psi(abs.(ψ) |> complex,X,K )
    wx,wy,wz = gradient(psia)

	cx = auto_correlate(wx,X,K)
    cy = auto_correlate(wy,X,K)
    cz = auto_correlate(wz,X,K)
    C = @. 0.5*(cx + cy + cz)
    return sinc_reduce(k,X...,C)
end

function qpressure_spectrum(P,k,psi::Psi{2})
    @unpack ψ,X,K = psi
    psia = Psi(abs.(ψ) |> complex,X,K)
    wx,wy = gradient(P[1],psia)

	cx = auto_correlate(wx,X,K,P[2])
	cy = auto_correlate(wy,X,K,P[2])
    C = @. 0.5*(cx + cy)
    return bessel_reduce(k,X...,C)
end

function qpressure_spectrum(P,k,psi::Psi{3})
    @unpack ψ,X,K = psi
    psia = Psi(abs.(ψ) |> complex,X,K)
    wx,wy,wz = gradient(P[1].psia)

	cx = auto_correlate(wx,X,K,P[2])[:,:,1:length(X[3])+1]
    cy = auto_correlate(wy,X,K,P[2])[:,:,1:length(X[3])+1]
    cz = auto_correlate(wz,X,K,P[2])[:,:,1:length(X[3])+1]
    C = @. 0.5*(cx + cy + cz)
    return sinc_reduce(k,X...,C)
end

function qpressure_spectrum(k,psi::Psi_qper2{2})
    @unpack ψ,X,K = psi
    psia = Psi(abs.(ψ) |> complex,X,K)
    wx,wy = gradient(psia)

	cx = auto_correlate(wx,X,K)
	cy = auto_correlate(wy,X,K)
    C = @. 0.5*(cx + cy)
    return bessel_reduce(k,X...,C)
end

function qpressure_spectrum(k,psi::Psi_qper3{3})
    @unpack ψ,X,K = psi
    psia = Psi(abs.(ψ) |> complex,X,K )
    wx,wy,wz = gradient(psia)

	cx = auto_correlate(wx,X,K)[:,:,1:length(X[3])+1]
    cy = auto_correlate(wy,X,K)[:,:,1:length(X[3])+1]
    cz = auto_correlate(wz,X,K)[:,:,1:length(X[3])+1]
    C = @. 0.5*(cx + cy + cz)
    return sinc_reduce_real(k,X...,C)
end

"""
    incompressible_density(k,ψ,X,K)

Calculates the kinetic energy density of the incompressible velocity field in the wavefunction ``\\psi``, at the
points `k`. Arrays `X`, `K` should be computed using `makearrays`.
"""
function incompressible_density(k,psi::Psi{2})
    @unpack ψ,X,K = psi 
    vx,vy = velocity(psi)
    a = abs.(ψ)
    ux = @. a*vx; uy = @. a*vy 
    Wi, Wc = helmholtz(ux,uy,K...)
    wix,wiy = Wi
    U = @. exp(im*angle(ψ))
    @. wix *= U # restore phase factors
    @. wiy *= U

	cx = auto_correlate(wix,X,K)
	cy = auto_correlate(wiy,X,K)
    C = @. 0.5*(cx + cy)
    return bessel_reduce(k,X...,C)
end

function incompressible_density(k,psi::Psi{3})
    @unpack ψ,X,K = psi 
    vx,vy,vz = velocity(psi)
    a = abs.(ψ)
    ux = @. a*vx; uy = @. a*vy; uz = @. a*vz
    Wi, Wc = helmholtz(ux,uy,uz,K...)
    wix,wiy,wiz = Wi
    U = @. exp(im*angle(ψ))
    @. wix *= U # restore phase factors
    @. wiy *= U
    @. wiz *= U

	cx = auto_correlate(wix,X,K)
    cy = auto_correlate(wiy,X,K)
    cz = auto_correlate(wiz,X,K)
    C = @. 0.5*(cx + cy + cz)
    return sinc_reduce(k,X...,C)
end

function incompressible_density(k,psi::Psi_plan{2})
    @unpack ψ,X,K,P = psi 
    vx,vy = velocity(psi)
    a = abs.(ψ)
    ux = @. a*vx; uy = @. a*vy 
    Wi, Wc = helmholtz(P[1],ux,uy,K...)
    wix,wiy = Wi
    U = @. exp(im*angle(ψ))
    @. wix *= U # restore phase factors
    @. wiy *= U

	cx = auto_correlate(wix,X,K,P)
	cy = auto_correlate(wiy,X,K,P)
    C = @. 0.5*(cx + cy)
    return bessel_reduce(k,X...,C)
end

function incompressible_density(k,psi::Psi_plan{3})
    @unpack ψ,X,K,P = psi 
    vx,vy,vz = velocity(psi)
    a = abs.(ψ)
    ux = @. a*vx; uy = @. a*vy; uz = @. a*vz
    Wi, Wc = helmholtz(P[1],ux,uy,uz,K...)
    wix,wiy,wiz = Wi
    U = @. exp(im*angle(ψ))
    @. wix *= U # restore phase factors
    @. wiy *= U
    @. wiz *= U

	cx = auto_correlate(wix,X,K,P)
    cy = auto_correlate(wiy,X,K,P)
    cz = auto_correlate(wiz,X,K,P)
    C = @. 0.5*(cx + cy + cz)
    return sinc_reduce(k,X...,C)
end

"""
    incompressible_density_qper(k,ψ,X,K)

Calculates the kinetic energy density of the incompressible velocity field in the wavefunction ``\\psi``, at the
points `k`. Arrays `X`, `K` should be computed using `makearrays`.
Uses quasiperiodic boundary conditions.
"""
function incompressible_density_qper(k,psi::Psi_qper2{2})
    @unpack ψ,X,K = psi 
    vx,vy = velocity_qper(psi)
    a = abs.(ψ)
    ux = @. a*vx; uy = @. a*vy 
    Wi, Wc = helmholtz(ux,uy,K...)
    wix,wiy = Wi
    U = @. exp(im*angle(ψ))
    @. wix *= U # restore phase factors
    @. wiy *= U

	cx = auto_correlate(wix,X,K)
	cy = auto_correlate(wiy,X,K)
    C = @. 0.5*(cx + cy)
    return bessel_reduce(k,X...,C)
end

function incompressible_density_qper(k,psi::Psi_qper3{3})
    @unpack ψ,X,K = psi 
    vx,vy,vz = velocity_qper(psi)
    a = abs.(ψ)
    ux = @. a*vx; uy = @. a*vy; uz = @. a*vz
    Wi, Wc = helmholtz(ux,uy,uz,K...)
    wix,wiy,wiz = Wi
    U = @. exp(im*angle(ψ))
    @. wix *= U # restore phase factors
    @. wiy *= U
    @. wiz *= U

	cx = auto_correlate(wix,X,K)
    cy = auto_correlate(wiy,X,K)
    cz = auto_correlate(wiz,X,K)
    C = @. 0.5*(cx + cy + cz)
    return sinc_reduce(k,X...,C)
end

"""
    compressible_density(k,ψ,X,K)

Calculates the kinetic energy density of the compressible velocity field in the wavefunction ``\\psi``, at the
points `k`. Arrays `X`, `K` should be computed using `makearrays`.
"""
function compressible_density(k,psi::Psi{2})
    @unpack ψ,X,K = psi 
    vx,vy = velocity(psi)
    a = abs.(ψ)
    ux = @. a*vx; uy = @. a*vy 
    Wi, Wc = helmholtz(ux,uy,K...)
    wcx,wcy = Wc
    U = @. exp(im*angle(ψ))
    @. wcx *= U # restore phase factors
    @. wcy *= U

	cx = auto_correlate(wcx,X,K)
	cy = auto_correlate(wcy,X,K)
    C = @. 0.5*(cx + cy)
    return bessel_reduce(k,X...,C)
end

function compressible_density(k,psi::Psi{3})
    @unpack ψ,X,K = psi 
    vx,vy,vz = velocity(psi)
    a = abs.(ψ)
    ux = @. a*vx; uy = @. a*vy; uz = @. a*vz
    Wi, Wc = helmholtz(ux,uy,uz,K...)
    wcx,wcy,wcz = Wc
    U = @. exp(im*angle(ψ))
    @. wcx *= U # restore phase factors
    @. wcy *= U
    @. wcz *= U

	cx = auto_correlate(wcx,X,K)
    cy = auto_correlate(wcy,X,K)
    cz = auto_correlate(wcz,X,K)
    C = @. 0.5*(cx + cy + cz)
    return sinc_reduce(k,X...,C)
end

function compressible_density(k,psi::Psi_plan{2})
    @unpack ψ,X,K,P = psi 
    vx,vy = velocity(psi)
    a = abs.(ψ)
    ux = @. a*vx; uy = @. a*vy 
    Wi, Wc = helmholtz(P[1],ux,uy,K...)
    wcx,wcy = Wc
    U = @. exp(im*angle(ψ))
    @. wcx *= U # restore phase factors
    @. wcy *= U

	cx = auto_correlate(wcx,X,K,P)
	cy = auto_correlate(wcy,X,K,P)
    C = @. 0.5*(cx + cy)
    return bessel_reduce(k,X...,C)
end

function compressible_density(k,psi::Psi_plan{3})
    @unpack ψ,X,K,P = psi 
    vx,vy,vz = velocity(psi)
    a = abs.(ψ)
    ux = @. a*vx; uy = @. a*vy; uz = @. a*vz
    Wi, Wc = helmholtz(P[1],ux,uy,uz,K...)
    wcx,wcy,wcz = Wc
    U = @. exp(im*angle(ψ))
    @. wcx *= U # restore phase factors
    @. wcy *= U
    @. wcz *= U

	cx = auto_correlate(wcx,X,K,P)
    cy = auto_correlate(wcy,X,K,P)
    cz = auto_correlate(wcz,X,K,P)
    C = @. 0.5*(cx + cy + cz)
    return sinc_reduce(k,X...,C)
end

"""
    compressible_density_qper(k,ψ,X,K)

Calculates the kinetic energy density of the compressible velocity field in the wavefunction ``\\psi``, at the
points `k`. Arrays `X`, `K` should be computed using `makearrays`.
Uses quasiperiodic boundary conditions.
"""
function compressible_density_qper(k,psi::Psi_qper2{2})
    @unpack ψ,X,K = psi 
    vx,vy = velocity_qper(psi)
    a = abs.(ψ)
    ux = @. a*vx; uy = @. a*vy 
    Wi, Wc = helmholtz(ux,uy,K...)
    wcx,wcy = Wc
    U = @. exp(im*angle(ψ))
    @. wcx *= U # restore phase factors
    @. wcy *= U

	cx = auto_correlate(wcx,X,K)
	cy = auto_correlate(wcy,X,K)
    C = @. 0.5*(cx + cy)
    return bessel_reduce(k,X...,C)
end

function compressible_density_qper(k,psi::Psi_qper3{3})
    @unpack ψ,X,K = psi 
    vx,vy,vz = velocity_qper(psi)
    a = abs.(ψ)
    ux = @. a*vx; uy = @. a*vy; uz = @. a*vz
    Wi, Wc = helmholtz(ux,uy,uz,K...)
    wcx,wcy,wcz = Wc
    U = @. exp(im*angle(ψ))
    @. wcx *= U # restore phase factors
    @. wcy *= U
    @. wcz *= U

	cx = auto_correlate(wcx,X,K)
    cy = auto_correlate(wcy,X,K)
    cz = auto_correlate(wcz,X,K)
    C = @. 0.5*(cx + cy + cz)
    return sinc_reduce(k,X...,C)
end

"""
    qpressure_density(k,ψ,X,K)

Energy density of the quantum pressure in the wavefunction ``\\psi``, at the
points `k`. Arrays `X`, `K` should be computed using `makearrays`.
"""
function qpressure_density(k,psi::Psi{2})
    @unpack ψ,X,K = psi
    psia = Psi(abs.(ψ) |> complex,X,K )
    rnx,rny = gradient(psia)
    U = @. exp(im*angle(ψ))
    @. rnx *= U # restore phase factors
    @. rny *= U 

	cx = auto_correlate(rnx,X,K)
	cy = auto_correlate(rny,X,K)
    C = @. 0.5*(cx + cy)
    return bessel_reduce(k,X...,C)
end

function qpressure_density(k,psi::Psi{3})
    @unpack ψ,X,K = psi
    psia = Psi(abs.(ψ) |> complex,X,K )
    rnx,rny,rnz = gradient(psia)
    U = @. exp(im*angle(ψ))
    @. rnx *= U # restore phase factors
    @. rny *= U 
    @. rnz *= U 

	cx = auto_correlate(rnx,X,K)
    cy = auto_correlate(rny,X,K)
    cz = auto_correlate(rnz,X,K)
    C = @. 0.5*(cx + cy + cz)
    return sinc_reduce(k,X...,C)
end

function qpressure_density(k,psi::Psi_plan{2})
    @unpack ψ,X,K,P = psi
    psia = Psi_plan(abs.(ψ) |> complex,X,K,P)
    rnx,rny = gradient(psia)
    U = @. exp(im*angle(ψ))
    @. rnx *= U # restore phase factors
    @. rny *= U 

	cx = auto_correlate(rnx,X,K,P)
	cy = auto_correlate(rny,X,K,P)
    C = @. 0.5*(cx + cy)
    return bessel_reduce(k,X...,C)
end

function qpressure_density(k,psi::Psi_plan{3})
    @unpack ψ,X,K,P = psi
    psia = Psi_plan(abs.(ψ) |> complex,X,K,P)
    rnx,rny,rnz = gradient(psia)
    U = @. exp(im*angle(ψ))
    @. rnx *= U # restore phase factors
    @. rny *= U 
    @. rnz *= U 

	cx = auto_correlate(rnx,X,K,P)
    cy = auto_correlate(rny,X,K,P)
    cz = auto_correlate(rnz,X,K,P)
    C = @. 0.5*(cx + cy + cz)
    return sinc_reduce(k,X...,C)
end

function qpressure_density(k,psi::Psi_qper2{2})
    @unpack ψ,X,K = psi
    psia = Psi(abs.(ψ) |> complex,X,K )
    rnx,rny = gradient(psia)
    U = @. exp(im*angle(ψ))
    @. rnx *= U # restore phase factors
    @. rny *= U 

	cx = auto_correlate(rnx,X,K)
	cy = auto_correlate(rny,X,K)
    C = @. 0.5*(cx + cy)
    return bessel_reduce(k,X...,C)
end

function qpressure_density(k,psi::Psi_qper3{3})
    @unpack ψ,X,K = psi
    psia = Psi(abs.(ψ) |> complex,X,K )
    rnx,rny,rnz = gradient(psia)
    U = @. exp(im*angle(ψ))
    @. rnx *= U # restore phase factors
    @. rny *= U 
    @. rnz *= U 

	cx = auto_correlate(rnx,X,K)
    cy = auto_correlate(rny,X,K)
    cz = auto_correlate(rnz,X,K)
    C = @. 0.5*(cx + cy + cz)
    return sinc_reduce(k,X...,C)
end

## coupling terms

"""
    ic_density(k,ψ,X,K)

Energy density of the incompressible-compressible interaction in the wavefunction ``\\psi``, at the
points `k`. Arrays `X`, `K` should be computed using `makearrays`.
"""
function ic_density(k,psi::Psi{2})
    @unpack ψ,X,K = psi 
    vx,vy = velocity(psi)
    a = abs.(ψ)
    ux = @. a*vx; uy = @. a*vy 
    Wi, Wc = helmholtz(ux,uy,K...)
    wix,wiy = Wi; wcx,wcy = Wc
    U = @. exp(im*angle(ψ))
    @. wix *= im*U # restore phase factors and make u -> w fields
    @. wiy *= im*U
    @. wcx *= im*U 
    @. wcy *= im*U

    cicx = convolve(wix,wcx,X,K) 
    ccix = convolve(wcx,wix,X,K)
    cicy = convolve(wiy,wcy,X,K) 
    cciy = convolve(wcy,wiy,X,K)
    C = @. 0.5*(cicx + ccix + cicy + cciy)  
    return bessel_reduce(k,X...,C)
end

function ic_density(k,psi::Psi{3})
    @unpack ψ,X,K = psi 
    vx,vy,vz = velocity(psi)
    a = abs.(ψ)
    ux = @. a*vx; uy = @. a*vy; uz = @. a*vz 
    Wi, Wc = helmholtz(ux,uy,uz,K...)
    wix,wiy,wiz = Wi; wcx,wcy,wcz = Wc
    U = @. exp(im*angle(ψ))
    @. wix *= im*U # restore phase factors and make u -> w fields
    @. wiy *= im*U
    @. wiz *= im*U   
    @. wcx *= im*U 
    @. wcy *= im*U
    @. wcz *= im*U

    cicx = convolve(wix,wcx,X,K) 
    ccix = convolve(wcx,wix,X,K)
    cicy = convolve(wiy,wcy,X,K) 
    cciy = convolve(wcy,wiy,X,K)
    cicz = convolve(wiz,wcz,X,K) 
    cciz = convolve(wcz,wiz,X,K)
    C = @. 0.5*(cicx + ccix + cicy + cciy + cicz + cciz)  
    return sinc_reduce(k,X...,C)
end

function ic_density(k,psi::Psi_plan{2})
    @unpack ψ,X,K,P = psi 
    vx,vy = velocity(psi)
    a = abs.(ψ)
    ux = @. a*vx; uy = @. a*vy 
    Wi, Wc = helmholtz(P[1],ux,uy,K...)
    wix,wiy = Wi; wcx,wcy = Wc
    U = @. exp(im*angle(ψ))
    @. wix *= im*U # restore phase factors and make u -> w fields
    @. wiy *= im*U
    @. wcx *= im*U 
    @. wcy *= im*U

    cicx = convolve(wix,wcx,X,K,P[2]) 
    ccix = convolve(wcx,wix,X,K,P[2])
    cicy = convolve(wiy,wcy,X,K,P[2]) 
    cciy = convolve(wcy,wiy,X,K,P[2])
    C = @. 0.5*(cicx + ccix + cicy + cciy)  
    return bessel_reduce(k,X...,C)
end

function ic_density(k,psi::Psi_plan{3})
    @unpack ψ,X,K,P = psi 
    vx,vy,vz = velocity(psi)
    a = abs.(ψ)
    ux = @. a*vx; uy = @. a*vy; uz = @. a*vz 
    Wi, Wc = helmholtz(P[1],ux,uy,uz,K...)
    wix,wiy,wiz = Wi; wcx,wcy,wcz = Wc
    U = @. exp(im*angle(ψ))
    @. wix *= im*U # restore phase factors and make u -> w fields
    @. wiy *= im*U
    @. wiz *= im*U   
    @. wcx *= im*U 
    @. wcy *= im*U
    @. wcz *= im*U

    cicx = convolve(wix,wcx,X,K,P[2]) 
    ccix = convolve(wcx,wix,X,K,P[2])
    cicy = convolve(wiy,wcy,X,K,P[2]) 
    cciy = convolve(wcy,wiy,X,K,P[2])
    cicz = convolve(wiz,wcz,X,K,P[2]) 
    cciz = convolve(wcz,wiz,X,K,P[2])
    C = @. 0.5*(cicx + ccix + cicy + cciy + cicz + cciz)  
    return sinc_reduce(k,X...,C)
end

"""
    ic_density_qper(k,ψ,X,K)

Energy density of the incompressible-compressible interaction in the wavefunction ``\\psi``, at the
points `k`. Arrays `X`, `K` should be computed using `makearrays`.
Uses quasiperiodic boundary conditions.
"""
function ic_density_qper(k,psi::Psi_qper2{2})
    @unpack ψ,X,K = psi 
    vx,vy = velocity_aper(psi)
    a = abs.(ψ)
    ux = @. a*vx; uy = @. a*vy 
    Wi, Wc = helmholtz(ux,uy,K...)
    wix,wiy = Wi; wcx,wcy = Wc
    U = @. exp(im*angle(ψ))
    @. wix *= im*U # restore phase factors and make u -> w fields
    @. wiy *= im*U
    @. wcx *= im*U 
    @. wcy *= im*U

    cicx = convolve(wix,wcx,X,K) 
    ccix = convolve(wcx,wix,X,K)
    cicy = convolve(wiy,wcy,X,K) 
    cciy = convolve(wcy,wiy,X,K)
    C = @. 0.5*(cicx + ccix + cicy + cciy)  
    return bessel_reduce(k,X...,C)
end

function ic_density_qper(k,psi::Psi_qper3{3})
    @unpack ψ,X,K = psi 
    vx,vy,vz = velocity_qper(psi)
    a = abs.(ψ)
    ux = @. a*vx; uy = @. a*vy; uz = @. a*vz 
    Wi, Wc = helmholtz(ux,uy,uz,K...)
    wix,wiy,wiz = Wi; wcx,wcy,wcz = Wc
    U = @. exp(im*angle(ψ))
    @. wix *= im*U # restore phase factors and make u -> w fields
    @. wiy *= im*U
    @. wiz *= im*U   
    @. wcx *= im*U 
    @. wcy *= im*U
    @. wcz *= im*U

    cicx = convolve(wix,wcx,X,K) 
    ccix = convolve(wcx,wix,X,K)
    cicy = convolve(wiy,wcy,X,K) 
    cciy = convolve(wcy,wiy,X,K)
    cicz = convolve(wiz,wcz,X,K) 
    cciz = convolve(wcz,wiz,X,K)
    C = @. 0.5*(cicx + ccix + cicy + cciy + cicz + cciz)  
    return sinc_reduce(k,X...,C)
end

"""
    iq_density(k,ψ,X,K)

Energy density of the incompressible-quantum pressure interaction in the wavefunction ``\\psi``, at the
points `k`. Arrays `X`, `K` should be computed using `xk_arrays`.
"""
function iq_density(k,psi::Psi{2})
    @unpack ψ,X,K = psi 
    vx,vy = velocity(psi)
    a = abs.(ψ)
    ux = @. a*vx; uy = @. a*vy 
    Wi, Wc = helmholtz(ux,uy,K...)
    wix,wiy = Wi 

    psia = Psi(abs.(ψ) |> complex,X,K )
    wqx,wqy = gradient(psia)

    U = @. exp(im*angle(ψ))
    @. wix *= im*U # phase factors and make u -> w fields
    @. wiy *= im*U
    @. wqx *= U
    @. wqy *= U

    ciqx = convolve(wix,wqx,X,K) 
    cqix = convolve(wqx,wix,X,K) 
    ciqy = convolve(wiy,wqy,X,K) 
    cqiy = convolve(wqy,wiy,X,K) 
    C = @. 0.5*(ciqx + cqix + ciqy + cqiy) 
    return bessel_reduce(k,X...,C)
end

function iq_density(k,psi::Psi{3})
    @unpack ψ,X,K = psi 
    vx,vy,vz = velocity(psi)
    a = abs.(ψ)
    ux = @. a*vx; uy = @. a*vy; uz = @. a*vz
    Wi, Wc = helmholtz(ux,uy,uz,K...)
    wix,wiy,wiz = Wi

    psia = Psi(abs.(ψ) |> complex,X,K )
    wqx,wqy,wqz = gradient(psia)

    U = @. exp(im*angle(ψ))
    @. wix *= im*U # phase factors and make u -> w fields
    @. wiy *= im*U
    @. wiz *= im*U
    @. wqx *= U
    @. wqy *= U
    @. wqz *= U

    ciqx = convolve(wix,wqx,X,K) 
    cqix = convolve(wqx,wix,X,K) 
    ciqy = convolve(wiy,wqy,X,K) 
    cqiy = convolve(wqy,wiy,X,K) 
    ciqz = convolve(wiz,wqz,X,K) 
    cqiz = convolve(wqz,wiz,X,K) 
    C = @. 0.5*(ciqx + cqix + ciqy + cqiy + ciqz + cqiz) 
    return sinc_reduce(k,X...,C)
end

function iq_density(k,psi::Psi_plan{2})
    @unpack ψ,X,K,P = psi 
    vx,vy = velocity(psi)
    a = abs.(ψ)
    ux = @. a*vx; uy = @. a*vy 
    Wi, Wc = helmholtz(P[1],ux,uy,K...)
    wix,wiy = Wi 

    psia = Psi_plan(abs.(ψ) |> complex,X,K,P)
    wqx,wqy = gradient(psia)

    U = @. exp(im*angle(ψ))
    @. wix *= im*U # phase factors and make u -> w fields
    @. wiy *= im*U
    @. wqx *= U
    @. wqy *= U

    ciqx = convolve(wix,wqx,X,K,P[2]) 
    cqix = convolve(wqx,wix,X,K,P[2]) 
    ciqy = convolve(wiy,wqy,X,K,P[2]) 
    cqiy = convolve(wqy,wiy,X,K,P[2]) 
    C = @. 0.5*(ciqx + cqix + ciqy + cqiy) 
    return bessel_reduce(k,X...,C)
end

function iq_density(k,psi::Psi_plan{3})
    @unpack ψ,X,K,P = psi 
    vx,vy,vz = velocity(psi)
    a = abs.(ψ)
    ux = @. a*vx; uy = @. a*vy; uz = @. a*vz
    Wi, Wc = helmholtz(P[1],ux,uy,uz,K...)
    wix,wiy,wiz = Wi

    psia = Psi_plan(abs.(ψ) |> complex,X,K,P)
    wqx,wqy,wqz = gradient(psia)

    U = @. exp(im*angle(ψ))
    @. wix *= im*U # phase factors and make u -> w fields
    @. wiy *= im*U
    @. wiz *= im*U
    @. wqx *= U
    @. wqy *= U
    @. wqz *= U

    ciqx = convolve(wix,wqx,X,K,P[2]) 
    cqix = convolve(wqx,wix,X,K,P[2]) 
    ciqy = convolve(wiy,wqy,X,K,P[2]) 
    cqiy = convolve(wqy,wiy,X,K,P[2]) 
    ciqz = convolve(wiz,wqz,X,K,P[2]) 
    cqiz = convolve(wqz,wiz,X,K,P[2]) 
    C = @. 0.5*(ciqx + cqix + ciqy + cqiy + ciqz + cqiz) 
    return sinc_reduce(k,X...,C)
end

"""
    iq_density_qper(k,ψ,X,K)

Energy density of the incompressible-quantum pressure interaction in the wavefunction ``\\psi``, at the
points `k`. Arrays `X`, `K` should be computed using `xk_arrays`.
Uses quasiperiodic boundary conditions.
"""
function iq_density_qper(k,psi::Psi_qper2{2})
    @unpack ψ,X,K = psi 
    vx,vy = velocity_qper(psi)
    a = abs.(ψ)
    ux = @. a*vx; uy = @. a*vy 
    Wi, Wc = helmholtz(ux,uy,K...)
    wix,wiy = Wi 

    psia = Psi(abs.(ψ) |> complex,X,K )
    wqx,wqy = gradient(psia)

    U = @. exp(im*angle(ψ))
    @. wix *= im*U # phase factors and make u -> w fields
    @. wiy *= im*U
    @. wqx *= U
    @. wqy *= U

    ciqx = convolve(wix,wqx,X,K) 
    cqix = convolve(wqx,wix,X,K) 
    ciqy = convolve(wiy,wqy,X,K) 
    cqiy = convolve(wqy,wiy,X,K) 
    C = @. 0.5*(ciqx + cqix + ciqy + cqiy) 
    return bessel_reduce(k,X...,C)
end

function iq_density_qper(k,psi::Psi_qper3{3})
    @unpack ψ,X,K = psi 
    vx,vy,vz = velocity_qper(psi)
    a = abs.(ψ)
    ux = @. a*vx; uy = @. a*vy; uz = @. a*vz
    Wi, Wc = helmholtz(ux,uy,uz,K...)
    wix,wiy,wiz = Wi

    psia = Psi(abs.(ψ) |> complex,X,K )
    wqx,wqy,wqz = gradient(psia)

    U = @. exp(im*angle(ψ))
    @. wix *= im*U # phase factors and make u -> w fields
    @. wiy *= im*U
    @. wiz *= im*U
    @. wqx *= U
    @. wqy *= U
    @. wqz *= U

    ciqx = convolve(wix,wqx,X,K) 
    cqix = convolve(wqx,wix,X,K) 
    ciqy = convolve(wiy,wqy,X,K) 
    cqiy = convolve(wqy,wiy,X,K) 
    ciqz = convolve(wiz,wqz,X,K) 
    cqiz = convolve(wqz,wiz,X,K) 
    C = @. 0.5*(ciqx + cqix + ciqy + cqiy + ciqz + cqiz) 
    return sinc_reduce(k,X...,C)
end


"""
    cq_density(k,ψ,X,K)

Energy density of the compressible-quantum pressure interaction in the wavefunction ``\\psi``, at the
points `k`. Arrays `X`, `K` should be computed using `makearrays`.
"""
function cq_density(k,psi::Psi{2})
    @unpack ψ,X,K = psi 
    vx,vy = velocity(psi)
    a = abs.(ψ)
    ux = @. a*vx; uy = @. a*vy 
    Wi, Wc = helmholtz(ux,uy,K...)
    wcx,wcy = Wc 

    psia = Psi(abs.(ψ) |> complex,X,K)
    wqx,wqy = gradient(psia)

    U = @. exp(im*angle(ψ))
    @. wcx *= im*U # phase factors and make u -> w fields
    @. wcy *= im*U
    @. wqx *= U
    @. wqy *= U

    ccqx = convolve(wcx,wqx,X,K) 
    cqcx = convolve(wqx,wcx,X,K) 
    ccqy = convolve(wcy,wqy,X,K) 
    cqcy = convolve(wqy,wcy,X,K) 
    C = @. 0.5*(ccqx + cqcx + ccqy + cqcy) 
    return bessel_reduce(k,X...,C)
end

function cq_density(k,psi::Psi{3})
    @unpack ψ,X,K = psi 
    vx,vy,vz = velocity(psi)
    a = abs.(ψ)
    ux = @. a*vx; uy = @. a*vy; uz = @. a*vz
    Wi, Wc = helmholtz(ux,uy,uz,K...)
    wcx,wcy,wcz = Wc  

    psia = Psi(abs.(ψ) |> complex,X,K)
    wqx,wqy,wqz = gradient(psia)

    U = @. exp(im*angle(ψ))
    @. wcx *= im*U # phase factors and make u -> w fields
    @. wcy *= im*U
    @. wcz *= im*U
    @. wqx *= U
    @. wqy *= U
    @. wqz *= U

    ccqx = convolve(wcx,wqx,X,K) 
    cqcx = convolve(wqx,wcx,X,K) 
    ccqy = convolve(wcy,wqy,X,K) 
    cqcy = convolve(wqy,wcy,X,K) 
    ccqz = convolve(wcz,wqz,X,K) 
    cqcz = convolve(wqz,wcz,X,K) 
    C = @. 0.5*(ccqx + cqcx + ccqy + cqcy + ccqz + cqcz) 
    return sinc_reduce(k,X...,C)
end

function cq_density(k,psi::Psi_plan{2})
    @unpack ψ,X,K,P = psi 
    vx,vy = velocity(psi)
    a = abs.(ψ)
    ux = @. a*vx; uy = @. a*vy 
    Wi, Wc = helmholtz(P[1],ux,uy,K...)
    wcx,wcy = Wc 

    psia = Psi_plan(abs.(ψ) |> complex,X,K,P)
    wqx,wqy = gradient(psia)

    U = @. exp(im*angle(ψ))
    @. wcx *= im*U # phase factors and make u -> w fields
    @. wcy *= im*U
    @. wqx *= U
    @. wqy *= U

    ccqx = convolve(wcx,wqx,X,K,P[2]) 
    cqcx = convolve(wqx,wcx,X,K,P[2]) 
    ccqy = convolve(wcy,wqy,X,K,P[2]) 
    cqcy = convolve(wqy,wcy,X,K,P[2]) 
    C = @. 0.5*(ccqx + cqcx + ccqy + cqcy) 
    return bessel_reduce(k,X...,C)
end

function cq_density(k,psi::Psi_plan{3})
    @unpack ψ,X,K,P = psi 
    vx,vy,vz = velocity(psi)
    a = abs.(ψ)
    ux = @. a*vx; uy = @. a*vy; uz = @. a*vz
    Wi, Wc = helmholtz(P[1],ux,uy,uz,K...)
    wcx,wcy,wcz = Wc  

    psia = Psi_plan(abs.(ψ) |> complex,X,K,P)
    wqx,wqy,wqz = gradient(psia)

    U = @. exp(im*angle(ψ))
    @. wcx *= im*U # phase factors and make u -> w fields
    @. wcy *= im*U
    @. wcz *= im*U
    @. wqx *= U
    @. wqy *= U
    @. wqz *= U

    ccqx = convolve(wcx,wqx,X,K,P[2]) 
    cqcx = convolve(wqx,wcx,X,K,P[2]) 
    ccqy = convolve(wcy,wqy,X,K,P[2]) 
    cqcy = convolve(wqy,wcy,X,K,P[2]) 
    ccqz = convolve(wcz,wqz,X,K,P[2]) 
    cqcz = convolve(wqz,wcz,X,K,P[2]) 
    C = @. 0.5*(ccqx + cqcx + ccqy + cqcy + ccqz + cqcz) 
    return sinc_reduce(k,X...,C)
end

"""
    cq_density_qper(k,ψ,X,K)

Energy density of the compressible-quantum pressure interaction in the wavefunction ``\\psi``, at the
points `k`. Arrays `X`, `K` should be computed using `makearrays`.
Uses quasiperiodic boundary conditions.
"""
function cq_density_qper(k,psi::Psi_qper2{2})
    @unpack ψ,X,K = psi 
    vx,vy = velocity_qper(psi)
    a = abs.(ψ)
    ux = @. a*vx; uy = @. a*vy 
    Wi, Wc = helmholtz(ux,uy,K...)
    wcx,wcy = Wc 

    psia = Psi(abs.(ψ) |> complex,X,K)
    wqx,wqy = gradient(psia)

    U = @. exp(im*angle(ψ))
    @. wcx *= im*U # phase factors and make u -> w fields
    @. wcy *= im*U
    @. wqx *= U
    @. wqy *= U

    ccqx = convolve(wcx,wqx,X,K) 
    cqcx = convolve(wqx,wcx,X,K) 
    ccqy = convolve(wcy,wqy,X,K) 
    cqcy = convolve(wqy,wcy,X,K) 
    C = @. 0.5*(ccqx + cqcx + ccqy + cqcy) 
    return bessel_reduce(k,X...,C)
end

function cq_density_qper(k,psi::Psi_qper3{3})
    @unpack ψ,X,K = psi 
    vx,vy,vz = velocity_qper(psi)
    a = abs.(ψ)
    ux = @. a*vx; uy = @. a*vy; uz = @. a*vz
    Wi, Wc = helmholtz(ux,uy,uz,K...)
    wcx,wcy,wcz = Wc  

    psia = Psi(abs.(ψ) |> complex,X,K)
    wqx,wqy,wqz = gradient(psia)

    U = @. exp(im*angle(ψ))
    @. wcx *= im*U # phase factors and make u -> w fields
    @. wcy *= im*U
    @. wcz *= im*U
    @. wqx *= U
    @. wqy *= U
    @. wqz *= U

    ccqx = convolve(wcx,wqx,X,K) 
    cqcx = convolve(wqx,wcx,X,K) 
    ccqy = convolve(wcy,wqy,X,K) 
    cqcy = convolve(wqy,wcy,X,K) 
    ccqz = convolve(wcz,wqz,X,K) 
    cqcz = convolve(wqz,wcz,X,K) 
    C = @. 0.5*(ccqx + cqcx + ccqy + cqcy + ccqz + cqcz) 
    return sinc_reduce(k,X...,C)
end

"""
    gv(r,k,ε)

Transform the power spectrum `ε(k)` defined at `k` to position space to give a system averaged velocity two-point correlation function on the spatial points `r`. The vector `r` can be chosen arbitrarily, provided `r ≥ 0`. 
"""
function gv(r,k,ε)
    dk = diff(k)
    push!(dk,last(dk))  # vanishing spectra at high k
    E = sum(@. ε*dk)
    gv = zero(r)
    @tullio gv[i] = ε[j]*besselj0(k[j]*r[i])*dk[j] avx=false
    return gv/E
end

"""
    gv3(r,k,ε)

Transform the power spectrum `ε(k)` defined at `k` to position space to give a system averaged velocity two-point correlation function on the spatial points `r`. The vector `r` can be chosen arbitrarily, provided `r ≥ 0`. 
"""
function gv3(r,k,ε)
    dk = diff(k)
    push!(dk,last(dk))  # vanishing spectra at high k
    E = sum(@. ε*dk)
    gv = zero(r)
    @tullio gv[i] = ε[j]*sinc(k[j]*r[i]/pi)*dk[j] avx=false
    return gv/E
end

function trap_spectrum(k,V,psi::Psi{2})
    @unpack ψ,X,K = psi; x,y = X
    f = @. abs(ψ)*sqrt(V(x,y',0.))
    C = auto_correlate(f,X,K)

    return bessel_reduce(k,X...,C)
end

function trap_spectrum(k,V,psi::Psi{3})
    @unpack ψ,X,K = psi; x,y,z = X
    f = @. abs(ψ)*sqrt(V(x,y',reshape(z,1,1,length(z)),0.))
    C = auto_correlate(f,X,K)

    return sinc_reduce(k,X...,C)
end

function trap_spectrum(k,V,psi::Psi_plan{2})
    @unpack ψ,X,K,P = psi; x,y = X
    f = @. abs(ψ)*sqrt(V(x,y',0.))
    C = auto_correlate(f,X,K,P)

    return bessel_reduce(k,X...,C)
end

function trap_spectrum(k,V,psi::Psi_plan{3})
    @unpack ψ,X,K,P = psi; x,y,z = X
    f = @. abs(ψ)*sqrt(V(x,y',reshape(z,1,1,length(z)),0.))
    C = auto_correlate(f,X,K,P)

    return sinc_reduce(k,X...,C)
end

function trap_spectrum(k,V,psi::Psi_qper2{2})
    @unpack ψ,X,K = psi; x,y = X
    f = @. abs(ψ)*sqrt(V(x,y',0.))
    C = auto_correlate(f,X,K)

    return bessel_reduce(k,X...,C)
end

function trap_spectrum(k,V,psi::Psi_qper3{3})
    @unpack ψ,X,K = psi; x,y,z = X
    f = @. abs(ψ)*sqrt(V(x,y',reshape(z,1,1,length(z)),0.))
    C = auto_correlate(f,X,K)

    return sinc_reduce(k,X...,C)
end

function density_spectrum(k,psi::Psi{2}) 
    @unpack ψ,X,K = psi 
    n = abs2.(ψ)
    C = auto_correlate(n,X,K) 

    return bessel_reduce(k,X...,C)
end

function density_spectrum(k,psi::Psi{3}) 
    @unpack ψ,X,K = psi 
    n = abs2.(ψ)
    C = auto_correlate(n,X,K) 

    return sinc_reduce(k,X...,C)
end

function density_spectrum(k,psi::Psi_plan{2}) 
    @unpack ψ,X,K,P = psi 
    n = abs2.(ψ)
    C = auto_correlate(n,X,K,P) 

    return bessel_reduce(k,X...,C)
end

function density_spectrum(k,psi::Psi_plan{3}) 
    @unpack ψ,X,K,P = psi 
    n = abs2.(ψ)
    C = auto_correlate(n,X,K,P) 

    return sinc_reduce(k,X...,C)
end

function density_spectrum(k,psi::Psi_qper2{2}) 
    @unpack ψ,X,K = psi 
    n = abs2.(ψ)
    C = auto_correlate(n,X,K) 

    return bessel_reduce(k,X...,C)
end

function density_spectrum(k,psi::Psi_qper3{3}) 
    @unpack ψ,X,K = psi 
    n = abs2.(ψ)
    C = auto_correlate(n,X,K) 

    return sinc_reduce(k,X...,C)
end
