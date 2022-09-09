using LinearAlgebra: norm,Diagonal,tr
using OMEinsum 
using Zygote
using DelimitedFiles
using JLD2

include("../src/autodiff.jl")

function GetIsingTensor(BetaT, J::Float64)
    H = - J * [1 -1; -1 1]
    BW = exp.(-BetaT * H)
    U, S,_ = mysvd(BW)
    W = U * Diagonal(sqrt.(S))
    s = [1; -1]
    E = (H .* BW * U) .* (1 ./ sqrt.(abs.(S)))'
    @ein T[i, j, k, l] := W[a, i] * W[a, j] * W[a, k] * W[a, l] 
    @ein Ts[i, j, k, l] := W[a, i] * W[a, j] * W[a, k] * W[a, l] * s[a]
    @ein U[i, j, k, l] := E[a, i] * W[a, j] * W[a, k] * W[a, l] 
    U = (U + permutedims(U, (4, 3, 1, 2)) + permutedims(U, (2, 1, 4, 3)) + permutedims(U, (3, 4, 2, 1))) / 4
    return T, Ts, U
end
function CTMiter(corner, edge, T, D::Int64, Dbond::Int64)
    
    @ein CE[b,d,k] := corner[a, b] * edge[a, d, k]
    @ein CEE[d,k,c,i] := CE[b,d,k] * edge[b, c, i]
    @ein cp[c,l,d,j]   := CEE[d,k,c,i] * T[i, j, k, l]
    cpmat = reshape(cp, Dbond*D, Dbond*D)
    u, s,_= mysvd(cpmat)
    z = reshape(u[:, 1:Dbond], Dbond, D, Dbond)
    @ein corner[m, n] :=z[d, j, n] * (cp[c, l, d, j] * z[c, l, m])
    @ein edge[c,d,j] := ((z[a,k,c] * edge[a,b,i]) * T[i,j,k,l] )* z[b,l,d]
    # indexperm_symmetrize
    corner += corner'
    edge += ein"ikj -> kij"(conj(edge))   
    corner /= norm(corner)
    edge /= norm(edge) 
    return corner, edge
end
function GetlnZ(BetaT,Dbond::Int64, D::Int64,J::Float64 , CTMRGstep::Int64, corner::Matrix{Float64}, edge::Array{Float64,3})
    T, Tm, Tu = GetIsingTensor(BetaT, J)
 
    corner = corner .* ones(eltype(T),Dbond,Dbond)
    edge = edge .* ones(eltype(T),Dbond,Dbond,D)
    for i in 1:CTMRGstep
        corner, edge = CTMiter(corner, edge, T, D, Dbond)
    end
    @ein CE[a,c,d] := corner[a,b] * edge[b,c,d]
    @ein ECE[i,b,d,l] := edge[i,a,b] * CE[a,l,d]
    @ein CEC[a,e,d] := CE[a,c,d] *corner[c,e]
    @ein CECE[a,d,g,f] := CEC[a,e,d] * edge[e,f,g]
    @ein CECEC[a,d,g,h] := CECE[a,d,g,f] * corner[f,h]
    @ein Up1t[a,j,i,h] := CECEC[a,d,g,h]* T[d,i,g,j]
    @ein Up1[] := Up1t[a,j,i,h]*ECE[h,i,j,a]
    Up2 = tr(corner^4)
    @ein Down[]  := CEC[a,b,c] * CEC[a,b,c]
    PartitionFunc = (Up1[] * Up2) / ((Down[])^2)
    return log(PartitionFunc)
end
function GetM(Tem::Float64,Dbond::Int64, D::Int64,J::Float64 , CTMRGstep::Int64, corner::Matrix{Float64}, edge::Array{Float64,3})
    T, Tm, Tu = GetIsingTensor(1/Tem, J)
    for i in 1:CTMRGstep
        corner, edge = CTMiter(corner, edge, T, D, Dbond)
    end
    @ein CE[a,c,d] := corner[a,b] * edge[b,c,d]
    @ein ECE[i,b,d,l] := edge[i,a,b] * CE[a,l,d]
    @ein CEC[a,e,d] := CE[a,c,d] *corner[c,e]
    @ein CECE[a,d,g,f] := CEC[a,e,d] * edge[e,f,g]
    @ein CECEC[a,d,g,h] := CECE[a,d,g,f] * corner[f,h]
    @ein Up1t[a,j,i,h] := CECEC[a,d,g,h]* T[d,i,g,j]
    @ein Up1[] := Up1t[a,j,i,h]*ECE[h,i,j,a]
    @ein Up1m[a,j,i,h] := CECEC[a,d,g,h]* Tm[d,i,g,j]
    @ein Up1m[] := Up1m[a,j,i,h]*ECE[h,i,j,a]
    m = abs.(Up1m[] / Up1[])
    return m
end

function GetU(Tem::Float64,Dbond::Int64, D::Int64,J::Float64 , CTMRGstep::Int64, corner::Matrix{Float64}, edge::Array{Float64,3})
    T, Tm, Tu = GetIsingTensor(1/Tem, J)
    for i in 1:CTMRGstep
        corner, edge = CTMiter(corner, edge, T, D, Dbond)
    end
    @ein CE[a,c,d] := corner[a,b] * edge[b,c,d]
    @ein ECE[i,b,d,l] := edge[i,a,b] * CE[a,l,d]
    @ein CEC[a,e,d] := CE[a,c,d] *corner[c,e]
    @ein CECE[a,d,g,f] := CEC[a,e,d] * edge[e,f,g]
    @ein CECEC[a,d,g,h] := CECE[a,d,g,f] * corner[f,h]
    @ein Up1t[a,j,i,h] := CECEC[a,d,g,h]* T[d,i,g,j]
    @ein Up1[] := Up1t[a,j,i,h]*ECE[h,i,j,a]
    @ein Up1u[a,j,i,h] := CECEC[a,d,g,h]* Tu[d,i,g,j]
    @ein Up1u[] := Up1u[a,j,i,h]*ECE[h,i,j,a]
    UEnergy = 2 * Up1u[] / Up1[]
    return UEnergy
end

function main(Tem::Float64,Dbond::Int64,CTMRGstep::Int64)
    D,J=2,1.0
    Env=jldopen("./Env/Env_Dbond=$(Dbond)_Tem=$(Tem).jld2","r")
    edge=read(Env,"edge")
    corner=read(Env,"corner")
    close(Env)

    BetaT = 1/Tem
    @show Tem
    Z(β) = GetlnZ(β,Dbond, D,J , CTMRGstep, corner, edge)
    @time lnz =  Z(BetaT)
    @show  lnz

    M(t) = GetM(t,Dbond, D,J , CTMRGstep, corner, edge)
    @time Magnetization = M(Tem)
    @show  Magnetization

    U(t) = GetU(t,Dbond, D,J , CTMRGstep, corner, edge)
    @time Uenergy = U(Tem)
    @show  Uenergy

    @time Cm = gradient( M, Tem)[1]
    @show  Cm 
    @time Cv = gradient( U, Tem)[1]
    @show  Cv

    @time U_AD = -gradient(Z, BetaT)[1]
    @show U_AD

    @time Cv_AD1 = BetaT*BetaT*  hessian(Z, BetaT)
    @show  Cv_AD1


    open( "./data/Ising_AD_CTMRGstep=$(CTMRGstep)_D=$(Dbond).txt", "a" ) do io  
        writedlm( io, [1/Tem  lnz  Magnetization  Uenergy Cm  Cv U_AD Cv_AD1  ] )
    end

    return nothing
end


function test()
    # tem = [2.0]
    tem = vcat(collect(2:0.005:2.268),collect(2.269:0.001:2.27),collect(2.271:0.005:2.5))
    for i in tem
        @time main(i,80,10)
    end
end
test()