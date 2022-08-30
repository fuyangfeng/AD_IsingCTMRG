using LinearAlgebra: norm,Diagonal,tr ,svd
using OMEinsum 
using DelimitedFiles
using JLD2
include("../src/CTM.jl")
include("../src/Ising_tensor.jl")
function GetFMU(Corner::Matrix{Float64}, Edge::Array{Float64,3}, T::Array{Float64,4}, Tm::Array{Float64,4}, Tu::Array{Float64,4})

    @ein CE[a,c,d] := Corner[a,b] * Edge[b,c,d]
    @ein ECE[i,b,d,l] := Edge[i,a,b] * CE[a,l,d]
    @ein CEC[a,e,d] := CE[a,c,d] *Corner[c,e]
    @ein CECE[a,d,g,f] := CEC[a,e,d] * Edge[e,f,g]
    @ein CECEC[a,d,g,h] := CECE[a,d,g,f] * Corner[f,h]

    @ein Up1t[a,j,i,h] := CECEC[a,d,g,h]* T[d,i,g,j]
    @ein Up1[] := Up1t[a,j,i,h]*ECE[h,i,j,a]
    @ein Up1m[a,j,i,h] := CECEC[a,d,g,h]* Tm[d,i,g,j]
    @ein Up1m[] := Up1m[a,j,i,h]*ECE[h,i,j,a]
    @ein Up1u[a,j,i,h] := CECEC[a,d,g,h]* Tu[d,i,g,j]
    @ein Up1u[] := Up1u[a,j,i,h]*ECE[h,i,j,a]

    Up2 = tr(Corner^4)
    @ein Down[]  := CEC[a,b,c] * CEC[a,b,c]
 
    PartitionFunc = (Up1[] * Up2[]) / (Down[]*Down[])
    m = abs.(Up1m[] / Up1[])
    FEnergy =  - log(PartitionFunc)

    UEnergy = 2 * Up1u[] / Up1[]
    return FEnergy, m, UEnergy

end
function main(Tem::Float64,Dbond::Int64)

    D,J , step =  2, 1.0 , 10000

    # step 1 : get Ising local Tensor
    T, Tm, Tu = GetIsingTensor(1/Tem, J)

    # step 2 : initial environment

    if "Env_Dbond=$(Dbond)_Tem=$(Tem).jld2" in readdir("../../Env/")
        Env=jldopen("../Env/Env_Dbond=$(Dbond)_Tem=$(Tem).jld2","r")
        edge=read(Env,"edge")
        corner=read(Env,"corner")
        close(Env)
    else
        corner = rand(Dbond, Dbond)
        corner += corner'
        edge = rand(Dbond, Dbond, D)
        edge += ein"ikj -> kij"(conj(edge))     
    end  

    # step 3 : perform CTM and save the lastest environment
    @time  edge, corner,EntangS, diff,TrunError = CTM(corner, edge,T,Tm,Tu, D,Dbond, step, 1e-16)
    Env = jldopen("../Env/Env_Dbond=$(Dbond)_Tem=$(Tem).jld2","w")
    write(Env, "edge", edge)
    write(Env, "corner", corner)
    close(Env)

    # step 4 : get observed value, such as F, M, U,S
    Fenergy,m,U=GetFMU(corner, edge, T, Tm,Tu)
    Fenergy = Tem * Fenergy
    open( "../data/Ising_FMUS_D=$(Dbond).txt", "a" ) do io  
        writedlm( io, [Tem  Fenergy  m U EntangS  ] )
    end
    open( "../data/Ising_TrunErro_D=$(Dbond).txt", "a" ) do io  
        writedlm( io, hcat([Tem] , TrunError) )
    end

    println( "Tem = $(Tem) , Fenergy = $(Fenergy) , magnetization = $m , TrunError = $(TrunError)" )

    return nothing
end


function test()
    tem =  vcat(collect(2:0.005:2.268),collect(2.269:0.001:2.27),collect(2.271:0.005:2.5))
    for i in tem
        @time main(i,80)
    end
end
test()