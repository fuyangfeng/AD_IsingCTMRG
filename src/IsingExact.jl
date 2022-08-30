using Integrals
using SciMLSensitivity
using OMEinsum 
using Zygote
using Zygote: @adjoint
using Zygote: bufferfrom
using DelimitedFiles
using JLD2
using LinearAlgebra
function GetlnZ(β)
    #K=(2*sinh(2*β))/((cosh(2*β))^2)
    f(x,β) = log(0.5*(1+sqrt(1-((2*sinh(2*β))/((cosh(2*β))^2))^2*(sin(x)^2)))) 
    lnλ = log(2*cosh(2*β)) + 1/pi * solve(IntegralProblem(f,0, pi/2,β),QuadGKJL(),abstol=1e-16)[1]
    lnλ
end
function ising(β)
    @time lnz = GetlnZ(β)
    @show lnz
    @time U = -gradient( GetlnZ,β)[1]
    @show  U

    @time Cv = β * β * hessian(GetlnZ,β)
    @show Cv
    open( "../data/IsingExact.txt", "a" ) do io  
        writedlm( io, [β  lnz   U Cv   ] )
    end
    return nothing
end
function test()
    tem =  vcat(collect(2:0.005:2.268),collect(2.269:0.001:2.27),collect(2.271:0.005:2.5))
    for Tem in tem
        @time ising(1/Tem)
    end
end
test()

