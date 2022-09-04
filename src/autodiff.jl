using ChainRulesCore
using LinearAlgebra
import LinearAlgebra: svd

mysvd(A) = svd(A)
"""
    svd_back(U, S, V, dU, dS, dV)
adjoint for SVD decomposition.
References:
    https://j-towns.github.io/papers/svd-derivative.pdf
    https://giggleliu.github.io/2019/04/02/einsumbp.html
"""
function ChainRulesCore.rrule(::typeof(mysvd), A::AbstractArray{T,2}) where {T}
    U, S, V = mysvd(A)
    function back((dU, dS, dV))
        res = svd_back(U, S, V, dU, dS, dV)
        return NoTangent(), res
    end
    return (U, S, V), back
end

function svd_back(U::AbstractArray, S::AbstractArray{T}, V, dU, dS, dV; η::Real=10^-2) where T
    η = T(η)
    S2 = S .^ 2
    Sinv = @. S/(S2+η)
    F = S2' .- S2
    F ./= (F .^ 2 .+ η)

    res = zero(Diagonal(S))
    if !(dU == ZeroTangent())
        UdU = U'*dU
        J = F.*(UdU)
        res += (J+J')*Diagonal(S)
    end
    if !(dV == ZeroTangent())
        VdV = V'*dV
        K = F.*(VdV)
        res += Diagonal(S) * (K+K')
    end
    if !(dS == ZeroTangent())
        res += Diagonal(dS)
    end

    res = U*res*V'

    if !(dU == ZeroTangent()) && size(U, 1) != size(U, 2)
        res += (dU - U* (U'*dU)) * Diagonal(Sinv) * V'
    end

    if !(dV == ZeroTangent()) && size(V, 1) != size(V, 2)
        res = res + U * Diagonal(Sinv) * (dV' - (dV'*V)*V')
    end
    res
end