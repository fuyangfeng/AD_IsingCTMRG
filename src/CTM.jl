function CTMiter(corner::Matrix{Float64}, edge::Array{Float64,3}, T::Array{Float64,4}, D::Int64, χ::Int64)
    # D is the dimension of each index of tensor T
    # χ is the dimension of each index of corner tensor
    @ein cp[c, l , d, j] := ((corner[a, b] * edge[a, d, k]) * edge[b, c, i]) * T[i, j, k, l]
    # renormalize
    cpmat = reshape(cp, χ*D, χ*D)
    u, s,_= svd(cpmat)
    z = reshape(u[:, 1:χ], χ, D, χ)
    @ein corner[m, n] :=z[d, j, n] * (cp[c, l, d, j] * z[c, l, m])
    @ein edge[c,d,j] := ((z[a,k,c] * edge[a,b,i]) * T[i,j,k,l] )* z[b,l,d]
    s = s ./ s[1]
    TrunError = 1 - sum( s[ 1 : χ] ) / sum( s )
    # indexperm_symmetrize
    corner += corner'
    edge += ein"ikj -> kij"(conj(edge))   
    corner /= norm(corner)
    edge /= norm(edge) 
    # corner /= maximum(abs.(corner))
    # edge /= maximum(abs.(edge))
    return corner, edge, s,TrunError 
end

function CTM(corner::Matrix{Float64}, edge::Array{Float64,3},T::Array{Float64,4},Tm::Array{Float64,4},Tu::Array{Float64,4}, D::Int64, χ::Int64, maxiter::Int64, tol::Float64)

    TrunErro = [-1.0 -1.0]
    vals = fill(1, χ)
    oldvals = fill(1, D * χ)
    oldlnz = 100
    oldm = 100
    oldu = 100
    ErrorZ = 100
    ErrorM = 100
    ErrorU = 100
    for iter = 1: maxiter
        corner, edge, vals,TrunErro[2] = CTMiter(corner, edge, T, D, χ)
        diff = norm(vals - oldvals)
        oldvals = vals
        (TrunErro[1]<TrunErro[2]) && (TrunErro[1]=TrunErro[2])
        #####################

        if diff <= tol
            println("     Iter = $(iter) , diff = $(diff) , TrunError = $(TrunErro[2]) , ErrorZ = $(ErrorZ) , ErrorM = $(ErrorM) , ErrorU = $(ErrorU) , ")
            vals = (vals .^ 4) / sum(vals .^ 4)
            EntangS= - sum( vals .* log.(vals))
            return edge, corner,EntangS,diff, TrunErro 
        end
        if mod(iter, 1000) == 0
            lnz, M,U = getCtmFMU(edge, corner,T, Tm, Tu)
            ErrorZ = abs(lnz - oldlnz)
            ErrorM = abs(M - oldm)
            ErrorU = abs(U - oldu)
            oldlnz = lnz
            oldm = M
            oldu = U
            println("     Iter = $(iter) , diff = $(diff) , TrunError = $(TrunErro[2]) , ErrorZ = $(ErrorZ) , ErrorM = $(ErrorM) , ErrorU = $(ErrorU) , ")
        end
    end
    vals = (vals .^ 4) / sum(vals .^ 4)
    EntangS= - sum( vals .* log.(vals))
    return edge, corner,EntangS,diff, TrunErro
end
 function getCtmFMU(Edge::Array{Float64,3}, Corner::Matrix{Float64},T::Array{Float64,4}, Tm::Array{Float64,4}, Tu::Array{Float64,4})

    
    @ein CE[a,c,d] := Corner[a,b] * Edge[b,c,d]
    @ein ECE[i,b,d,l] := Edge[i,a,b] * CE[a,l,d]
    @ein CEC[a,e,d] := CE[a,c,d] *Corner[c,e]
    @ein CECE[a,d,g,f] := CEC[a,e,d] * Edge[e,f,g]
    @ein CECEC[a,d,g,h] := CECE[a,d,g,f] * Corner[f,h]
    @ein Up1t[a,j,i,h] := CECEC[a,d,g,h]* T[d,i,g,j]
    @ein Up1[] := Up1t[a,j,i,h]*ECE[h,i,j,a]
    Up2 = tr(Corner^4)
    @ein Down[]  := CEC[a,b,c] * CEC[a,b,c]
    PartitionFunc = (Up1[] * Up2) / ((Down[])^2)

    @ein Up1m[a,j,i,h] := CECEC[a,d,g,h]* Tm[d,i,g,j]
    @ein Up1m[] := Up1m[a,j,i,h]*ECE[h,i,j,a]
    m = abs.(Up1m[] / Up1[])

    @ein halfupu[a,j,i,f]  :=  CECE[a,d,g,f] * Tu[i,d,j,g]
    @ein halfupu[h,f,i]  :=   halfupu[a,j,i,f] * Edge[h,a,j]
    @ein upu[] := halfupu[m,f,j] * halfupu[f,m,j]

    @ein downu[a,h,j,f]  :=  CECE[a,d,g,f] * T[d,j,g,h]
    @ein downu[m,f,j]  :=   downu[a,h,j,f] * Edge[m,a,h]
    @ein downu[] := downu[m,f,j] * downu[f,m,j]
    U = -2*upu[]/downu[] #each point

    return log(PartitionFunc), m,U
 
 end



