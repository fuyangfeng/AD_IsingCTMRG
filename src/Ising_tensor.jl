function GetIsingTensor(BetaT::Float64, J::Float64)

    H = - J * [1 -1; -1 1]
    BW = exp.(-BetaT * H)
    U, S,_ = svd(BW)
    W = U * Diagonal(sqrt.(S))
    s = [1; -1]
    E = (H .* BW * U) .* (1 ./ sqrt.(abs.(S)))'
    @ein T[i, j, k, l] := W[a, i] * W[a, j] * W[a, k] * W[a, l] 
    @ein Ts[i, j, k, l] := W[a, i] * W[a, j] * W[a, k] * W[a, l] * s[a]
    @ein U[i, j, k, l] := E[a, i] * W[a, j] * W[a, k] * W[a, l] 
    U = (U + permutedims(U, (4, 3, 1, 2)) + permutedims(U, (2, 1, 4, 3)) + permutedims(U, (3, 4, 2, 1))) / 4

    return T, Ts, U

end

