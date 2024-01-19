## 12/15/2023
# Compute the von Nuemann entanglement entropy 
using ITensors
using ITensors: orthocenter, sites, copy, complex, real

# Measure von Neumann entanglment entropy on a sequence of bonds
function entanglement_entropy(tmp_ψ :: MPS, length :: Int)
    entropy = Vector{Float64}()
    for site_index in 1 : length - 1 
        orthogonalize!(tmp_ψ, site_index)
        if site_index == 1
            i₁ = siteind(tmp_ψ, site_index)
            _, C1, _ = svd(tmp_ψ[site_index], i₁)
        else
            i₁, j₁ = siteind(tmp_ψ, site_index), linkind(tmp_ψ, site_index - 1)
            _, C1, _ = svd(tmp_ψ[site_index], i₁, j₁)
        end
        C1 = matrix(C1)
        SvN₁ = compute_entropy(C1)
        
        # @show site_index, SvN₁
        push!(entropy, SvN₁)
    end
    return entropy
end


# Measure von Neumann entanglment entropy on a sequence of bonds
function entanglement_entropy_bonds(tmp_ψ :: MPS, bonds)
    entropy = Vector{Float64}()
    for tmp_bond in bonds
        site_index = tmp_bond.s2
        orthogonalize!(tmp_ψ, site_index)

        i₁, j₁ = siteind(tmp_ψ, site_index), linkind(tmp_ψ, tmp_bond.s1)
        _, C1, _ = svd(tmp_ψ[site_index], i₁, j₁)

        C1 = matrix(C1)
        SvN₁ = compute_entropy(C1)
        
        # @show site_index, SvN₁
        push!(entropy, SvN₁)
    end

    return entropy
end


# Compute von Neumann entanglement entropy given the eigen values
function compute_entropy(input_matrix)
    local tmpEntropy = 0.0
    for index in 1 : size(input_matrix, 1) 
        tmp = input_matrix[index, index]^2
        if tmp > 1E-8
            tmpEntropy += -tmp * log(tmp)
        end
    end
    return tmpEntropy
end