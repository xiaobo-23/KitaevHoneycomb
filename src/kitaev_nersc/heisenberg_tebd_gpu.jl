# 12/02/2025 
# Test Metal.jl backend using time evolution of the Heisenebrg model as an example
# TO-DO: test the CUDA.jl backend as well

using ITensors, ITensorMPS
using Metal

let
  N = 10
  cutoff = 1e-10
  tau = 0.1
  ttotal = 1.0

  # Make an array of 'site' indices
  sites = siteinds("S=1/2", N; conserve_qns=true)
  state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
  psi = MPS(sites, states=state)
  psi_mtl = mtl(psi)


  # Make gates (1,2),(2,3),(3,4),...
  gates = ITensor[]
  for j in 1:(N - 1)
    s₁, s₂ = sites[j], sites[j + 1]
    hj = op("Sz", s₁) * op("Sz", s₂) + 1/2 * op("S+", s₁) * op("S-", s₂) + 1/2 * op("S-", s₁) * op("S+", s₂)
    Gj = exp(-im * tau / 2 * hj)
    Gj_mtl = mtl(Gj)
    push!(gates, Gj_mtl)
  end
  append!(gates, reverse(gates))


  # Compute and print <Sz> at each time step then apply the gates to go to the next time
  
  for t in 0.0:tau:ttotal
    Sz = expect(psi, "Sz"; sites=div(N, 2))
    println("$t $Sz")

    t≈ttotal && break

    psi_mtl = apply(gates, psi_mtl; cutoff)
    normalize!(psi_mtl)
  end

  return
end