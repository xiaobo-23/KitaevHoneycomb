# 1/23/2024
# Simulate the 2d Kitaev model on a honeycomb lattice

# using Pkg
using ITensors
using HDF5

include("src/heisenberg/Honeycomb_Lattice.jl")
include("src/heisenberg/Entanglement.jl")


# Define a custom observer
mutable struct energyObserver <: AbstractObserver
  ehistory::Vector{Float64}

  energyObserver() = new(Float64[])
end

# # Overload the checkdone! method
# function ITensors.checkdone!(tmp_energy::energyObserver; kwargs...)
#   energy = kwargs[:energy]
#   println("The energy is $energy")
#   push!(tmp_energy.ehistory, energy)
# end

# Overloading the measure! method
function ITensors.measure!(tmpObs::energyObserver; kwargs...)
  energy = kwargs[:energy]
  sweep = kwargs[:sweep]
  bond = kwargs[:bond]
  outputlevel = kwargs[:outputlevel]

  if bond == 1
    push!(tmpObs.ehistory, energy)
  end
end

let
  # Set up the lattice parameters
  Nx_unit_cell = 4
  Ny_unit_cell = 4
  Nx = 2 * Nx_unit_cell
  Ny = Ny_unit_cell
  N = Nx * Ny

  # Set up the interaction parameters for the Hamiltonian
  # Fine tunning these parameters can lead to different topological properties
  Jx = 1.0
  Jy = 1.0
  Jz = 1.0

  # honeycomb lattice
  # lattice = honeycomb_lattice_Cstyle(Nx, Ny; yperiodic=true)
  # lattice = honeycomb_lattice_rings(Nx, Ny; yperiodic=true)
  lattice = honeycomb_lattice_rings_pbc(Nx, Ny; yperiodic=true)
  number_of_bonds = length(lattice)
  @show number_of_bonds
  @show lattice
  
  os = OpSum()
  for b in lattice
    tmp_x = div(b.s1 - 1, Ny) + 1
    if mod(tmp_x, 2) == 0
      os .+= Jz, "Sz", b.s1, "Sz", b.s2
    else
      if mod(b.s1, 2) == mod(b.s2, 2)
        os .+= Jx, "Sx", b.s1, "Sx", b.s2
      else
        os .+= Jy, "Sy", b.s1, "Sy", b.s2
      end 
    end
  end
  sites = siteinds("S=1/2", N; conserve_qns=false)
  H = MPO(os, sites)
  
  # Initialize wavefunction to a random MPS
  # of bond-dimension 10 with same quantum
  # numbers as `state`
  state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
  ψ₀ = randomMPS(sites, state, 20)

  # 1/5/2024
  # Add noise terms to prevent DMRG from getting stuck in a local minimum
  nsweeps = 12
  maxdim  = [20, 60, 100, 100, 200, 400, 800, 1000, 1500, 3000]
  cutoff  = [1E-8]
  # noise   = [1E-6, 1E-7, 1E-8, 0.0]

  tmp_observer = energyObserver()
  Sz₀ = expect(ψ₀, "Sz", sites = 1 : N)
  energy, ψ = dmrg(H, ψ₀; nsweeps, maxdim, cutoff, observer = tmp_observer)
  Sz = expect(ψ, "Sz", sites = 1 : N)
  zzcorr = correlation_matrix(ψ, "Sz", "Sz", sites = 1 : N)
  
  # Sz₀ = expect(ψ₀, "Sz", sites = 1 : N)
  # # Sx₀ = expect(ψ₀, "Sx", sites = 1 : N)
  # energy, ψ = dmrg(H, ψ₀; nsweeps, maxdim, cutoff)
  # Sz = expect(ψ, "Sz", sites = 1 : N)
  # # Sx = expect(ψ, "Sx", sites = 1 : N)
  # zzcorr = correlation_matrix(ψ, "Sz", "Sz", sites = 1 : N)

  # Check the variance of the energy
  H2 = inner(H, ψ, H, ψ)
  E₀ = inner(ψ', H, ψ)
  variance = H2 - E₀^2
  @show variance

  # # 1/22/2024
  # # Check von Neumann entanglement entropy per bond
  # SvN = entanglement_entropy_bonds(ψ, lattice)
  # @show SvN

  # Compute energy per bound
  @show number_of_bonds, energy / number_of_bonds
  @show N, energy / N
  @show E₀
  @show tmp_observer.ehistory
  
  h5open("data/2d_kitaev_honeycomb_lattice_pbc_rings_L$(Nx)W$(Ny).h5", "w") do file
    write(file, "psi", ψ)
    write(file, "NormalizedE0", energy / number_of_bonds)
    write(file, "E0", energy)
    write(file, "E0variance", variance)
    # write(file, "Entropy", SvN)
    write(file, "Sz0", Sz₀)
    write(file, "Sz",  Sz)
    write(file, "Czz", zzcorr)
    write(file, "Ehist", tmp_observer.ehistory)
    # write(file, "Sx0", Sx₀)
    # write(file, "Sx",  Sx)
  end

  return
end