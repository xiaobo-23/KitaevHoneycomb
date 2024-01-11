## 1/10/2024
## Code up the Kitaev-Heisenberg model on the honeycomb lattice

# using Pkg
using ITensors
using HDF5

include("src/heisenberg/Honeycomb_Lattice.jl")

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
  Nx = 16
  Ny = 4
  N = Nx * Ny

  # Set up the Hamiltonian parameters
  ϕ = 0.0

  # Configure the lattice geometery
  # square lattice
  # lattice = square_lattice(Nx, Ny; yperiodic=true)
  # @show length(lattice)
  # @show lattice

  # triangular lattice
  # lattice = triangular_lattice(Nx, Ny; yperiodic=true)
  # @show length(lattice)
  # @show lattice
  
  # honeycomb lattice
  # lattice = honeycomb_lattice_Cstyle(Nx, Ny; yperiodic=true)
  lattice = honeycomb_lattice_rings(Nx, Ny; yperiodic=true)
  # lattice = honeycomb_lattice_rings_pbc(Nx, Ny; yperiodic=true)
  number_of_bonds = length(lattice)
  @show number_of_bonds
  @show lattice

  
  # Set up the Hamiltonian by constructing the MPO
  os = OpSum()
  for b in lattice
    os .+= 0.5 * cos(ϕ), "S+", b.s1, "S-", b.s2
    os .+= 0.5 * cos(ϕ), "S-", b.s1, "S+", b.s2
    os .+= cos(ϕ), "Sz", b.s1, "Sz", b.s2
  end
  # Add Kitaev terms
  for b in lattice
    column_number = div(b.s1 - 1, Ny) + 1
    
    if iseven(column_number)
      println("even column & type 1 bond")
      @show column_number, b.s1, b.s2
      os .+= 2 * sin(ϕ), "Sz", b.s1, "Sz", b.s2
    else
      if mod(b.s1, 2) == mod(b.s2, 2)
        println("odd column & type 2 bond")
        @show column_number, b.s1, b.s2
        os .+= 2 * sin(ϕ), "Sx", b.s1, "Sx", b.s2
      else
        println("odd column & type 3 bond")
        @show column_number, b.s1, b.s2
        os .+= 2 * sin(ϕ), "Sy", b.s1, "Sy", b.s2
      end
    end
  end
  sites = siteinds("S=1/2", N; conserve_qns=false)
  H = MPO(os, sites)
  
  # # Initialize wavefunction to a random MPS
  # # of bond-dimension 10 with same quantum
  # # numbers as `state`
  # state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
  # ψ₀ = randomMPS(sites, state, 20)

  # # 1/5/2024
  # # Add noise terms to prevent DMRG from getting stuck in a local minimum
  # nsweeps = 12
  # maxdim  = [20, 60, 100, 100, 200, 400, 800, 1000, 1500, 2000]
  # cutoff  = [1E-8]
  # # noise   = [1E-6, 1E-7, 1E-8, 0.0]

  # tmp_observer = energyObserver()
  # Sz₀ = expect(ψ₀, "Sz", sites = 1 : N)
  # energy, ψ = dmrg(H, ψ₀; nsweeps, maxdim, cutoff, observer = tmp_observer)
  # Sz = expect(ψ, "Sz", sites = 1 : N)
  # zzcorr = correlation_matrix(ψ, "Sz", "Sz", sites = 1 : N)

  # # Sz₀ = expect(ψ₀, "Sz", sites = 1 : N)
  # # # Sx₀ = expect(ψ₀, "Sx", sites = 1 : N)
  # # energy, ψ = dmrg(H, ψ₀; nsweeps, maxdim, cutoff)
  # # Sz = expect(ψ, "Sz", sites = 1 : N)
  # # # Sx = expect(ψ, "Sx", sites = 1 : N)
  # # zzcorr = correlation_matrix(ψ, "Sz", "Sz", sites = 1 : N)

  # # Check the variance of the energy
  # H2 = inner(H, ψ, H, ψ)
  # E₀ = inner(ψ', H, ψ)
  # variance = H2 - E₀^2
  # @show variance

  # # Compute energy per bound
  # @show number_of_bonds, energy / number_of_bonds
  # @show N, 4 * energy / N
  # @show E₀
  # @show tmp_observer.ehistory
  
  # h5open("data/2d_heisenberg_honeycomb_lattice_pbc_rings_L$(Nx)W$(Ny).h5", "w") do file
  #   write(file, "psi", ψ)
  #   write(file, "NormalizedE0", energy / number_of_bonds)
  #   write(file, "E0", energy)
  #   write(file, "E0variance", variance)
  #   write(file, "Sz0", Sz₀)
  #   write(file, "Sz",  Sz)
  #   write(file, "Czz", zzcorr)
  #   write(file, "Ehist", tmp_observer.ehistory)
  #   # write(file, "Sx0", Sx₀)
  #   # write(file, "Sx",  Sx)
  # end

  return
end