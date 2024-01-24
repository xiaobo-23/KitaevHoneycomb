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

# # Overload the checkdone! method
# function ITensors.checkdone!(tmp_energy::energyObserver; kwargs...)
#   energy = kwargs[:energy]
#   println("The energy is $energy")
#   push!(tmp_energy.ehistory, energy)
# end

let
  # Set up the parameters for the lattice
  # Number of unit cells in x and y directions
  Nx_unit_cell = 3
  Ny_unit_cell = 3
  Nx = 2 * Nx_unit_cell
  Ny = Ny_unit_cell
  N = Nx * Ny

  # Set up the interaction parameters for the Hamiltonian
  # |Jx| <= |Jy| + |Jz| in the A-phase
  # |Jx| > |Jy| + |Jz| in the B-phase
  Jx = -3.0
  Jy = -1.0
  Jz = -1.0

  # honeycomb lattice
  # lattice = honeycomb_lattice_Cstyle(Nx, Ny; yperiodic=true)
  # lattice = honeycomb_lattice_rings(Nx, Ny; yperiodic=true)
  lattice = honeycomb_lattice_rings_pbc(Nx, Ny; yperiodic=true)
  number_of_bonds = length(lattice)
  # @show number_of_bonds
  # @show lattice
  
  # Construct the Hamiltonian as an MPO
  # This part explicitly depends on how the lattice is mapped from a 2d honeycomb lattice to a 1d chain
  os = OpSum()
  # count_bonds = 0
  for b in lattice
    tmp_x = div(b.s1 - 1, Ny) + 1
    tmp_y = mod(b.s1 - 1, Ny) + 1
    tmp_x_prime = div(b.s2 - 1, Ny) + 1
    tmp_y_prime = mod(b.s2 - 1, Ny) + 1
    
    if mod(tmp_x, 2) == 0
      os .+= -Jz, "Sz", b.s1, "Sz", b.s2
      # @show b.s1, b.s2
      # count_bonds += 1
    else
      if (mod(b.s1, 2) == mod(b.s2, 2)) || (mod(b.s1, 2) != mod(b.s2, 2) && tmp_y == 1 && tmp_y_prime == Ny)
        os .+= -Jx, "Sx", b.s1, "Sx", b.s2
        # @show b.s1, b.s2
        # count_bonds += 1
      elseif mod(b.s1, 2) != mod(b.s2, 2) && abs(tmp_x - tmp_x_prime) == 1
        os .+= -Jy, "Sy", b.s1, "Sy", b.s2
        # @show b.s1, b.s2
        # count_bonds += 1
      end 
    end

    if tmp_x == 1 && tmp_x_prime == Nx
      os .+= -Jz, "Sz", b.s2, "Sz", b.s1
      # @show b.s2, b.s1
      # count_bonds += 1
    end
  end
# @show count_bonds
# @show os
  sites = siteinds("S=1/2", N; conserve_qns=false)
  H = MPO(os, sites)
  
  # Initialize wavefunction to a random MPS
  # of bond-dimension 10 with same quantum 
  # numbers as `state`
  state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
  ψ₀ = randomMPS(sites, state, 20)


  # Add noise terms to prevent DMRG from getting stuck in a local minimum
  nsweeps = 15
  maxdim  = [20, 60, 100, 100, 200, 400, 800, 1000, 1500, 3000]
  cutoff  = [1E-8]
  noise   = [1E-6, 1E-7, 1E-8, 0.0]



  tmp_observer = energyObserver()
  Sx₀ = expect(ψ₀, "Sx", sites = 1 : N)
  Sy₀ = expect(ψ₀, "iSy", sites = 1 : N)
  Sz₀ = expect(ψ₀, "Sz", sites = 1 : N)
  energy, ψ = dmrg(H, ψ₀; nsweeps, maxdim, cutoff, observer = tmp_observer)
  Sz = expect(ψ, "Sz", sites = 1 : N)
  Sx = expect(ψ, "Sx", sites = 1 : N)
  Sy = expect(ψ, "iSy", sites = 1 : N)
  xxcorr = correlation_matrix(ψ, "Sx", "Sx", sites = 1 : N)
  yycorr = correlation_matrix(ψ, "Sy", "Sy", sites = 1 : N)
  zzcorr = correlation_matrix(ψ, "Sz", "Sz", sites = 1 : N)
  

  # Check the variance of the energy
  H2 = inner(H, ψ, H, ψ)
  E₀ = inner(ψ', H, ψ)
  variance = H2 - E₀^2
  @show variance


  # 1/24/2024
  # Check the eigenvalue of the plaquette operator
  orthogonalize!(ψ, 4)
  plaquette_operator = Vector{String}(["Y", "Z", "X", "Y", "Z", "X"])
  plaquette_operator_imaginary = Vector{String}(["Z", "X", "iY", "iY", "Z", "X"])
  loop_inds = Vector{Int64}([4, 2, 5, 8, 10, 7])
  # plaquette_operator = Vector{String}(["σy", "σz", "σx", "σy"])
  # loop_inds = Vector{Int64}([4, 2, 5, 8])
  
  os_w = OpSum()
  os_w .+= plaquette_operator[1], loop_inds[1], plaquette_operator[2], loop_inds[2], 
    plaquette_operator[3], loop_inds[3], plaquette_operator[4], loop_inds[4], 
    plaquette_operator[5], loop_inds[5], plaquette_operator[6], loop_inds[6]
  W = MPO(os_w, sites)
  W_eigenvalue = inner(ψ', W, ψ)
  ψ_updated = W * ψ
  W_eigenvalue_updated = inner(ψ', ψ_updated)
  
  os_w_imaginary = OpSum()
  os_w_imaginary .+= plaquette_operator_imaginary[1], loop_inds[1], plaquette_operator_imaginary[2], loop_inds[2], 
    plaquette_operator_imaginary[3], loop_inds[3], plaquette_operator_imaginary[4], loop_inds[4], 
    plaquette_operator_imaginary[5], loop_inds[5], plaquette_operator_imaginary[6], loop_inds[6]
  W_imaginary = MPO(os_w_imaginary, sites)
  W_eigenvalue_imaginary = inner(ψ', W_imaginary, ψ)
  
  # Debug the plaquette operator
  @show os_w
  @show os_w_imaginary
  @show W_eigenvalue
  @show W_eigenvalue_updated
  @show W_eigenvalue_imaginary
  @show 4 * xxcorr[4, 2]

  # # 1/22/2024
  # # Check von Neumann entanglement entropy per bond
  # SvN = entanglement_entropy_bonds(ψ, lattice)
  # @show SvN


  # Compute energy per bound
  @show number_of_bonds, energy / number_of_bonds
  @show N, energy / N
  @show E₀
  @show tmp_observer.ehistory
  # @show Sy
  
  # h5open("data/2d_kitaev_honeycomb_lattice_pbc_rings_L$(Nx)W$(Ny)_AFM.h5", "w") do file
  #   write(file, "psi", ψ)
  #   write(file, "NormalizedE0", energy / number_of_bonds)
  #   write(file, "E0", energy)
  #   write(file, "E0variance", variance)
  #   # write(file, "Entropy", SvN)
  #   write(file, "Sx0", Sx₀)
  #   write(file, "Sx",  Sx)
  #   write(file, "Cxx", xxcorr)
  #   # write(file, "Sy0", Sy₀)
  #   # write(file, "Sy", Sy)
  #   # write(file, "Cyy", yycorr)
  #   write(file, "Sz0", Sz₀)
  #   write(file, "Sz",  Sz)
  #   write(file, "Czz", zzcorr)
  #   write(file, "Ehist", tmp_observer.ehistory)
  # end

  return
end