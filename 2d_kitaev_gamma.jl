# Simulate the 2d Kitaev model on a honeycomb lattice with magnetic field and bond-dependent interactions

# using Pkg
using ITensors
using HDF5

include("src/kitaev_heisenberg/Honeycomb_Lattice.jl")
include("src/kitaev_heisenberg/Entanglement.jl")
include("src/kitaev_heisenberg/TopologicalLoops.jl")



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
  Nx_unit_cell = 4
  Ny_unit_cell = 4
  Nx = 2 * Nx_unit_cell
  Ny = Ny_unit_cell
  N = Nx * Ny

  # Set up the interaction parameters for the Hamiltonian
  # |Jx| <= |Jy| + |Jz| in the gapless A-phase
  # |Jx| > |Jy| + |Jz| in the gapped B-phase
  Jx=Jy=Jz=1.0
  h=0.0
  gamma=0.4
  @show Jx, Jy, Jz, h, gamma

  # honeycomb lattice
  x_direction_periodic = false
  if x_direction_periodic
    lattice = honeycomb_lattice_rings_pbc(Nx, Ny; yperiodic=true)
  else
    lattice = honeycomb_lattice_rings(Nx, Ny; yperiodic=true)
  end
  # lattice = honeycomb_lattice_Cstyle(Nx, Ny; yperiodic=true)
  number_of_bonds = length(lattice)
  @show number_of_bonds
  @show lattice
  
  
  # Select the site where vacancies are introduced
  sites_to_delete = Set{Int64}([])
  lattice_sites = Set{Int64}()

  
  # Construct the Hamiltonian using the OpSum system
  os = OpSum()
  enumerate_bonds = 0
  for b in lattice
    tmp_x = div(b.s1 - 1, Ny) + 1
    if in(b.s1, sites_to_delete) || in(b.s2, sites_to_delete)
      coefficient_Jx = 0.01 * Jx
      coefficient_Jy = 0.01 * Jy
      coefficient_Jz = 0.01 * Jz
    else
      coefficient_Jx = Jx
      coefficient_Jy = Jy
      coefficient_Jz = Jz
    end
    @show b.s1, b.s2, coefficient_Jx, coefficient_Jy, coefficient_Jz  

    if mod(tmp_x, 2) == 0
      os .+= -coefficient_Jz, "Sz", b.s1, "Sz", b.s2
      
      # Introduce the bond-dependent interactions
      os .+= gamma, "Sx", b.s1, "Sy", b.s2
      os .+= gamma, "Sy", b.s1, "Sx", b.s2

      # @show b.s1, b.s2
      # enumerate_bonds += 1
    else
      if b.s2 == b.s1 + Ny
        os .+= -coefficient_Jx, "Sx", b.s1, "Sx", b.s2

        # Introduce the bond-dependent interactions
        os .+= gamma, "Sy", b.s1, "Sz", b.s2
        os .+= gamma, "Sz", b.s1, "Sy", b.s2

        # @show b.s1, b.s2
        # enumerate_bonds += 1
      else
        os .+= -coefficient_Jy, "Sy", b.s1, "Sy", b.s2

        # Introduce the bond-dependent interactions
        os .+= gamma, "Sx", b.s1, "Sz", b.s2
        os .+= gamma, "Sz", b.s1, "Sx", b.s2

        # @show b.s1, b.s2
        # enumerate_bonds += 1
      end 
    end

    if !in(b.s1, lattice_sites)
      push!(lattice_sites, b.s1)
    end

    if !in(b.s2, lattice_sites)
      push!(lattice_sites, b.s2)
    end
  end
  @show enumerate_bonds

  
  # Add the Zeeman coupling of the spins to a magnetic field applied in [111] direction
  # The magnetic field breaks integrability 
  @show length(lattice_sites)
  @show lattice_sites
  if h > 1e-8
    for tmp_site in lattice_sites
      os .+= h, "Sx", tmp_site
      os .+= h, "Sy", tmp_site
      os .+= h, "Sz", tmp_site
    end
  end
  # @show os
 
  sites = siteinds("S=1/2", N; conserve_qns=false)
  H = MPO(os, sites)

  
  # Initialize wavefunction to a random MPS
  # of bond-dimension 10 with same quantum 
  # numbers as `state`
  state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
  ψ₀ = randomMPS(sites, state, 20)

  
  # Set up the parameters for DMRG including the maximum bond dimension, truncation error cutoff, etc.
  nsweeps = 10
  maxdim  = [20, 60, 100, 100, 200, 400, 800, 1000, 1500, 2000]
  cutoff  = [1E-8]
  # Add noise terms to prevent DMRG from getting stuck in a local minimum
  # noise   = [1E-6, 1E-7, 1E-8, 0.0]

  
  # Run DMRG and measure the energy, one-point functions, and two-point functions
  tmp_observer = energyObserver()
  Sx₀ = expect(ψ₀, "Sx", sites = 1 : N)
  Sy₀ = expect(ψ₀, "iSy", sites = 1 : N)
  Sz₀ = expect(ψ₀, "Sz", sites = 1 : N)
  energy, ψ = dmrg(H, ψ₀; nsweeps, maxdim, cutoff, observer = tmp_observer)
  Sx = expect(ψ, "Sx", sites = 1 : N)
  Sy = expect(ψ, "iSy", sites = 1 : N)
  Sz = expect(ψ, "Sz", sites = 1 : N)
  xxcorr = correlation_matrix(ψ, "Sx", "Sx", sites = 1 : N)
  yycorr = correlation_matrix(ψ, "Sy", "Sy", sites = 1 : N)
  zzcorr = correlation_matrix(ψ, "Sz", "Sz", sites = 1 : N)
  

  # Construct the plaquette operators
  # normalize!(ψ)
  plaquette_operator = Vector{String}(["Z", "iY", "X", "X", "iY", "Z"])
  loop_inds = PlaquetteList(Nx_unit_cell, Ny_unit_cell, "rings", false)
  for index in 1 : size(loop_inds)[1]
    @show loop_inds[index, :]
  end
  W_operator_eigenvalues = Vector{Float64}(undef, size(loop_inds)[1])
  
  
  # Compute the eigenvalues of the plaquette operator
  for loop_index in 1 : size(loop_inds)[1]
    os_w = OpSum()
    os_w += plaquette_operator[1], loop_inds[loop_index, 1], plaquette_operator[2], loop_inds[loop_index, 2], 
      plaquette_operator[3], loop_inds[loop_index, 3], plaquette_operator[4], loop_inds[loop_index, 4], 
      plaquette_operator[5], loop_inds[loop_index, 5], plaquette_operator[6], loop_inds[loop_index, 6]
    W = MPO(os_w, sites)
    W_operator_eigenvalues[loop_index] = real(inner(ψ', W, ψ))
    # @show inner(ψ', W, ψ) / inner(ψ', ψ)
  end
  

  # Construct the loop operators in the y direction
  loop_operator_y = Vector{String}([])
  for index in 1 : 2 * Ny
    push!(loop_operator_y, "Z")
  end
  @show loop_operator_y

  # Construct the loop indices in the y direction
  y_inds = LoopList(Nx_unit_cell, Ny_unit_cell, "rings", "y")
  y_loop_eigenvalues = Vector{Float64}(undef, size(y_inds)[1])

  # Compute eigenvalues of the loop operators in the y direction
  for loop_index in 1 : size(y_inds)[1]
    # @show y_inds[loop_index, :]
    os_wl = OpSum()
    os_wl += loop_operator_y[1], y_inds[loop_index, 1], 
      loop_operator_y[2], y_inds[loop_index, 2], 
      loop_operator_y[3], y_inds[loop_index, 3], 
      loop_operator_y[4], y_inds[loop_index, 4], 
      loop_operator_y[5], y_inds[loop_index, 5], 
      loop_operator_y[6], y_inds[loop_index, 6],
      loop_operator_y[7], y_inds[loop_index, 7],
      loop_operator_y[8], y_inds[loop_index, 8]
    Wl = MPO(os_wl, sites)
    y_loop_eigenvalues[loop_index] = real(inner(ψ', Wl, ψ))
  end

  # Check the variance of the energy
  H2 = inner(H, ψ, H, ψ)
  E₀ = inner(ψ', H, ψ)
  variance = H2 - E₀^2


  # Print out several quantities of interest including the energy per site etc.
  @show number_of_bonds, energy / number_of_bonds
  @show N, energy / N
  @show E₀
  @show tmp_observer.ehistory
  println("")
  println("Eigenvalues of the plaquette operator:")
  @show W_operator_eigenvalues
  println("")

  print("")
  println("Eigenvalues of the loop operator(s):")
  @show y_loop_eigenvalues
  println("")


  println("")
  println("Variance of the energy is $variance")
  println("")
  
  # h5open("data/bond-dependent/2d_kitaev_L$(Nx)W$(Ny)_FM.h5", "w") do file
  #   write(file, "psi", ψ)
  #   write(file, "NormalizedE0", energy / number_of_bonds)
  #   write(file, "E0", energy)
  #   write(file, "E0variance", variance)
  #   write(file, "Ehist", tmp_observer.ehistory)
  #   # write(file, "Entropy", SvN)
  #   write(file, "Sx0", Sx₀)
  #   write(file, "Sx",  Sx)
  #   write(file, "Cxx", xxcorr)
  #   write(file, "Sy0", Sy₀)
  #   write(file, "Sy", Sy)
  #   write(file, "Cyy", yycorr)
  #   write(file, "Sz0", Sz₀)
  #   write(file, "Sz",  Sz)
  #   write(file, "Czz", zzcorr)
  #   write(file, "Plaquette", W_operator_eigenvalues)
  #   write(file, "Wly", y_loop_eigenvalues)
  # end

  return
end