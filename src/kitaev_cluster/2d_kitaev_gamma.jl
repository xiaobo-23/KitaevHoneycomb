# Simulate the 2d Kitaev model on a honeycomb lattice with magnetic field and bond-dependent interactions

using ITensors
using HDF5

include("../HoneycombLattice.jl")
include("../Entanglement.jl")
include("../TopologicalLoops.jl")
include("../PlaquetteGenerator.jl")



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
  NxUnitCell = 9
  NyUnitCell = 3
  Nx = 2 * NxUnitCell
  Ny = NyUnitCell
  N = Nx * Ny

  # Set up the interaction parameters for the Hamiltonian
  # |Jx| <= |Jy| + |Jz| in the gapless A-phase
  # |Jx| > |Jy| + |Jz| in the gapped B-phase
  Jx=Jy=Jz=1.0
  h=0.0
  gamma=0.8
  gamma_prime=0.0
  # Coefficient for the interaction strengths near the defects
  alpha=0.01
  @show Jx, Jy, Jz, h, gamma, gamma_prime

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
      coefficient_Jx = alpha * Jx
      coefficient_Jy = alpha * Jy
      coefficient_Jz = alpha * Jz
    else
      coefficient_Jx = Jx
      coefficient_Jy = Jy
      coefficient_Jz = Jz
    end
    @show b.s1, b.s2, coefficient_Jx, coefficient_Jy, coefficient_Jz  

    if mod(tmp_x, 2) == 0
      os .+= -coefficient_Jz, "Sz", b.s1, "Sz", b.s2
      
      # Introduce the bond-dependent interactions with γ terms
      os .+= gamma, "Sx", b.s1, "Sy", b.s2
      os .+= gamma, "Sy", b.s1, "Sx", b.s2

      # # Introduce the bond-dependent interactions with γ' terms
      # os .+= gamma_prime, "Sx", b.s1, "Sz", b.s2
      # os .+= gamma_prime, "Sz", b.s1, "Sx", b.s2
      # os .+= gamma_prime, "Sy", b.s1, "Sz", b.s2
      # os .+= gamma_prime, "Sz", b.s1, "Sy", b.s2

      @show b.s1, b.s2
      enumerate_bonds += 1
    else
      if b.s2 == b.s1 + Ny
        os .+= -coefficient_Jx, "Sx", b.s1, "Sx", b.s2

        # Introduce the bond-dependent interactions with γ terms
        os .+= gamma, "Sy", b.s1, "Sz", b.s2
        os .+= gamma, "Sz", b.s1, "Sy", b.s2

        # # Introduce the bond-dependent interactions with γ' terms
        # os .+= gamma_prime, "Sz", b.s1, "Sx", b.s2
        # os .+= gamma_prime, "Sx", b.s1, "Sz", b.s2
        # os .+= gamma_prime, "Sy", b.s1, "Sx", b.s2
        # os .+= gamma_prime, "Sx", b.s1, "Sy", b.s2

        # @show b.s1, b.s2
        # enumerate_bonds += 1
      else
        os .+= -coefficient_Jy, "Sy", b.s1, "Sy", b.s2

        # Introduce the bond-dependent interactions with γ terms
        os .+= gamma, "Sx", b.s1, "Sz", b.s2
        os .+= gamma, "Sz", b.s1, "Sx", b.s2
        
        # # Introduce the bond-dependent interactions with γ' terms
        # os .+= gamma_prime, "Sx", b.s1, "Sy", b.s2
        # os .+= gamma_prime, "Sy", b.s1, "Sx", b.s2
        # os .+= gamma_prime, "Sz", b.s1, "Sy", b.s2
        # os .+= gamma_prime, "Sy", b.s1, "Sz", b.s2

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
  nsweeps = 20
  maxdim  = [20, 60, 100, 100, 200, 400, 800, 1000, 1500, 2000]
  cutoff  = [1E-10]
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
  loop_inds = PlaquetteList(NxUnitCell, NyUnitCell, "rings", false)
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
  LoopY = Vector{String}([])
  for index in 1 : 2 * Ny
    push!(LoopY, "Z")
  end
  @show LoopY

  # Construct the loop indices in the y direction
  y_inds = LoopList(NxUnitCell, NyUnitCell, "rings", "y")
  y_loop_eigenvalues = Vector{Float64}(undef, size(y_inds)[1])

  # Compute eigenvalues of the loop operators in the y direction
  for loop_index in 1 : size(y_inds)[1]
    # @show y_inds[loop_index, :]
    os_wl = OpSum()
    os_wl += LoopY[1], y_inds[loop_index, 1], 
      LoopY[2], y_inds[loop_index, 2], 
      LoopY[3], y_inds[loop_index, 3], 
      LoopY[4], y_inds[loop_index, 4], 
      LoopY[5], y_inds[loop_index, 5], 
      LoopY[6], y_inds[loop_index, 6]
      # LoopY[7], y_inds[loop_index, 7],
      # LoopY[8], y_inds[loop_index, 8]
    Wl = MPO(os_wl, sites)
    y_loop_eigenvalues[loop_index] = real(inner(ψ', Wl, ψ))
  end


  # Compute the plaquette correlation function
  SiteList = Vector{Int64}([])
  for index in 1 : NxUnitCell - 1
    push!(SiteList, (2 * index - 1) * NyUnitCell)
  end
  @show SiteList
  
  # Define the string vector of the plaquette correlator
  # It should be a twelve-point correlator
  plaquette_correlator = Vector{String}(["Z", "Y", "X", "Z", "Y", "X", 
  "Z", "Y", "X", "Z", "Y", "X"])
  plaquette_correlation = Matrix{Float64}(undef, length(SiteList), length(SiteList))

  for (index1, Seed_Index1) in enumerate(SiteList)
    for (index2, Seed_Index2) in enumerate(SiteList)
      tmp_indices = Vector{Int64}([])
      Plaquette1 = GeneratePlaquetteRings(Seed_Index1, NyUnitCell; edge=false)
      Plaquette2 = GeneratePlaquetteRings(Seed_Index2, NyUnitCell; edge=false)
      append!(tmp_indices, Plaquette1)
      append!(tmp_indices, Plaquette2)
      @show index1, Seed_Index1, index2, Seed_Index2
      @show tmp_indices

      tmp_op = OpSum()
      tmp_op += plaquette_correlator[1], tmp_indices[1], 
        plaquette_correlator[2], tmp_indices[2], 
        plaquette_correlator[3], tmp_indices[3], 
        plaquette_correlator[4], tmp_indices[4], 
        plaquette_correlator[5], tmp_indices[5], 
        plaquette_correlator[6], tmp_indices[6],
        plaquette_correlator[7], tmp_indices[7], 
        plaquette_correlator[8], tmp_indices[8],
        plaquette_correlator[9], tmp_indices[9], 
        plaquette_correlator[10], tmp_indices[10],
        plaquette_correlator[11], tmp_indices[11], 
        plaquette_correlator[12], tmp_indices[12]
      WpWp = MPO(tmp_op, sites)
      plaquette_correlation[index1, index2] = real(inner(ψ', WpWp, ψ))
    end
  end
  

  # Print out several quantities of interest including the energy per site etc.
  @show number_of_bonds, energy / number_of_bonds
  @show N, energy / N
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
  println("Plaquette correlation function:")
  @show plaquette_correlation
  println("")


  # Check the variance of the energy
  H2 = inner(H, ψ, H, ψ)
  E₀ = inner(ψ', H, ψ)
  variance = H2 - E₀^2

  println("")
  @show E₀
  println("Variance of the energy is $variance")
  println("")
  
  h5open("../data/2d_kitaev_FM_h$(h).h5", "w") do file
    write(file, "psi", ψ)
    write(file, "NormalizedE0", energy / number_of_bonds)
    write(file, "E0", energy)
    write(file, "E0variance", variance)
    write(file, "Ehist", tmp_observer.ehistory)
    # write(file, "Entropy", SvN)
    write(file, "Sx0", Sx₀)
    write(file, "Sx",  Sx)
    write(file, "Cxx", xxcorr)
    write(file, "Sy0", Sy₀)
    write(file, "Sy", Sy)
    write(file, "Cyy", yycorr)
    write(file, "Sz0", Sz₀)
    write(file, "Sz",  Sz)
    write(file, "Czz", zzcorr)
    write(file, "Plaquette", W_operator_eigenvalues)
    write(file, "Wly", y_loop_eigenvalues)
    write(file, "PlaquetteCorrelation", plaquette_correlation)
  end

  return
end