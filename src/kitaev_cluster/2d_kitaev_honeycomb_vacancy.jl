# Simulate the 2d Kitaev model with Zeeman fields & vacancies on a honeycomb lattice

# using Pkg
using ITensors
using HDF5

include("../honeycomb_lattice.jl")
include("../entanglement.jl")
include("../TopologicalLoops.jl")


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
  Nx_unit_cell = 12
  Ny_unit_cell = 4
  Nx = 2 * Nx_unit_cell
  Ny = Ny_unit_cell
  N = Nx * Ny

  # Set up the interaction parameters for the Hamiltonian
  # |Jx| <= |Jy| + |Jz| in the gapless A-phase
  # |Jx| > |Jy| + |Jz| in the gapped B-phase
  Jx=Jy=Jz=1.0
  h=0.0
  @show Jx, Jy, Jz, h

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
  
  # Select the site where the vacancy is put into the system
  site1_to_delete = 35
  site2_to_delete = 67

  # Construct the Hamiltonian using the OpSum system
  # How the interactions are set up depends on the ordering scheme
  # The following Hamiltonain is based on the rings ordering scheme with PBC in both directions
  os = OpSum()
  lattice_sites = Set{Int64}()
  enumerate_bonds = 0
  for b in lattice
    tmp_x = div(b.s1 - 1, Ny) + 1
    if b.s1 == site1_to_delete || b.s2 == site1_to_delete || b.s1 == site2_to_delete || b.s2 == site2_to_delete
      coefficient_Jx = 0.001 * Jx
      coefficient_Jy = 0.001 * Jy
      coefficient_Jz = 0.001 * Jz
    else
      coefficient_Jx = Jx
      coefficient_Jy = Jy
      coefficient_Jz = Jz
    end
    @show b.s1, b.s2, coefficient_Jx, coefficient_Jy, coefficient_Jz  

    if mod(tmp_x, 2) == 0
      os .+= -coefficient_Jz, "Sz", b.s1, "Sz", b.s2
      # @show b.s1, b.s2
      # enumerate_bonds += 1
    else
      if b.s2 == b.s1 + Ny
        os .+= -coefficient_Jx, "Sx", b.s1, "Sx", b.s2
        # @show b.s1, b.s2
        # enumerate_bonds += 1
      else
        os .+= -coefficient_Jy, "Sy", b.s1, "Sy", b.s2
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
  

  # Check the variance of the energy
  H2 = inner(H, ψ, H, ψ)
  E₀ = inner(ψ', H, ψ)
  variance = H2 - E₀^2
 

  # # Compute the eigenvalues of the plaquette operator and its avaverage
  # # On the three-by-three lattice, there are nine plaquettes in total
  # # normalize!(ψ)
  # # plaquette_operator_im = Vector{String}(["X", "Y", "Z", "Z", "Y", "X"])
  # plaquette_operator = Vector{String}(["iY", "X", "Z", "Z", "X", "iY"])
  # loop_inds = Matrix{Int64}(undef, 9, 6)
  # loop_inds[1, :] = [5, 8, 10, 2, 4, 7]
  # loop_inds[2, :] = [6, 9, 11, 3, 5, 8]
  # loop_inds[3, :] = [4, 7, 12, 1, 6, 9]
  # loop_inds[4, :] = [11, 14, 16, 8, 10, 13]
  # loop_inds[5, :] = [12, 15, 17, 9, 11, 14]
  # loop_inds[6, :] = [10, 13, 18, 7, 12, 15]
  # loop_inds[7, :] = [17, 2, 4, 14, 16, 1]
  # loop_inds[8, :] = [18, 3, 5, 15, 17, 2]
  # loop_inds[9, :] = [16, 1, 6, 13, 18, 3]
  # @show size(loop_inds)[1]
  # W_operator_eigenvalues = Vector{Float64}(undef, size(loop_inds)[1])
  # # W_operator_im = Vector{Float64}(undef, size(loop_inds)[1])


  # # Compute the eigenvalues of the loop operator in the x-direction
  # if x_direction_periodic
  #   loop_operator_x = Vector{String}(["Y", "Y", "Y", "Y", "Y", "Y"])
  #   loop_x_inds = Matrix{Int64}(undef, 3, 6)
  #   loop_x_inds[1, :] = [3, 6, 9, 12, 15, 18]
  #   loop_x_inds[2, :] = [2, 5, 8, 11, 14, 17]
  #   loop_x_inds[3, :] = [1, 4, 7, 10, 13, 16]
  #   @show size(loop_x_inds)[1]  
  #   loop_operator_x_eigenvalues = Vector{Float64}(undef, size(loop_x_inds)[1])
  # end


  # # Compute the eigenvalues of the loop operator in y-direction
  # loop_operator_y = Vector{String}(["Z", "Z", "Z", "Z", "Z", "Z"])
  # loop_y_inds = Matrix{Int64}(undef, 3, 6)
  # loop_y_inds[1, :] = [1, 4, 2, 5, 3, 6]
  # loop_y_inds[2, :] = [7, 10, 8, 11, 9, 12]
  # loop_y_inds[3, :] = [13, 16, 14, 17, 15, 18]
  # @show size(loop_y_inds)[1]  
  # loop_operator_y_eigenvalues = Vector{Float64}(undef, size(loop_y_inds)[1])

  
  # # On a four-by-four lattice with PBC in both directions, compute the eigenvalues of the plaquette operator in all 16 plaquettes
  # # normalize!(ψ)
  # plaquette_operator = Vector{String}(["X", "iY", "Z", "Z", "iY", "X"])
  # loop_inds = Matrix{Int64}(undef, 16, 6)
  # loop_inds[1, :] = [9, 5, 1, 16, 12, 8]
  # loop_inds[2, :] = [10, 6, 2, 13, 9, 5]
  # loop_inds[3, :] = [11, 7, 3, 14, 10, 6]
  # loop_inds[4, :] = [12, 8, 4, 15, 11, 7]
  # loop_inds[5, :] = [17, 13, 9, 24, 20, 16]
  # loop_inds[6, :] = [18, 14, 10, 21, 17, 13]
  # loop_inds[7, :] = [19, 15, 11, 22, 18, 14]
  # loop_inds[8, :] = [20, 16, 12, 23, 19, 15]
  # loop_inds[9, :] = [25, 21, 17, 32, 28, 24]
  # loop_inds[10, :] = [26, 22, 18, 29, 25, 21]
  # loop_inds[11, :] = [27, 23, 19, 30, 26, 22]
  # loop_inds[12, :] = [28, 24, 20, 31, 27, 23]
  # loop_inds[13, :] = [1, 29, 25, 8, 4, 32]
  # loop_inds[14, :] = [2, 30, 26, 5, 1, 29]
  # loop_inds[15, :] = [3, 31, 27, 6, 2, 30]
  # loop_inds[16, :] = [4, 32, 28, 7, 3, 31] 
  # @show size(loop_inds)[1]
  # W_operator_eigenvalues = Vector{Float64}(undef, size(loop_inds)[1])


  # On a four-by-four lattice, compute the eigenvalues of the plaquette operator in all 16 plaquettes
  # normalize!(ψ)
  # plaquette_operator = Vector{String}(["X", "iY", "Z", "Z", "iY", "X"])
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
  
  # KEEP THIS FOR REFERENCE
  # loop_inds = Matrix{Int64}(undef, 12, 6)
  # loop_inds[1, :] = [9, 5, 1, 16, 12, 8]
  # loop_inds[2, :] = [10, 6, 2, 13, 9, 5]
  # loop_inds[3, :] = [11, 7, 3, 14, 10, 6]
  # loop_inds[4, :] = [12, 8, 4, 15, 11, 7]
  # loop_inds[5, :] = [17, 13, 9, 24, 20, 16]
  # loop_inds[6, :] = [18, 14, 10, 21, 17, 13]
  # loop_inds[7, :] = [19, 15, 11, 22, 18, 14]
  # loop_inds[8, :] = [20, 16, 12, 23, 19, 15]
  # loop_inds[9, :] = [25, 21, 17, 32, 28, 24]
  # loop_inds[10, :] = [26, 22, 18, 29, 25, 21]
  # loop_inds[11, :] = [27, 23, 19, 30, 26, 22]
  # loop_inds[12, :] = [28, 24, 20, 31, 27, 23]


  # Construct the loop operators in the y direction
  # loop_operator_y = Vector{String}(["Z", "Z", "Z", "Z", "Z", "Z", "Z", "Z"])
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

  # Compute the eigenvalue of the order parameter near the first vacancy
  order_parameter_loop = Vector{String}(["Z", "Z", "Y", "X", "X", "X", "Z", "Y", "Y", "Y", "X", "Z"])
  order_parameter_inds = Matrix{Int64}(undef, 1, 12)
  order_parameter_inds[1, :] = [31, 28, 32, 36, 39, 43, 46, 42, 38, 34, 30, 27]
  order_parameter = Vector{Float64}(undef, size(order_parameter_inds)[1])

  @show size(order_parameter_inds)[1]
  for index in 1 : size(order_parameter_inds)[1]
    os_parameter = OpSum()
    os_parameter += order_parameter_loop[1], order_parameter_inds[index, 1], 
      order_parameter_loop[2], order_parameter_inds[index, 2], 
      order_parameter_loop[3], order_parameter_inds[index, 3], 
      order_parameter_loop[4], order_parameter_inds[index, 4], 
      order_parameter_loop[5], order_parameter_inds[index, 5], 
      order_parameter_loop[6], order_parameter_inds[index, 6],
      order_parameter_loop[7], order_parameter_inds[index, 7],
      order_parameter_loop[8], order_parameter_inds[index, 8],
      order_parameter_loop[9], order_parameter_inds[index, 9],
      order_parameter_loop[10], order_parameter_inds[index, 10],
      order_parameter_loop[11], order_parameter_inds[index, 11],
      order_parameter_loop[12], order_parameter_inds[index, 12]
    W_parameter = MPO(os_parameter, sites)
    order_parameter[index] = real(inner(ψ', W_parameter, ψ))
  end

  # Compute the eigenvalue of the order parameter near the second vacancy
  order_index2 = Matrix{Int64}(undef, 1, 12)
  order_index2[1, :] = [63, 60, 64, 68, 71, 75, 78, 74, 70, 66, 62, 59]
  order_parameter2 = Vector{Float64}(undef, size(order_index2)[1])

  @show size(order_parameter_inds)[1]
  for index in 1 : size(order_index2)[1]
    os_parameter = OpSum()
    os_parameter += order_parameter_loop[1], order_index2[index, 1], 
      order_parameter_loop[2], order_index2[index, 2], 
      order_parameter_loop[3], order_index2[index, 3], 
      order_parameter_loop[4], order_index2[index, 4], 
      order_parameter_loop[5], order_index2[index, 5], 
      order_parameter_loop[6], order_index2[index, 6],
      order_parameter_loop[7], order_index2[index, 7],
      order_parameter_loop[8], order_index2[index, 8],
      order_parameter_loop[9], order_index2[index, 9],
      order_parameter_loop[10], order_index2[index, 10],
      order_parameter_loop[11], order_index2[index, 11],
      order_parameter_loop[12], order_index2[index, 12]
    W_parameter = MPO(os_parameter, sites)
    order_parameter2[index] = real(inner(ψ', W_parameter, ψ))
  end

  # # Compute the eigenvalues of the loop operators in the x-direction
  # if x_direction_periodic
  #   for loop_index in 1 : size(loop_x_inds)[1]
  #     os_wl = OpSum()
  #     os_wl += loop_operator_x[1], loop_x_inds[loop_index, 1], 
  #       loop_operator_x[2], loop_x_inds[loop_index, 2], 
  #       loop_operator_x[3], loop_x_inds[loop_index, 3], 
  #       loop_operator_x[4], loop_x_inds[loop_index, 4], 
  #       loop_operator_x[5], loop_x_inds[loop_index, 5], 
  #       loop_operator_x[6], loop_x_inds[loop_index, 6]
  #     Wl = MPO(os_wl, sites)
  #     loop_operator_x_eigenvalues[loop_index] = real(inner(ψ', Wl, ψ))
  #     # @show inner(ψ', W, ψ) / inner(ψ', ψ)
  #   end
  # end


  # # # Compute the von Neumann entanglement Entropy
  # # # TO-DO: Fix the bonds that are cut to compute the entanglement entropy
  # # SvN = entanglement_entropy_bonds(ψ, lattice)
  # # @show SvN


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
  if x_direction_periodic
    @show loop_operator_x_eigenvalues
  end
  @show y_loop_eigenvalues
  println("")

  println("")
  println("Eigenvalues of the twelve-point correlator near the first vacancy:")
  @show order_parameter
  println("")

  println("")
  println("Eigenvalues of the twelve-point correlator near the second vacancy:")
  @show order_parameter2
  println("")

  println("")
  println("Variance of the energy is $variance")
  println("")
  
  h5open("../data/2d_kitaev_AFM_h$(h).h5", "w") do file
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
    write(file, "plaquette", W_operator_eigenvalues)
    if x_direction_periodic
      write(file, "Wlx", loop_operator_x_eigenvalues)
    end
    write(file, "Wly", y_loop_eigenvalues)
    write(file, "OrderParameter", order_parameter)
    write(file, "OrderParameter2", order_parameter2)
  end

  return
end