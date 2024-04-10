# Simulate the 2d Kitaev model on a honeycomb lattice 
# Introducing vacancies, magnetic field, and string operators
using HDF5
using ITensors
using MKL
using TimerOutputs
using LinearAlgebra


include("../HoneycombLattice.jl")
include("../Entanglement.jl")
include("../TopologicalLoops.jl")


# Set up parameters for multithreading for BLAS/LAPACK and Block sparse multithreading
MKL_NUM_THREADS = 8
OPENBLAS_NUM_THREADS = 8
OMP_NUM_THREADS = 8

# Timing and profiling
const time_machine = TimerOutput()

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


let
  # Monitor the number of threads used by BLAS and LAPACK
  @show BLAS.get_config()
  @show BLAS.get_num_threads()


  # Set up the parameters for the lattice
  # Number of unit cells in x and y directions
  Nx_unit_cell = 15
  Ny_unit_cell = 3
  Nx = 2 * Nx_unit_cell
  Ny = Ny_unit_cell
  N = Nx * Ny

  # Set up the interaction parameters for the Hamiltonian
  # |Jx| <= |Jy| + |Jz| in the gapless A-phase
  # |Jx| > |Jy| + |Jz| in the gapped B-phase
  Jx = Jy = Jz = 1.0
  alpha = 0.001
  h=0.0
  lambda_left  = -0.1
  lambda_right = 1.0 * lambda_left
  @show Jx, Jy, Jz, alpha, lambda_left, lambda_right, h

  
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
  # @show lattice
  
  
  # Select the position(s) of the vacancies
  sites_to_delete = Set{Int64}([44])
  lattice_sites   = Set{Int64}()
  pinning_ptr = collect(1 : Nx_unit_cell)
  deleteat!(pinning_ptr, 8)
  @show pinning_ptr

  
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
  # @show length(lattice_sites)
  # @show lattice_sites
  if h > 1e-8
    for tmp_site in lattice_sites
      os .+= -1.0 * h, "Sx", tmp_site
      os .+= -1.0 * h, "Sy", tmp_site
      os .+= -1.0 * h, "Sz", tmp_site
    end
  end
  # @show os
  
  # Add the string operators as perturbations in the left and right edges of the cylinder
  string_operators = Vector{String}([])
  for index in 1 : 2 * Ny
    push!(string_operators, "Z")
  end
  @show string_operators

  
  # Add the index of the pinning sites into a Matrix
  pinning_sites = Matrix{Int64}(undef, length(pinning_ptr), 2 * Ny)
  for index1 in 1 : size(pinning_sites, 1)
    for index2 in 1 : 2 * Ny
      pinning_sites[index1, index2] = 2 * (pinning_ptr[index1] - 1) * Ny + index2
    end
    @show pinning_sites[index1, :]
  end
  
  
  for index in 1 : Int(size(pinning_sites, 1) / 2)
    @show index
    os .+= -1.0 * lambda_left, string_operators[1], pinning_sites[index, 1], 
      string_operators[2], pinning_sites[index, 2], string_operators[3], pinning_sites[index, 3], 
      string_operators[4], pinning_sites[index, 4], string_operators[5], pinning_sites[index, 5], 
      string_operators[6], pinning_sites[index, 6]
  end
  
  
  for index in Int(size(pinning_sites, 1) / 2) + 1 : size(pinning_sites, 1)
    @show index
    os .+= -1.0 * lambda_right, string_operators[1], pinning_sites[index, 1], 
      string_operators[2], pinning_sites[index, 2], string_operators[3], pinning_sites[index, 3], 
      string_operators[4], pinning_sites[index, 4], string_operators[5], pinning_sites[index, 5], 
      string_operators[6], pinning_sites[index, 6]
  end


  # Increase the maximum dimension of Krylov space used to locally solve the eigenvalues problem.
  sites = siteinds("S=1/2", N; conserve_qns=false)
  H = MPO(os, sites)


  # Initialize wavefunction to a random MPS of bond-dimension 20 with same quantum 
  # numbers as `state`
  state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
  ψ₀ = randomMPS(sites, state, 20)


  # Set up the parameters including bond dimensions and truncation error
  nsweeps = 35
  maxdim  = [20, 60, 100, 500, 800, 1000, 1500, 3000]
  cutoff  = [1E-8]
  eigsolve_krylovdim = 50
  # Add noise terms to prevent DMRG from getting stuck in a local minimum
  # noise = [1E-6, 1E-7, 1E-8, 0.0]

  
  # Run DMRG and measure the energy, one-point functions, and two-point functions
  tmp_observer = energyObserver()
  Sx₀ = expect(ψ₀, "Sx", sites = 1 : N)
  Sy₀ = expect(ψ₀, "iSy", sites = 1 : N)
  Sz₀ = expect(ψ₀, "Sz", sites = 1 : N)
  
  @timeit time_machine "dmrg simulation" begin
    energy, ψ = dmrg(H, ψ₀; nsweeps, maxdim, cutoff, eigsolve_krylovdim, observer = tmp_observer) 
  end

  
  @timeit time_machine "one-point functions" begin
    Sx = expect(ψ, "Sx", sites = 1 : N)
    Sy = expect(ψ, "iSy", sites = 1 : N)
    Sz = expect(ψ, "Sz", sites = 1 : N)
  end

  # @timeit time_machine to "two-point functions" begin
  #   xxcorr = correlation_matrix(ψ, "Sx", "Sx", sites = 1 : N)
  #   yycorr = correlation_matrix(ψ, "Sy", "Sy", sites = 1 : N)
  #   zzcorr = correlation_matrix(ψ, "Sz", "Sz", sites = 1 : N)
  # end


  # Compute the eigenvalues of all plaquette operators
  # normalize!(ψ)
  @timeit time_machine "plaquette operators" begin
    plaquette_operator = Vector{String}(["Z", "iY", "X", "X", "iY", "Z"])
    loop_inds = PlaquetteList(Nx_unit_cell, Ny_unit_cell, "rings", false)
    for index in 1 : size(loop_inds)[1]
      @show loop_inds[index, :]
    end
    W_operator_eigenvalues = Vector{Float64}(undef, size(loop_inds)[1])
    
    
    # Compute the eigenvalues of the plaquette operator
    for loop_index in 1 : size(loop_inds)[1]
      os_w = OpSum()
      os_w += plaquette_operator[1], loop_inds[loop_index, 1], 
        plaquette_operator[2], loop_inds[loop_index, 2], 
        plaquette_operator[3], loop_inds[loop_index, 3], 
        plaquette_operator[4], loop_inds[loop_index, 4], 
        plaquette_operator[5], loop_inds[loop_index, 5], 
        plaquette_operator[6], loop_inds[loop_index, 6]
      W = MPO(os_w, sites)
      W_operator_eigenvalues[loop_index] = -1.0 * real(inner(ψ', W, ψ))
      # @show inner(ψ', W, ψ) / inner(ψ', ψ)
    end
  end

  
  @timeit time_machine "loop operators" begin
    # Construct the loop indices in the direction with PBC
    y_inds = LoopList(Nx_unit_cell, Ny_unit_cell, "rings", "y")
    y_loop_eigenvalues = Vector{Float64}(undef, size(y_inds)[1])

    
    # Compute eigenvalues of the loop operators in the direction with PBC.
    for loop_index in 1 : size(y_inds)[1]
      @show y_inds[loop_index, :]
      os_wl = OpSum()
      os_wl += string_operators[1], y_inds[loop_index, 1], 
        string_operators[2], y_inds[loop_index, 2], 
        string_operators[3], y_inds[loop_index, 3], 
        string_operators[4], y_inds[loop_index, 4], 
        string_operators[5], y_inds[loop_index, 5], 
        string_operators[6], y_inds[loop_index, 6]
      Wl = MPO(os_wl, sites)
      y_loop_eigenvalues[loop_index] = real(inner(ψ', Wl, ψ))
    end
  end


  # Compute the eigenvalues of the order parameters near vacancies
  @timeit time_machine "twelve-point correlator(s)" begin
    order_loop = Vector{String}(["Z", "Y", "Y", "Y", "X", "Z", "Z", "Z", "Y", "X", "X", "X"])
    order_indices = Matrix{Int64}(undef, 1, 12)
    # Complete the loop indices near vacancies
    order_indices[1, :] = [52, 49, 46, 43, 40, 38, 41, 39, 42, 45, 47, 50]
    order_parameter = Vector{Float64}(undef, size(order_indices)[1])

    
    @show size(order_indices)[1]
    for index in 1 : size(order_indices)[1]
      os_parameter = OpSum()
      os_parameter += order_loop[1], order_indices[index, 1], 
        order_loop[2], order_indices[index, 2], 
        order_loop[3], order_indices[index, 3], 
        order_loop[4], order_indices[index, 4], 
        order_loop[5], order_indices[index, 5], 
        order_loop[6], order_indices[index, 6],
        order_loop[7], order_indices[index, 7],
        order_loop[8], order_indices[index, 8],
        order_loop[9], order_indices[index, 9],
        order_loop[10], order_indices[index, 10],
        order_loop[11], order_indices[index, 11],
        order_loop[12], order_indices[index, 12]
      W_parameter = MPO(os_parameter, sites)
      order_parameter[index] = real(inner(ψ', W_parameter, ψ))
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
  println("Eigenvalues of the twelve-point correlator near the first vacancy:")
  @show order_parameter
  println("")


  # Check the variance of the energy
  @timeit time_machine "compaute the variance" begin
    H2 = inner(H, ψ, H, ψ)
    E₀ = inner(ψ', H, ψ)
    variance = H2 - E₀^2
  end
 
  println("")
  @show E₀
  println("Variance of the energy is $variance")
  println("")


  @show time_machine
  h5open("../data/2d_kitaev_honeycomb_h$(h).h5", "w") do file
    write(file, "psi", ψ)
    write(file, "NormalizedE0", energy / number_of_bonds)
    write(file, "E0", energy)
    write(file, "E0variance", variance)
    write(file, "Ehist", tmp_observer.ehistory)
    # write(file, "Entropy", SvN)
    write(file, "Sx0", Sx₀)
    write(file, "Sx",  Sx)
    # write(file, "Cxx", xxcorr)
    write(file, "Sy0", Sy₀)
    write(file, "Sy", Sy)
    # write(file, "Cyy", yycorr)
    write(file, "Sz0", Sz₀)
    write(file, "Sz",  Sz)
    # write(file, "Czz", zzcorr)
    write(file, "Plaquette", W_operator_eigenvalues)
    write(file, "Loop", y_loop_eigenvalues)
    write(file, "OrderParameter", order_parameter)
  end

  return
end