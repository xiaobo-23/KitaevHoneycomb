# Simulate the 2d Kitaev model on a honeycomb lattice 
# Introducing vacancies, magnetic field, and string operators
using HDF5
using ITensors
using MKL
using TimerOutputs
using LinearAlgebra
import ITensors: energies
include("src/kitaev_heisenberg/HoneycombLattice.jl")
include("src/kitaev_heisenberg/Entanglement.jl")
include("src/kitaev_heisenberg/TopologicalLoops.jl")
include("src/kitaev_heisenberg/CustomObserver.jl")


# Set up parameters for multithreading for BLAS/LAPACK and Block sparse multithreading
MKL_NUM_THREADS = 8
OPENBLAS_NUM_THREADS = 8
OMP_NUM_THREADS = 8

# Timing and profiling
const time_machine = TimerOutput()

let
  # Monitor the number of threads used by BLAS and LAPACK
  @show BLAS.get_config()
  @show BLAS.get_num_threads()

  # Set up the parameters for the lattice
  # Number of unit cells in x and y directions
  Nx_unit_cell = 16
  Ny_unit_cell = 3
  Nx = 2 * Nx_unit_cell
  Ny = Ny_unit_cell
  N = Nx * Ny

  # Set up the interaction parameters for the Hamiltonian
  # |Jx| <= |Jy| + |Jz| in the gapless A-phase
  # |Jx| > |Jy| + |Jz| in the gapped B-phase
  Jx = Jy = Jz = 1.0
  alpha = 1.0
  h=0.0
  @show Jx, Jy, Jz, alpha, h

  # Set up the perturbation strength for loop operators
  lambda_left  = 0.05
  lambda_right = 1.0 * lambda_left

  # The strength of the plaquette perturbation
  # Use a positive sign here to lower the energy, given the plaquette operator is negative
  eta = abs(lambda_left) 
  @show lambda_left, lambda_right, eta


  # honeycomb lattice implemented in the ring ordering scheme
  x_direction_periodic = false
  y_direction_twist = true

  if x_direction_periodic
    lattice = honeycomb_lattice_rings_pbc(Nx, Ny; yperiodic=true)
    @show length(lattice)
    # @show lattice
  else
    if y_direction_twist
      lattice = honeycomb_lattice_rings_right_twist(Nx, Ny; yperiodic=true)
      @show length(lattice)
      # @show lattice
    else
      lattice = honeycomb_lattice_rings(Nx, Ny; yperiodic=true)
      @show length(lattice)
      # @show lattice
    end
  end
  number_of_bonds = length(lattice)
  # @show number_of_bonds

  # honeycomb lattice implemented in the C-style ordering scheme
  # lattice = honeycomb_lattice_Cstyle(Nx, Ny; yperiodic=true)
  

  # Select the position(s) of the vacancies
  sites_to_delete = Set{Int64}([44])
  lattice_sites   = Set{Int64}()
  
  # Add pinning fields to the lattice in a symmetric format
  pinning_seeds = collect(1 : Nx_unit_cell)
  deleteat!(pinning_seeds, 16)
  deleteat!(pinning_seeds, 8) 
  @show pinning_seeds

  
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
  # @show enumerate_bonds

  
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
  
  # Add the string operators as perturbations into the cylinder
  string_operators = Vector{String}([])
  
  for index in 1 : 2
    if y_direction_twist
      push!(string_operators, "X")
    else
      push!(string_operators, "Z")
    end 
  end

  for index in 3 : 2 * Ny
    push!(string_operators, "Z")
  end
  @show string_operators


  # Add the index of the pinning sites into a Matrix
  pinning_sites = Matrix{Int64}(undef, length(pinning_seeds), 2 * Ny)
  for index in eachindex(pinning_seeds)
    pinning_sites[index, 1] = pinning_seeds[index] * 2 * Ny + 1
    pinning_sites[index, 2] = (2 * pinning_seeds[index] - 1) * Ny + 1
    pinning_sites[index, 3] = pinning_sites[index, 2] - 2
    pinning_sites[index, 4] = pinning_sites[index, 2] + 1
    pinning_sites[index, 5] = pinning_sites[index, 2] - 1
    pinning_sites[index, 6] = pinning_sites[index, 2] + 2

    # for index2 in 1 : 2 * Ny
    #   pinning_sites[index1, index2] = 2 * (pinning_seeds[index1] - 1) * Ny + index2 + 1
    # end
    # println("")
    # @show pinning_sites[index, :]
    # println("")
  end
  
  # Add perturbation to the left of the vacancy
  if lambda_left > 1E-8
    for index in 1 : Int(size(pinning_sites, 1) / 2)
      @show index, lambda_left
      os .+= -1.0 * lambda_left, string_operators[1], pinning_sites[index, 1], 
        string_operators[2], pinning_sites[index, 2], string_operators[3], pinning_sites[index, 3], 
        string_operators[4], pinning_sites[index, 4], string_operators[5], pinning_sites[index, 5], 
        string_operators[6], pinning_sites[index, 6]
    end
  end


  # Add perturbation to the right of the vacancy
  if lambda_right > 1E-8
    for index in Int(size(pinning_sites, 1) / 2) + 1 : size(pinning_sites, 1)
      @show index, lambda_right
      os .+= -1.0 * lambda_right, string_operators[1], pinning_sites[index, 1], 
        string_operators[2], pinning_sites[index, 2], string_operators[3], pinning_sites[index, 3], 
        string_operators[4], pinning_sites[index, 4], string_operators[5], pinning_sites[index, 5], 
        string_operators[6], pinning_sites[index, 6]
    end
  end 

  
  # Add the plaquette perturbation to the cylinder
  plaquette_operator = Vector{String}(["Z", "iY", "X", "X", "iY", "Z"])
  if y_direction_twist
    plaquette_indices = PlaquetteList_RightTiwst(Nx_unit_cell, Ny_unit_cell, "rings", false)
  else
    plaquette_indices = PlaquetteList(Nx_unit_cell, Ny_unit_cell, "rings", false)
  end


  # Remove plaquette perturbations near the vacancy
  println("The size of the orginal list of plaquettes:")
  @show size(plaquette_indices)
  println("")
  tmp_plaquette_indices = plaquette_indices[setdiff(1 : size(plaquette_indices, 1), range(13, 30, step = 1)), :]
  println("The size of the truncated list of plaquettes:")
  @show size(tmp_plaquette_indices)
  println("")

  if eta > 1E-8
    for index in 1 : size(tmp_plaquette_indices, 1)
      println("")
      @show tmp_plaquette_indices[index, :]
      println("")

      os .+= eta, plaquette_operator[1], tmp_plaquette_indices[index, 1], 
      plaquette_operator[2], tmp_plaquette_indices[index, 2], 
      plaquette_operator[3], tmp_plaquette_indices[index, 3], 
      plaquette_operator[4], tmp_plaquette_indices[index, 4], 
      plaquette_operator[5], tmp_plaquette_indices[index, 5], 
      plaquette_operator[6], tmp_plaquette_indices[index, 6]

      # Only remove three plaquette perturbations near the vacancy
      # if 44 ∉ plaquette_indices[index, :]
      #   # println("")
      #   # @show plaquette_indices[index, :]
      #   # println("")
      #   # @show size(os)

      #   os .+= eta, plaquette_operator[1], plaquette_indices[index, 1], 
      #   plaquette_operator[2], plaquette_indices[index, 2], 
      #   plaquette_operator[3], plaquette_indices[index, 3], 
      #   plaquette_operator[4], plaquette_indices[index, 4], 
      #   plaquette_operator[5], plaquette_indices[index, 5], 
      #   plaquette_operator[6], plaquette_indices[index, 6]
      # end 
    end
  end

  
  # Increase the maximum dimension of Krylov space used to locally solve the eigenvalues problem.
  sites = siteinds("S=1/2", N; conserve_qns=false)
  H = MPO(os, sites)


  # Initialize wavefunction to a random MPS of bond-dimension 20 with same quantum 
  # numbers as `state`
  state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
  ψ₀ = randomMPS(sites, state, 20)

  
  # Set up the parameters including bond dimensions and truncation error
  nsweeps = 1
  maxdim  = [20, 60, 100, 500, 800, 1000, 1500, 3000]
  cutoff  = [1E-8]
  eigsolve_krylovdim = 50
  
  # Add noise terms to prevent DMRG from getting stuck in a local minimum
  # noise = [1E-6, 1E-7, 1E-8, 0.0]

  # Measure the initial local observables (one-point functions)
  Sx₀ = expect(ψ₀, "Sx", sites = 1 : N)
  Sy₀ = expect(ψ₀, "iSy", sites = 1 : N)
  Sz₀ = expect(ψ₀, "Sz", sites = 1 : N)
  
  
  # Construct a custom observer and stop the DMRG calculation early if needed
  custom_observer = CustomObserver()
  @show custom_observer.etolerance
  @show custom_observer.minsweeps
  @timeit time_machine "dmrg simulation" begin
    energy, ψ = dmrg(H, ψ₀; nsweeps, maxdim, cutoff, eigsolve_krylovdim, observer = custom_observer)
  end

  #***************************************************************************************************************
  # # Construct a DMRGobserver to measure local observables and stop the calculation early if needed
  # Sz_observer = DMRGObserver(["Sz"], sites, minsweeps=2, energy_tol = 1E-7)
  
  # @timeit time_machine "dmrg simulation" begin
  #   energy, ψ = dmrg(H, ψ₀; nsweeps, maxdim, cutoff, eigsolve_krylovdim, observer = Sz_observer) 
  # end

  # for (sweep, Szs) in enumerate(measurements(Sz_observer)["Sz"])
  #   println("Total Sz after sweep $sweep= ", sum(Szs) / (2 * N))
  # end
  # @show energies(Sz_observer)
  # @show maxlinkdim(ψ)
  #***************************************************************************************************************
  
  
  # Measure local observables (one-point functions) after finish the DMRG simulation
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
  # The plaquette operators are always six-point correlators
  # normalize!(ψ)
  @timeit time_machine "plaquette operators" begin
    W_operator_eigenvalues = Vector{Float64}(undef, size(plaquette_indices, 1))
    
    # Compute the eigenvalues of the plaquette operator
    for index in 1 : size(plaquette_indices, 1)
      os_w = OpSum()
      os_w += plaquette_operator[1], plaquette_indices[index, 1], 
        plaquette_operator[2], plaquette_indices[index, 2], 
        plaquette_operator[3], plaquette_indices[index, 3], 
        plaquette_operator[4], plaquette_indices[index, 4], 
        plaquette_operator[5], plaquette_indices[index, 5], 
        plaquette_operator[6], plaquette_indices[index, 6]
      W = MPO(os_w, sites)
      W_operator_eigenvalues[index] = -1.0 * real(inner(ψ', W, ψ))
      # @show inner(ψ', W, ψ) / inner(ψ', ψ)
    end
  end

  
  # Compute the eigenvalues of the loop operators 
  # The loop operators depend on the width of the cylinder  
  @timeit time_machine "loop operators" begin
    # Construct the loop indices along the y direction with/without y_direction_twist
    if y_direction_twist
      yloop_indices = LoopList_RightTwist(Nx_unit_cell, Ny_unit_cell, "rings", "y"); @show yloop_indices
    else
      yloop_indices = LoopList(Nx_unit_cell, Ny_unit_cell, "rings", "y"); @show yloop_indices
    end
    yloop_eigenvalues = Vector{Float64}(undef, size(yloop_indices)[1])
    
    # Compute eigenvalues of the loop operators in the direction with PBC.
    for loop_index in 1 : size(yloop_indices)[1]
      @show yloop_indices[loop_index, :]
      
      os_wl = OpSum()
      # Construct the loop operator(s) along the y direction for three-leg cylinder
      os_wl += string_operators[1], yloop_indices[loop_index, 1], 
        string_operators[2], yloop_indices[loop_index, 2], 
        string_operators[3], yloop_indices[loop_index, 3], 
        string_operators[4], yloop_indices[loop_index, 4], 
        string_operators[5], yloop_indices[loop_index, 5], 
        string_operators[6], yloop_indices[loop_index, 6]

      # # Construct the loop operator(s) along the y direction for four-leg cylinder
      # os_wl += string_operators[1], yloop_indices[loop_index, 1], 
      #   string_operators[2], yloop_indices[loop_index, 2], 
      #   string_operators[3], yloop_indices[loop_index, 3], 
      #   string_operators[4], yloop_indices[loop_index, 4], 
      #   string_operators[5], yloop_indices[loop_index, 5], 
      #   string_operators[6], yloop_indices[loop_index, 6],
      #   string_operators[7], yloop_indices[loop_index, 7],
      #   string_operators[8], yloop_indices[loop_index, 8]

      
      Wl = MPO(os_wl, sites)
      yloop_eigenvalues[loop_index] = real(inner(ψ', Wl, ψ))
    end
  end


  # Compute the eigenvalues of the order parameters near vacancies
  # TO-DO: The order parameter (twelve-point correlator) loop is hard-coded for now
  # need to genealize in the future to automatically generate the loop indices near vacancies

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
  println("")
  println("Visualize the optimization history of the energy and bond dimensions:")
  @show custom_observer.ehistory_full
  @show custom_observer.ehistory
  @show custom_observer.chi
  # @show number_of_bonds, energy / number_of_bonds
  # @show N, energy / N
  

  println("")
  println("Eigenvalues of the plaquette operator:")
  @show W_operator_eigenvalues
  println("")


  print("")
  println("Eigenvalues of the loop operator(s):")
  @show yloop_eigenvalues
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
  
  # h5open("./data/test/BC/2d_kitaev_FM_L$(Nx)W$(Ny)_epsilon1E-8.h5", "w") do file
  #   write(file, "psi", ψ)
  #   write(file, "NormalizedE0", energy / number_of_bonds)
  #   write(file, "E0", energy)
  #   write(file, "E0variance", variance)
  #   write(file, "Ehist", custom_observer.ehistory)
  #   write(file, "Bond", custom_observer.chi)
  #   # write(file, "Entropy", SvN)
  #   write(file, "Sx0", Sx₀)
  #   write(file, "Sx",  Sx)
  #   # write(file, "Cxx", xxcorr)
  #   write(file, "Sy0", Sy₀)
  #   write(file, "Sy", Sy)
  #   # write(file, "Cyy", yycorr)
  #   write(file, "Sz0", Sz₀)
  #   write(file, "Sz",  Sz)
  #   # write(file, "Czz", zzcorr)
  #   write(file, "Plaquette", W_operator_eigenvalues)
  #   write(file, "Loop", yloop_eigenvalues)
  #   write(file, "OrderParameter", order_parameter)
  # end

  return
end