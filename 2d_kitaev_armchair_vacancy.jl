## 01/05/2025
## Simulate the 2d Kitaev model on a honeycomb lattice 
## Introducing vacancies, magnetic field, string and plaquette operators


using ITensors
using ITensorMPS
using HDF5
using MKL
using LinearAlgebra
using TimerOutputs


include("src/kitaev_heisenberg/HoneycombLattice.jl")
include("src/kitaev_heisenberg/Entanglement.jl")
include("src/kitaev_heisenberg/TopologicalLoops.jl")
include("src/kitaev_heisenberg/CustomObserver.jl")


# Set up parameters for multithreading for BLAS/LAPACK and Block sparse multithreading
MKL_NUM_THREADS = 8
OPENBLAS_NUM_THREADS = 8
OMP_NUM_THREADS = 8

# Monitor the number of threads used by BLAS and LAPACK
@show BLAS.get_config()
@show BLAS.get_num_threads()


const Nx_unit = 15
const Ny_unit = 4
const Nx = 2 * Nx_unit
const Ny = Ny_unit
const N = Nx * Ny
# Timing and profiling
const time_machine = TimerOutput()


let
  # Set up the interaction parameters for the Hamiltonian
  # |Jx| <= |Jy| + |Jz| in the gapless A-phase
  # |Jx| > |Jy| + |Jz| in the gapped B-phase
  Jx, Jy, Jz = -1.0, -1.0, - 1.0
  alpha = 1E-4
  h = 0.02
  @show Jx, Jy, Jz, alpha, h

  
  # Set up the perturbation strength for the string and plaquette operators
  lambda_left  = 0.1
  lambda_right = 1.0 * lambda_left
  eta = abs(lambda_left)  # The strength of the plaquette perturbation
  @show lambda_left, lambda_right, eta  # Use a positive sign here in order to lower the eneergy, given that the plaquette operator is negative 
  

  # honeycomb lattice implemented in the ring ordering scheme
  x_periodic = false
  y_direction_twist = true

  
  # Construct the honeycomb lattice with armchair geometry if OBC is used in the x direction
  if x_periodic
    lattice = honeycomb_lattice_rings_pbc(Nx, Ny; yperiodic=true)
    @show length(lattice)
  else
    lattice = honeycomb_lattice_armchair(Nx, Ny; yperiodic=true)
    @show length(lattice)
  end 
  number_of_bonds = length(lattice)

  
  
  # Select the position(s) of the vacancies
  # sites_to_delete = Set{Int64}()
  sites_to_delete = Set{Int64}([58])  # The site number of the vacancy depends on the lattice width
  lattice_sites   = Set{Int64}()
  
  
  # Construct the Hamiltonian using OpSum
  os = OpSum()
  xbond = 0
  ybond = 0
  zbond = 0
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

    if abs(b.s1 - b.s2) == 1 || abs(b.s1 - b.s2) == Ny - 1
      os .+= -coefficient_Jz, "Sz", b.s1, "Sz", b.s2
      zbond += 1
      @show b.s1, b.s2, "Sz"
    else
      if mod(tmp_x, 2) == 1 
        if mod(b.s1, 2) == 1 && mod(b.s2, 2) == 1
          os .+= -coefficient_Jx, "Sx", b.s1, "Sx", b.s2
          xbond += 1
          @show b.s1, b.s2, "Sx"
        elseif mod(b.s1, 2) == 0 && mod(b.s2, 2) == 0
          os .+= -coefficient_Jy, "Sy", b.s1, "Sy", b.s2
          ybond += 1
          @show b.s1, b.s2, "Sy"
        end
      else
        if mod(b.s1, 2) == 0 && mod(b.s2, 2) == 0
          os .+= -coefficient_Jx, "Sx", b.s1, "Sx", b.s2
          xbond += 1
          @show b.s1, b.s2, "Sx"
        elseif mod(b.s1, 2) == 1 && mod(b.s2, 2) == 1
          os .+= -coefficient_Jy, "Sy", b.s1, "Sy", b.s2
          ybond += 1
          @show b.s1, b.s2, "Sy"
        end
      end
    end

    if !in(b.s1, lattice_sites)
      push!(lattice_sites, b.s1)
    end

    if !in(b.s2, lattice_sites)
      push!(lattice_sites, b.s2)
    end
  end
  @show xbond, ybond, zbond
  
  
  # Add the Zeeman coupling of the spins to a magnetic field applied in [111] direction
  # The magnetic field breaks integrability 
  # @show length(lattice_sites), lattice_sites
  if h > 1e-8
    for tmp_site in lattice_sites
      os .+= -1.0 * h, "Sx", tmp_site
      os .+= -1.0 * h, "Sy", tmp_site
      os .+= -1.0 * h, "Sz", tmp_site
    end
  end
  

  
  #*************************************************************************************************************************
  #*************************************************************************************************************************
  # Add loop perturbations to the Hamiltonian
  # Generate the indices for all loop operators along the cylinder
  loop_operator = Vector{String}(["iY", "X", "iY", "X", "iY", "X", "iY", "X"])  # Hard-coded for width-4 cylinders
  loop_indices = LoopListArmchair(Nx_unit, Ny_unit, "armchair", "y")  
  

  # Define the loop operators wrapping around the cylinder
  string_operator = Vector{String}([])
  for index in 1 : 2 * Ny
    if index % 2 == 1
      push!(string_operator, "iY")
    else
      push!(string_operator, "X")
    end
  end
  # @show string_operator, loop_indices 

  
  # Generate the indices for all string operators along the cylinder
  pinning_sites = loop_indices[1 : end .!= 8, :]
  @show size(pinning_sites), size(loop_indices)
  

  # Add perturbation to the left of the vacancy
  if abs(lambda_left) > 1E-8
    for index in 1 : div(size(pinning_sites, 1), 2)
      os .+= -1.0 * lambda_left, string_operator[1], pinning_sites[index, 1], 
        string_operator[2], pinning_sites[index, 2], string_operator[3], pinning_sites[index, 3], 
        string_operator[4], pinning_sites[index, 4], string_operator[5], pinning_sites[index, 5], 
        string_operator[6], pinning_sites[index, 6], string_operator[7], pinning_sites[index, 7],
        string_operator[8], pinning_sites[index, 8]

      mirror_index = index + div(size(pinning_sites, 1), 2)
      os .+= -1.0 * lambda_right, string_operator[1], pinning_sites[mirror_index, 1], 
        string_operator[2], pinning_sites[mirror_index, 2], string_operator[3], pinning_sites[mirror_index, 3], 
        string_operator[4], pinning_sites[mirror_index, 4], string_operator[5], pinning_sites[mirror_index, 5], 
        string_operator[6], pinning_sites[mirror_index, 6], string_operator[7], pinning_sites[mirror_index, 7],
        string_operator[8], pinning_sites[mirror_index, 8]

      # @show index, lambda_left, mirror_index, lambda_right
    end
  end
  #*************************************************************************************************************************
  #*************************************************************************************************************************


  # Generate the plaquette indices for all the plaquettes in the cylinder
  plaquette_operator = Vector{String}(["iY", "Z", "X", "X", "Z", "iY"])
  plaquette_indices = PlaquetteListArmchair(Nx_unit, Ny_unit, "armchair", false)
  # @show plaquette_indices

  # # Remove plaquette perturbations near the vacancy
  # println("The size of the orginal list of plaquettes:")
  # @show size(plaquette_indices)
  # println("")
  # # tmp_plaquette_indices = plaquette_indices[setdiff(1 : size(plaquette_indices, 1), range(19, 20, step = 1)), :]
  # tmp_plaquette_indices = plaquette_indices[setdiff(1 : size(plaquette_indices, 1), [19, 20, 22]), :]
  # println("The size of the truncated list of plaquettes:")
  # @show size(tmp_plaquette_indices)
  # println("")

  # if abs(eta) > 1E-8
  #   for index in 1 : size(tmp_plaquette_indices, 1)
  #     println("")
  #     @show tmp_plaquette_indices[index, :]
  #     println("")

  #     os .+= eta, plaquette_operator[1], tmp_plaquette_indices[index, 1], 
  #     plaquette_operator[2], tmp_plaquette_indices[index, 2], 
  #     plaquette_operator[3], tmp_plaquette_indices[index, 3], 
  #     plaquette_operator[4], tmp_plaquette_indices[index, 4], 
  #     plaquette_operator[5], tmp_plaquette_indices[index, 5], 
  #     plaquette_operator[6], tmp_plaquette_indices[index, 6]

  #     # Only remove three plaquette perturbations near the vacancy
  #     # if 44 ∉ plaquette_indices[index, :]
  #     #   # println("")
  #     #   # @show plaquette_indices[index, :]
  #     #   # println("")
  #     #   # @show size(os)

  #     #   os .+= eta, plaquette_operator[1], plaquette_indices[index, 1], 
  #     #   plaquette_operator[2], plaquette_indices[index, 2], 
  #     #   plaquette_operator[3], plaquette_indices[index, 3], 
  #     #   plaquette_operator[4], plaquette_indices[index, 4], 
  #     #   plaquette_operator[5], plaquette_indices[index, 5], 
  #     #   plaquette_operator[6], plaquette_indices[index, 6]
  #     # end 
  #   end
  # end


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
  cutoff  = [1E-10]
  eigsolve_krylovdim = 50
  
  # Add noise terms to prevent DMRG from getting stuck in a local minimum
  # noise = [1E-6, 1E-7, 1E-8, 0.0]

  # Measure the initial local observables (one-point functions)
  Sx₀ = expect(ψ₀, "Sx", sites = 1 : N)
  Sy₀ = expect(ψ₀, "iSy", sites = 1 : N)
  Sz₀ = expect(ψ₀, "Sz", sites = 1 : N)
  
  
  # Construct a custom observer and stop the DMRG calculation early if needed
  
  # custom_observer = DMRGObserver(; energy_tol=1E-9, minsweeps=2, energy_type=Float64)
  custom_observer = CustomObserver()
  @show custom_observer.etolerance
  @show custom_observer.minsweeps
  @timeit time_machine "dmrg simulation" begin
    energy, ψ = dmrg(H, ψ₀; nsweeps, maxdim, cutoff, eigsolve_krylovdim, observer = custom_observer)
  end
  
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


  # Compute the eigenvalues of plaquette operators
  # normalize!(ψ)
  @timeit time_machine "plaquette operators" begin
    W_operator_eigenvalues = zeros(Float64, size(plaquette_indices, 1))
    
    # Compute the eigenvalues of the plaquette operator
    for index in 1 : size(plaquette_indices, 1)
      @show plaquette_indices[index, :]
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
  @show W_operator_eigenvalues
  
  # Compute the eigenvalues of the loop operators 
  # The loop operators depend on the width of the cylinder  
  @timeit time_machine "loop operators" begin
    yloop_eigenvalues = zeros(Float64, size(loop_indices)[1])
    
    # Compute eigenvalues of the loop operators in the direction with PBC.
    for index in 1 : size(loop_indices)[1]
      ## Construct loop operators along the y direction with PBC
      os_wl = OpSum()
      os_wl += loop_operator[1], loop_indices[index, 1], 
        loop_operator[2], loop_indices[index, 2], 
        loop_operator[3], loop_indices[index, 3], 
        loop_operator[4], loop_indices[index, 4], 
        loop_operator[5], loop_indices[index, 5], 
        loop_operator[6], loop_indices[index, 6],
        loop_operator[7], loop_indices[index, 7],
        loop_operator[8], loop_indices[index, 8]

      Wl = MPO(os_wl, sites)
      yloop_eigenvalues[index] = real(inner(ψ', Wl, ψ))
    end
  end



  # # Compute the eigenvalues of the order parameters near vacancies
  # # TO-DO: The order parameter (twelve-point correlator) loop is hard-coded for now
  # # need to genealize in the future to automatically generate the loop indices near vacancies

  # @timeit time_machine "twelve-point correlator(s)" begin
  #   order_loop = Vector{String}(["Z", "Y", "Y", "Y", "X", "Z", "Z", "Z", "Y", "X", "X", "X"])
  #   order_indices = Matrix{Int64}(undef, 1, 12)
  #   # Complete the loop indices near vacancies
  #   # order_indices[1, :] = [52, 49, 46, 43, 40, 38, 41, 39, 42, 45, 47, 50]      # On the width-3 cylinders  
  #   order_indices[1, :] = [70, 66, 62, 58, 54, 51, 55, 52, 56, 60, 63, 67]      # On the width-4 cylinders
  #   order_parameter = Vector{Float64}(undef, size(order_indices)[1])

    
  #   @show size(order_indices)[1]
  #   for index in 1 : size(order_indices)[1]
  #     os_parameter = OpSum()
  #     os_parameter += order_loop[1], order_indices[index, 1], 
  #       order_loop[2], order_indices[index, 2], 
  #       order_loop[3], order_indices[index, 3], 
  #       order_loop[4], order_indices[index, 4], 
  #       order_loop[5], order_indices[index, 5], 
  #       order_loop[6], order_indices[index, 6],
  #       order_loop[7], order_indices[index, 7],
  #       order_loop[8], order_indices[index, 8],
  #       order_loop[9], order_indices[index, 9],
  #       order_loop[10], order_indices[index, 10],
  #       order_loop[11], order_indices[index, 11],
  #       order_loop[12], order_indices[index, 12]
  #     W_parameter = MPO(os_parameter, sites)
  #     order_parameter[index] = real(inner(ψ', W_parameter, ψ))
  #   end
  # end

  
  # Print out useful information of physical quantities
  println("")
  println("Visualize the optimization history of the energy and bond dimensions:")
  @show custom_observer.ehistory_full
  @show custom_observer.ehistory
  @show custom_observer.chi
  # @show number_of_bonds, energy / number_of_bonds
  # @show N, energy / N
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
  

  println("")
  println("Eigenvalues of the plaquette operator:")
  @show W_operator_eigenvalues
  println("")


  print("")
  println("Eigenvalues of the loop operator(s):")
  @show yloop_eigenvalues
  println("")

  # println("")
  # println("Eigenvalues of the twelve-point correlator near the first vacancy:")
  # @show order_parameter
  # println("")

  @show time_machine
  
  h5open("data/test/armchair_geometery/2d_kitaev_honeycomb_armchair_FM_Lx$(Nx_unit)_h$(h).h5", "w") do file
    write(file, "psi", ψ)
    write(file, "NormalizedE0", energy / number_of_bonds)
    write(file, "E0", energy)
    write(file, "E0variance", variance)
    write(file, "Ehist", custom_observer.ehistory)
    write(file, "Bond", custom_observer.chi)
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
    write(file, "Loop", yloop_eigenvalues)
    # write(file, "OrderParameter", order_parameter)
  end

  return
end