# 04/24/2025
# Simulate the 2D tJ-Kitaev honeycomb model to design topologucal qubits based on quantum spin liquids (QSLs)
# Introduce three-spin interaction, electron hopping, and Kitaev interaction; remove the spin vacancy


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


const Nx_unit = 4
const Ny_unit = 4
const Nx = 2 * Nx_unit
const Ny = Ny_unit
const N = Nx * Ny
# Timing and profiling
const time_machine = TimerOutput()


let
  # Set up the parameters in the Hamiltonian
  Jx, Jy, Jz = -1.0, -1.0, -1.0               # The Kitaev interaction 
  κ = -1.0                       # The three-spin interaction strength       
  t = 1.0                                      # The hopping amplitude 
  h = 0
  alpha = 1E-4
  @show Jx, Jy, Jz, alpha, κ, t, h

  
  # Set up the perturbation strength for the string and plaquette operators
  lambda_left  = 0.1
  lambda_right = 1.0 * lambda_left
  eta = abs(lambda_left)  # The strength of the plaquette perturbation
  @show lambda_left, lambda_right, eta  # Use a positive sign here in order to lower the eneergy, given that the plaquette operator is negative 
  

  # Bondary conditions and the mapping scheme
  x_periodic = false
  y_direction_twist = true


  # Construct a honeycomb lattice using armchair geometry
  # TO-DO: Implement the armchair geometery with periodic boundary condition
  if x_periodic
    lattice = honeycomb_lattice_rings_pbc(Nx, Ny; yperiodic=true)
    @show length(lattice)
  else
    lattice = honeycomb_lattice_armchair(Nx, Ny; yperiodic=true)
    @show length(lattice)
  end 
  number_of_bonds = length(lattice)

  
  # Construct the wedges in order to set up three-body spin interactions
  wedge = honeycomb_armchair_wedge(Nx, Ny; yperiodic=true)
  @show length(wedge), wedge 

  
  # Select the position(s) of the vacancies
  # sites_to_delete = Set{Int64}([59])            # The site number of the vacancy depends on the lattice width
  # sites_to_delete = Set{Int64}()
  lattice_sites = Set{Int64}(1 : N)                  # The set of all sites in the lattice
  @show lattice_sites

  
  #***************************************************************************************************************
  #***************************************************************************************************************  
  # Construct the Kitaev interaction and the hopping terms
  os = OpSum()
  xbond = 0
  ybond = 0
  zbond = 0
  for b in lattice
    # # Set up the hopping terms for spin-up and spin-down electrons
    # os .+= -t, "Cdagup", b.s1, "Cup", b.s2
    # os .+= -t, "Cdagup", b.s2, "Cup", b.s1
    # os .+= -t, "Cdagdn", b.s1, "Cdn", b.s2
    # os .+= -t, "Cdagdn", b.s2, "Cdn", b.s1

    # Set up the anisotropic Kitaev interaction
    tmp_x = div(b.s1 - 1, Ny) + 1
    if abs(b.s1 - b.s2) == 1 || abs(b.s1 - b.s2) == Ny - 1
      os .+= -Jz, "Sz", b.s1, "Sz", b.s2
      zbond += 1
      @show b.s1, b.s2, "Sz"
    else
      if mod(tmp_x, 2) == 1 
        if mod(b.s1, 2) == 1 && mod(b.s2, 2) == 1
          os .+= -Jx, "Sx", b.s1, "Sx", b.s2
          xbond += 1
          @show b.s1, b.s2, "Sx"
        elseif mod(b.s1, 2) == 0 && mod(b.s2, 2) == 0
          # Set up the Sy * Sy interaction using two different ways
          # os .+= -Jy, "Sy", b.s1, "Sy", b.s2
          
          os .+= -0.25 * Jy, "S+", b.s1, "S-", b.s2
          os .+= -0.25 * Jy, "S-", b.s1, "S+", b.s2
          os .+=  0.25 * Jy, "S+", b.s1, "S+", b.s2
          os .+=  0.25 * Jy, "S-", b.s1, "S-", b.s2
          
          ybond += 1
          @show b.s1, b.s2, "Sy"
        end
      else
        if mod(b.s1, 2) == 0 && mod(b.s2, 2) == 0
          os .+= -Jx, "Sx", b.s1, "Sx", b.s2
          xbond += 1
          @show b.s1, b.s2, "Sx"
        elseif mod(b.s1, 2) == 1 && mod(b.s2, 2) == 1
          # Set up the Sy * Sy interaction using two different ways
          # os .+= -Jy, "Sy", b.s1, "Sy", b.s2
          
          os .+= -0.25 * Jy, "S+", b.s1, "S-", b.s2
          os .+= -0.25 * Jy, "S-", b.s1, "S+", b.s2
          os .+=  0.25 * Jy, "S+", b.s1, "S+", b.s2
          os .+=  0.25 * Jy, "S-", b.s1, "S-", b.s2

          ybond += 1
          @show b.s1, b.s2, "Sy"
        end
      end
    end
  end
  
  @show xbond, ybond, zbond
  @show lattice_sites
  
  
  # Add the Zeeman coupling of the spins to a magnetic field applied in [111] direction, which breaks the integrability
  if h > 1e-8
    for site in lattice_sites
      os .+= -h, "Sx", site
      os .+= -0.5h, "iS-", site
      os .+=  0.5h, "iS+", site
      os .+= -h, "Sz", site
    end
  end
  

  # Implement the three-spin interaction terms in the Hamiltonian
  horizontal_wedge = 0
  vertical_wedge = 0
  for w in wedge
    @show w.s1, w.s2, w.s3
    x_coordinate = div(w.s2 - 1, Ny) + 1
    y_coordinate = mod(w.s2 - 1, Ny) + 1

    if abs(w.s1 - w.s2) == abs(w.s2 - w.s3)
      same_parity = (isodd(x_coordinate) && isodd(y_coordinate)) || (iseven(x_coordinate) && iseven(y_coordinate))

      if same_parity
        # Three-spin interaction: Sy(w.s1) * Sz(w.s2) * Sx(w.s3)
        os .+=  0.5im * κ, "S-", w.s1, "Sz", w.s2, "Sx", w.s3
        os .+= -0.5im * κ, "S+", w.s1, "Sz", w.s2, "Sx", w.s3
        @info "Added three-spin term" term = ("Sy", w.s1, "Sz", w.s2, "Sx", w.s3)
      else
        # Three-spin interaction: Sx(w.s1) * Sz(w.s2) * Sy(w.s3)
        os .+=  0.5im * κ, "Sx", w.s1, "Sz", w.s2, "S-", w.s3
        os .+= -0.5im * κ, "Sx", w.s1, "Sz", w.s2, "S+", w.s3
        @info "Added three-spin term" term = ("Sx", w.s1, "Sz", w.s2, "Sy", w.s3)
      end
      horizontal_wedge += 1
    end

    if abs(w.s1 - w.s2) == 3
      if w.s1 < w.s2
        # Three-spin interaction: Sz(w.s1) * Sx(w.s2) * Sy(w.s3)
        os .+=  0.5im * κ, "Sz", w.s1, "Sx", w.s2, "S-", w.s3
        os .+= -0.5im * κ, "Sz", w.s1, "Sx", w.s2, "S+", w.s3
        @info "Added three-spin term" term = ("Sz", w.s1, "Sx", w.s2, "Sy", w.s3)
      else
        # Three-spin interaction: Sz(w.s1) * Sy(w.s2) * Sx(w.s3)
        os .+=  0.5im * κ, "Sz", w.s1, "S-", w.s2, "Sx", w.s3
        os .+= -0.5im * κ, "Sz", w.s1, "S+", w.s2, "Sx", w.s3
        @info "Added three-spin term" term = ("Sz", w.s1, "Sy", w.s2, "Sx", w.s3)
      end
      vertical_wedge += 1
    end
    
    if abs(w.s2 - w.s1) == 1
      same_parity = (isodd(x_coordinate) && isodd(y_coordinate)) || (iseven(x_coordinate) && iseven(y_coordinate))
      
      if same_parity
        if w.s2 > w.s3
          # Sy(w.s3) * Sx(w.s2) * Sz(w.s1)
          os .+=  0.5im * κ, "S-", w.s3, "Sx", w.s2, "Sz", w.s1
          os .+= -0.5im * κ, "S+", w.s3, "Sx", w.s2, "Sz", w.s1
          @info "Added three-spin term (same_parity, w.s2 > w.s3)" term = ("Sy", w.s3, "Sx", w.s2, "Sz", w.s1)
        else
          # Sx(w.s3) * Sy(w.s2) * Sz(w.s1)
          os .+=  0.5im * κ, "Sx", w.s3, "S-", w.s2, "Sz", w.s1
          os .+= -0.5im * κ, "Sx", w.s3, "S+", w.s2, "Sz", w.s1
          @info "Added three-spin term (same_parity, w.s2 < w.s3)" term = ("Sx", w.s3, "Sy", w.s2, "Sz", w.s1)
        end
      else
        if w.s2 > w.s3
          # Sx(w.s3) * Sy(w.s2) * Sz(w.s1)
          os .+=  0.5im * κ, "Sx", w.s3, "S-", w.s2, "Sz", w.s1
          os .+= -0.5im * κ, "Sx", w.s3, "S+", w.s2, "Sz", w.s1
          @info "Added three-spin term (mixed_parity, w.s2 > w.s3)" term = ("Sx", w.s3, "Sy", w.s2, "Sz", w.s1)
        else
          # Sy(w.s3) * Sx(w.s2) * Sz(w.s1)
          os .+=  0.5im * κ, "S-", w.s3, "Sx", w.s2, "Sz", w.s1
          os .+= -0.5im * κ, "S+", w.s3, "Sx", w.s2, "Sz", w.s1
          @info "Added three-spin term (mixed_parity, w.s2 < w.s3)" term = ("Sy", w.s3, "Sx", w.s2, "Sz", w.s1)
        end
      end
      vertical_wedge += 1
    end
  end
  
  @show horizontal_wedge, vertical_wedge

  
  # #***************************************************************************************************************
  # #***************************************************************************************************************  
  # Generate the indices for all loop operators along the cylinder
  loop_operator = Vector{String}(["iY", "X", "iY", "X", "iY", "X", "iY", "X"])  # Hard-coded for width-4 cylinders
  loop_indices = LoopListArmchair(Nx_unit, Ny_unit, "armchair", "y")  
  # @show loop_indices

  
  # Generate the plaquette indices for all the plaquettes in the cylinder
  plaquette_operator = Vector{String}(["iY", "Z", "X", "X", "Z", "iY"])
  plaquette_indices = PlaquetteListArmchair(Nx_unit, Ny_unit, "armchair", false)
  # @show plaquette_indices

  #*****************************************************************************************************
  #*****************************************************************************************************  
  # Increase the maximum dimension of Krylov space used to locally solve the eigenvalues problem.
  sites = siteinds("tJ", N; conserve_qns=false)
  H = MPO(os, sites)

  # Initialize wavefunction to a random MPS of bond-dimension 20 with same quantum 
  # numbers as `state`
  state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
  ψ₀ = randomMPS(sites, state, 20)
  
  # Set up the parameters including bond dimensions and truncation error
  nsweeps = 10
  maxdim  = [20, 60, 100, 500, 800, 1000, 1500, 3000]
  cutoff  = [1E-10]
  eigsolve_krylovdim = 50
  
  # # Add noise terms to prevent DMRG from getting stuck in a local minimum
  # # noise = [1E-6, 1E-7, 1E-8, 0.0]
  # #*****************************************************************************************************
  # #*****************************************************************************************************

  #*****************************************************************************************************
  #*****************************************************************************************************
  # Measure one-point functions of the initial state
  Sx₀ = expect(ψ₀, "Sx", sites = 1 : N)
  Splus₀  = expect(ψ₀, "S+", sites = 1 : N)
  Sminus₀ = expect(ψ₀, "S-", sites = 1 : N)
  Sy₀ = 0.5im * (Splus₀ - Sminus₀)
  @show Sy₀ 
  Sz₀ = expect(ψ₀, "Sz", sites = 1 : N)
  #*****************************************************************************************************
  #*****************************************************************************************************
  

  # # Construct a custom observer and stop the DMRG calculation early if needed 
  # # custom_observer = DMRGObserver(; energy_tol=1E-9, minsweeps=2, energy_type=Float64)
  # custom_observer = CustomObserver()
  # @show custom_observer.etolerance
  # @show custom_observer.minsweeps
  # @timeit time_machine "dmrg simulation" begin
  #   energy, ψ = dmrg(H, ψ₀; nsweeps, maxdim, cutoff, eigsolve_krylovdim, observer = custom_observer)
  # end

  
  # # Measure local observables (one-point functions) after finish the DMRG simulation
  # @timeit time_machine "one-point functions" begin
  #   Sx = expect(ψ, "Sx", sites = 1 : N)
  #   Splus  = expect(ψ, "S+", sites = 1 : N)
  #   Sminus = expect(ψ, "S-", sites = 1 : N)
  #   Sy = 0.5im * (Splus - Sminus)
  #   Sz = expect(ψ, "Sz", sites = 1 : N)
  # end

  # # @timeit time_machine to "two-point functions" begin
  # #   xxcorr = correlation_matrix(ψ, "Sx", "Sx", sites = 1 : N)
  # #   yycorr = correlation_matrix(ψ, "Sy", "Sy", sites = 1 : N)
  # #   zzcorr = correlation_matrix(ψ, "Sz", "Sz", sites = 1 : N)
  # # end


  # # Compute the eigenvalues of plaquette operators
  # # normalize!(ψ)
  # @timeit time_machine "plaquette operators" begin
  #   W_operator_eigenvalues = zeros(Float64, size(plaquette_indices, 1))
    
  #   # Compute the eigenvalues of the plaquette operator
  #   for index in 1 : size(plaquette_indices, 1)
  #     @show plaquette_indices[index, :]
  #     os_w = OpSum()
  #     os_w += plaquette_operator[1], plaquette_indices[index, 1], 
  #       plaquette_operator[2], plaquette_indices[index, 2], 
  #       plaquette_operator[3], plaquette_indices[index, 3], 
  #       plaquette_operator[4], plaquette_indices[index, 4], 
  #       plaquette_operator[5], plaquette_indices[index, 5], 
  #       plaquette_operator[6], plaquette_indices[index, 6]
  #     W = MPO(os_w, sites)
  #     W_operator_eigenvalues[index] = -1.0 * real(inner(ψ', W, ψ))
  #     # @show inner(ψ', W, ψ) / inner(ψ', ψ)
  #   end
  # end
  # @show W_operator_eigenvalues
  
  # # # Compute the eigenvalues of the loop operators 
  # # # The loop operators depend on the width of the cylinder  
  # # @timeit time_machine "loop operators" begin
  # #   yloop_eigenvalues = zeros(Float64, size(loop_indices)[1])
    
  # #   # Compute eigenvalues of the loop operators in the direction with PBC.
  # #   for index in 1 : size(loop_indices)[1]
  # #     ## Construct loop operators along the y direction with PBC
  # #     os_wl = OpSum()
  # #     os_wl += loop_operator[1], loop_indices[index, 1], 
  # #       loop_operator[2], loop_indices[index, 2], 
  # #       loop_operator[3], loop_indices[index, 3], 
  # #       loop_operator[4], loop_indices[index, 4], 
  # #       loop_operator[5], loop_indices[index, 5], 
  # #       loop_operator[6], loop_indices[index, 6],
  # #       loop_operator[7], loop_indices[index, 7],
  # #       loop_operator[8], loop_indices[index, 8]

  # #     Wl = MPO(os_wl, sites)
  # #     yloop_eigenvalues[index] = real(inner(ψ', Wl, ψ))
  # #   end
  # # end


  # # # Compute the eigenvalues of the order parameters near vacancies
  # # # TO-DO: The order parameter (twelve-point correlator) loop is hard-coded for now
  # # # need to genealize in the future to automatically generate the loop indices near vacancies

  # # @timeit time_machine "twelve-point correlator(s)" begin
  # #   order_loop = Vector{String}(["Z", "Y", "Y", "Y", "X", "Z", "Z", "Z", "Y", "X", "X", "X"])
  # #   order_indices = Matrix{Int64}(undef, 1, 12)
  # #   # Complete the loop indices near vacancies
  # #   # order_indices[1, :] = [52, 49, 46, 43, 40, 38, 41, 39, 42, 45, 47, 50]      # On the width-3 cylinders  
  # #   order_indices[1, :] = [70, 66, 62, 58, 54, 51, 55, 52, 56, 60, 63, 67]      # On the width-4 cylinders
  # #   order_parameter = Vector{Float64}(undef, size(order_indices)[1])

    
  # #   @show size(order_indices)[1]
  # #   for index in 1 : size(order_indices)[1]
  # #     os_parameter = OpSum()
  # #     os_parameter += order_loop[1], order_indices[index, 1], 
  # #       order_loop[2], order_indices[index, 2], 
  # #       order_loop[3], order_indices[index, 3], 
  # #       order_loop[4], order_indices[index, 4], 
  # #       order_loop[5], order_indices[index, 5], 
  # #       order_loop[6], order_indices[index, 6],
  # #       order_loop[7], order_indices[index, 7],
  # #       order_loop[8], order_indices[index, 8],
  # #       order_loop[9], order_indices[index, 9],
  # #       order_loop[10], order_indices[index, 10],
  # #       order_loop[11], order_indices[index, 11],
  # #       order_loop[12], order_indices[index, 12]
  # #     W_parameter = MPO(os_parameter, sites)
  # #     order_parameter[index] = real(inner(ψ', W_parameter, ψ))
  # #   end
  # # end

  
  # # Print out useful information of physical quantities
  # println("")
  # println("Visualize the optimization history of the energy and bond dimensions:")
  # @show custom_observer.ehistory_full
  # @show custom_observer.ehistory
  # @show custom_observer.chi
  # # @show number_of_bonds, energy / number_of_bonds
  # # @show N, energy / N
  # println("")

  # # Check the variance of the energy
  # @timeit time_machine "compaute the variance" begin
  #   H2 = inner(H, ψ, H, ψ)
  #   E₀ = inner(ψ', H, ψ)
  #   variance = H2 - E₀^2
  # end
  # println("")
  # @show E₀
  # println("Variance of the energy is $variance")
  # println("")
  

  # println("")
  # println("Eigenvalues of the plaquette operator:")
  # @show W_operator_eigenvalues
  # println("")


  # print("")
  # println("Eigenvalues of the loop operator(s):")
  # @show yloop_eigenvalues
  # println("")

  # # println("")
  # # println("Eigenvalues of the twelve-point correlator near the first vacancy:")
  # # @show order_parameter
  # # println("")


  @show time_machine
  

  # h5open("data/test/armchair_geometery/2d_kitaev_honeycomb_armchair_FM_Lx$(Nx_unit)_h$(h).h5", "w") do file
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
  #   # write(file, "OrderParameter", order_parameter)
  # end

  return
end