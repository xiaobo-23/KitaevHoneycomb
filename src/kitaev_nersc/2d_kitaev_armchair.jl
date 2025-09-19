# 09/18/2025
# Simulate the Kitaev model on a 2D honeycomb lattice with armchair geometry using DMRG
# Introduce loop perturbations on both edges of the cylinder

using ITensors
using ITensorMPS
using HDF5
using MKL
using LinearAlgebra
using TimerOutputs

include("HoneycombLattice.jl")
include("Entanglement.jl")
include("TopologicalLoops.jl")
include("CustomObserver.jl")


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
const number_of_bonds = 6 * Nx - 4

# Timing and profiling
const time_machine = TimerOutput()

let
  # Set up the parameters in the Hamiltonian
  Jx, Jy, Jz = -1.0, -1.0, -1.0     # Kitaev interaction strengths
  kappa=-0.4                        # Three-spin interaction strength
  t=0                               # Electron hopping amplitude
  P=-10.0                           # Chemical potential on the edges of the cylinder
  λ₁, λ₂ = -64.0, -64.0             # Perturbation strengths for the loop operators on both edges of the cylinder
  println(repeat("#", 100))
  println("# Parameters simulated in the Hamiltonian #")
  @show Jx, Jy, Jz, kappa, t, P, λ₁, λ₂
  println(repeat("#", 100))


  # honeycomb lattice implemented in the ring ordering scheme
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
  # number_of_bonds = length(lattice)


  # Construct the wedges ito set up three-spin interactions 
  wedge = honeycomb_armchair_wedge(Nx, Ny; yperiodic=true)
  @show length(wedge)
  # @show wedge

  #***************************************************************************************************************
  #***************************************************************************************************************  
  # Construct the Hamiltonian using OpSum
  # Construct the two-body interaction temrs in the Kitaev Hamiltonian
  os = OpSum()
  xbond = 0
  ybond = 0
  zbond = 0
  for b in lattice
    # Set up the electron hopping terms
    os .+= -t, "Cdagup", b.s1, "Cup", b.s2
    os .+= -t, "Cdagup", b.s2, "Cup", b.s1
    os .+= -t, "Cdagdn", b.s1, "Cdn", b.s2
    os .+= -t, "Cdagdn", b.s2, "Cdn", b.s1

    # Set up the anisotropic two-body Kitaev interaction terms
    x_coordinate = div(b.s1 - 1, Ny) + 1
    
    # Set up the Sz-Sz bond interaction 
    if abs(b.s1 - b.s2) == 1 || abs(b.s1 - b.s2) == Ny - 1
      os .+= -Jz, "Sz", b.s1, "Sz", b.s2
      zbond += 1
      @info "Added Sz-Sz bond" s1=b.s1 s2=b.s2
    end

    # Set up the Sx-Sx and Sy-Sy bond interactions
    if (isodd(x_coordinate) && isodd(b.s1) && isodd(b.s2)) || (iseven(x_coordinate) && iseven(b.s1) && iseven(b.s2))
      os .+= -Jx, "Sx", b.s1, "Sx", b.s2
      xbond += 1
      @info "Added Sx-Sx bond" s1=b.s1 s2=b.s2
    elseif (isodd(x_coordinate) && iseven(b.s1) && iseven(b.s2)) || (iseven(x_coordinate) && isodd(b.s1) && isodd(b.s2))
      # os .+= -Jy, "Sy", b.s1, "Sy", b.s2
      os .+= -0.25 * Jy, "S+", b.s1, "S-", b.s2
      os .+= -0.25 * Jy, "S-", b.s1, "S+", b.s2
      os .+=  0.25 * Jy, "S+", b.s1, "S+", b.s2
      os .+=  0.25 * Jy, "S-", b.s1, "S-", b.s2
      ybond += 1
      @info "Added Sy-Sy bond" s1=b.s1 s2=b.s2
    end
      
  end
  

  if xbond + ybond + zbond != number_of_bonds
    error("The number of bonds in the Hamiltonian does not match the expected number of bonds!")
  end
  @show xbond, ybond, zbond
  
  
  # # Implement the three-body interaction terms in the Hamiltonian
  # horizontal_wedge = 0
  # vertical_wedge = 0
  # for w in wedge
  #   @show w.s1, w.s2, w.s3
  #   x_coordinate = div(w.s2 - 1, Ny) + 1
  #   y_coordinate = mod(w.s2 - 1, Ny) + 1
  #   if abs(w.s1 - w.s2) == abs(w.s2 - w.s3)
  #     if (mod(x_coordinate, 2) == 1 && mod(y_coordinate, 2) == 1) || (mod(x_coordinate, 2) == 0 && mod(y_coordinate, 2) == 0)
  #       os .+= K, "Sy", w.s1, "Sz", w.s2, "Sx", w.s3
  #     elseif (mod(x_coordinate, 2) == 1 && mod(y_coordinate, 2) == 0) || (mod(x_coordinate, 2) == 0 && mod(y_coordinate, 2) == 1)
  #       os .+= K, "Sx", w.s1, "Sz", w.s2, "Sy", w.s3
  #     end
  #     horizontal_wedge += 1
  #   end

  #   if abs(w.s1 - w.s2) == 3
  #     if w.s1 < w.s2
  #       os .+= K, "Sz", w.s1, "Sy", w.s2, "Sx", w.s3
  #     else
  #       os .+= K, "Sz", w.s1, "Sx", w.s2, "Sy", w.s3
  #     end
  #     vertical_wedge += 1
  #   end


  #   if abs(w.s2 - w.s1) == 1
  #     if (mod(x_coordinate, 2) == 1 && mod(y_coordinate, 2) == 1) || (mod(x_coordinate, 2) == 0 && mod(y_coordinate, 2) == 0)
  #       if w.s2 > w.s3
  #         os .+= K, "Sy", w.s3, "Sx", w.s2, "Sz", w.s1
  #       else
  #         os .+= K, "Sx", w.s3, "Sy", w.s2, "Sz", w.s1
  #       end
  #     elseif (mod(x_coordinate, 2) == 1 && mod(y_coordinate, 2) == 0) || (mod(x_coordinate, 2) == 0 && mod(y_coordinate, 2) == 1) 
  #       if w.s2 > w.s3
  #         os .+= K, "Sx", w.s3, "Sy", w.s2, "Sz", w.s1
  #       else
  #         os .+= K, "Sy", w.s3, "Sx", w.s2, "Sz", w.s1
  #       end
  #     end
  #     vertical_wedge += 1
  #   end
  # end
  # # @show horizontal_wedge, vertical_wedge

  #***************************************************************************************************************
  #***************************************************************************************************************  


  # Generate the indices for all loop operators along the cylinder
  loop_operator = Vector{String}(["iY", "X", "iY", "X", "iY", "X", "iY", "X"])  # Hard-coded for width-4 cylinders
  loop_indices = LoopListArmchair(Nx_unit, Ny_unit, "armchair", "y")  
  @show loop_indices

  # Generate the plaquette indices for all the plaquettes in the cylinder
  plaquette_operator = Vector{String}(["iY", "Z", "X", "X", "Z", "iY"])
  plaquette_indices = PlaquetteListArmchair(Nx_unit, Ny_unit, "armchair", false)
  @show plaquette_indices

  #*****************************************************************************************************
  #*****************************************************************************************************  
  # Increase the maximum dimension of Krylov space used to locally solve the eigenvalues problem.
  sites = siteinds("tJ", N; conserve_nf=true)
  H = MPO(os, sites)

  # Initialize wavefunction to a random MPS of bond-dimension 20 with same quantum 
  # numbers as `state`
  state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
  ψ₀ = randomMPS(sites, state, 8)
  
  # Set up the parameters including bond dimensions and truncation error
  nsweeps = 1
  maxdim  = [20, 60, 100, 500, 800, 1000, 1500, 3000]
  cutoff  = [1E-10]
  eigsolve_krylovdim = 50
  
  # Add noise terms to prevent DMRG from getting stuck in a local minimum
  # noise = [1E-6, 1E-7, 1E-8, 0.0]
  #*****************************************************************************************************
  #*****************************************************************************************************

  #*****************************************************************************************************
  #*****************************************************************************************************
  # Measure one-point functions of the initial state
  Sx₀ = expect(ψ₀, "Sx", sites = 1 : N)
  Sy₀ = expect(ψ₀, "iSy", sites = 1 : N)
  Sz₀ = expect(ψ₀, "Sz", sites = 1 : N)
  #*****************************************************************************************************
  #*****************************************************************************************************
  

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