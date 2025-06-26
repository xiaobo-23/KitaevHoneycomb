# 04/24/2025
# Simulate the 2D tJ-Kitaev honeycomb model to design topologucal qubits based on quantum spin liquids (QSLs)
# Introduce three-spin interaction, electron hopping, and Kitaev interaction; remove the spin vacancy


using ITensors
using ITensorMPS
using HDF5
using MKL
using LinearAlgebra
using TimerOutputs


include("../HoneycombLattice.jl")
include("../Entanglement.jl")
include("../TopologicalLoops.jl")
include("../CustomObserver.jl")


# Set up parameters for multithreading for BLAS/LAPACK and Block sparse multithreading
MKL_NUM_THREADS = 8
OPENBLAS_NUM_THREADS = 8
OMP_NUM_THREADS = 8


# Monitor the number of threads used by BLAS and LAPACK
@show BLAS.get_config()
@show BLAS.get_num_threads()


const Nx_unit = 12
const Ny_unit = 3
const Nx = 2 * Nx_unit
const Ny = Ny_unit
const N = Nx * Ny
# Timing and profiling
const time_machine = TimerOutput()


let
  # Set up the parameters in the Hamiltonian
  Jx, Jy, Jz = 1.0, 1.0, 1.0                      # Kitaev interactions
  kappa=0                                         # Three-spin interaction
  t = 0.005                                       # Electron hopping amplitude
  P = -10.0                                       # Chemical potential for the edge sites
  h = 0.0                                         # Zeeman field
  @show Jx, Jy, Jz, kappa, t, P, h

  # Reduce the Kitaev interaction connected to the vacancy
  # alpha = 1E-4
  # @show Jx, Jy, Jz, alpha, kappa, t, h

  #***************************************************************************************************************
  #***************************************************************************************************************
  # Set up the honeycomb lattice with proper boundary conditions
  # Return the two types of objects: bonds (for two-body interactions) and wedges (for three-body interactions)
  #***************************************************************************************************************
  #***************************************************************************************************************
  
  # Set up the boundary conditions for the cylinder
  x_periodic = false
  y_periodic = true
  y_direction_twist = true
  
  # Construct a honeycomb lattice using XC geometry with a twist 
  # TO-DO: Implement the armchair geometery with periodic boundary condition
  if x_periodic
    lattice = honeycomb_lattice_rings_pbc(Nx, Ny; yperiodic=true)
    @show length(lattice)
  else
    # lattice = honeycomb_lattice_armchair(Nx, Ny; yperiodic=true)
    lattice = honeycomb_lattice_rings_right_twist(Nx, Ny; yperiodic=true)
    @show length(lattice)
  end 
  number_of_bonds = length(lattice)
  
  # Construct the wedges in order to set up three-body spin interactions
  wedge = honeycomb_twist_wedge(Nx, Ny; yperiodic=true)
  # @show length(wedge), wedge 
  
  # Select the position(s) of the vacancies
  # sites_to_delete = Set{Int64}([59])            # The site number of the vacancy depends on the lattice width
  # sites_to_delete = Set{Int64}()
  lattice_sites = Set{Int64}(1 : N)               # The set of all sites in the lattice
  @show lattice_sites

  # Identify edge sites: first 7 and last 7 sites of the lattice
  edge_sites = Set(1 : 6*Ny) ∪ Set(N - 6*Ny + 1 : N)
  @show length(edge_sites), edge_sites
  #***************************************************************************************************************
  #***************************************************************************************************************  
  

  #***************************************************************************************************************
  #***************************************************************************************************************
  # Construct the Hamiltonian as MPO
  #***************************************************************************************************************
  #***************************************************************************************************************

  # Construct the Kitaev interaction and the hopping terms
  os = OpSum()
  xbond = 0
  ybond = 0
  zbond = 0
  
  for b in lattice
    # Set up the hopping terms for spin-up and spin-down electrons
    os .+= -t, "Cdagup", b.s1, "Cup", b.s2
    os .+= -t, "Cdagup", b.s2, "Cup", b.s1
    os .+= -t, "Cdagdn", b.s1, "Cdn", b.s2
    os .+= -t, "Cdagdn", b.s2, "Cdn", b.s1

    # Set up the anisotropic two-body Kitaev interaction
    tmp_x = div(b.s1 - 1, Ny) + 1
    if iseven(tmp_x)
      os .+= -Jz, "Sz", b.s1, "Sz", b.s2
      zbond += 1
      @info "Added Sz-Sz bond" s1=b.s1 s2=b.s2
    else
      if abs(b.s1 - b.s2) == Ny
        os .+= -Jx, "Sx", b.s1, "Sx", b.s2
        xbond += 1
        @info "Added Sx-Sx bond" s1=b.s1 s2=b.s2
      else
        os .+= -0.25 * Jy, "S+", b.s1, "S-", b.s2
        os .+= -0.25 * Jy, "S-", b.s1, "S+", b.s2
        os .+=  0.25 * Jy, "S+", b.s1, "S+", b.s2
        os .+=  0.25 * Jy, "S-", b.s1, "S-", b.s2
        ybond += 1 
        @info "Added Sy-Sy bond" s1=b.s1 s2=b.s2
      end
    end
  end
  
  # Check the number of bonds in the Hamiltonian 
  total_bonds = trunc(Int, 3/2 * N) - Ny + (y_periodic ? -1 : -trunc(Int, N / 2))
  @show xbond + ybond + zbond == total_bonds
  if xbond + ybond + zbond != total_bonds
    error("The number of bonds in the Hamiltonian is not correct!")
  end

  # Set Up the three-spin interactions in the Hamiltonian
  count_wedge = 0
  for w in wedge
    @show w.s1, w.s2, w.s3
    x_coordinate = div(w.s2 - 1, Ny) + 1
    y_coordinate = mod(w.s2 - 1, Ny) + 1

    if abs(w.s1 - w.s2) == abs(w.s2 - w.s3)
      if isodd(x_coordinate)
        # os .+= kappa, "Sz", w.s1, "Sy", w.s2, "Sx", w.s3
        os .+=  0.5im * kappa, "Sz", w.s1, "S-", w.s2, "Sx", w.s3 
        os .+= -0.5im * kappa, "Sz", w.s1, "S+", w.s2, "Sx", w.s3
        @info "Added three-spin term" term = ("Sz", w.s1, "Sy", w.s2, "Sx", w.s3)
      else
        # os .+= kappa, "Sx", w.s1, "Sy", w.s2, "Sz", w.s3
        os .+=  0.5im * kappa, "Sx", w.s1, "S-", w.s2, "Sz", w.s3
        os .+= -0.5im * kappa, "Sx", w.s1, "S+", w.s2, "Sz", w.s3
        @info "Added three-spin term" term = ("Sx", w.s1, "Sy", w.s2, "Sz", w.s3)
      end
      count_wedge += 1
    end

    if (abs(w.s1 - w.s2) == 1 || abs(w.s1 - w.s2) == 2) && abs(w.s2 - w.s3) == 3
      if isodd(x_coordinate)
        # os .+= kappa, "Sy", w.s1, "Sz", w.s2, "Sx", w.s3
        os .+=  0.5im * kappa, "S-", w.s1, "Sz", w.s2, "Sx", w.s3
        os .+= -0.5im * kappa, "S+", w.s1, "Sz", w.s2, "Sx", w.s3
        @info "Added three-spin term" term = ("Sy", w.s1, "Sz", w.s2, "Sx", w.s3)
        count_wedge += 1
      else
        # os .+= kappa, "Sy", w.s1, "Sx", w.s2, "Sz", w.s3
        os .+=  0.5im * kappa, "S-", w.s1, "Sx", w.s2, "Sz", w.s3
        os .+= -0.5im * kappa, "S+", w.s1, "Sx", w.s2, "Sz", w.s3
        @info "Added three-spin term" term = ("Sy", w.s1, "Sx", w.s2, "Sz", w.s3)
        count_wedge += 1
      end
    end

    if abs(w.s1 - w.s2) == 3 && (abs(w.s2 - w.s3) == 1 || abs(w.s2 - w.s3) == 2)
      if iseven(x_coordinate)
        # os .+= kappa, "Sx", w.s1, "Sz", w.s2, "Sy", w.s3
        os .+=  0.5im * kappa, "Sx", w.s1, "Sz", w.s2, "S-", w.s3
        os .+= -0.5im * kappa, "Sx", w.s1, "Sz", w.s2, "S+", w.s3
        @info "Added three-spin term" term = ("Sx", w.s1, "Sz", w.s2, "Sy", w.s3)
        count_wedge += 1
      else
        # os .+= kappa, "Sz", w.s1, "Sx", w.s2, "Sy", w.s3
        os .+=  0.5im * kappa, "Sz", w.s1, "Sx", w.s2, "S-", w.s3
        os .+= -0.5im * kappa, "Sz", w.s1, "Sx", w.s2, "S+", w.s3
        @info "Added three-spin term" term = ("Sz", w.s1, "Sx", w.s2, "Sy", w.s3)
        count_wedge += 1
      end
    end
  end
  
  # Add edge chemical potentiial to avoid the hole being trapped in the edge
  if abs(P) > 1e-8
    for site in edge_sites
      os .+= P, "Ntot", site
      @info "Added edge chemical potential" site=site P=P
    end
  end

  # # Add the Zeeman coupling of the spins to a magnetic field applied in [111] direction, which breaks the integrability
  # if h > 1e-8
  #   for site in lattice_sites
  #     os .+= -h, "Sx", site
  #     os .+= -0.5h, "iS-", site
  #     os .+=  0.5h, "iS+", site
  #     os .+= -h, "Sz", site
  #   end
  # end
  
  # @show count_wedge, 3 * N - 4 * Ny - 2
  if count_wedge != 3 * N - 4 * Ny - 2
    error("The number of three-spin interactions is not correct!")
  end
  
  # Set up the loop operators and loop indices 
  loop_operator = ["Sx", "Sx", "Sz", "Sz", "Sz", "Sz"]            # Hard-coded for width-3 cylinders
  loop_indices = LoopList_RightTwist(Nx_unit, Ny_unit, "rings", "y")  
  @show loop_indices


  # Generate the plaquette indices for all the plaquettes in the cylinder
  # plaquette_operator = Vector{String}(["iY", "Z", "X", "X", "Z", "iY"])
  # plaquette_operator = Vector{String}(["Z", "iY", "X", "X", "iY", "Z"]) 
  plaquette_operator = [
    ["Sz", "S+", "Sx", "Sx", "S+", "Sz"],
    ["Sz", "S+", "Sx", "Sx", "S-", "Sz"],
    ["Sz", "S-", "Sx", "Sx", "S-", "Sz"],
    ["Sz", "S-", "Sx", "Sx", "S+", "Sz"]
  ]
  plaquette_indices = PlaquetteList_RightTwist(Nx_unit, Ny_unit, "rings", false)
  @show plaquette_indices
  #***************************************************************************************************************
  #***************************************************************************************************************  
  
  
  #********************************************************************************************************
  #********************************************************************************************************
  # Read in the ground-state wavefunction from a file and sample the wavefunction
  #********************************************************************************************************
  #********************************************************************************************************
  println("*************************************************************************************")
  println("Read in the wavefunction from a file and start the sampling process.")
  println("*************************************************************************************")

  file = h5open("../../t0.005/data/2d_tK_Lx12_Ly3_kappa-0.375_doped.h5", "r")
  ψ₀ = read(file, "psi", MPS)

  # Measure one-point functions of the initial state
  Sx₀ = expect(ψ₀, "Sx", sites = 1 : N)
  Splus₀  = expect(ψ₀, "S+", sites = 1 : N)
  Sminus₀ = expect(ψ₀, "S-", sites = 1 : N)
  Sy₀ = 0.5im * (Splus₀ - Sminus₀)
  # @show Sy₀ 
  Sz₀ = expect(ψ₀, "Sz", sites = 1 : N)
  n₀ = expect(ψ₀, "Ntot", sites = 1 : N)
  println("")
  @show sum(n₀)
  # @show n₀
  println("")

  # Check if the system is properly doped before running the DMRG simulation
  if abs(N - sum(n₀) - 1) > 1E-6
    error("The system is not properly doped!")
  end

  #*****************************************************************************************************
  #*****************************************************************************************************
  
  
  #*****************************************************************************************************
  #*****************************************************************************************************  
  # Set up the initial MPS and parameters for the DMRG simulation
  #*****************************************************************************************************
  #*****************************************************************************************************

  # Increase the maximum dimension of Krylov space used to locally solve the eigenvalues problem.
  # sites = siteinds("tJ", N; conserve_qns=false)
  # sites = siteinds("tJ", N; conserve_nf=true)
  sites = siteinds(ψ₀)
  H = MPO(os, sites)

  # # Initialize wavefunction to a random MPS of bond-dimension 10 with same quantum 
  # # numbers as `state`
  # # state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
  # state = []
  # hole_idx = 36
  # for (idx, n) in enumerate(1 : N)
  #   if n == hole_idx
  #     push!(state, "Emp")
  #   else
  #     if isodd(idx)
  #       push!(state, "Up")
  #     else
  #       push!(state, "Dn")
  #     end
  #   end
  # end
  # @show state
  # ψ₀ = randomMPS(sites, state, 10)
  #********************************************************************************************************
  #********************************************************************************************************
 

  #******************************************************************************************************
  #******************************************************************************************************
  # Run the DMRG simulation and obtain the ground state wavefunction
  #******************************************************************************************************
  #******************************************************************************************************
  # Set up the parameters including bond dimensions and truncation error
  nsweeps = 20
  maxdim  = [20, 100, 200, 500, 800, 1000, 1500, 3000]
  cutoff  = [1E-10]
  eigsolve_krylovdim = 35
  
  # Add noise terms to prevent DMRG from getting stuck in a local minimum
  noise = [1E-6, 1E-7, 1E-8, 0.0]

  # Construct a custom observer and stop the DMRG calculation early if criteria are met
  # custom_observer = DMRGObserver(; energy_tol=1E-9, minsweeps=2, energy_type=Float64)
  custom_observer = CustomObserver()
  @show custom_observer.etolerance
  @show custom_observer.minsweeps
  @timeit time_machine "dmrg simulation" begin
    energy, ψ = dmrg(H, ψ₀; nsweeps, maxdim, cutoff, eigsolve_krylovdim, observer = custom_observer)
  end
  #*****************************************************************************************************
  #*****************************************************************************************************

  #******************************************************************************************************
  #******************************************************************************************************
  # Take measurements of the wavefunction after the DMRG simulation
  #******************************************************************************************************
  #******************************************************************************************************
  # Measure local observables (one-point functions)
  @timeit time_machine "one-point functions" begin
    Sx = expect(ψ, "Sx", sites = 1 : N)
    Splus  = expect(ψ, "S+", sites = 1 : N)
    Sminus = expect(ψ, "S-", sites = 1 : N)
    Sy = 0.5im * (Splus - Sminus)
    Sz = expect(ψ, "Sz", sites = 1 : N)
    n = expect(ψ, "Ntot", sites = 1 : N)
  end

  # Check if the system is properly doped after the DMRG simulation
  println("")
  @show sum(n)
  @show n
  println("")

  if abs(N - sum(n) - 1) > 1E-6
    error("The system is not properly doped!")
  end
  

  # Measure spin correlation functions (two-point functions)  
  @timeit time_machine "two-point functions" begin
    xxcorr = correlation_matrix(ψ, "Sx", "Sx", sites = 1 : N)
    zzcorr = correlation_matrix(ψ, "Sz", "Sz", sites = 1 : N)
    # yycorr = correlation_matrix(ψ, "Sy", "Sy", sites = 1 : N)
  end

  # Measure the loop operators along the y direction of the cylinder
  # The number of terms in the loop operator depends on the width of the cylinder 
  @timeit time_machine "loop operators" begin
    nloops = size(loop_indices, 1)
    yloop_eigenvalues = zeros(Float64, nloops)
    
    for idx in 1 : nloops
      indices = loop_indices[idx, :]
      # Construct the loop operators as MPOs and compute the eigenvalues
      os_wl = OpSum()
      os_wl += loop_operator[1], indices[1], 
        loop_operator[2], indices[2], 
        loop_operator[3], indices[3], 
        loop_operator[4], indices[4], 
        loop_operator[5], indices[5], 
        loop_operator[6], indices[6]
      Wl = MPO(os_wl, sites)

      # The normalize factor is due to the difference between Pauli operators and spin operators
      yloop_eigenvalues[idx] = real(inner(ψ', Wl, ψ))
      yloop_eigenvalues[idx] *= 2^6 
    end
  end
  # @show yloop_eigenvalues

  # Measure the eigenvalues of plaquette operators
  # Decompose the plaquette operators into four terms for tJ type of sites
  @timeit time_machine "plaquette operators" begin
    nplaquettes = size(plaquette_indices, 1)
    plaquette_eigenvalues = zeros(Float64, nplaquettes)
    
    for idx1 in 1:nplaquettes
      indices  = plaquette_indices[idx1, :]
      
      for idx2 in 1:4
        operator = plaquette_operator[idx2]
        # @show operator, indices
        os_w = OpSum()
        os_w += operator[1], indices[1], 
          operator[2], indices[2], 
          operator[3], indices[3], 
          operator[4], indices[4], 
          operator[5], indices[5], 
          operator[6], indices[6]
        W = MPO(os_w, sites)
        # @show (-1.0)^idx2 * real(inner(ψ', W, ψ)) * 2^6
        plaquette_eigenvalues[idx1] += (-1.0)^idx2 * real(inner(ψ', W, ψ))
      end
      plaquette_eigenvalues[idx1] *= 2^6 / 4
      # @show inner(ψ', W, ψ) / inner(ψ', ψ)
    end
  end
  # @show plaquette_eigenvalues


  # Set up and measure the eigenvalues of the order parameter(s)
  # Define the central sites, excluding a margin of 2*Ny sites from both boundaries
  centers = collect((2 * Ny + 2):(N - 2 * Ny - 1))
  @info "Central sites selected for measurement" centers=centers
  
  order_loops = []
  for center in centers
    tmp_x = div(center - 1, Ny) + 1
    tmp_y = mod(center - 1, Ny) + 1
    tmp_loop = []

    if isodd(tmp_x)
      if tmp_y == 1
        append!(tmp_loop, [
          center + 1,
          center + Ny,
          center + 2 * Ny,
          center + 2 * Ny - 1,
          center + Ny - 1,
          center - 1,
          center - Ny - 1,
          center - 2 * Ny - 1,
          center - 2 * Ny,
          center - Ny,
          center - 2 * Ny + 1, 
          center - Ny + 1
        ])
      elseif tmp_y == Ny
        append!(tmp_loop, [
          center + Ny + 1,
          center + Ny,
          center + 2 * Ny,
          center + 2 * Ny + 2,
          center + 2 * Ny - 1,
          center + Ny - 1,
          center - 1,
          center - Ny - 1,
          center - 2 * Ny,
          center - Ny,
          center - Ny + 1,
          center + 1
        ])
      else
        # Construct the loop for odd x and tmp_y ≠ 1 and tmp_y ≠ Ny
        append!(tmp_loop, [
          center + 1,
          center + Ny,
          center + 2 * Ny,
          center + 2 * Ny + 2,
          center + 2 * Ny - 1,
          center + Ny - 1,
          center - 1,
          center - Ny - 1,
          center - 2 * Ny,
          center - Ny,
          center - 2 * Ny + 1,
          center - Ny + 1
        ])
      end
    else
      if tmp_y == 1
        # Construct the loop for even x and tmp_y == 1
        append!(tmp_loop, [
          center - Ny - 1,
          center - Ny,
          center - 2 * Ny,
          center - 3 * Ny + 1,
          center - 2 * Ny + 1,
          center - Ny + 1,
          center + 1,
          center + Ny + 1,
          center + 2 * Ny,
          center + Ny,
          center + Ny - 1,
          center - 1
        ])
      elseif tmp_y == Ny
        # Construct the loop for even x and tmp_y == Ny
        append!(tmp_loop, [
          center - 1,
          center - Ny,
          center - 2 * Ny,
          center - 2 * Ny + 1,
          center - Ny + 1,
          center + 1,
          center + Ny + 1,
          center + 2 * Ny + 1,
          center + 2 * Ny,
          center + Ny,
          center + 2 * Ny - 1,
          center + Ny - 1
        ]) 
      else
        # Construct the loop for even x and tmp_y != 1 and tmp_y != Ny
        append!(tmp_loop, [
          center - 1,
          center - Ny,
          center - 2 * Ny,
          center - 3 * Ny + 1,
          center - 2 * Ny + 1,
          center - Ny + 1,
          center + 1,
          center + Ny + 1,
          center + 2 * Ny,
          center + Ny,
          center + 2 * Ny - 1,
          center + Ny - 1
        ])
      end
    end
    push!(order_loops, tmp_loop)
  end
  
  for idx in eachindex(order_loops)
    @show centers[idx], order_loops[idx]
  end

  function configure_signs(input_string)
    return [(-1.0)^count(==( "S-" ), row) for row in input_string]
  end

  order_string = [["Sx", "Sx", "Sx", "Sz", "S+", "S+", "S+", "Sx", "Sz", "Sz", "Sz", "S+"], 
  ["Sx", "Sx", "Sx", "Sz", "S+", "S+", "S+", "Sx", "Sz", "Sz", "Sz", "S-"], 
  ["Sx", "Sx", "Sx", "Sz", "S+", "S+", "S-", "Sx", "Sz", "Sz", "Sz", "S+"],
  ["Sx", "Sx", "Sx", "Sz", "S+", "S+", "S-", "Sx", "Sz", "Sz", "Sz", "S-"], 
  ["Sx", "Sx", "Sx", "Sz", "S+", "S-", "S+", "Sx", "Sz", "Sz", "Sz", "S+"], 
  ["Sx", "Sx", "Sx", "Sz", "S+", "S-", "S+", "Sx", "Sz", "Sz", "Sz", "S-"], 
  ["Sx", "Sx", "Sx", "Sz", "S+", "S-", "S-", "Sx", "Sz", "Sz", "Sz", "S+"], 
  ["Sx", "Sx", "Sx", "Sz", "S+", "S-", "S-", "Sx", "Sz", "Sz", "Sz", "S-"], 
  ["Sx", "Sx", "Sx", "Sz", "S-", "S+", "S+", "Sx", "Sz", "Sz", "Sz", "S+"], 
  ["Sx", "Sx", "Sx", "Sz", "S-", "S+", "S+", "Sx", "Sz", "Sz", "Sz", "S-"], 
  ["Sx", "Sx", "Sx", "Sz", "S-", "S+", "S-", "Sx", "Sz", "Sz", "Sz", "S+"],
  ["Sx", "Sx", "Sx", "Sz", "S-", "S+", "S-", "Sx", "Sz", "Sz", "Sz", "S-"], 
  ["Sx", "Sx", "Sx", "Sz", "S-", "S-", "S+", "Sx", "Sz", "Sz", "Sz", "S+"], 
  ["Sx", "Sx", "Sx", "Sz", "S-", "S-", "S+", "Sx", "Sz", "Sz", "Sz", "S-"], 
  ["Sx", "Sx", "Sx", "Sz", "S-", "S-", "S-", "Sx", "Sz", "Sz", "Sz", "S+"], 
  ["Sx", "Sx", "Sx", "Sz", "S-", "S-", "S-", "Sx", "Sz", "Sz", "Sz", "S-"]]
  
  # Reference sign structure for the order parameter 
  # sign = [1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0]
  sign = configure_signs(order_string)
  @show sign

  @timeit time_machine "order parameter(s)" begin
    order_parameter = zeros(Float64, length(order_loops))
    order₀ = zeros(Float64, length(order_loops))

    for idx1 in 1 : size(order_loops)[1]
      loop = order_loops[idx1]
      for idx2 in 1 : size(order_string)[1]
        operator = order_string[idx2]
        os_order = OpSum()
        os_order +=  "Ntot", centers[idx1], 
          operator[1], loop[1], 
          operator[2], loop[2], 
          operator[3], loop[3], 
          operator[4], loop[4], 
          operator[5], loop[5], 
          operator[6], loop[6],
          operator[7], loop[7],
          operator[8], loop[8],
          operator[9], loop[9],
          operator[10], loop[10],
          operator[11], loop[11],
          operator[12], loop[12]
        W_order = MPO(os_order, sites)

        os_order_identity = OpSum()
        os_order_identity += operator[1], loop[1], 
          operator[2], loop[2], 
          operator[3], loop[3], 
          operator[4], loop[4], 
          operator[5], loop[5], 
          operator[6], loop[6],
          operator[7], loop[7],
          operator[8], loop[8],
          operator[9], loop[9],
          operator[10], loop[10],
          operator[11], loop[11],
          operator[12], loop[12]
        W_order_identity = MPO(os_order_identity, sites)

        order_parameter[idx1] += (1/2)^4 * 2^12 * sign[idx2] * (real(inner(ψ', W_order_identity, ψ)) - real(inner(ψ', W_order, ψ)))
        order₀[idx1] += (1/2)^4 * 2^12 * sign[idx2] * real(inner(ψ', W_order, ψ))
      end
    end
  end

  for idx in eachindex(order_parameter)
    @show order_parameter[idx]
  end

  for idx in eachindex(order₀)
    @show order₀[idx]
  end

  # # Print out useful information of physical quantities
  # println("")
  # println("Visualize the optimization history of the energy and bond dimensions:")
  # @show custom_observer.ehistory_full
  # @show custom_observer.ehistory
  # @show custom_observer.chi
  # # @show number_of_bonds, energy / number_of_bonds
  # # @show N, energy / N
  # println("")

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
  @show plaquette_eigenvalues
  println("")

  print("")
  println("Eigenvalues of the loop operator(s):")
  @show yloop_eigenvalues
  println("")

  # # println("")
  # # println("Eigenvalues of the twelve-point correlator near the first vacancy:")
  # # @show order_parameter
  # # println("")


  # @show time_machine
  h5open("../data/2d_tK_Lx$(Nx_unit)_Ly$(Ny_unit)_kappa$(kappa)_doped.h5", "w") do file
    write(file, "psi", ψ)
    write(file, "NormalizedE0", energy / number_of_bonds)
    write(file, "E0", energy)
    write(file, "E0variance", variance)
    write(file, "Ehist", custom_observer.ehistory)
    write(file, "Bond", custom_observer.chi)
    # write(file, "Entropy", SvN)
    write(file, "Sx0", Sx₀)
    write(file, "Sx",  Sx)
    write(file, "Cxx", xxcorr)
    write(file, "Sy0", Sy₀)
    write(file, "Sy", Sy)
    # # write(file, "Cyy", yycorr)
    write(file, "Sz0", Sz₀)
    write(file, "Sz",  Sz)
    write(file, "Czz", zzcorr)
    write(file, "N0", n₀)
    write(file, "N", n)
    write(file, "Plaquette", plaquette_eigenvalues)
    write(file, "Loop", yloop_eigenvalues)
    write(file, "OrderParameter", order_parameter)
  end

  return
end