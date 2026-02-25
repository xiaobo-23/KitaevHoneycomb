# 02/23/2026
# Simulate the Kitaev model on a 2D honeycomb lattice with zigzag geometry using DMRG
# Introduce loop perturbations on edges of the cylinder & string pontential

using ITensors
using ITensorMPS
using HDF5
using MKL
using LinearAlgebra
using TimerOutputs
using Random


include("HoneycombLattice.jl")
include("Entanglement.jl")
include("TopologicalLoops.jl")
include("CustomObserver.jl")


# Set up parameters for multithreading for BLAS/LAPACK and Block sparse multithreading
const MKL_NUM_THREADS = 8
const OPENBLAS_NUM_THREADS = 8
const OMP_NUM_THREADS = 8

# Monitor the number of threads used by BLAS and LAPACK
@info "BLAS configuration" config=BLAS.get_config() num_threads=BLAS.get_num_threads()


# Lattice parameters for the honeycom lattice with zigzag geometry
const Nx_unit = 11
const Ny_unit = 3
const Nx = 2 * Nx_unit
const Ny = Ny_unit
const N = Nx * Ny


# Set up the parameters in the Hamiltonian
const Jx = 1.0                                     # Kitaev interaction (x-bond)
const Jy = 1.0                                     # Kitaev interaction (y-bond)
const Jz = 1.0                                     # Kitaev interaction (z-bond)
const kappa = -0.4                                 # Three-spin interaction strength
const t = 0.0                                      # Electron hopping amplitude
const P = -10.0                                    # Chemical potential for edge sites
const lambda₁ = 64.0                               # Loop perturbation (left edge)
const lambda₂ = 64.0                               # Loop perturbation (right edge)
@info "Hamiltonian parameters" Jx Jy Jz kappa t P lambda₁ lambda₂


# Validate parameter ranges
@assert abs(Jx) > 0 && abs(Jy) > 0 && abs(Jz) > 0  "Kitaev couplings must be non-zero"
@assert abs(lambda₁) > 0 && abs(lambda₂) > 0       "Loop perturbation strengths must be non-zero"


# Set up the boundary conditions for the honeycomb lattice
const x_periodic = false
const y_periodic = true


# Timing and profiling
const time_machine = TimerOutput()




let
  header = repeat('#', 200)
  println(header)
  println("Running DMRG simulation to obtain the ground-state wavefunction of the Kitaev model")
  println(header, "\n")

  #****************************************************************************************************************
  """
    Set up the honeycomb lattice with proper boundary conditions.
    Returns bonds (two-body interactions) and wedges (three-spin interactions).
  """
  
  # Construct a vector of bonds to set up the two-body interactions
  lattice = if x_periodic
    honeycomb_zigzag_pbc(Nx, Ny; yperiodic=y_periodic)
  else
    honeycomb_zigzag(Nx, Ny; yperiodic=y_periodic)
  end
  number_of_bonds = length(lattice)
 
  # Validate expected bond count
  expected_bonds = trunc(Int, 3/2 * N) - Ny + (y_periodic ? 0 : -trunc(Int, N / 2))
  if number_of_bonds != expected_bonds
    error("Unexpected number of bonds! Expected: $expected_bonds, Found: $number_of_bonds")
  end

  # for (i, b) in enumerate(lattice)
  #   @info "Bond $i" s1=b.s1 s2=b.s2
  # end
  
  
  # Construct a vector of three-point objects to set up the three-spin interactions
  wedge = honeycomb_zigzag_wedge(Nx, Ny; yperiodic=true)
  
  # Validate expected wedge count
  if length(wedge) != 3 * N - 4 * Ny
    error("The number of three-spin interaction terms is not correct! Expected: $(3 * N - 4 * Ny), Found: $(length(wedge))")
  end

  # for (idx, w) in enumerate(wedge)
  #   @info "Wedge $idx" s1=w.s1 s2=w.s2 s3=w.s3
  # end  


  # Identify edge sites: first 6 and last 6 columns
  edge_sites = sort(collect(Set(1 : 6*Ny) ∪ Set(N - 6*Ny + 1 : N)))
  # @show length(edge_sites), edge_sites
  #****************************************************************************************************************
  
  
  
  
  #****************************************************************************************************************
  """Set the Hamiltonian as an MPO"""
  os = OpSum()
  

  # Set up two-body interactions in the Hamiltonian
  println("\nSetting up two-body terms including the hopping and Kitaev interaction...")
  xbond = Ref(0)
  ybond = Ref(0)
  zbond = Ref(0)
  
  for b in lattice
    # Set up the hopping terms for spin-up and spin-down electrons seperately
    os .+= -t, "Cdagup", b.s1, "Cup", b.s2
    os .+= -t, "Cdagup", b.s2, "Cup", b.s1
    os .+= -t, "Cdagdn", b.s1, "Cdn", b.s2
    os .+= -t, "Cdagdn", b.s2, "Cdn", b.s1

    # Set up the anisotropic two-body Kitaev interaction
    xcoordinate = div(b.s1 - 1, Ny) + 1
    if iseven(xcoordinate)
      os .+= -Jz, "Sz", b.s1, "Sz", b.s2
      zbond[] += 1
      # @info "Added Sz-Sz bond" s1=b.s1 s2=b.s2
    else
      if abs(b.s1 - b.s2) == Ny
        os .+= -Jx, "Sx", b.s1, "Sx", b.s2
        xbond[] += 1
        # @info "Added Sx-Sx bond" s1=b.s1 s2=b.s2
      else
        os .+= -0.25 * Jy, "S+", b.s1, "S-", b.s2
        os .+= -0.25 * Jy, "S-", b.s1, "S+", b.s2
        os .+=  0.25 * Jy, "S+", b.s1, "S+", b.s2
        os .+=  0.25 * Jy, "S-", b.s1, "S-", b.s2
        ybond[] += 1 
        # @info "Added Sy-Sy bond" s1=b.s1 s2=b.s2
      end
    end
  end
  
  # Validate the number of bonds in the Hamiltonian
  total_bonds = xbond[] + ybond[] + zbond[]
  @info "Two-body interaction count summary" xbond=xbond[] ybond=ybond[] zbond=zbond[] total=total_bonds expected=number_of_bonds
  if total_bonds != number_of_bonds
    error("Bond count mismatch! Expected: $number_of_bonds, Found: $total_bonds (x=$( xbond[]), y=$(ybond[]), z=$(zbond[]))")
  end
  println("\n")


  # Set up three-spin interactions in the Hamiltonian
  println("\nSetting up three-spin interactions...")
  count_wedge = Ref(0)
  for w in wedge
    x_coordinate = div(w.s2 - 1, Ny) + 1
    y_coordinate = mod(w.s2 - 1, Ny) + 1

    if abs(w.s3 - w.s1) == 2 * Ny
      if iseven(x_coordinate)
        os .+=  0.5im * kappa, "Sx", w.s1, "S-", w.s2, "Sz", w.s3 
        os .+= -0.5im * kappa, "Sx", w.s1, "S+", w.s2, "Sz", w.s3
        @info "Added three-spin term" term = ("Sx", w.s1, "Sy", w.s2, "Sz", w.s3)
      else
        os .+=  0.5im * kappa, "Sz", w.s1, "S-", w.s2, "Sx", w.s3
        os .+= -0.5im * kappa, "Sz", w.s1, "S+", w.s2, "Sx", w.s3
        @info "Added three-spin term" term = ("Sz", w.s1, "Sy", w.s2, "Sx", w.s3)
      end
      count_wedge[] += 1
    end


    if abs(w.s3 - w.s1) == 1
      if isodd(x_coordinate)
        os .+=  0.5im * kappa, "S-", w.s1, "Sz", w.s2, "Sx", w.s3
        os .+= -0.5im * kappa, "S+", w.s1, "Sz", w.s2, "Sx", w.s3
        @info "Added three-spin term" term = ("Sy", w.s1, "Sz", w.s2, "Sx", w.s3)
      else
        os .+=  0.5im * kappa, "Sx", w.s1, "Sz", w.s2, "S-", w.s3
        os .+= -0.5im * kappa, "Sx", w.s1, "Sz", w.s2, "S+", w.s3
        @info "Added three-spin term" term = ("Sx", w.s1, "Sz", w.s2, "Sy", w.s3)
      end
      count_wedge[] += 1
    end


    if abs(w.s3 - w.s1) == 2
      if iseven(x_coordinate)
        os .+=  0.5im * kappa, "S-", w.s1, "Sz", w.s2, "Sx", w.s3
        os .+= -0.5im * kappa, "S+", w.s1, "Sz", w.s2, "Sx", w.s3
        @info "Added three-spin term" term = ("Sy", w.s1, "Sz", w.s2, "Sx", w.s3)
      else
        os .+=  0.5im * kappa, "Sx", w.s1, "Sz", w.s2, "S-", w.s3
        os .+= -0.5im * kappa, "Sx", w.s1, "Sz", w.s2, "S+", w.s3
        @info "Added three-spin term" term = ("Sx", w.s1, "Sz", w.s2, "Sy", w.s3)
      end
      count_wedge[] += 1
    end


    if abs(w.s3 - w.s1) == 2 * Ny - 1
      if isodd(x_coordinate)
        os .+=  0.5im * kappa, "Sz", w.s1, "Sx", w.s2, "S-", w.s3
        os .+= -0.5im * kappa, "Sz", w.s1, "Sx", w.s2, "S+", w.s3
        @info "Added three-spin term" term = ("Sz", w.s1, "Sx", w.s2, "Sy", w.s3)
      else
        os .+=  0.5im * kappa, "S-", w.s1, "Sx", w.s2, "Sz", w.s3
        os .+= -0.5im * kappa, "S+", w.s1, "Sx", w.s2, "Sz", w.s3
        @info "Added three-spin term" term = ("Sy", w.s1, "Sx", w.s2, "Sz", w.s3)
      end
      count_wedge[] += 1
    end


    if abs(w.s3 - w.s1) == 3 * Ny - 1 
      if iseven(x_coordinate)
        os .+=  0.5im * kappa, "S-", w.s1, "Sx", w.s2, "Sz", w.s3
        os .+= -0.5im * kappa, "S+", w.s1, "Sx", w.s2, "Sz", w.s3
        @info "Added three-spin term" term = ("Sy", w.s1, "Sx", w.s2, "Sz", w.s3)
      else
        os .+=  0.5im * kappa, "Sz", w.s1, "Sx", w.s2, "S-", w.s3
        os .+= -0.5im * kappa, "Sz", w.s1, "Sx", w.s2, "S+", w.s3
        @info "Added three-spin term" term = ("Sz", w.s1, "Sx", w.s2, "Sy", w.s3)
      end
      count_wedge[] += 1
    end

  end
  
  # Validate the number of three-spin interaction terms
  if count_wedge[] != 3 * N - 4 * Ny
    error("The number of three-spin interaction terms is not correct! Expected: $(3 * N - 4 * Ny), Found: $(count_wedge[])")
  end
  println("\n")
  
  
  # Add chemical potential on edges to prevent the hole being trapped on the edges
  println("\nSetting up chemical potential on edge sites...")
  if abs(P) > 1e-8
    for site in edge_sites
      os .+= -abs(P), "Ntot", site
      @info "Added edge chemical potential" site=site P=-abs(P)
    end
  end
  println("\n")
  

  #**************************************************************************************************************** 
  """Add perturbation terms in the Hamiltonian"""

  println("\nSetting up loop perturbations on the edges of the cylinder...")
  # Set up the loop operators and loop indices 
  loop_operator = ["Sz", "Sz", "Sz", "Sz", "Sz", "Sz"]      # Hard-coded for width-3 cylinders with zigzag geometry
  
  # Generate loop indices for the left edge and right edge of the cylinder
  loop_indices_left  = hcat([collect((idx - 1) * 2 * Ny + 1 : idx * 2 * Ny) for idx in 1:3]...)'
  loop_indices_right = hcat([N + 1 .- reverse(loop_indices_left[idx, :]) for idx in 1:3]...)'
  nloops = size(loop_indices_left, 1)
  # @info loop_indices_left, loop_indices_right 


  if abs(lambda₁) > 1e-8 && abs(lambda₂) > 1e-8
    for idx in 1 : 3
      @info "Adding loop perturbation terms" loop_index=idx left_indices=loop_indices_left[idx, :] right_indices=loop_indices_right[idx, :]
      os .+= -1.0 * lambda₁, loop_operator[1], loop_indices_left[idx, 1], 
            loop_operator[2], loop_indices_left[idx, 2], 
            loop_operator[3], loop_indices_left[idx, 3], 
            loop_operator[4], loop_indices_left[idx, 4], 
            loop_operator[5], loop_indices_left[idx, 5], 
            loop_operator[6], loop_indices_left[idx, 6]

      os .+= -1.0 * lambda₂, loop_operator[1], loop_indices_right[idx, 1], 
            loop_operator[2], loop_indices_right[idx, 2], 
            loop_operator[3], loop_indices_right[idx, 3], 
            loop_operator[4], loop_indices_right[idx, 4], 
            loop_operator[5], loop_indices_right[idx, 5], 
            loop_operator[6], loop_indices_right[idx, 6]
    end
  end

  println("\n")
  # # Generate the plaquette indices for all the plaquettes in the cylinder
  # # plaquette_operator = Vector{String}(["iY", "Z", "X", "X", "Z", "iY"])
  # # plaquette_operator = Vector{String}(["Z", "iY", "X", "X", "iY", "Z"]) 
  # plaquette_operator = [
  #   ["Sz", "S+", "Sx", "Sx", "S+", "Sz"],
  #   ["Sz", "S+", "Sx", "Sx", "S-", "Sz"],
  #   ["Sz", "S-", "Sx", "Sx", "S-", "Sz"],
  #   ["Sz", "S-", "Sx", "Sx", "S+", "Sz"]
  # ]
  # plaquette_indices = PlaquetteList_RightTwist(Nx_unit, Ny_unit, "rings", false)
  # @show plaquette_indices
  #**************************************************************************************************************** 
  #****************************************************************************************************************  
 
  



  
  
  #**************************************************************************************************************** 
  """
    Setting up the initial state and the Hamiltonian as an MPO for the DMRG simulation
  """
  
  println(header)
  println("Initialize the starting MPS for the DMRG simulation")
  println(header, "\n")


  
  # """Read in the wavefunction from a file and start the DMRG process"""
  # println("Read in the wavefunction from a file and start the DMRG process, "\n")
  # file = h5open("data/input.h5", "r")
  # ψ₀ = read(file, "psi", MPS)
  # sites = siteinds(ψ₀)

  

  """Set up the initial wavefunction as a random MPS"""
  println("Set up the initial wavefunction as a random MPS", "\n")
  # Increase the maximum dimension of Krylov space used to locally solve the eigenvalues problem
  # sites = siteinds("tJ", N; conserve_qns=false)
  sites = siteinds("tJ", N; conserve_nf=true)

  # Set up the initial state of a random MPS with bond dimension 10
  # state = [isodd(n) ? "Up" : "Dn" for n in 1:N]   # Half-filling without doping
  hole_idx = div(N, 2)                              # Doping one hole at the center of the cylinder
  state = [n == hole_idx ? "Emp" : isodd(n) ? "Up" : "Dn" for n in 1:N]
  @assert count(==("Emp"), state) == 1 "Initial state must contain exactly one hole"
  
  
  """Initialize the wavefunction as a random MPS"""
  ψ₀ = randomMPS(sites, state, 10)
  

  """Intiialize the Hamiltonian as an MPO"""
  H = MPO(os, sites)
  
  # Measure one-point functions of the initial state
  Sx₀ = expect(ψ₀, "Sx", sites = 1 : N)
  Splus₀  = expect(ψ₀, "S+", sites = 1 : N)
  Sminus₀ = expect(ψ₀, "S-", sites = 1 : N)
  Sy₀ = 0.5im * (Sminus₀ - Splus₀)
  Sz₀ = expect(ψ₀, "Sz", sites = 1 : N)
  
  n₀ = expect(ψ₀, "Ntot", sites = 1 : N)
  # Validate if the initial state is properly doped before running the DMRG simulation
  if abs(N - sum(n₀) - 1) > 1e-6
    error("The system is not properly doped!")
  end 
  #**************************************************************************************************************** 
  #****************************************************************************************************************  
 
  

  #**************************************************************************************************************** 
  """
    Running DMRG simulation to obtain the ground-state wavefunction of the Kitaev model
  """

  println(header)
  println("Running DMRG simulation to obtain the ground-state wavefunction of the Kitaev model")
  println(header, "\n")

  # Set up the parameters including bond dimensions and truncation error
  nsweeps = 2
  maxdim  = [20, 100, 500, 1500, 3500]
  cutoff  = [1E-10]
  eigsolve_krylovdim = 50
  
  # # Add noise terms to prevent DMRG from getting stuck in a local minimum
  # noise = [1E-6, 1E-7, 0.0]

  # Construct a custom observer and stop the DMRG calculation early if criteria are met
  # custom_observer = DMRGObserver(; energy_tol=1E-9, minsweeps=2, energy_type=Float64)
  custom_observer = CustomObserver()
  @show custom_observer.etolerance
  @show custom_observer.minsweeps
  @timeit time_machine "dmrg simulation" begin
    energy, ψ = dmrg(H, ψ₀; nsweeps, maxdim, cutoff, eigsolve_krylovdim, observer = custom_observer)
  end
  println("\n")
  #**************************************************************************************************************** 
  #****************************************************************************************************************  
 




  #**************************************************************************************************************** 
  """
    Running DMRG simulation to obtain the ground-state wavefunction of the Kitaev model
  """

  println(header)
  println("Running DMRG simulation to obtain the ground-state wavefunction of the Kitaev model")
  println(header, "\n")
  
  
  """Measure local observables (one-point functions)"""
  @timeit time_machine "one-point functions" begin
    Sx = expect(ψ, "Sx", sites = 1 : N)
    Splus  = expect(ψ, "S+", sites = 1 : N)
    Sminus = expect(ψ, "S-", sites = 1 : N)
    Sy = 0.5im * (Sminus - Splus)
    Sz = expect(ψ, "Sz", sites = 1 : N)
    n = expect(ψ, "Ntot", sites = 1 : N)
  end

  # Validate electron density after DMRG simulation
  println("\nThe electron density at each site is:")
  @show n
  println("\n")
  
  total_electrons = sum(n)
  n_holes = N - total_electrons
  if abs(n_holes - 1) > 1e-6
    error("Doping validation failed! Expected exactly 1 hole, found $(n_holes) holes (total electrons: $(total_electrons))")
  end
  

  
  """Measure spin-spin correlation functions (two-point functions)"""
  @timeit time_machine "two-point functions" begin
    xxcorr = correlation_matrix(ψ, "Sx", "Sx", sites = 1 : N)
    zzcorr = correlation_matrix(ψ, "Sz", "Sz", sites = 1 : N)
    # yycorr = correlation_matrix(ψ, "Sy", "Sy", sites = 1 : N)
  end


  
  """Measure loop operators"""
  @timeit time_machine "loop operators" begin
    loop_indices = [collect((idx - 1) * 2 * Ny + 1 : idx * 2 * Ny) for idx in 1:Nx_unit]
    yloop_eigenvalues = zeros(Float64, Nx_unit)

    for (idx, tmp_idx) in enumerate(loop_indices)
      os_wl = OpSum()
      os_wl += loop_operator[1], tmp_idx[1],
               loop_operator[2], tmp_idx[2],
               loop_operator[3], tmp_idx[3],
               loop_operator[4], tmp_idx[4],
               loop_operator[5], tmp_idx[5],
               loop_operator[6], tmp_idx[6]
      Wl = MPO(os_wl, sites)

      # Normalize by 2^6 to convert from spin operators to Pauli operators
      yloop_eigenvalues[idx] = 2^6 * real(inner(ψ', Wl, ψ))
    end
  end
  @info "Loop operator eigenvalues" yloop_eigenvalues



  # # Measure the eigenvalues of plaquette operators
  # # Decompose the plaquette operators into four terms for tJ type of sites
  # @timeit time_machine "plaquette operators" begin
  #   nplaquettes = size(plaquette_indices, 1)
  #   plaquette_eigenvalues = zeros(Float64, nplaquettes)
    
  #   for idx1 in 1:nplaquettes
  #     indices  = plaquette_indices[idx1, :]
      
  #     for idx2 in 1:4
  #       operator = plaquette_operator[idx2]
  #       # @show operator, indices
  #       os_w = OpSum()
  #       os_w += operator[1], indices[1], 
  #         operator[2], indices[2], 
  #         operator[3], indices[3], 
  #         operator[4], indices[4], 
  #         operator[5], indices[5], 
  #         operator[6], indices[6]
  #       W = MPO(os_w, sites)
  #       # @show (-1.0)^idx2 * real(inner(ψ', W, ψ)) * 2^6
  #       plaquette_eigenvalues[idx1] += (-1.0)^idx2 * real(inner(ψ', W, ψ))
  #     end
  #     plaquette_eigenvalues[idx1] *= 2^6 / 4
  #     # @show inner(ψ', W, ψ) / inner(ψ', ψ)
  #   end
  # end
  # # @show plaquette_eigenvalues


  # # Set up and measure the eigenvalues of the order parameter(s)
  # # Define the central sites, excluding a margin of 2*Ny sites from both boundaries
  # centers = collect((2 * Ny + 2):(N - 2 * Ny - 1))
  # @info "Central sites selected for measurement" centers=centers
  
  # order_loops = []
  # for center in centers
  #   tmp_x = div(center - 1, Ny) + 1
  #   tmp_y = mod(center - 1, Ny) + 1
  #   tmp_loop = []

  #   if isodd(tmp_x)
  #     if tmp_y == 1
  #       append!(tmp_loop, [
  #         center + 1,
  #         center + Ny,
  #         center + 2 * Ny,
  #         center + 2 * Ny - 1,
  #         center + Ny - 1,
  #         center - 1,
  #         center - Ny - 1,
  #         center - 2 * Ny - 1,
  #         center - 2 * Ny,
  #         center - Ny,
  #         center - 2 * Ny + 1, 
  #         center - Ny + 1
  #       ])
  #     elseif tmp_y == Ny
  #       append!(tmp_loop, [
  #         center + Ny + 1,
  #         center + Ny,
  #         center + 2 * Ny,
  #         center + 2 * Ny + 2,
  #         center + 2 * Ny - 1,
  #         center + Ny - 1,
  #         center - 1,
  #         center - Ny - 1,
  #         center - 2 * Ny,
  #         center - Ny,
  #         center - Ny + 1,
  #         center + 1
  #       ])
  #     else
  #       # Construct the loop for odd x and tmp_y ≠ 1 and tmp_y ≠ Ny
  #       append!(tmp_loop, [
  #         center + 1,
  #         center + Ny,
  #         center + 2 * Ny,
  #         center + 2 * Ny + 2,
  #         center + 2 * Ny - 1,
  #         center + Ny - 1,
  #         center - 1,
  #         center - Ny - 1,
  #         center - 2 * Ny,
  #         center - Ny,
  #         center - 2 * Ny + 1,
  #         center - Ny + 1
  #       ])
  #     end
  #   else
  #     if tmp_y == 1
  #       # Construct the loop for even x and tmp_y == 1
  #       append!(tmp_loop, [
  #         center - Ny - 1,
  #         center - Ny,
  #         center - 2 * Ny,
  #         center - 3 * Ny + 1,
  #         center - 2 * Ny + 1,
  #         center - Ny + 1,
  #         center + 1,
  #         center + Ny + 1,
  #         center + 2 * Ny,
  #         center + Ny,
  #         center + Ny - 1,
  #         center - 1
  #       ])
  #     elseif tmp_y == Ny
  #       # Construct the loop for even x and tmp_y == Ny
  #       append!(tmp_loop, [
  #         center - 1,
  #         center - Ny,
  #         center - 2 * Ny,
  #         center - 2 * Ny + 1,
  #         center - Ny + 1,
  #         center + 1,
  #         center + Ny + 1,
  #         center + 2 * Ny + 1,
  #         center + 2 * Ny,
  #         center + Ny,
  #         center + 2 * Ny - 1,
  #         center + Ny - 1
  #       ]) 
  #     else
  #       # Construct the loop for even x and tmp_y != 1 and tmp_y != Ny
  #       append!(tmp_loop, [
  #         center - 1,
  #         center - Ny,
  #         center - 2 * Ny,
  #         center - 3 * Ny + 1,
  #         center - 2 * Ny + 1,
  #         center - Ny + 1,
  #         center + 1,
  #         center + Ny + 1,
  #         center + 2 * Ny,
  #         center + Ny,
  #         center + 2 * Ny - 1,
  #         center + Ny - 1
  #       ])
  #     end
  #   end
  #   push!(order_loops, tmp_loop)
  # end
  
  # for idx in eachindex(order_loops)
  #   @show centers[idx], order_loops[idx]
  # end

  # function configure_signs(input_string)
  #   return [(-1.0)^count(==( "S-" ), row) for row in input_string]
  # end

  # order_string = [["Sx", "Sx", "Sx", "Sz", "S+", "S+", "S+", "Sx", "Sz", "Sz", "Sz", "S+"], 
  # ["Sx", "Sx", "Sx", "Sz", "S+", "S+", "S+", "Sx", "Sz", "Sz", "Sz", "S-"], 
  # ["Sx", "Sx", "Sx", "Sz", "S+", "S+", "S-", "Sx", "Sz", "Sz", "Sz", "S+"],
  # ["Sx", "Sx", "Sx", "Sz", "S+", "S+", "S-", "Sx", "Sz", "Sz", "Sz", "S-"], 
  # ["Sx", "Sx", "Sx", "Sz", "S+", "S-", "S+", "Sx", "Sz", "Sz", "Sz", "S+"], 
  # ["Sx", "Sx", "Sx", "Sz", "S+", "S-", "S+", "Sx", "Sz", "Sz", "Sz", "S-"], 
  # ["Sx", "Sx", "Sx", "Sz", "S+", "S-", "S-", "Sx", "Sz", "Sz", "Sz", "S+"], 
  # ["Sx", "Sx", "Sx", "Sz", "S+", "S-", "S-", "Sx", "Sz", "Sz", "Sz", "S-"], 
  # ["Sx", "Sx", "Sx", "Sz", "S-", "S+", "S+", "Sx", "Sz", "Sz", "Sz", "S+"], 
  # ["Sx", "Sx", "Sx", "Sz", "S-", "S+", "S+", "Sx", "Sz", "Sz", "Sz", "S-"], 
  # ["Sx", "Sx", "Sx", "Sz", "S-", "S+", "S-", "Sx", "Sz", "Sz", "Sz", "S+"],
  # ["Sx", "Sx", "Sx", "Sz", "S-", "S+", "S-", "Sx", "Sz", "Sz", "Sz", "S-"], 
  # ["Sx", "Sx", "Sx", "Sz", "S-", "S-", "S+", "Sx", "Sz", "Sz", "Sz", "S+"], 
  # ["Sx", "Sx", "Sx", "Sz", "S-", "S-", "S+", "Sx", "Sz", "Sz", "Sz", "S-"], 
  # ["Sx", "Sx", "Sx", "Sz", "S-", "S-", "S-", "Sx", "Sz", "Sz", "Sz", "S+"], 
  # ["Sx", "Sx", "Sx", "Sz", "S-", "S-", "S-", "Sx", "Sz", "Sz", "Sz", "S-"]]
  
  # # Reference sign structure for the order parameter 
  # # sign = [1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0]
  # sign = configure_signs(order_string)
  # @show sign

  # @timeit time_machine "order parameter(s)" begin
  #   order_parameter = zeros(Float64, length(order_loops))
  #   order₀ = zeros(Float64, length(order_loops))

  #   for idx1 in 1 : size(order_loops)[1]
  #     loop = order_loops[idx1]
  #     for idx2 in 1 : size(order_string)[1]
  #       operator = order_string[idx2]
  #       os_order = OpSum()
  #       os_order +=  "Ntot", centers[idx1], 
  #         operator[1], loop[1], 
  #         operator[2], loop[2], 
  #         operator[3], loop[3], 
  #         operator[4], loop[4], 
  #         operator[5], loop[5], 
  #         operator[6], loop[6],
  #         operator[7], loop[7],
  #         operator[8], loop[8],
  #         operator[9], loop[9],
  #         operator[10], loop[10],
  #         operator[11], loop[11],
  #         operator[12], loop[12]
  #       W_order = MPO(os_order, sites)

  #       os_order_identity = OpSum()
  #       os_order_identity += operator[1], loop[1], 
  #         operator[2], loop[2], 
  #         operator[3], loop[3], 
  #         operator[4], loop[4], 
  #         operator[5], loop[5], 
  #         operator[6], loop[6],
  #         operator[7], loop[7],
  #         operator[8], loop[8],
  #         operator[9], loop[9],
  #         operator[10], loop[10],
  #         operator[11], loop[11],
  #         operator[12], loop[12]
  #       W_order_identity = MPO(os_order_identity, sites)

  #       order_parameter[idx1] += (1/2)^4 * 2^12 * sign[idx2] * (real(inner(ψ', W_order_identity, ψ)) - real(inner(ψ', W_order, ψ)))
  #       order₀[idx1] += (1/2)^4 * 2^12 * sign[idx2] * real(inner(ψ', W_order, ψ))
  #     end
  #   end
  # end

  # for idx in eachindex(order_parameter)
  #   @show order_parameter[idx]
  # end

  # for idx in eachindex(order₀)
  #   @show order₀[idx]
  # end

  # # # Print out useful information of physical quantities
  # # println("")
  # # println("Visualize the optimization history of the energy and bond dimensions:")
  # # @show custom_observer.ehistory_full
  # # @show custom_observer.ehistory
  # # @show custom_observer.chi
  # # # @show number_of_bonds, energy / number_of_bonds
  # # # @show N, energy / N
  # # println("")

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
  # @show plaquette_eigenvalues
  # println("")

  # print("")
  # println("Eigenvalues of the loop operator(s):")
  # @show yloop_eigenvalues
  # @show yloop_eigenvalues_symmetric
  # println("")

  # # # println("")
  # # # println("Eigenvalues of the twelve-point correlator near the first vacancy:")
  # # # @show order_parameter
  # # # println("")

  # # # @show time_machine
  # # h5open("/pscratch/sd/x/xiaobo23/TensorNetworks/non_abelian_anyons/t-Kitaev/FM/W3/Lx10/perturbation/WL+1_WR+1/kappa-0.4/data/2d_tK_Lx$(Nx_unit)_Ly$(Ny_unit)_t$(t).h5", "w") do file
  # #   write(file, "psi", ψ)
  # #   write(file, "NormalizedE0", energy / number_of_bonds)
  # #   write(file, "E0", energy)
  # #   write(file, "E0variance", variance)
  # #   write(file, "Ehist", custom_observer.ehistory)
  # #   write(file, "Bond", custom_observer.chi)
  # #   # write(file, "Entropy", SvN)
  # #   write(file, "Sx0", Sx₀)
  # #   write(file, "Sx",  Sx)
  # #   write(file, "Cxx", xxcorr)
  # #   write(file, "Sy0", Sy₀)
  # #   write(file, "Sy", Sy)
  # #   # # write(file, "Cyy", yycorr)
  # #   write(file, "Sz0", Sz₀)
  # #   write(file, "Sz",  Sz)
  # #   write(file, "Czz", zzcorr)
  # #   write(file, "N0", n₀)
  # #   write(file, "N", n)
  # #   write(file, "Plaquette", plaquette_eigenvalues)
  # #   write(file, "Loop", yloop_eigenvalues)
  # #   write(file, "LoopSymmetric", yloop_eigenvalues_symmetric)
  # #   write(file, "OrderParameter", order_parameter)
  # # end

  
  return
end