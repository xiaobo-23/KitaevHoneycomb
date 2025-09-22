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
const number_of_wedges = 3 * N - 4 * Ny

# Timing and profiling
const time_machine = TimerOutput()

let
  # Set up the parameters in the Hamiltonian
  Jx, Jy, Jz = 1.0, 1.0, 1.0        # Kitaev interaction strengths
  kappa=-0.4                        # Three-spin interaction strength
  t=0                               # Electron hopping amplitude
  P=-10.0                           # Chemical potential on the edges of the cylinder
  λ₁, λ₂ = 256.0, 256.0             # Perturbation strengths for the loop operators on both edges of the cylinder
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
    # @show length(lattice)
  else
    lattice = honeycomb_lattice_armchair(Nx, Ny; yperiodic=true)
    if length(lattice) != number_of_bonds
      error("The number of bonds in the lattice does not match the expected number of bonds!")
    end
    # @show length(lattice)
  end 

  # Construct the wedges to set up three-spin interactions 
  wedge = honeycomb_armchair_wedge(Nx, Ny; yperiodic=true)
  if length(wedge) != number_of_wedges
    error("The number of wedges in the lattice does not match the expected number of wedges!")
  end
  # @show wedge

  # Identify edge sites: first 6*Ny and last 6*Ny sites of the lattice
  edge_sites = Set(1 : 6*Ny) ∪ Set(N - 6*Ny + 1 : N)
  # @show length(edge_sites), edge_sites  

  #***************************************************************************************************************
  #***************************************************************************************************************  
  # Construct the Hamiltonian using OpSum

  # Construct the two-body interaction temrs in the Kitaev Hamiltonian
  os = OpSum()

  # Initialize counters for the number of bonds in each direction
  bond_counts = Dict("xbond" => 0, "ybond" => 0, "zbond" => 0)
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
      bond_counts["zbond"] += 1
      @info "Added Sz-Sz bond" s1=b.s1 s2=b.s2
    end

    # Set up the Sx-Sx and Sy-Sy bond interactions
    if (isodd(x_coordinate) && isodd(b.s1) && isodd(b.s2)) || (iseven(x_coordinate) && iseven(b.s1) && iseven(b.s2))
      os .+= -Jx, "Sx", b.s1, "Sx", b.s2
      bond_counts["xbond"] += 1
      @info "Added Sx-Sx bond" s1=b.s1 s2=b.s2
    elseif (isodd(x_coordinate) && iseven(b.s1) && iseven(b.s2)) || (iseven(x_coordinate) && isodd(b.s1) && isodd(b.s2))
      os .+= -0.25 * Jy, "S+", b.s1, "S-", b.s2
      os .+= -0.25 * Jy, "S-", b.s1, "S+", b.s2
      os .+=  0.25 * Jy, "S+", b.s1, "S+", b.s2
      os .+=  0.25 * Jy, "S-", b.s1, "S-", b.s2
      bond_counts["ybond"] += 1
      @info "Added Sy-Sy bond" s1=b.s1 s2=b.s2
    end
  end
  
  total_bonds = bond_counts["xbond"] + bond_counts["ybond"] + bond_counts["zbond"]
  if total_bonds != number_of_bonds
    error("Mismatch in the number of bonds: expected $number_of_bonds, but found $total_bonds.")
  end
  @info "Bond counts by type" xbond=bond_counts["xbond"] ybond=bond_counts["ybond"] zbond=bond_counts["zbond"]


  # Set up the three-spin interaction terms in the Kitaev Hamiltonian
  edge_counts = Dict("horizontal" => 0, "vertical" => 0)
  for w in wedge
    # @show w.s1, w.s2, w.s3
    x_coordinate = div(w.s2 - 1, Ny) + 1
    y_coordinate = mod(w.s2 - 1, Ny) + 1
    
    
    # Set up the horizontal three-spin interaction terms
    if abs(w.s1 - w.s2) == abs(w.s2 - w.s3) == Ny_unit
      if (isodd(x_coordinate) && isodd(y_coordinate)) || (iseven(x_coordinate) && iseven(y_coordinate))
        os .+= -0.5im * kappa, "S+", w.s1, "Sz", w.s2, "Sx", w.s3
        os .+=  0.5im * kappa, "S-", w.s1, "Sz", w.s2, "Sx", w.s3
        @info "Added three-spin interaction" term = ("Sy", w.s1, "Sz", w.s2, "Sx", w.s3)
      elseif (isodd(x_coordinate) && iseven(y_coordinate)) || (iseven(x_coordinate) && isodd(y_coordinate))
        # os .+= kappa, "Sx", w.s1, "Sz", w.s2, "Sy", w.s3
        os .+= -0.5im * kappa, "Sx", w.s1, "Sz", w.s2, "S+", w.s3
        os .+=  0.5im * kappa, "Sx", w.s1, "Sz", w.s2, "S-", w.s3
        @info "Added three-spin interaction" term = ("Sx", w.s1, "Sz", w.s2, "Sy", w.s3)
      end
      edge_counts["horizontal"] += 1
    end



    # Set up the vertical three-spin interaction terms through periodic boundary condition along the y direction
    if abs(w.s1 - w.s2) == Ny - 1
      if (y_coordinate == 1 && w.s3 < w.s2) || (y_coordinate == Ny && w.s2 < w.s3)
        os .+= -0.5im * kappa, "Sz", w.s1, "S+", w.s2, "Sx", w.s3
        os .+=  0.5im * kappa, "Sz", w.s1, "S-", w.s2, "Sx", w.s3
        @info "Added three-spin interaction" term = ("Sz", w.s1, "Sy", w.s2, "Sx", w.s3)
      elseif (y_coordinate == 1 && w.s2 < w.s3) || (y_coordinate == Ny && w.s3 < w.s2)
        os .+= -0.5im * kappa, "Sz", w.s1, "Sx", w.s2, "S+", w.s3
        os .+=  0.5im * kappa, "Sz", w.s1, "Sx", w.s2, "S-", w.s3
        @info "Added three-spin interaction" term = ("Sz", w.s1, "Sx", w.s2, "Sy", w.s3)
      end
      edge_counts["vertical"] += 1
    end


    # Set up the vertical three-spin interaction terms within the bulk of the cylinder 
    if abs(w.s2 - w.s1) == 1
      if (isodd(x_coordinate) && isodd(y_coordinate)) || (iseven(x_coordinate) && iseven(y_coordinate))
        if w.s2 > w.s3
          os .+= -0.5im * kappa, "Sz", w.s1, "Sx", w.s2, "S+", w.s3 
          os .+=  0.5im * kappa, "Sz", w.s1, "Sx", w.s2, "S-", w.s3
          @info "Added three-spin interaction" term = ("Sz", w.s1, "Sx", w.s2, "Sy", w.s3)
        else
          os .+= -0.5im * kappa, "Sz", w.s1, "S+", w.s2, "Sx", w.s3
          os .+=  0.5im * kappa, "Sz", w.s1, "S-", w.s2, "Sx", w.s3
          @info "Added three-spin interaction" term = ("Sz", w.s1, "Sy", w.s2, "Sx", w.s3)
        end
      elseif (isodd(x_coordinate) && iseven(y_coordinate)) || (iseven(x_coordinate) && isodd(y_coordinate))
        if w.s2 > w.s3
          os .+= -0.5im * kappa, "Sz", w.s1, "S+", w.s2, "Sx", w.s3
          os .+=  0.5im * kappa, "Sz", w.s1, "S-", w.s2, "Sx", w.s3
          @info "Added three-spin interaction" term = ("Sz", w.s1, "Sy", w.s2, "Sx", w.s3)
        else
          os .+= -0.5im * kappa, "Sz", w.s1, "Sx", w.s2, "S+", w.s3 
          os .+=  0.5im * kappa, "Sz", w.s1, "Sx", w.s2, "S-", w.s3
          @info "Added three-spin interaction" term = ("Sz", w.s1, "Sx", w.s2, "Sy", w.s3)
        end
      end
      edge_counts["vertical"] += 1
    end
  end
  
  total_edges = edge_counts["horizontal"] + edge_counts["vertical"]
  if total_edges != number_of_wedges
    error("Mismatch in the number of wedges: expected $number_of_wedges, but found $total_edges.")
  end
  @info "Wedge counts by type" horizontal=edge_counts["horizontal"] vertical=edge_counts


  # Set up the edge chemical potential walls to confine the hole in the bulk of the cylinder
  if abs(P) > 1e-8
    for site in edge_sites
      os .+= P, "Ntot", site
      @info "Added chemical potential wall" site=site potential=P
    end
  end

  # Set up loop operators along the periodic direction of the cylinder
  # loop_operator = ["iSy", "Sx", "iSy", "Sx", "iSy", "Sx", "iSy", "Sx"]  # Hard-coded for width-4 cylinders
  loop_operator = [
    ["S+", "Sx", "S+", "Sx", "S+", "Sx", "S+", "Sx"],
    ["S+", "Sx", "S+", "Sx", "S+", "Sx", "S-", "Sx"],
    ["S+", "Sx", "S+", "Sx", "S-", "Sx", "S+", "Sx"],
    ["S+", "Sx", "S+", "Sx", "S-", "Sx", "S-", "Sx"],
    ["S+", "Sx", "S-", "Sx", "S+", "Sx", "S+", "Sx"],
    ["S+", "Sx", "S-", "Sx", "S+", "Sx", "S-", "Sx"],
    ["S+", "Sx", "S-", "Sx", "S-", "Sx", "S+", "Sx"],
    ["S+", "Sx", "S-", "Sx", "S-", "Sx", "S-", "Sx"],
    ["S-", "Sx", "S+", "Sx", "S+", "Sx", "S+", "Sx"],
    ["S-", "Sx", "S+", "Sx", "S+", "Sx", "S-", "Sx"],
    ["S-", "Sx", "S+", "Sx", "S-", "Sx", "S+", "Sx"],
    ["S-", "Sx", "S+", "Sx", "S-", "Sx", "S-", "Sx"],
    ["S-", "Sx", "S-", "Sx", "S+", "Sx", "S+", "Sx"],
    ["S-", "Sx", "S-", "Sx", "S+", "Sx", "S-", "Sx"],
    ["S-", "Sx", "S-", "Sx", "S-", "Sx", "S+", "Sx"],
    ["S-", "Sx", "S-", "Sx", "S-", "Sx", "S-", "Sx"],
  ]
  loop_indices = LoopListArmchair(Nx_unit, Ny_unit, "armchair", "y")  
  nloops = size(loop_indices, 1)
  for idx in 1 : nloops
    @show idx, loop_indices[idx, :]
  end
  
  loops_signs = [(-1.0)^count(==("S-"), loop) for loop in loop_operator]
  @show loops_signs
  # for loop in loop_operator
  #   @show loop
  #   @show (-1.0)^count(==("S-"), loop)
  # end
  println("")
  println("")

  if abs(λ₁) > 1e-8 && abs(λ₂) > 1e-8
    for idx in 1 : 3
      @show "Adding loop perturbation terms for loop index: $idx, $(nloops - idx + 1)"

      for idx_op in 1 : 16
        operator = loop_operator[idx_op]

        os .+= -1.0/16 * λ₁ * loops_signs[idx_op], 
          operator[1], loop_indices[idx, 1], 
          operator[2], loop_indices[idx, 2], 
          operator[3], loop_indices[idx, 3], 
          operator[4], loop_indices[idx, 4], 
          operator[5], loop_indices[idx, 5], 
          operator[6], loop_indices[idx, 6],
          operator[7], loop_indices[idx, 7],
          operator[8], loop_indices[idx, 8]

        os .+= -1.0/16 * λ₂ * loops_signs[idx_op], 
          operator[1], loop_indices[nloops - idx + 1, 1], 
          operator[2], loop_indices[nloops - idx + 1, 2], 
          operator[3], loop_indices[nloops - idx + 1, 3], 
          operator[4], loop_indices[nloops - idx + 1, 4], 
          operator[5], loop_indices[nloops - idx + 1, 5], 
          operator[6], loop_indices[nloops - idx + 1, 6],
          operator[7], loop_indices[nloops - idx + 1, 7],
          operator[8], loop_indices[nloops - idx + 1, 8]
      end
    end
  end
  #***************************************************************************************************************
  #*************************************************************************************************************** 


  #***************************************************************************************************************
  #*************************************************************************************************************** 
  # Increase the maximum dimension of Krylov space used to locally solve the eigenvalues problem.
  sites = siteinds("tJ", N; conserve_nf=true)
  H = MPO(os, sites)

  # Initialize wavefunction as a random MPS with same quantum numbers as `state`
  # state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
  state = []
  hole_index = div(N, 2)
  for (idx, n) in enumerate(1 : N)
    if n == hole_index 
      push!(state, "Emp")
    else
      if isodd(n)
        push!(state, "Up")
      else
        push!(state, "Dn")
      end
    end
  end
  if count(==("Emp"), state) != 1
    error("The system is not proper doped with one hole!")
  end
  println(repeat("#", 100))
  println("Initial state used in DMRG simulation:")
  @show state
  println(repeat("#", 100))
  println("")
  println("")
  ψ₀ = randomMPS(sites, state, 8)
  

  # Set up the parameters including bond dimensions and truncation error
  nsweeps = 1
  maxdim  = [20, 100, 500, 800, 1000, 1500, 5000]
  cutoff  = [1E-10]
  eigsolve_krylovdim = 100
  
  # Add noise terms to prevent DMRG from getting stuck in a local minimum
  # noise = [1E-6, 1E-7, 1E-8, 0.0]
  #***************************************************************************************************************
  #*************************************************************************************************************** 

  #***************************************************************************************************************
  #*************************************************************************************************************** 
  # Measure local observables (one-point functions) before starting the DMRG simulation
  Sx₀ = expect(ψ₀, "Sx", sites = 1 : N)
  Splus₀ = expect(ψ₀, "S+", sites = 1 : N)
  Sminus₀ = expect(ψ₀, "S-", sites = 1 : N)
  Sy₀ = -0.5im * (Splus₀ - Sminus₀)
  Sz₀ = expect(ψ₀, "Sz", sites = 1 : N)

  n₀ = expect(ψ₀, "Ntot", sites = 1 : N)
  if abs(N - sum(n₀) - 1) > 1e-8
    error("The initial state does not have the correct number of electrons!")
  end
  println(repeat("#", 100))
  println("Initial electron density before DMRG simulation:")
  println("n₀ = $n₀")
  println(repeat("#", 100))
  #***************************************************************************************************************
  #*************************************************************************************************************** 
  

  #***************************************************************************************************************
  #*************************************************************************************************************** 
  # Run the DMRG optimization to the find the ground state of the Kitaev Hamiltonian
  # Construct a custom observer and stop the DMRG calculation early if needed 
  # custom_observer = DMRGObserver(; energy_tol=1E-9, minsweeps=2, energy_type=Float64)
  custom_observer = CustomObserver()
  @show custom_observer.etolerance
  @show custom_observer.minsweeps
  @timeit time_machine "dmrg simulation" begin
    energy, ψ = dmrg(H, ψ₀; nsweeps, maxdim, cutoff, eigsolve_krylovdim, observer = custom_observer)
  end
  #***************************************************************************************************************
  #*************************************************************************************************************** 
  
  
  #***************************************************************************************************************
  #*************************************************************************************************************** 
  # Measure local observables (one-point functions) after finish the DMRG simulation
  @timeit time_machine "one-point functions" begin
    Sx = expect(ψ, "Sx", sites = 1 : N)
    Splus = expect(ψ, "S+", sites = 1 : N)
    Sminus = expect(ψ, "S-", sites = 1 : N)
    Sy = -0.5im * (Splus - Sminus)
    Sz = expect(ψ, "Sz", sites = 1 : N)
    
    n = expect(ψ, "Ntot", sites = 1 : N)
    if abs(N - sum(n) - 1) > 1e-8
      error("The optimized state does not have the correct number of electrons!")
    end
    println(repeat("#", 100))
    println("Electron density computed based on the optimized wavefunction: ")
    println("n = $n")
    println(repeat("#", 100))
  end
  #***************************************************************************************************************
  #*************************************************************************************************************** 

  
  
  #***************************************************************************************************************
  #*************************************************************************************************************** 
  # Measure two-point correlation functions based on the optimized ground state wavefunction
  @timeit time_machine "two-point functions" begin
    xxcorr = correlation_matrix(ψ, "Sx", "Sx", sites = 1 : N)
    yyplusplus = correlation_matrix(ψ, "S+", "S+", sites = 1 : N)
    yyplusminus = correlation_matrix(ψ, "S+", "S-", sites = 1 : N)
    yyminusplus = correlation_matrix(ψ, "S-", "S+", sites = 1 : N)
    yyminusminus = correlation_matrix(ψ, "S-", "S-", sites = 1 : N)
    yycorr = -0.25 * (yyplusplus - yyplusminus - yyminusplus + yyminusminus)
    zzcorr = correlation_matrix(ψ, "Sz", "Sz", sites = 1 : N)
  end
  #***************************************************************************************************************
  #***************************************************************************************************************
  
  
  #***************************************************************************************************************
  #*************************************************************************************************************** 
  # Set up plaquette operators for each hexagonal plaquette
  # plaquette_operator = ["iSy", "Sz", "Sx", "Sx", "Sz", "iSy"]
  plaquette = [
    ["S+", "Sz", "Sx", "Sx", "Sz", "S+"],
    ["S+", "Sz", "Sx", "Sx", "Sz", "S-"],
    ["S-", "Sz", "Sx", "Sx", "Sz", "S+"],
    ["S-", "Sz", "Sx", "Sx", "Sz", "S-"],  
  ]
  plaquette_indices = PlaquetteListArmchair(Nx_unit, Ny_unit, "armchair", false)
  nplaquettes = size(plaquette_indices, 1)
  for idx in 1 : nplaquettes
    @show idx, plaquette_indices[idx, :]
  end
  plaquette_signs = [(-1.0)^count(==("S-"), plaq) for plaq in plaquette]
  @show plaquette_signs
  println("")
  println("")


  # Compute the expectation values of plaquette operators
  # normalize!(ψ)
  @timeit time_machine "Plaquette Operators" begin
    plaquette_vals = zeros(Float64, nplaquettes)
    
    # Compute the eigenvalues of the plaquette operator
    for idx1 in 1 : nplaquettes
      tmp_indices = plaquette_indices[idx1, :]

      for idx2 in 1 : 4
        operator = plaquette[idx2]
        
        os_w = OpSum()
        os_w += operator[1], tmp_indices[1], 
          operator[2], tmp_indices[2], 
          operator[3], tmp_indices[3], 
          operator[4], tmp_indices[4], 
          operator[5], tmp_indices[5], 
          operator[6], tmp_indices[6]
        W = MPO(os_w, sites)

        plaquette_vals[idx1] += -1.0 * plaquette_signs[idx2] * real(inner(ψ', W, ψ))
      end
      
      # Normalize the eigenvalues of the plaquette operator
      plaquette_vals[idx1] *= 2^6/4
    end
  end
  @show plaquette_vals
  #***************************************************************************************************************
  #*************************************************************************************************************** 

 
  #***************************************************************************************************************
  #*************************************************************************************************************** 
  # Compute the expectation values of loop operators along the periodic direction of the cylinder
  @timeit time_machine "Loop Operators" begin
    loop_vals = zeros(Float64, nloops)
    
    # Compute eigenvalues of the loop operators in the direction with PBC.
    for idx1 in 1 : nloops
      tmp_indices = loop_indices[idx1, :]

      for idx2 in 1 : 16
        operator = loop_operator[idx2]
        
        os_w = OpSum()
        os_w += operator[1], tmp_indices[1], 
          operator[2], tmp_indices[2], 
          operator[3], tmp_indices[3], 
          operator[4], tmp_indices[4], 
          operator[5], tmp_indices[5], 
          operator[6], tmp_indices[6],
          operator[7], tmp_indices[7],
          operator[8], tmp_indices[8]
        W = MPO(os_w, sites)

        loop_vals[idx1] += 2^8 * loops_signs[idx2] * real(inner(ψ', W, ψ))
      end
      
      loop_vals[idx1] *= 1/16
    end
  end
  #***************************************************************************************************************
  #*************************************************************************************************************** 


  #***************************************************************************************************************
  #*************************************************************************************************************** 
  # Compute the eigenvalues of the order parameters near vacancies
  @timeit time_machine "twelve-point correlator(s)" begin
    centers = collect((2 * Ny + 1):(N - 2 * Ny))
    @info "Central sites selected for measurement" centers=centers

    order_loops = []
    for center in centers
      tmp_x = div(center - 1, Ny) + 1
      tmp_y = mod(center - 1, Ny) + 1
      tmp_loop = []

      if isodd(tmp_x)
        if isodd(tmp_y)
          if tmp_y == 1
            tmp₁ = center + Ny - 1
            tmp₂ = center - 1
            tmp₃ = center + 2 * Ny - 1
          else
            tmp₁ = center - 1
            tmp₂ = center - Ny - 1
            tmp₃ = center + Ny - 1
          end

          append!(tmp_loop, [
            tmp₁,
            tmp₂,
            center - Ny,
            center - 2 * Ny,
            center - 2 * Ny + 1,
            center - Ny + 1,
            center + 1,
            center + Ny + 1,
            center + 2 * Ny + 1,
            center + 2 * Ny,
            center + Ny, 
            tmp₃
          ])
        elseif iseven(tmp_y)
          if tmp_y == Ny
            tmp₁ = center - Ny + 1
            tmp₂ = center + 1
            tmp₃ = center - 2 * Ny + 1
          else
            tmp₁ = center + 1
            tmp₂ = center + Ny + 1
            tmp₃ = center - Ny + 1
          end

          append!(tmp_loop, [
            tmp₁,
            tmp₂,
            center + Ny,
            center + 2 * Ny,
            center + 2 * Ny - 1,
            center + Ny - 1,
            center - 1,
            center - Ny - 1,
            center - 2 * Ny - 1,
            center - 2 * Ny,
            center - Ny,
            tmp₃
          ])
        end
      else
        if isodd(tmp_y)
          if tmp_y == 1
            tmp₁ = center + 3 * Ny - 1
            tmp₂ = center + 2 * Ny - 1
            tmp₃ = center + Ny - 1
            tmp₄ = center - 1
            tmp₅ = center - Ny - 1
          else
            tmp₁ = center + 2 * Ny - 1
            tmp₂ = center + Ny - 1
            tmp₃ = center - 1
            tmp₄ = center - Ny - 1
            tmp₅ = center - 2 * Ny - 1
          end

          append!(tmp_loop, [
            center + 1,
            center + Ny + 1,
            center + Ny,
            center + 2 * Ny,
            tmp₁,
            tmp₂,
            tmp₃,
            tmp₄,
            tmp₅,
            center - 2 * Ny,
            center - Ny,
            center - Ny + 1
          ])
        else
          if tmp_y == Ny
            tmp₁ = center - 3 * Ny + 1
            tmp₂ = center - 2 * Ny + 1
            tmp₃ = center - Ny + 1
            tmp₄ = center + 1
            tmp₅ = center + Ny + 1
          else
            tmp₁ = center - 2 * Ny + 1
            tmp₂ = center - Ny + 1
            tmp₃ = center + 1
            tmp₄ = center + Ny + 1
            tmp₅ = center + 2 * Ny + 1
          end

          append!(tmp_loop, [
            center - 1,
            center - Ny - 1,
            center - Ny,
            center - 2 * Ny,
            tmp₁,
            tmp₂,
            tmp₃,
            tmp₄,
            tmp₅,
            center + 2 * Ny,
            center + Ny,
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
    
    # order_loop = Vector{String}(["Z", "Y", "Y", "Y", "X", "Z", "Z", "Z", "Y", "X", "X", "X"])
    order_string = [["Sz", "S+", "S+", "S+", "Sz", "Sz", "Sz", "Sz", "S+", "Sx", "Sx", "Sx"],
      ["Sz", "S+", "S+", "S+", "Sz", "Sz", "Sz", "Sz", "S-", "Sx", "Sx", "Sx"],
      ["Sz", "S+", "S+", "S-", "Sz", "Sz", "Sz", "Sz", "S+", "Sx", "Sx", "Sx"],
      ["Sz", "S+", "S+", "S-", "Sz", "Sz", "Sz", "Sz", "S-", "Sx", "Sx", "Sx"],
      ["Sz", "S+", "S-", "S+", "Sz", "Sz", "Sz", "Sz", "S+", "Sx", "Sx", "Sx"],
      ["Sz", "S+", "S-", "S+", "Sz", "Sz", "Sz", "Sz", "S-", "Sx", "Sx", "Sx"],
      ["Sz", "S+", "S-", "S-", "Sz", "Sz", "Sz", "Sz", "S+", "Sx", "Sx", "Sx"],
      ["Sz", "S+", "S-", "S-", "Sz", "Sz", "Sz", "Sz", "S-", "Sx", "Sx", "Sx"],
      ["Sz", "S-", "S+", "S+", "Sz", "Sz", "Sz", "Sz", "S+", "Sx", "Sx", "Sx"],
      ["Sz", "S-", "S+", "S+", "Sz", "Sz", "Sz", "Sz", "S-", "Sx", "Sx", "Sx"],
      ["Sz", "S-", "S+", "S-", "Sz", "Sz", "Sz", "Sz", "S+", "Sx", "Sx", "Sx"],
      ["Sz", "S-", "S+", "S-", "Sz", "Sz", "Sz", "Sz", "S-", "Sx", "Sx", "Sx"],
      ["Sz", "S-", "S-", "S+", "Sz", "Sz", "Sz", "Sz", "S+", "Sx", "Sx", "Sx"],
      ["Sz", "S-", "S-", "S+", "Sz", "Sz", "Sz", "Sz", "S-", "Sx", "Sx", "Sx"],
      ["Sz", "S-", "S-", "S-", "Sz", "Sz", "Sz", "Sz", "S+", "Sx", "Sx", "Sx"],
      ["Sz", "S-", "S-", "S-", "Sz", "Sz", "Sz", "Sz", "S-", "Sx", "Sx", "Sx"],
    ]

    order_signs = configure_signs(order_string)
    @show order_signs

    if length(centers) != size(order_loops, 1)
      error("The size of the order parameters is not initialized correctly!")
    end
    order_parameter = zeros(Float64, length(centers))
    order₀ = zeros(Float64, length(centers))

    for idx1 in eachindex(order_loops)
      loop = order_loops[idx1]
      for idx2 in eachindex(order_string)
        operator = order_string[idx2]

        # println("")
        # @show idx1, loop
        # @show idx2, operator
        # println("")

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

        order_parameter[idx1] += (1/2)^4 * 2^12 * order_signs[idx2] * (real(inner(ψ', W_order_identity, ψ)) - real(inner(ψ', W_order, ψ)))
        order₀[idx1] += (1/2)^4 * 2^12 * order_signs[idx2] * real(inner(ψ', W_order, ψ))
      end
    end
  end
  #***************************************************************************************************************
  #*************************************************************************************************************** 



  #***************************************************************************************************************
  #*************************************************************************************************************** 
  # Check the variance of the energy based on the optimized ground state wavefunction
  @timeit time_machine "compaute the variance" begin
    H2 = inner(H, ψ, H, ψ)
    E₀ = inner(ψ', H, ψ)
    variance = H2 - E₀^2
  end
  #***************************************************************************************************************
  #*************************************************************************************************************** 
 

  #***************************************************************************************************************
  #*************************************************************************************************************** 
  # Print out the values of various physical quantities measured
  println(repeat("#", 100))
  println(repeat("#", 100))
  println("Output data for the entire simulation:")

  println("")
  println("Visualize the optimization history of the energy and bond dimensions:")
  @show custom_observer.ehistory_full
  @show custom_observer.ehistory
  @show custom_observer.chi
  println("")

  
  println("")
  println("Ground state energy is:")
  @show E₀
  println("Variance of the energy is $variance")
  println("")
  

  println("")
  println("Expectation values of plaquette operators: ")
  @show plaquette_vals
  println("")


  print("")
  println("Expectation values of loop operators:")
  @show loop_vals
  println("")


  # println("")
  # println("Eigenvalues of the twelve-point correlator near the first vacancy:")
  # @show order_parameter
  # println("")

  println(repeat("#", 100))
  println(repeat("#", 100))
  #***************************************************************************************************************
  #*************************************************************************************************************** 

  @show time_machine
  # h5open("data/2d_tK_armchair_FM_Lx$(Nx_unit)_Ly$(Ly_unit)_kappa$(kappa).h5", "w") do file
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
  #   write(file, "N0", n₀)
  #   write(file, "N", n)
  #   write(file, "Plaquette", plaquette_vals)
  #   write(file, "Loop", loop_vals)
  #   # write(file, "OrderParameter", order_parameter)
  # end

  return
end