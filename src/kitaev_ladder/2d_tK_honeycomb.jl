# 08/17/2025
# Simulating the doped Kitaev-Heisenberg ladder

using ITensors
using ITensorMPS
using HDF5
# using MKL
using LinearAlgebra
using TimerOutputs

include("HoneycombLattice.jl")
include("CustomObserver.jl")


# Set up parameters for multithreading for BLAS/LAPACK and Block sparse multithreading
MKL_NUM_THREADS = 8
OPENBLAS_NUM_THREADS = 8
OMP_NUM_THREADS = 8


# Monitor the number of threads used by BLAS and LAPACK
@show BLAS.get_config()
@show BLAS.get_num_threads()


#******************************************************************************************************
# Set up the parameters in the Hamiltonian
#******************************************************************************************************
const Nx::Int = 9                     # Number of rungs along x (must be odd, >= 3)
const Ny::Int = 2                     # Number of legs (fixed to 2 for two-leg ladder code path)
const N::Int  = Nx * Ny               # Total number of sites
const ϕ::Float64 = 0.0
const t::Float64 = 0.1                # Electron hopping amplitude
const K::Float64 = sin(ϕ)             # Anisotropic Kitaev interaction 
const J::Float64 = cos(ϕ)             # Heisenberg interaction

# Timer for profiling distinct simulation stages
const time_machine = TimerOutput()                                    # Timing and profiling

#******************************************************************************************************
# Check the input parameters of the system to see if they are consistent with the setup
#******************************************************************************************************
@assert Nx ≥ 3 "Nx must be at least 3."
@assert isodd(Nx) "Nx must be odd for the two-leg honeycomb ladder construction."
@assert Ny == 2 "Current ladder implementation assumes Ny == 2."

@info "Setting up the parameters in the Hamiltonian" t, K, J
@info "Number of honeycomb unit cells is" Int((Nx - 1) / 2)


let
  #***************************************************************************************************************
  # Set up the two-leg honeycomb ladder 
  # Return the two types of objects: bonds (for two-body interactions)
  #***************************************************************************************************************
  x_periodic = false
  y_periodic = true

  if x_periodic
    lattice = honeycomb_lattice_rings_pbc(Nx, Ny; yperiodic=true)
  else
    lattice = honeycomb_lattice_ladder(Nx, Ny; yperiodic=true)
  end 
  number_of_bonds = length(lattice)
  @show number_of_bonds
  #***************************************************************************************************************
  #***************************************************************************************************************  
  


  #***************************************************************************************************************
  # Construct the Hamiltonian as MPO
  #***************************************************************************************************************
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
    tmp_y = mod(b.s1 - 1, Ny) + 1

    if abs(b.s1 - b.s2) == 1
      os .+= K, "Sz", b.s1, "Sz", b.s2
      zbond += 1
      @info "Added Sz-Sz bond" s1=b.s1 s2=b.s2
    end

    if abs(b.s1 - b.s2) == 2
      if isodd(tmp_x)
        if isodd(tmp_y)
          os .+= K, "Sx", b.s1, "Sx", b.s2
          xbond += 1
          @info "Added Sx-Sx bond" s1=b.s1 s2=b.s2
        else
          os .+=  0.25 * K, "S+", b.s1, "S-", b.s2
          os .+=  0.25 * K, "S-", b.s1, "S+", b.s2
          os .+= -0.25 * K, "S+", b.s1, "S+", b.s2
          os .+= -0.25 * K, "S-", b.s1, "S-", b.s2 
          ybond += 1
          @info "Added Sy-Sy bond" s1=b.s1 s2=b.s2
        end
      else
        if iseven(tmp_y)
          os .+= K, "Sx", b.s1, "Sx", b.s2
          xbond += 1
          @info "Added Sx-Sx bond" s1=b.s1 s2=b.s2
        else
          os .+=  0.25 * K, "S+", b.s1, "S-", b.s2
          os .+=  0.25 * K, "S-", b.s1, "S+", b.s2
          os .+= -0.25 * K, "S+", b.s1, "S+", b.s2
          os .+= -0.25 * K, "S-", b.s1, "S-", b.s2
          ybond += 1
          @info "Added Sy-Sy bond" s1=b.s1 s2=b.s2
        end
      end
    end


    # Set up the Heisenberg interaction
    os .+= J, "Sz", b.s1, "Sz", b.s2
    os .+= 0.5 * J, "S+", b.s1, "S-", b.s2
    os .+= 0.5 * J, "S-", b.s1, "S+", b.s2
  end
  
  
  # Check the number of bonds in the Hamiltonian 
  total_bonds = trunc(Int, 3/2 * N) - 2
  if xbond + ybond + zbond != total_bonds
    error("The number of bonds in the Hamiltonian is not correct!")
  end
  # @show xbond, ybond, zbond
  #***************************************************************************************************************
  #***************************************************************************************************************  
  
  
  
  #*************************************************************************************************************** 
  # Set up the initial MPS and DMRG simulations parameters
  #***************************************************************************************************************
  # Increase the maximum dimension of Krylov space used to locally solve the eigenvalues problem
  sites = siteinds("tJ", N; conserve_nf=true)

  # Initialize wavefunction to a random MPS of bond-dimension 10 with same quantum numbers as `state`
  state = []                                              # Put the hole in the middle of the system
  hole_idx = Int(N / 2)
  for (idx, n) in enumerate(1 : N)
    if n == hole_idx
      push!(state, "Emp")
    else
      if isodd(idx)
        push!(state, "Up")
      else
        push!(state, "Dn")
      end
    end
  end
  # @show state
  ψ₀ = randomMPS(sites, state, 10)
  
  # Set up the Hamiltonian as an MPO
  H = MPO(os, sites)


  # Measure one-point functions of the initial state
  Sx₀ = expect(ψ₀, "Sx", sites = 1 : N)
  Splus₀  = expect(ψ₀, "S+", sites = 1 : N)
  Sminus₀ = expect(ψ₀, "S-", sites = 1 : N)
  Sy₀ = 0.5im * (Splus₀ - Sminus₀)
  Sz₀ = expect(ψ₀, "Sz", sites = 1 : N)
  n₀ = expect(ψ₀, "Ntot", sites = 1 : N)
  println("")
  @show sum(n₀)
  println("")
  
  # Check if the system is properly doped before running the DMRG simulation
  if abs(N - sum(n₀) - 1) > 1E-6
    error("The system is not properly doped!")
  end 
  #********************************************************************************************************
  #********************************************************************************************************
 


  #******************************************************************************************************
  # Run the DMRG simulation and obtain the ground state wavefunction
  #******************************************************************************************************
  # Set up the parameters including bond dimensions and truncation error
  nsweeps = 20
  maxdim  = [20, 100, 200, 500, 800, 1000, 1500, 3000]
  cutoff  = [1E-10]
  eigsolve_krylovdim = 35
  
  # # Add noise terms to prevent DMRG from getting stuck in a local minimum
  # noise = [1E-6, 1E-7, 1E-8, 0.0]

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
  # Take measurements of the wavefunction after the DMRG simulation
  #******************************************************************************************************
  # Measure local observables (i.e. one-point functions)
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
  # @show n
  println("")

  doping_tol = 1e-6
  if abs(N - sum(n) - 1) > doping_tol
    error("The system is not properly doped!")
  end

  
  # Measure spin-spin correlation functions (i.e. two-point functions)
  @timeit time_machine "two-point functions" begin
    xxcorr = correlation_matrix(ψ, "Sx", "Sx", sites = 1 : N)
    zzcorr = correlation_matrix(ψ, "Sz", "Sz", sites = 1 : N)
    # yycorr = correlation_matrix(ψ, "Sy", "Sy", sites = 1 : N)
  end

  
  # Generate the plaquette indices for all the plaquettes in the cylinder
  # plaquette_operator = Vector{String}(["iY", "X", "Z", "Z", "X", "iY"]) 
  plaquette_operator = [
    ["S+", "Sx", "Sz", "Sz", "Sx", "S+"],
    ["S+", "Sx", "Sz", "Sz", "Sx", "S-"],
    ["S-", "Sx", "Sz", "Sz", "Sx", "S-"],
    ["S-", "Sx", "Sz", "Sz", "Sx", "S+"]
  ]
  number_of_plaquettes = Int((Nx - 1) / 2)
  @info "Number of plaquettes in the cylinder" number_of_plaquettes


  # Generate the plaquette indices for all the plaquettes in the cylinder
  plaquette_indices = zeros(Int, number_of_plaquettes, 6)
  for idx in 1 : number_of_plaquettes
    plaquette_indices[idx, :] = [
      4 * (idx - 1) + 1,
      4 * (idx - 1) + 2,
      4 * (idx - 1) + 3,
      4 * (idx - 1) + 4,
      4 * (idx - 1) + 5,
      4 * (idx - 1) + 6
    ]
  end
  @show plaquette_indices

  
  # Measure the eigenvalues of plaquette operators
  # Decompose the plaquette operators into four terms for tJ type of sites
  @timeit time_machine "plaquette operators" begin
    plaquette_eigenvalues = zeros(Float64, number_of_plaquettes)

    for idx1 in 1:number_of_plaquettes
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
        plaquette_eigenvalues[idx1] += (-1.0)^idx2 * real(inner(ψ', W, ψ))
      end
      # Normalize the plaquette eigenvalues because the plaquette operator is in sigma notation
      plaquette_eigenvalues[idx1] *= 2^6 / 4
    end
  end
  println("")
  println("Eigenvalues of the plaquette operator:")
  @show plaquette_eigenvalues
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
  #********************************************************************************************************
  #********************************************************************************************************
 
  
  # # @show time_machine
  # h5open("2d_Kitaev_Heisenberg_Lx$(Nx_unit)_phi$(ϕ).h5", "w") do file
  #   write(file, "psi", ψ)
  #   write(file, "NormalizedE0", energy / number_of_bonds)
  #   write(file, "E0", energy)
  #   write(file, "E0variance", variance)
  #   write(file, "Ehist", custom_observer.ehistory)
  #   write(file, "Bond", custom_observer.chi)
  #   write(file, "Sx0", Sx₀)
  #   write(file, "Sx",  Sx)
  #   write(file, "Cxx", xxcorr)
  #   write(file, "Sy0", Sy₀)
  #   write(file, "Sy", Sy)
  #   # # write(file, "Cyy", yycorr)
  #   write(file, "Sz0", Sz₀)
  #   write(file, "Sz",  Sz)
  #   write(file, "Czz", zzcorr)
  #   write(file, "N0", n₀)
  #   write(file, "N", n)
  #   write(file, "Plaquette", plaquette_eigenvalues)
  # end

  return
end