# 1/2/2024
# Set up and simulate the Heisenberg model on a square or triangular lattice with edge pinning fields

using Pkg
using ITensors
using HDF5

let
  # Set up the lattice parameters
  Nx = 32
  Ny = 4
  N = Nx * Ny

  # Set up the pinning field
  pinning_strength = 0.5
  if pinning_strength > 1E-8
    # Pinning field breaks the conservation of Sz
    sites = siteinds("S=1/2", N; conserve_qns=false)
  else
    sites = siteinds("S=1/2", N; conserve_qns=true)
  end

  # Configure the lattice geometery
  lattice = square_lattice(Nx, Ny; yperiodic=true)
  # lattice = triangular_lattice(Nx, Ny; yperiodic=true)
  
  # Set up the Hamiltonian
  @show lattice
  os = OpSum()
  for b in lattice
    os .+= 0.5, "S+", b.s1, "S-", b.s2
    os .+= 0.5, "S-", b.s1, "S+", b.s2
    os .+= "Sz", b.s1, "Sz", b.s2
  end

  # Adding pinning fields to the left edge of the lattice
  left_edge = [(1, 5), (2, 6), (3, 7), (4, 8)]
  for b in lattice
    if (b.s1, b.s2) in left_edge
      println("Adding pinning filed to the left edge")
      @show b.s1, b.s2
      os .+= pinning_strength*(-1)^b.s1, "Sz", b.s1
    end
    # @show b.s1, b.s2
  end
  H = MPO(os, sites)

  # Initialize wavefunction to a random MPS
  # of bond-dimension 10 with same quantum
  # numbers as `state`
  state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
  ψ₀ = randomMPS(sites, state, 20)

  # Running parameters for DMRG 
  nsweeps = 10
  maxdim = [20, 60, 100, 100, 200, 400, 800, 1000, 1500, 2000]
  cutoff = [1E-8]

  Sz₀ = expect(ψ₀, "Sz", sites = 1 : N)
  # Sx₀ = expect(ψ₀, "Sx", sites = 1 : N)
  energy, ψ = dmrg(H, ψ₀; nsweeps, maxdim, cutoff)
  Sz = expect(ψ, "Sz", sites = 1 : N)
  # Sx = expect(ψ, "Sx", sites = 1 : N)
  zzcorr = correlation_matrix(ψ, "Sz", "Sz", sites = 1 : N)

  # Compute energy per bound
  number_of_bonds = length(lattice)
  @show number_of_bonds, energy / number_of_bonds
  
  h5open("data/2d_heisenberg_square_lattice_pinning_p05_L$(Nx)W$(Ny).h5", "w") do file
    write(file, "psi", ψ)
    write(file, "E0", energy / number_of_bonds)
    write(file, "Sz0", Sz₀)
    write(file, "Sz",  Sz)
    write(file, "Czz", zzcorr)
    # write(file, "Sx0", Sx₀)
    # write(file, "Sx",  Sx)
  end

  return
end