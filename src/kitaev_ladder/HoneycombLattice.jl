# 1/4/2024
# Implement the honeycomb lattice geometery using different ordering schemes

# """
# A LatticeBond is a struct which represents
# a single bond in a geometrical lattice or
# else on interaction graph defining a physical
# model such as a quantum Hamiltonian.

# LatticeBond has the following data fields:

#   - s1::Int -- number of site 1
#   - s2::Int -- number of site 2
#   - x1::Float64 -- x coordinate of site 1
#   - y1::Float64 -- y coordinate of site 1
#   - x2::Float64 -- x coordinate of site 2
#   - y2::Float64 -- y coordinate of site 2
#   - type::String -- optional description of bond type
# """
struct LatticeBond
  s1::Int
  s2::Int
  x1::Float64
  y1::Float64
  x2::Float64
  y2::Float64
  type::String
end

"""
    LatticeBond(s1::Int,s2::Int)

    LatticeBond(s1::Int,s2::Int,
                x1::Real,y1::Real,
                x2::Real,y2::Real,
                type::String="")

Construct a LatticeBond struct by
specifying just the numbers of sites
1 and 2, or additional details including
the (x,y) coordinates of the two sites and
an optional type string.
"""
function LatticeBond(s1::Int, s2::Int)
  return LatticeBond(s1, s2, 0.0, 0.0, 0.0, 0.0, "")
end

function LatticeBond(
  s1::Int, s2::Int, x1::Real, y1::Real, x2::Real, y2::Real, bondtype::String=""
)
  cf(x) = convert(Float64, x)
  return LatticeBond(s1, s2, cf(x1), cf(y1), cf(x2), cf(y2), bondtype)
end

"""
Lattice is an alias for Vector{LatticeBond}
"""
const Lattice = Vector{LatticeBond}

# """
#     honeycomb_lattice(Nx::Int,
#                        Ny::Int;
#                        kwargs...)::Lattice

#     Return a Lattice (array of LatticeBond
#     objects) corresponding to the two-dimensional
#     honeycomb lattice of dimensions (Nx, Ny).
#     By default the lattice has open boundaries,
#     but can be made periodic in the y direction
#     by specifying the keyword argument
#     `yperiodic=true`.
# """

function honeycomb_lattice_rings(Nx::Int, Ny::Int; yperiodic=false)::Lattice
	"""
		Using the ring ordering scheme
		Nx needs to be an even number
	"""
	yperiodic = yperiodic && (Ny > 2)
	N = Nx * Ny
	Nbond = trunc(Int, 3/2 * N) - Ny + (yperiodic ? 0 : -trunc(Int, Nx / 2))
	@show Nbond
  
  	latt = Lattice(undef, Nbond)
  	b = 0
		for n in 1:N
			x = div(n - 1, Ny) + 1
			y = mod(n - 1, Ny) + 1

			# x-direction bonds for A sublattice
			if mod(x, 2) == 0 && x < Nx
				latt[b += 1] = LatticeBond(n, n + Ny)
			end

			# bonds for B sublattice
			if Ny > 1
				if mod(x, 2) == 1 && x < Nx
					# @show latt
					latt[b += 1] = LatticeBond(n, n + Ny)
					if y != 1
						latt[b += 1] = LatticeBond(n, n + Ny - 1)
					end
				end
			
				# periodic bonds 
				if mod(x, 2) == 1 && yperiodic && y == 1
					latt[b += 1] = LatticeBond(n, n + 2 * Ny - 1)
				end
			end

		# @show latt
	end

	return latt
end


function honeycomb_lattice_rings_pbc(Nx::Int, Ny::Int; yperiodic=false)::Lattice
	"""
	  Using the ring ordering scheme
	  Nx needs to be an even number
	"""
	yperiodic = yperiodic && (Ny > 2)
	N = Nx * Ny
	Nbond = trunc(Int, 3/2 * N) + (yperiodic ? 0 : -trunc(Int, Nx / 2))
	@show Nbond
	
	latt = Lattice(undef, Nbond)
	b = 0
	for n in 1:N
		x = div(n - 1, Ny) + 1
		y = mod(n - 1, Ny) + 1
		# @show Nx, Ny, x, y, b
  
	  	# x-direction bonds for A sublattice
		if mod(x, 2) == 0 && x < Nx
			latt[b += 1] = LatticeBond(n, n + Ny)
		end
  
		# bonds for B sublattice
		if Ny > 1
			if mod(x, 2) == 1 && x < Nx
				# @show latt
				latt[b += 1] = LatticeBond(n, n + Ny)
				if y != 1
					latt[b += 1] = LatticeBond(n, n + Ny - 1)
				end
			end

			# periodic bonds along the x direction
			if mod(x, 2) == 1 && x == 1
				# latt[b += 1] = LatticeBond(n, n + (Nx - 1) * Ny)
				latt[b += 1] = LatticeBond(n + (Nx - 1) * Ny, n)
			end

			# periodic bonds along the y direction
			if mod(x, 2) == 1 && yperiodic && y == 1
				latt[b += 1] = LatticeBond(n, n + 2 * Ny - 1)
			end
		end
	
	# @show latt
	end

	return latt
end


function honeycomb_lattice_ladder(Nx::Int, Ny::Int; yperiodic=false)::Lattice
	"""
	  Using the ring ordering scheme & Ny = 2
	"""
	# @assert iseven(Nx) "Nx must be even for honeycomb_lattice_ladder"
	N = Nx * Ny
	Nbond = trunc(Int, 3/2 * N) - 2
	@show Nbond
	
	latt = Lattice(undef, Nbond)
	b = 0
	for n in 1:N
		x = div(n - 1, Ny) + 1
		y = mod(n - 1, Ny) + 1
		# @show n, x, y, b
  
		# Set up the bonds for the first column
		if x < Nx
			if y == 1
				latt[b += 1] = LatticeBond(n, n + 1)
				latt[b += 1] = LatticeBond(n, n + Ny)
			else
				latt[b += 1] = LatticeBond(n, n + Ny)
			end
		end

		# Set up the bonds for the last column
		if x == Nx && y == 1
			latt[b += 1] = LatticeBond(n, n + 1)
		end
	end
	# @show latt
	if b != Nbond
		diff = Nbond - b
		throw(ArgumentError("honeycomb_lattice_ladder: bond count mismatch (expected $Nbond, got $b, Δ=$diff)"))
	end
	@show latt 
	return latt
end