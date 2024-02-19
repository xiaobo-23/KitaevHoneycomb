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
# struct LatticeBond
#   s1::Int
#   s2::Int
#   x1::Float64
#   y1::Float64
#   x2::Float64
#   y2::Float64
#   type::String
# end

# """
#     LatticeBond(s1::Int,s2::Int)

#     LatticeBond(s1::Int,s2::Int,
#                 x1::Real,y1::Real,
#                 x2::Real,y2::Real,
#                 type::String="")

# Construct a LatticeBond struct by
# specifying just the numbers of sites
# 1 and 2, or additional details including
# the (x,y) coordinates of the two sites and
# an optional type string.
# """
# function LatticeBond(s1::Int, s2::Int)
#   return LatticeBond(s1, s2, 0.0, 0.0, 0.0, 0.0, "")
# end

# function LatticeBond(
#   s1::Int, s2::Int, x1::Real, y1::Real, x2::Real, y2::Real, bondtype::String=""
# )
#   cf(x) = convert(Float64, x)
#   return LatticeBond(s1, s2, cf(x1), cf(y1), cf(x2), cf(y2), bondtype)
# end

# """
# Lattice is an alias for Vector{LatticeBond}
# """
# const Lattice = Vector{LatticeBond}


# # include("/Users/boxiao/.julia/packages/ITensors/MnaxI/src/physics/lattices.jl")

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
	Nbond = trunc(Int, 3/2 * N) - Ny + (yperiodic ? 0 : trunc(Int, Nx / 2))
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
	Nbond = trunc(Int, 3/2 * N) + (yperiodic ? 0 : trunc(Int, Nx / 2))
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


function honeycomb_lattice_rings_twist(Nx::Int, Ny::Int; yperiodic=false)::Lattice
	"""
	  Using the ring ordering scheme
	  Nx needs to be an even number
	"""
	yperiodic = yperiodic && (Ny > 2)
	N = Nx * Ny - 2
	Nbond = trunc(Int, 3/2 * N) - 4 + (yperiodic ? 0 : trunc(Int, Nx / 2))
	@show Nbond
	
	latt = Lattice(undef, Nbond)
	b = 0
	for n in 1:N
		# Configure the x coordinate
		if n < Ny
			x = 1
		else
			x = div(n - Ny, Ny) + 2
		end
		# Configure the y coordinate
		if n < Ny
			y = n + 1
		else
			y = mod(n, Ny) + 1
		end
		@show n, x, y, b
  
	  	# horizontal bonds
		if mod(x, 2) == 0 && x < Nx
			latt[b += 1] = LatticeBond(n, n + Ny)
		end
  
		# bonds to accomodate simga_x * sigma_x and sigma_y * sigma_y interactions
		if Ny > 1
			if mod(x, 2) == 1 
				if y > 1
					latt[b += 1] = LatticeBond(n, n + Ny - 1)
					if n + Ny <= N
						latt[b += 1] = LatticeBond(n, n + Ny)
					end
				else
					latt[b += 1] = LatticeBond(n, n + Ny)
					latt[b += 1] = LatticeBond(n, n - 1)
				end
			end
		end
	# @show latt
	end
	return latt
end


function honeycomb_lattice_Cstyle(Nx::Int, Ny::Int; yperiodic=false)::Lattice
  """
	Using the C-style ordering scheme
	Nx needs to be an even number
  """
	yperiodic = yperiodic && (Ny > 2)
	N = Nx * Ny
	Nbond = trunc(Int, 3/2 * N) - Ny + (yperiodic ? 0 : trunc(Int, Nx / 2))
	@show Nbond
  
	latt = Lattice(undef, Nbond)
	b = 0
	for n in 1:N
		tmp = div(n - 1, 2 * Ny)
		x = 2 * tmp + mod(n - 1, 2) + 1
		y = mod(div(n - 1, 2), Ny) + 1
		@show n, x, y

		# x-direction bonds for A sublattice
		if mod(x, 2) == 0 && x < Nx
			latt[b += 1] = LatticeBond(n, n + 2 * Ny - 1)
			@show n, x, y, b
		end

		# bonds for B sublattice
		if Ny > 1
			if mod(x, 2) == 1 && x < Nx
				# @show latt
				latt[b += 1] = LatticeBond(n, n + 1)
				if y != 1
					latt[b += 1] = LatticeBond(n, n - 1)
				end
			end

			# periodic bonds 
			if mod(x, 2) == 1 && yperiodic && y == 1
				latt[b += 1] = LatticeBond(n, n + 2 * Ny - 1)
			end
		end
	end
	# @show latt
	return latt
end


# 1/22/2024
# Turn off interactions on selected bonds and benchmark against the one-dimensional DRMG result
function honeycomb_lattice_rings_map_to_1d_chains(Nx::Int, Ny::Int)::Lattice
	"""
		Using the ring ordering scheme 
		Nx needs to be an even number
	"""

	N = Nx * Ny
	Nbond = N - 1
	
	latt = Lattice(undef, Nbond)
	b = 0
	for n in 1:N
		x = div(n - 1, Ny) + 1
		y = mod(n - 1, Ny) + 1
		
		# Use the ring ordering scheme to set up an one-dimensional chain
		if Ny > 1 && n < N
			if mod(x, 2) == 1 && x < Nx
				latt[b += 1] = LatticeBond(n, n + Ny)
				if y != 1
					latt[b += 1] = LatticeBond(n, n + Ny - 1)
				end
			end

			if mod(x, 2) == 0 && y == Ny
				latt[b += 1] = LatticeBond(n, n + 1)
			end
		end
	end

	return latt
end