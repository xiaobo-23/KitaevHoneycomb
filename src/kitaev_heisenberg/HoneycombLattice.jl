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


# 01/06/2025
# Implement the honeycomb lattice geometry using the armchair pattern
function honeycomb_lattice_armchair(Nx::Int, Ny::Int; yperiodic=false)::Lattice
	"""
		Use the armchair pattern 
		The number of rows needs to be an even number 
	"""
	
	if Ny % 2 != 0
		error("The number of rows (Ny) needs to be an even number.")
	end

	yperiodic = yperiodic && (Ny > 2)
	N = Nx * Ny
	Nbond = trunc(Int, 3/2 * N) - Ny + (yperiodic ? 0 : -trunc(Int, Nx / 2))
	@show Nbond
  
  	latt = Lattice(undef, Nbond)
  	b = 0
	for n in 1:N
		x = div(n - 1, Ny) + 1
		y = mod(n - 1, Ny) + 1

		# Set up the vertical bonds at odd column in the armchair geometry
		if mod(x, 2) == 1 && mod(y, 2) == 1
			latt[b += 1] = LatticeBond(n, n + 1)
		end
		
		# Set up the vertical bonds at even column in the armchair geometry
		if mod(x, 2) == 0 && mod(y, 2) == 0
			if mod(y, Ny) == 0
				latt[b += 1] = LatticeBond(n, n + 1 - Ny)
			else
				latt[b += 1] = LatticeBond(n, n + 1)
			end
		end

		# Set up the non-vertical bonds in the bulk
		if x != 1 && x != Nx
			latt[b += 1] = LatticeBond(n, n + Ny)
		end

		# Set up the non-vertical bonds at the left edge
		if x == 1
			latt[b += 1] = LatticeBond(n, n + Ny)
		end

		# # Set up the non-vertical bonds the right edge
		# if x == Nx
		# 	latt[b += 1] = LatticeBond(n, n - Ny)
		# end
		# @show latt 
	end
	
	# @show latt 
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


function honeycomb_lattice_rings_right_twist(Nx::Int, Ny::Int; yperiodic=false)::Lattice
	"""
	  Using the ring ordering scheme
	  Nx needs to be an even number
	"""
	yperiodic = yperiodic && (Ny > 2)
	N = Nx * Ny
	Nbond = trunc(Int, 3/2 * N) - Ny + (yperiodic ? -1 : -trunc(Int, Nx / 2))
	@show Nbond
	
	latt = Lattice(undef, Nbond)
	b = 0
	for n in 1:N
		x = div(n - 1, Ny) + 1
		y = mod(n - 1, Ny) + 1
		@show n, x, y, b
  
	  	# horizontal bonds
		if mod(x, 2) == 0 && x < Nx
			latt[b += 1] = LatticeBond(n, n + Ny)
		end
  
		# bonds to accomodate simga_x * sigma_x and sigma_y * sigma_y interactions
		if Ny > 1
			if mod(x, 2) == 1
				latt[b += 1] = LatticeBond(n, n + Ny)
				# twisted boundary condition along the y direction
				if y == 1
					if x != 1
						latt[b += 1] = LatticeBond(n, n - 1)
					end
				else
					latt[b += 1] = LatticeBond(n, n + Ny - 1)
				end 
			end
		end
	# @show latt
	end
	return latt
end



## 07/23/2024
function honeycomb_lattice_rings_reorder(Nx::Int, Ny::Int; yperiodic=false)::Lattice
	"""
		Using the ring ordering scheme
		Nx needs to be an even number
	"""
	yperiodic = yperiodic && (Ny > 2)
	N = Nx * Ny
	Nbond = trunc(Int, 3/2 * N) - Ny + (yperiodic ? 0 : -trunc(Int, Nx / 2))
	# @show Nbond
  
  	latt = Lattice(undef, Nbond)
  	b = 0
	for n in 1:N
		x = div(n - 1, Ny) + 1
		if mod(x, 2) == 0
			seed_position = Ny - mod(trunc(Int, x / 2) - 1, Ny)
			y = mod(seed_position + mod(n - 1, Ny), Ny)
			if y == 0
				y = Ny
			end
			# @show n, x, y, seed_position
		elseif mod(x, 2) == 1 && x > 1 
			seed_position = Ny - mod(trunc(Int, (x - 1) / 2) - 1, Ny)
			y = mod(seed_position + mod(n - 1, Ny), Ny)
			if y == 0
				y = Ny
			end
			# @show n, x, y, seed_position
		else
			y = mod(n - 1, Ny) + 1
			# @show n, x, y
		end

		# x-direction bonds for A sublattice
		if mod(x, 2) == 0 && x < Nx
			@show n, n + Ny
			latt[b += 1] = LatticeBond(n, n + Ny)
		end

		# bonds for B sublattice
		if Ny > 1
			if mod(x, 2) == 1 && x < Nx
				@show n, n + Ny
				latt[b += 1] = LatticeBond(n, n + Ny)
				if n == Ny || mod(n - Ny, 2 * Ny) == 0
					@show n, n + 1
					latt[b += 1] = LatticeBond(n, n + 1)
				else
					@show n, n + Ny + 1
					latt[b += 1] = LatticeBond(n, n + Ny + 1)
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
	Nbond = trunc(Int, 3/2 * N) - Ny + (yperiodic ? 0 : -trunc(Int, Nx / 2))
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



# 01/06/2025
# Define a wedge bond to introduce the three-body interaction
struct WedgeBond
  s1::Int
  s2::Int
  s3::Int
  x1::Float64
  y1::Float64
  x2::Float64
  y2::Float64
  x3::Float64
  y3::Float64
  type::String
end


function WedgeBond(s1::Int, s2::Int, s3::Int)
	return WedgeBond(s1, s2, s3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "")
end


function WedgeBond(
  s1::Int, s2::Int, s3::Int, x1::Real, y1::Real, x2::Real, y2::Real, x3::Real, y3::Real, bondtype::String=""
)
  cf(x) = convert(Float64, x)
  return WedgeBond(s1, s2, s3, cf(x1), cf(y1), cf(x2), cf(y2), cf(x3), cf(y3), bondtype)
end


# """
# Wedge is an alias for Vector{WedgeBond}
# """
# const Wedge = Vector{WedgeBond}


# 01/06/2025
#  Implement the wedge object to introduce the three-body interaction on the armchair geometry
function honeycomb_armchair_wedge(Nx::Int, Ny::Int; yperiodic=false)
	"""
		Use the armchair geometery
	""" 
	
	yperiodic = yperiodic && (Ny > 2)
	N = Nx * Ny									# Number of sites	
	Nwedge = 3 * Nx * Ny - 4 * Ny  				# Number of wedges

	wedge = Vector{WedgeBond}(undef, Nwedge)
	# wedge = Wedge(undef, Nwedge)

	b = 0
	for n in 1 : N
		x = div(n - 1, Ny) + 1
		y = mod(n - 1, Ny) + 1
		@show n, x, y
		
		if x == 1
			if mod(y, 2) == 1
				wedge[b += 1] = WedgeBond(n + 1, n, n + Ny)
			else
				wedge[b += 1] = WedgeBond(n - 1, n, n + Ny)
			end
		end

		if x == Nx
			if mod(y, 2) == 1
				if y == 1
					n_next = n + Ny - 1
				else
					n_next = n - 1
				end
			else
				if y == Ny
					n_next = n - Ny + 1
				else
					n_next = n + 1
				end
			end
			wedge[b += 1] = WedgeBond(n_next, n, n - Ny)
		end

		if 1 < x < Nx
			if mod(x, 2) == 1
				if mod(y, 2) == 1
					n_next = n + 1
				else
					n_next = n - 1
				end
			else
				if mod(y, 2) == 1
					if y == 1
						n_next = n + Ny - 1
					else
						n_next = n - 1
					end
				else
					if y == Ny
						n_next = n - Ny + 1
					else
						n_next = n + 1
					end
				end
			end
			wedge[b += 1] = WedgeBond(n - Ny, n, n + Ny)
			wedge[b += 1] = WedgeBond(n_next, n, n - Ny)
			wedge[b += 1] = WedgeBond(n_next, n, n + Ny)
		end
	end

	# @show wedge
	return wedge
end


# 05/21/2025
# Implement the wedge object to introduce the three-body interaction on the XC geometry
function honeycomb_twist_wedge(Nx::Int, Ny::Int; yperiodic=false)
	"""
		Use the XC geometry with a twist
	"""
	yperiodic = yperiodic && (Ny > 2)
	N = Nx * Ny									# Number of sites
	Nwedge = 3 * Nx * Ny - 2 * 2 * Ny - 2		# Number of wedges
	@show Nwedge

	wedge = Vector{WedgeBond}(undef, Nwedge)
	# wedge = Wedge(undef, Nwedge)

	b = 0
	for n in 1 : N
		x = div(n - 1, Ny) + 1
		y = mod(n - 1, Ny) + 1

		if isodd(x)
			@show n, x, y
			if x == 1
				if y != 1
					wedge[b += 1] = WedgeBond(n + Ny - 1, n, n + Ny)
				end
			else
				wedge[b += 1] = WedgeBond(n - Ny, n, n + Ny)
				if y == 1
					wedge[b += 1] = WedgeBond(n - 1, n, n - Ny)
					wedge[b += 1] = WedgeBond(n - 1, n, n + Ny)
				else
					wedge[b += 1] = WedgeBond(n + 2, n, n - Ny)
					wedge[b += 1] = WedgeBond(n + 2, n, n + Ny)
				end
			end
		end

		if iseven(x)
			@show n, x, y
			if x == Nx
				if y != Ny  
					wedge[b += 1] = WedgeBond(n - Ny, n, n - Ny + 1)
				end
			else
				wedge[b += 1] = WedgeBond(n - Ny, n, n + Ny)
				if y == Ny
					wedge[b += 1] = WedgeBond(n + 1, n, n - Ny)
					wedge[b += 1] = WedgeBond(n + 1, n, n + Ny)
				else
					wedge[b += 1] = WedgeBond(n - Ny + 1, n, n - Ny)
					wedge[b += 1] = WedgeBond(n - Ny + 1, n, n + Ny)
				end
			end
		end
	end

	# @show wedge
	return wedge
end