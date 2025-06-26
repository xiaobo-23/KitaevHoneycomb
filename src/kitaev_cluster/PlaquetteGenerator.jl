# Generate the plaquette indices on a honeycomb lattice

function GeneratePlaquetteRings(Seed::Int, Ny::Int; edge=false)
	"""
		Generate a list of site indices for the plaquettes
		Handle the bulk case and throw an error for the edge case 
	"""
	
	plaqutte = Vector{Int}(undef, 6)
	if edge
		error("Edge case not implemented!")
	end
	
	# Assign the site indices to the plaquette
	# Use a clockwise ordering and the rings ordering scheme
	plaquette = [Seed, Seed + Ny, Seed + 2 * Ny, Seed + 3 * Ny - 1, Seed + 2 * Ny - 1, Seed + Ny - 1]
	
	return plaquette
end