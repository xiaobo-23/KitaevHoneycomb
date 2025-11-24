## 11/24/2025
## Set up the Kitaev Hamiltonian as an MPO

using ITensors
using ITensorMPS

include("HoneycombLattice.jl")
include("TopologicalLoops.jl")

# Function to construct the two-body interaction terms in the Kitaev Hamiltonian
function construct_two_body_interactions(input_os, input_lattice::Vector{LatticeBond},
    hopping::Float64, Jx::Float64, Jy::Float64, Jz::Float64, input_Ny::Int, input_bonds::Int)
    
    
    # Initialize a dictionary to count the number of each type of bond
    bond_counts = Dict("xbond" => 0, "ybond" => 0, "zbond" => 0)
  

    # Construct the two-body interaction terms in the Kitaev Hamiltonian
    for b in input_lattice
        # Set up the electron hopping terms
        input_os .+= -hopping, "Cdagup", b.s1, "Cup", b.s2
        input_os .+= -hopping, "Cdagup", b.s2, "Cup", b.s1
        input_os .+= -hopping, "Cdagdn", b.s1, "Cdn", b.s2
        input_os .+= -hopping, "Cdagdn", b.s2, "Cdn", b.s1

        # Set up the anisotropic two-body Kitaev interaction terms
        x_coordinate = div(b.s1 - 1, input_Ny) + 1

        # Set up the Sz-Sz bond interaction 
        if abs(b.s1 - b.s2) == 1 || abs(b.s1 - b.s2) == input_Ny - 1
            input_os .+= -Jz, "Sz", b.s1, "Sz", b.s2
            bond_counts["zbond"] += 1
            @info "Added Sz-Sz bond" s1=b.s1 s2=b.s2
        end

        # Set up the Sx-Sx and Sy-Sy bond interactions
        if (isodd(x_coordinate) && isodd(b.s1) && isodd(b.s2)) || 
            (iseven(x_coordinate) && iseven(b.s1) && iseven(b.s2))
            input_os .+= -Jx, "Sx", b.s1, "Sx", b.s2
            bond_counts["xbond"] += 1
            @info "Added Sx-Sx bond" s1=b.s1 s2=b.s2
        elseif (isodd(x_coordinate) && iseven(b.s1) && iseven(b.s2)) || 
            (iseven(x_coordinate) && isodd(b.s1) && isodd(b.s2))
            input_os .+= -0.25 * Jy, "S+", b.s1, "S-", b.s2
            input_os .+= -0.25 * Jy, "S-", b.s1, "S+", b.s2
            input_os .+=  0.25 * Jy, "S+", b.s1, "S+", b.s2
            input_os .+=  0.25 * Jy, "S-", b.s1, "S-", b.s2
            bond_counts["ybond"] += 1
            @info "Added Sy-Sy bond" s1=b.s1 s2=b.s2
        end
    end


    # Check the total number of bonds added; throw an error if there is a mismatch
    total_bonds = bond_counts["xbond"] + bond_counts["ybond"] + bond_counts["zbond"]
    if total_bonds != input_bonds
        error("Mismatch in the number of bonds: expected $input_bonds, but found $total_bonds.")
    end
    # @info "Bond counts by type" xbond=bond_counts["xbond"] ybond=bond_counts["ybond"] zbond=bond_counts["zbond"]
    
    return input_os
end


# Function to construc the three-spin interaction terms in the Kitaev Hamiltonian
function construct_three_spin_interaction(input_os, input_wedge::Vector{WedgeBond}, input_Nx::Int, input_Ny::Int, 
    input_kappa::Float64)
    
    # Initialize a dictionary to count the number of each type of edge
    edge_counts = Dict("horizontal" => 0, "vertical" => 0)


    # Loop over each wedge to set up the three-spin interaction terms
    for w in input_wedge
        x_coordinate = div(w.s2 - 1, input_Ny) + 1
        y_coordinate = mod(w.s2 - 1, input_Ny) + 1    
        
        # Set up the horizontal three-spin interaction terms
        if abs(w.s1 - w.s2) == abs(w.s2 - w.s3) == input_Ny
            if (isodd(x_coordinate) && isodd(y_coordinate)) || (iseven(x_coordinate) && iseven(y_coordinate))
                input_os .+= -0.5im * input_kappa, "S+", w.s1, "Sz", w.s2, "Sx", w.s3
                input_os .+=  0.5im * input_kappa, "S-", w.s1, "Sz", w.s2, "Sx", w.s3
                # @info "Added three-spin interaction" term = ("Sy", w.s1, "Sz", w.s2, "Sx", w.s3)
            elseif (isodd(x_coordinate) && iseven(y_coordinate)) || (iseven(x_coordinate) && isodd(y_coordinate))
                input_os .+= -0.5im * input_kappa, "Sx", w.s1, "Sz", w.s2, "S+", w.s3
                input_os .+=  0.5im * input_kappa, "Sx", w.s1, "Sz", w.s2, "S-", w.s3
                # @info "Added three-spin interaction" term = ("Sx", w.s1, "Sz", w.s2, "Sy", w.s3)
            end
            edge_counts["horizontal"] += 1
        end



        # Set up the vertical three-spin interaction terms through periodic boundary condition along the y direction
        if abs(w.s1 - w.s2) == input_Ny - 1
            if (y_coordinate == 1 && w.s3 < w.s2) || (y_coordinate == input_Ny && w.s2 < w.s3)
                input_os .+= -0.5im * input_kappa, "Sz", w.s1, "S+", w.s2, "Sx", w.s3
                input_os .+=  0.5im * input_kappa, "Sz", w.s1, "S-", w.s2, "Sx", w.s3
                # @info "Added three-spin interaction" term = ("Sz", w.s1, "Sy", w.s2, "Sx", w.s3)
            elseif (y_coordinate == 1 && w.s2 < w.s3) || (y_coordinate == input_Ny && w.s3 < w.s2)
                input_os .+= -0.5im * input_kappa, "Sz", w.s1, "Sx", w.s2, "S+", w.s3
                input_os .+=  0.5im * input_kappa, "Sz", w.s1, "Sx", w.s2, "S-", w.s3
                # @info "Added three-spin interaction" term = ("Sz", w.s1, "Sx", w.s2, "Sy", w.s3)
            end
            edge_counts["vertical"] += 1
        end


        # Set up the vertical three-spin interaction terms within the bulk of the cylinder 
        if abs(w.s2 - w.s1) == 1
            if (isodd(x_coordinate) && isodd(y_coordinate)) || (iseven(x_coordinate) && iseven(y_coordinate))
            if w.s2 > w.s3
                input_os .+= -0.5im * input_kappa, "Sz", w.s1, "Sx", w.s2, "S+", w.s3 
                input_os .+=  0.5im * input_kappa, "Sz", w.s1, "Sx", w.s2, "S-", w.s3
                # @info "Added three-spin interaction" term = ("Sz", w.s1, "Sx", w.s2, "Sy", w.s3)
            else
                input_os .+= -0.5im * input_kappa, "Sz", w.s1, "S+", w.s2, "Sx", w.s3
                input_os .+=  0.5im * input_kappa, "Sz", w.s1, "S-", w.s2, "Sx", w.s3
                # @info "Added three-spin interaction" term = ("Sz", w.s1, "Sy", w.s2, "Sx", w.s3)
            end
            elseif (isodd(x_coordinate) && iseven(y_coordinate)) || (iseven(x_coordinate) && isodd(y_coordinate))
            if w.s2 > w.s3
                input_os .+= -0.5im * input_kappa, "Sz", w.s1, "S+", w.s2, "Sx", w.s3
                input_os .+=  0.5im * input_kappa, "Sz", w.s1, "S-", w.s2, "Sx", w.s3
                # @info "Added three-spin interaction" term = ("Sz", w.s1, "Sy", w.s2, "Sx", w.s3)
            else
                input_os .+= -0.5im * input_kappa, "Sz", w.s1, "Sx", w.s2, "S+", w.s3 
                input_os .+=  0.5im * input_kappa, "Sz", w.s1, "Sx", w.s2, "S-", w.s3
                # @info "Added three-spin interaction" term = ("Sz", w.s1, "Sx", w.s2, "Sy", w.s3)
            end
            end
            edge_counts["vertical"] += 1
        end
    end


    if (edge_counts["horizontal"] + edge_counts["vertical"]) != number_of_wedges
    error("Mismatch in the number of wedges: expected $number_of_wedges, but found $(edge_counts["horizontal"] + edge_counts["vertical"]).")
    end
    # @info "Wedge counts by type" horizontal=edge_counts["horizontal"] vertical=edge_counts["vertical"]
end