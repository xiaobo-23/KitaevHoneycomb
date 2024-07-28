# Construct a list of loops that have non-trivial topological properties
using ITensors

function LoopList(input_Nx:: Int, input_Ny:: Int, ordering_scheme:: String, direction:: String)
    # '''
    #     Use periodic boundary condition in y direction and therefore
    #     the size of the output list is determined by the length and width of the cylinder
    #     Nx: the number of unit cells in the x direction
    #     Ny: the number of unit cells in the y direction
    # '''

    if ordering_scheme != "rings"
        error("Ordering scheme not supported!")
    end

    if ordering_scheme == "rings" && direction == "y"
        tmp_list = Matrix{Int64}(undef, input_Nx, 2 * input_Ny)
        for index1 in 1 : input_Nx
            for index2 in 1 : input_Ny
                tmp_list[index1, 2 * index2 - 1] = index2 + 2 * (index1 - 1) * input_Ny
                tmp_list[index1, 2 * index2] = index2 + (2 * index1 - 1) * input_Ny
            end
        end
    end     
    # @show tmp_list
    return tmp_list
end


function LoopList_RightTwist(input_Nx:: Int, input_Ny:: Int, ordering_scheme:: String, direction:: String)
    # '''
    #     Use periodic boundary condition in y direction and therefore
    #     the size of the output list is determined by the length and width of the cylinder
    #     Nx: the number of unit cells in the x direction
    #     Ny: the number of unit cells in the y direction
    # '''

    if ordering_scheme != "rings"
        error("Ordering scheme not supported!")
    end

    if ordering_scheme == "rings" && direction == "y"
        tmp_list = Matrix{Int64}(undef, input_Nx - 1, 2 * input_Ny)
        for index1 in 1 : input_Nx - 1
            for index2 in 1 : input_Ny
                if index2 == 1
                    tmp_list[index1, 2 * index2 - 1] = index2 + 2 * index1 * input_Ny
                else
                    tmp_list[index1, 2 * index2 - 1] = index2 + 2 * (index1 - 1) * input_Ny
                end
                tmp_list[index1, 2 * index2] = index2 + (2 * index1 - 1) * input_Ny
            end
        end
    end     
    # @show tmp_list
    return tmp_list
end


function PlaquetteList(input_Nx:: Int, input_Ny:: Int, ordering_scheme:: String, PBC_in_x:: Bool)
    # '''
    #     Assume using periodic boundary condition in y direction 
    #     Implement the list of plaquettes for open boundary condition in x direction
    #     Nx: the number of unit cells in the x direction
    #     Ny: the number of unit cells in the y direction
    # '''

    if ordering_scheme != "rings"
        error("Ordering scheme not supported!")
    end

    if ordering_scheme == "rings" && PBC_in_x == false
        number_of_plaquettes = (input_Nx - 1) * input_Ny
        tmp_list = Matrix{Int64}(undef, number_of_plaquettes, 6)

        for index in 1 : number_of_plaquettes
            x_index = div(index - 1, input_Ny) + 1
            y_index = mod(index - 1, input_Ny) + 1
            site_index = 2 * (x_index - 1) * input_Ny + y_index
            if y_index == 1
                symmetric_site_index = site_index + 2 * input_Ny - 1
            else
                symmetric_site_index = site_index + input_Ny - 1
            end
            # @show index, x_index, y_index, site_index, symmetric_site_index
            
            tmp_list[index, 1] = site_index
            tmp_list[index, 2] = site_index + input_Ny
            tmp_list[index, 3] = site_index + 2 * input_Ny
            tmp_list[index, 4] = symmetric_site_index
            tmp_list[index, 5] = symmetric_site_index + input_Ny
            tmp_list[index, 6] = symmetric_site_index + 2 * input_Ny
        end
    elseif ordering_scheme == "rings" && PBC_in_x == true
       error("Periodic boundary condition in x direction needs to be implemented!")
    end     

    return tmp_list
end


function PlaquetteListReordering(input_Nx:: Int, input_Ny:: Int, ordering_scheme:: String, PBC_in_x:: Bool, input_seeds:: Array{Int64, 1})
    # '''
    #     Assume using periodic boundary condition in y direction 
    #     Implement the list of plaquettes for open boundary condition in x direction
    #     Nx: the number of unit cells in the x direction
    #     Ny: the number of unit cells in the y direction
    # '''

    if ordering_scheme != "rings"
        error("Ordering scheme not supported!")
    end

    if ordering_scheme == "rings" && PBC_in_x == false
        number_of_plaquettes = (input_Nx - 1) * input_Ny
        tmp_list = Matrix{Int64}(undef, number_of_plaquettes, 6)

        for index in 1 : number_of_plaquettes
            coordinate = input_seeds[index]
            tmp_list[index, 1] = coordinate
            tmp_list[index, 2] = coordinate + input_Ny
            tmp_list[index, 3] = coordinate + 2 * input_Ny
            if coordinate == input_Ny || mod(coordinate - input_Ny, 2 * input_Ny) == 0
                tmp_list[index, 4] = coordinate + 1 
                tmp_list[index, 5] = coordinate + input_Ny + 1
                tmp_list[index, 6] = coordinate + 2 * input_Ny + 1
            else
                tmp_list[index, 4] = coordinate + input_Ny + 1 
                tmp_list[index, 5] = coordinate + 2 * input_Ny + 1
                tmp_list[index, 6] = coordinate + 3 * input_Ny + 1
            end 
            
        end
    elseif ordering_scheme == "rings" && PBC_in_x == true
       error("Periodic boundary condition in x direction needs to be implemented!")
    end     

    return tmp_list
end


function PlaquetteList_RightTiwst(input_Nx:: Int, input_Ny:: Int, ordering_scheme:: String, PBC_in_x:: Bool)
    # '''
    #     Assume using periodic boundary condition in y direction 
    #     Implement the list of plaquettes for open boundary condition in x direction
    #     Nx: the number of unit cells in the x direction
    #     Ny: the number of unit cells in the y direction
    # '''

    println("")
    println("Generate the list of indices for plaquettes with twisted boundary condition in the y direction")
    println("")
    
    if ordering_scheme != "rings"
        error("Ordering scheme not supported!")
    end

    if ordering_scheme == "rings" && PBC_in_x == false
        number_of_plaquettes = (input_Nx - 1) * input_Ny - 1
        tmp_list = Matrix{Int64}(undef, number_of_plaquettes, 6)

        for index in 2 : number_of_plaquettes + 1
            x_index = div(index - 1, input_Ny) + 1
            y_index = mod(index - 1, input_Ny) + 1
            site_index = 2 * (x_index - 1) * input_Ny + y_index
            if y_index == 1
                symmetric_site_index = site_index - 1
            else
                symmetric_site_index = site_index + input_Ny - 1
            end

            tmp_list[index - 1, 1] = site_index
            tmp_list[index - 1, 2] = site_index + input_Ny
            tmp_list[index - 1, 3] = site_index + 2 * input_Ny
            tmp_list[index - 1, 4] = symmetric_site_index
            tmp_list[index - 1, 5] = symmetric_site_index + input_Ny
            tmp_list[index - 1, 6] = symmetric_site_index + 2 * input_Ny
        end
    elseif ordering_scheme == "rings" && PBC_in_x == true
       error("Periodic boundary condition in x direction needs to be implemented!")
    end     
    # @show tmp_list
    return tmp_list
end