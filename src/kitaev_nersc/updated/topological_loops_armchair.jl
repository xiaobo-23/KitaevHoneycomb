# 11/30/2025
# Construct topological loops for armchair geometery


using ITensors, ITensorMPS


# Generate the list of indices for each plaquette using armchair geometery
function PlaquetteListArmchair(inputNx:: Int, inputNy:: Int, geometery:: String, x_periodic=false)
    """
        Assume using periodic boundary condition in y direction 
        Implement the list of plaquettes for open boundary condition in x direction
        inputNx: the number of unit cells in the x direction
        inputNy: the number of unit cells in the y direction
    """

    # println(repeat("#", 200))
    # println("Generate the list of indices for each plaquette using armchair geometery")
    # println(repeat("#", 200))

    if geometery != "armchair"
        error("Simualting the Kitaev model on armchair geometery; input geometery is not supported!")
    end

    if geometery == "armchair" && x_periodic == false
        totalNumber = (inputNx - 1) * inputNy
        plaquette = Matrix{Int64}(undef, totalNumber, 6)

        for index in 1 : totalNumber
            x_index = div(index - 1, inputNy) + 1
            y_index = mod(index - 1, inputNy) + 1

            site_index1 = 2 * (x_index - 1) * inputNy + 2 * (y_index - 1) + 1
            if y_index <= div(inputNy, 2)
                site_index2 = site_index1 + 1
                plaquette[index, 1] = site_index1
                plaquette[index, 2] = site_index1 + inputNy
                plaquette[index, 3] = site_index1 + 2 * inputNy
                plaquette[index, 4] = site_index2
                plaquette[index, 5] = site_index2 + inputNy
                plaquette[index, 6] = site_index2 + 2 * inputNy 
            else
                if y_index == div(inputNy, 2) + 1
                    site_index2 = site_index1 + inputNy - 1
                else
                    site_index2 = site_index1 - 1
                end
                plaquette[index, 1] = site_index2
                plaquette[index, 2] = site_index2 + inputNy
                plaquette[index, 3] = site_index2 + 2 * inputNy
                plaquette[index, 4] = site_index1
                plaquette[index, 5] = site_index1 + inputNy
                plaquette[index, 6] = site_index1 + 2 * inputNy
            end 
        end
    elseif geometery == "armchair" && x_periodic == true
       error("Periodic boundary condition in x direction needs to be implemented!")
    end     

    return plaquette 
end


# Generate the list of indices for each loop along the y direction using armchair geometery
function LoopListArmchair(inputNx::Int, inputNy::Int, ordering_geometery::String, direction::String)
    """
        Assume using periodic boundary condition in y direction
        Nx: number of unit cells in the x direction
        Ny: number of unit cells in the y direction
    """

    # println(repeat("#", 200))
    # println("Generate the list of indices for each loop using armchair geometery")
    # println(repeat("#", 200))


    # Check the ordering geometery
    if ordering_geometery != "armchair"
        error("Simualting the Kitaev model on armchair geometery; ordering geometery has not been implemented!")
    end

    
    # Check the direction of the loops
    if direction != "y"
        error("Simulating the Kitaev model on armchair geometery; direction of the loops along the x direction has not been implemented!")
    end

    
    if ordering_geometery == "armchair" && direction == "y"
        loop_list = Matrix{Int64}(undef, inputNx, 2 * inputNy)
        for idx1 in 1 : inputNx
            for idx2 in 1 : 2 * inputNy
                loop_list[idx1, idx2] = 2 * (idx1 - 1) * inputNy + idx2
            end
        end
    end     
    
    return loop_list
end



# Generate the list of indices for each enlarged loop based on armchair geometery
function OrderParameterLoopListArmchair(input_N::Int, input_Ny::Int, ordering_scheme::String)
    """
        Assume using open boundary condition in x direction && periodic boundary condition in y direction
        Implement the list of enlarged loops for open boundary condition in x direction
        input_N: number of total sites
        input_Ny: number of unit cells in the y direction
    """
    
    if ordering_scheme != "armchair"
        error("Setting up topological loops for armchair geometery; ordering scheme not supported!")
    end

    
    # Set up the list of reference sites as the centers of enlarged loops
    centers = collect((2 * input_Ny + 1):(input_N - 2 * input_Ny))
    

    order_loops = []
    for center in centers
        tmp_x = div(center - 1, input_Ny) + 1
        tmp_y = mod(center - 1, input_Ny) + 1
        tmp_loop = []

        if isodd(tmp_x)
            if isodd(tmp_y)
                if tmp_y == 1
                    tmp₁ = center + input_Ny - 1
                    tmp₂ = center - 1
                    tmp₃ = center + 2 * input_Ny - 1
                else
                    tmp₁ = center - 1
                    tmp₂ = center - input_Ny - 1
                    tmp₃ = center + input_Ny - 1
                end

                append!(tmp_loop, [
                    tmp₁,
                    tmp₂,
                    center - input_Ny,
                    center - 2 * input_Ny,
                    center - 2 * input_Ny + 1,
                    center - input_Ny + 1,
                    center + 1,
                    center + input_Ny + 1,
                    center + 2 * input_Ny + 1,
                    center + 2 * input_Ny,
                    center + input_Ny, 
                    tmp₃
                ])
            elseif iseven(tmp_y)
                if tmp_y == input_Ny
                    tmp₁ = center - input_Ny + 1
                    tmp₂ = center + 1
                    tmp₃ = center - 2 * input_Ny + 1
                else
                    tmp₁ = center + 1
                    tmp₂ = center + input_Ny + 1
                    tmp₃ = center - input_Ny + 1
                end

                append!(tmp_loop, [
                    tmp₁,
                    tmp₂,
                    center + input_Ny,
                    center + 2 * input_Ny,
                    center + 2 * input_Ny - 1,
                    center + input_Ny - 1,
                    center - 1,
                    center - input_Ny - 1,
                    center - 2 * input_Ny - 1,
                    center - 2 * input_Ny,
                    center - input_Ny,
                    tmp₃
                ])
            end
        else
            if isodd(tmp_y)
                if tmp_y == 1
                    tmp₁ = center + 3 * input_Ny - 1
                    tmp₂ = center + 2 * input_Ny - 1
                    tmp₃ = center + input_Ny - 1
                    tmp₄ = center - 1
                    tmp₅ = center - input_Ny - 1
                else
                    tmp₁ = center + 2 * input_Ny - 1
                    tmp₂ = center + input_Ny - 1
                    tmp₃ = center - 1
                    tmp₄ = center - input_Ny - 1
                    tmp₅ = center - 2 * input_Ny - 1
                end

                append!(tmp_loop, [
                    center + 1,
                    center + input_Ny + 1,
                    center + input_Ny,
                    center + 2 * input_Ny,
                    tmp₁,
                    tmp₂,
                    tmp₃,
                    tmp₄,
                    tmp₅,
                    center - 2 * input_Ny,
                    center - input_Ny,
                    center - input_Ny + 1
                ])
            else
                if tmp_y == input_Ny
                    tmp₁ = center - 3 * input_Ny + 1
                    tmp₂ = center - 2 * input_Ny + 1
                    tmp₃ = center - input_Ny + 1
                    tmp₄ = center + 1
                    tmp₅ = center + input_Ny + 1
                else
                    tmp₁ = center - 2 * input_Ny + 1
                    tmp₂ = center - input_Ny + 1
                    tmp₃ = center + 1
                    tmp₄ = center + input_Ny + 1
                    tmp₅ = center + 2 * input_Ny + 1
                end

                append!(tmp_loop, [
                    center - 1,
                    center - input_Ny - 1,
                    center - input_Ny,
                    center - 2 * input_Ny,
                    tmp₁,
                    tmp₂,
                    tmp₃,
                    tmp₄,
                    tmp₅,
                    center + 2 * input_Ny,
                    center + input_Ny,
                    center + input_Ny - 1
                ])
            end
        end
        push!(order_loops, tmp_loop)
    end


    return centers, order_loops
end