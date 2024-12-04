# 04/11/2024
# Defining a custom observer for the DMRG calculation to keep track of 
# the energy, bonod dimension etc.
# Using early stopping to stop the calculation when the energy converges 

using ITensors
using ITensorMPS
# using ITensors: energies

# Defining a custom observer and make this struct a subtype of AbstractObserver
mutable struct CustomObserver <: AbstractObserver
    ehistory::Vector{Float64}
    ehistory_full::Vector{Float64}
    chi::Vector{Int64}            # Bond dimensions  
    etolerance::Float64
    last_energy::Float64
    minsweeps::Int64
    # CustomObserver(etolerance=0.0) = new(Float64[], Int64[], etolerance, 0.0, 0.0, 0)
  end
  
  
  # function CustomObserver(; etolerance=1E-8, minsweeps=2)
  #   return CustomObserver(
  #     Float64[], 
  #     Float64[],
  #     Int[], 
  #     etolerance, 
  #     1000.0,
  #     minsweeps,
  #   )
  # end
  
  # Overloading the measure! method
  function ITensorMPS.measure!(tmpObs::DMRGObserver; kwargs...)
    half_sweep = kwargs[:half_sweep]
    energy = kwargs[:energy]
    sweep = kwargs[:sweep]
    bond = kwargs[:bond]
    psi = kwargs[:psi]
    outputlevel = kwargs[:outputlevel]
  
    if bond == 1
      push!(tmpObs.ehistory_full, energy)
    end
  
    if half_sweep == 2
      if bond == 1
        push!(tmpObs.ehistory, energy)
        push!(tmpObs.chi, maxlinkdim(psi))
      end
    end
  end
  
  
  function ITensorMPS.checkdone!(tmpObs::DMRGObserver; kwargs...)
    sweep = kwargs[:sweep]
    energy = kwargs[:energy]  
    if abs(energy - tmpObs.last_energy) < tmpObs.energy_tol
      println("Stopping DRMG after sweep $sweep: energy converged to $energy")
      return true
    end
    # @show abs(energy - tmpObs.last_energy)
    # @show tmpObs.etolerance
  
    # println("Updating the last energy to $energy  after sweep $sweep")
    tmpObs.last_energy = energy
    return false
  end