"""
    1/18/2024
    Set up the benchmark code for the Heisenberg model on a honeycomb lattice 
    to compare againt the code written in ITensor
"""
import numpy as np

from tenpy.networks.mps import MPS
from tenpy.models.lattice import Honeycomb
from tenpy.networks.site import SpinSite
from tenpy.models.spins import SpinModel
from tenpy.algorithms import dmrg


def example_DMRG_heisenberg_xxx_honeycomb(input_Lx, input_Ly):
    print("finite DMRG, simulate the Heisenberg model on a honeycomb lattice")
    # print("Jz={Jz:.2f}, conserve={conserve!r}".format(Jz=Jz, conserve=conserve))
    print("the length and width of the cylinder are Lx={Lx:.2f}, Ly={Ly:.2f}".format(Lx=input_Lx, Ly=input_Ly))
    
    spinSite = SpinSite(S=0.5, conserve='Sz')
    lattice = Honeycomb(Lx=input_Lx, Ly=input_Ly, sites=spinSite, bc=['open', 'periodic'], bc_MPS='finite')
    model_params = {
        "lattice": lattice, 
        "Jx": 1.0, 
        "Jy": 1.0, 
        "Jz": 1.0
    }
    model = SpinModel(model_params)

    number_of_sites = 2 * input_Lx * input_Ly
    product_state = (["up", "down"] * number_of_sites)[ : number_of_sites]  # initial Neel state
    psi = MPS.from_product_state(model.lat.mps_sites(), product_state, bc=model.lat.bc_MPS)
    dmrg_params = {
        'mixer': True,              # setting this to True helps to escape local minima
        'trunc_params': {
            'chi_max': 2000,
            'svd_min': 1.e-10,
        },
        'max_E_err': 1.e-10,
    }
    info = dmrg.run(psi, model, dmrg_params)
    
    E = info['E']
    print("E = {E:.13f}".format(E=E))
    print("final bond dimensions: ", psi.chi)
    Sz = psi.expectation_value("Sz")  # Sz instead of Sigma z: spin-1/2 operators!
    mag_z = np.mean(Sz)
    print("<S_z> = [{Sz0:.5f}, {Sz1:.5f}]; mean ={mag_z:.5f}".format(Sz0=Sz[0],
                                                                     Sz1=Sz[1],
                                                                     mag_z=mag_z))
    # note: it's clear that mean(<Sz>) is 0: the model has Sz conservation!
    
    # print("correlation length:", psi.correlation_length())
    corrs = psi.correlation_function("Sz", "Sz", sites1=range(10))
    print("correlations <Sz_i Sz_j> =")
    print(corrs)

    
    # Save measurements of the wavefunction into a dictionary
    output_data = measurement(psi, None)
    output_data['E0'] = E
    return E, psi, model, output_data

def measurement(input_psi, data):
    keys = ['entropy', 'Sx', 'Sz', 'corr_XX', 'corr_ZZ']
    if data is None:
        data = dict([(k, []) for k in keys])
    data['entropy'].append(input_psi.entanglement_entropy())
    data['Sz'].append(input_psi.expectation_value('Sz'))
    data['corr_ZZ'].append(input_psi.correlation_function('Sz', 'Sz'))
    # data['Sx'].append(input_psi.expectation_value('Sx'))
    # data['corr_XX'].append(input_psi.correlation_function('Sx', 'Sx'))
    
    return data

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("-" * 100)
    E0, converged_psi, _, measuremnet_data = example_DMRG_heisenberg_xxx_honeycomb(3, 3)
    
    import pickle
    with open('2d_heisenberg_honeycomb_L3W3.pkl', 'wb') as file:
        pickle.dump(measuremnet_data, file)
        print('Output data has been successfully saved to file.')