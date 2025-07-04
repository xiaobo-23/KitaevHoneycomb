##################################################################
## Generate input file for disBatch
## Run DMRG for different magnetic field
##################################################################

import numpy as np

def generate_input_file(input_kappa, task_file):
    '''Generate corresponding folders and input files based on chemical potential'''

    folder_name = "kappa" + "{:.3f}".format(input_kappa) + "/"
    task_file.write("cd " + folder_name \
        + " &&  julia --threads=1 2d_tK_honeycomb.jl" + " &> 2K_honeycomb" \
        + "{:.3f}".format(input_kappa) + ".log" + "\n")


def main():
    kappa_values = np.arange(-0.125, -0.525, -0.025)
    kappa_values = np.around(kappa_values, decimals = 3)

    submit_file = open("tK", "a")
    for tmp in kappa_values:
        generate_input_file(tmp, submit_file)
    submit_file.close() 


if __name__ == "__main__":
    main()