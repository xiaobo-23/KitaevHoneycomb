##################################################################
## Generate input file for disBatch
## Run DMRG for different magnetic field
##################################################################

import numpy as np

def generate_input_file(input_field, task_file):
    '''Generate corresponding folders and input files based on chemical potential'''

    folder_name = "gamma" + "{:.2f}".format(input_field) + "/"
    task_file.write("cd " + folder_name \
        + " &&  julia --threads=1 2d_kitaev_gamma.jl" + " &> kitaev_honeycomb_gamma" \
        + "{:.2f}".format(input_field) + ".log" + "\n")
    

def main():
    field_strength = np.arange(0.0, 1.01, 0.01)
    field_strength = np.around(field_strength, decimals=2)

    submit_file = open("kitaev", "a")
    for tmp in field_strength:
        generate_input_file(tmp, submit_file)
    submit_file.close() 

if __name__ == "__main__":
    main()