##################################################################
## Generate input file for disBatch
## Run DMRG for different magnetic field
##################################################################

import numpy as np

def generate_input_file(input_field, task_file):
    '''Generate corresponding folders and input files based on chemical potential'''

    folder_name = "h" + "{:.3f}".format(input_field) + "/"
    task_file.write("cd " + folder_name \
        + " &&  julia --threads=1 2d_kitaev_honeycomb_vacancy.jl" + " &> kitaev_honeycomb_h" \
        + "{:.3f}".format(input_field) + ".log" + "\n")
    

def main():
    field_strength = np.arange(0.0, 0.026, 0.001)
    field_strength = np.around(field_strength, decimals=3)

    submit_file = open("kitaev", "a")
    for tmp in field_strength:
        generate_input_file(tmp, submit_file)
    submit_file.close() 

    # field_strength = np.arange(0.251, 0.501, 0.001)
    # field_strength = np.around(field_strength, decimals=3)

    # submit_file = open("kitaev2", "a")
    # for tmp in field_strength:
    #     generate_input_file(tmp, submit_file)
    # submit_file.close()    


if __name__ == "__main__":
    main()