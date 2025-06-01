using ITensors

i, j, k = Index.((2, 2, 2))
A = random_itensor(i, j)
B = random_itensor(j, k)

# Perform tensor operations on CPU
A * B

###########################################
using Metal # This will trigger the loading of `NDTensorsMetalExt` in the background

# Move tensors to Apple GPU
Amtl = mtl(A)
Bmtl = mtl(B)

# Perform tensor operations on Apple GPU
Amtl * Bmtl