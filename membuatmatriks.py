# numpy = numerical python
# membuat list
print('-'*50)
cth_list = [5, 'enam', 7,8,0, True]
print ("ini adalah list:")
print(cth_list)
print('\n')

# mengindeksasi element list
print("elemen kedua dari list tsb adalah: ", cth_list[1])
print ("\n")

# membuat matriks menggunakan list
matriks_list = ([7, 8, 9,],
                [10, 11, 12])
print ("ini adalah matriks dari list")
print(matriks_list)
print("\n")
print("elemen baris kedua kolom ketiga dari matriks tersebut adalah", matriks_list [1][2])
print("\n")

import numpy as np
# membuat array
cth_array = np.array([7,8,9,10,11])
print ("ini adalah array:")
print(cth_array)
print("\n")

# Membuat matriks menggunakan Array
matriks_array = np.array ([[-2, 1, -1], 
                           [-3,-1, 2],
                           [-2, 1, 2]])
print("ini adalah matriks array")
print(matriks_array)
