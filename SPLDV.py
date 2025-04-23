# Program untuk mencari hasil menggunakan  library NumPy
import numpy as np
# Sistem 1
A1 = np.array([[2, 3],
			[1, -1]])
b1 = np.array([7, 1])
solusi1 = np.linalg.solve(A1, b1)
# Sistem 2
A2 = np.array([[1, 2, 1],
		[3, -1, 2],
		[-2, 3, -1]])
b2 = np.array([10, 5, -9])
solusi2 = np.linalg.solve(A2, b2)

print("Solusi sistem 1 (NumPy):", solusi1)
print("Solusi sistem 2 (NumPy):", solusi2)

# Program untuk mencari hasil menggunakan library Sympy
import sympy as sp
# Sistem 1
x, y = sp.symbols('x y')
pers1 = sp.Eq(2*x + 3*y, 7)
pers2 = sp.Eq(x - y, 1)
solusi1_sym = sp.solve((pers1, pers2), (x, y))

# Sistem 2
x, y, z = sp.symbols('x y z')
pers3 = sp.Eq(x + 2*y + z, 10)
pers4 = sp.Eq(3*x - y + 2*z, 5)
pers5 = sp.Eq(-2*x + 3*y - z, -9)
solusi2_sym = sp.solve((pers3, pers4, pers5), (x, y, z))

print("Solusi sistem 1 (SymPy):", solusi1_sym)
print("Solusi sistem 2 (SymPy):", solusi2_sym)
