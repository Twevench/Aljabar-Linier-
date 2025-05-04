import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =====================
# Solusi dengan NumPy
# =====================
A1, b1 = np.array([[2, 3], [1, -1]]), np.array([7, 1])
A2, b2 = np.array([[1, 2, 1], [3, -1, 2], [-2, 3, -1]]), np.array([10, 5, -9])
solusi1, solusi2 = np.linalg.solve(A1, b1), np.linalg.solve(A2, b2)

# =====================
# Visualisasi
# =====================

# Sistem 1 (2D)
x_vals = np.linspace(-5, 5, 400)
y1_vals = (7 - 2*x_vals) / 3
y2_vals = x_vals - 1

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(x_vals, y1_vals, label='2x + 3y = 7')
plt.plot(x_vals, y2_vals, label='x - y = 1')
plt.plot(*solusi1, 'ro', label='Solusi')
plt.text(solusi1[0] + 0.2, solusi1[1], f'({solusi1[0]:.2f}, {solusi1[1]:.2f})', color='red')
plt.xlabel('x'); plt.ylabel('y'); plt.title('Sistem 1 (2D)')
plt.grid(True); plt.legend(); plt.axis('equal')

# Sistem 2 (3D)
ax = plt.subplot(1, 2, 2, projection='3d')
xg, yg = np.meshgrid(np.linspace(-5, 5, 10), np.linspace(-5, 5, 10))
zg1 = 10 - xg - 2*yg
zg2 = (5 - 3*xg + yg) / 2
zg3 = -9 + 2*xg - 3*yg

ax.plot_surface(xg, yg, zg1, alpha=0.5, color='red')
ax.plot_surface(xg, yg, zg2, alpha=0.5, color='green')
ax.plot_surface(xg, yg, zg3, alpha=0.5, color='blue')
ax.scatter(*solusi2, color='black', s=50, label='Solusi')
ax.text(solusi2[0]+0.2, solusi2[1], solusi2[2], f'({solusi2[0]:.2f}, {solusi2[1]:.2f}, {solusi2[2]:.2f})', color='black')
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
ax.set_title('Sistem 2 (3D)')

plt.tight_layout()
plt.show()
