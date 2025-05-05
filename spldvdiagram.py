import numpy as np
import sympy as sp
import matplotlib.pyplot as plt 

# soal 1 numpy
A = np.array([[2, 3],
              [1, -1]])
b = np.array([7, 1])

solusi_1 = np.linalg.solve(A, b)
print(f"Solusi soal 1 dengan NumPy = x: {solusi_1[0]} y: {solusi_1[1]}")

# Visualisasi untuk soal 1 dengan NumPy
def plot_soal_1_numpy():
    # Definisikan range x
    x_vals = np.linspace(-5, 5, 100)

    # Persamaan garis
    y1_vals = (7 - 2 * x_vals) / 3  # Dari persamaan 2x + 3y = 7
    y2_vals = x_vals - 1           # Dari persamaan x - y = 1

    # Plot garis
    plt.plot(x_vals, y1_vals, label="2x + 3y = 7", color="blue")
    plt.plot(x_vals, y2_vals, label="x - y = 1", color="green")

    # Plot solusi
    plt.scatter(solusi_1[0], solusi_1[1], color="red", label=f"Solusi: ({solusi_1[0]:.2f}, {solusi_1[1]:.2f})")

    # Tambahkan label dan legend
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.title("Diagram Soal 1 dengan NumPy")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

# Panggil fungsi untuk menampilkan diagram
plot_soal_1_numpy()

# soal 1 sympy
x, y = sp.symbols('x y')

persamaan1 = sp.Eq(2*x + 3*y, 7)
persamaan2 = sp.Eq(x - y, 1)

solusi_2 = sp.solve((persamaan1, persamaan2), (x, y))
print(f"Solusi soal 1 dengan SymPy = {solusi_2}")

def plot_soal_1_sympy():
# Visualisasi untuk soal 1 dengan SymPy
    x_sol = float(solusi_2[x])
    y_sol = float(solusi_2[y])

    # Buat titik-titik untuk x (pakai list biasa)
    x_vals = [i/10 for i in range(-50, 51)]  # dari -5 sampai 5

    # Hitung y untuk masing-masing garis
    y1_vals = [(7 - 2*i)/3 for i in x_vals]  # dari 2x + 3y = 7 → y = (7 - 2x)/3
    y2_vals = [i - 1 for i in x_vals]       # dari x - y = 1 → y = x - 1

    # Gambar grafik
    plt.plot(x_vals, y1_vals, label="2x + 3y = 7", color="purple")
    plt.plot(x_vals, y2_vals, label="x - y = 1", color="blue")

    # Gambar titik solusi
    plt.scatter(x_sol, y_sol, color="red", label=f"Solusi: ({x_sol:.2f}, {y_sol:.2f})")

    # Tambahan styling
    plt.axhline(0, color='gray', linewidth=0.5)
    plt.axvline(0, color='gray', linewidth=0.5)
    plt.grid(True)
    plt.legend()
    plt.title("Visualisasi soal 1 dengan SymPy")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

# Panggil fungsi untuk menampilkan diagram
plot_soal_1_sympy()

# soal 2 numpy
Matriks_A = np.array([[1, 2, 1],
                      [3, -1, 2],
                      [-2, 3, -1]])
Matriks_B = np.array([10, 5, -9])

solusi_3 = np.linalg.solve(Matriks_A, Matriks_B)
print(f"Solusi soal 2 dengan NumPy = x: {solusi_3[0]}, y: {solusi_3[1]}, z: {solusi_3[2]}")

# Visualisasi untuk soal 2
def plot_soal_2():
    # Definisikan range untuk x dan y
    x_vals = np.linspace(-10, 10, 100)
    y_vals = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Definisikan persamaan bidang
    Z1 = (10 - X - 2 * Y)  # Dari persamaan 1x + 2y + 1z = 10
    Z2 = (5 - 3 * X + Y) / 2  # Dari persamaan 3x - 1y + 2z = 5
    Z3 = (-9 + 2 * X - 3 * Y)  # Dari persamaan -2x + 3y - 1z = -9

    # Buat plot 3D
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot bidang
    ax.plot_surface(X, Y, Z1, alpha=0.5, rstride=100, cstride=100, color='blue', label="1x + 2y + 1z = 10")
    ax.plot_surface(X, Y, Z2, alpha=0.5, rstride=100, cstride=100, color='green', label="3x - 1y + 2z = 5")
    ax.plot_surface(X, Y, Z3, alpha=0.5, rstride=100, cstride=100, color='red', label="-2x + 3y - 1z = -9")

    # Plot solusi
    ax.scatter(solusi_3[0], solusi_3[1], solusi_3[2], color="black", s=50, label=f"Solusi: ({solusi_3[0]:.2f}, {solusi_3[1]:.2f}, {solusi_3[2]:.2f})")

    # Tambahkan label dan legend
    ax.set_title("Visualisasi Soal 2 dengan Numpy")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.legend()
    plt.show()

# Panggil fungsi untuk menampilkan diagram
plot_soal_2()

Matriks_C = sp.Matrix([[1, 2, 1],
              [3, -1, 2],
              [-2, 3, -1]])
Matriks_D = sp.Matrix([10, 5, -9])

if Matriks_C.det() == 0:
    print ("Solusi soal 2 dengan SymPy = Matriks tidak memiliki solusi")
else:
    hasil = Matriks_C.solve(Matriks_D)
    print(f"Solusi soal 2 dengan SymPy = x: {hasil[0]}, y: {hasil[1]}, z: {hasil[2]}")

# Visualisasi untuk soal 2 dengan SymPy

x, y, z = sp.symbols('x y z')

persamaan1 = sp.Eq(x + 2*y + z, 10)
persamaan2 = sp.Eq(3*x - y + 2*z, 5)
persamaan3 = sp.Eq(-2*x + 3*y - z, -9)

# Langkah 2: Selesaikan sistem
solusi = sp.solve((persamaan1, persamaan2, persamaan3), (x, y, z))
print(f"Solusi sistem = {solusi}")

# Check if a solution was found
if solusi:
    # If a solution exists, it will be a dictionary
    x_sol = float(solusi[x])
    y_sol = float(solusi[y])
    z_sol = float(solusi[z])
else:
    print("Sistem persamaan tidak memiliki solusi unik.")
    # Handle the case where no solution is found, e.g., set default values
    x_sol, y_sol, z_sol = 0, 0, 0  

# Langkah 3: Siapkan grid untuk x dan y

# Buat grid X dan Y
x = np.linspace(-10, 10, 20)
y = np.linspace(-10, 10, 20)
X, Y = np.meshgrid(x, y)

# Bidang 1: x + 2y + z = 10 -> z = 10 - x - 2y
Z1 = 10 - X - 2*Y

# Bidang 2: 3x - y + 2z = 5 -> z = (5 - 3x + y) / 2
Z2 = (5 - 3*X + Y) / 2

# Bidang 3: -2x + 3y - z = -9 -> z = -2x + 3y + 9
Z3 = -2*X + 3*Y + 9

# Plot bidang-bidang
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z1, alpha=0.5, color='red', label='x + 2y + z = 10')
ax.plot_surface(X, Y, Z2, alpha=0.5, color='green', label='3x - y + 2z = 5')
ax.plot_surface(X, Y, Z3, alpha=0.5, color='blue', label='-2x + 3y - z = -9')

# Label sumbu
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Visualisasi Tiga Bidang dari Sistem Linear')

# Legend manual karena plot_surface tidak mendukung label langsung
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='red', edgecolor='r', label='x + 2y + z = 10'),
    Patch(facecolor='green', edgecolor='g', label='3x - y + 2z = 5'),
    Patch(facecolor='blue', edgecolor='b', label='-2x + 3y - z = -9')
]
ax.legend(handles=legend_elements)

plt.show()
