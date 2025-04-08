import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

def crossLearningProject(barz, barzg, epsilon):
    N = len(barz)
    z = [cp.Variable(2) for _ in range(N)]
    zg = cp.Variable(2)

    # Objective function
    objective = cp.Minimize(sum(cp.norm(barz[i] - z[i], 2) for i in range(N)) + cp.norm(barzg - zg, 2))

    # Constraints
    constraints = [cp.norm(z[i] - zg, 2) <= epsilon for i in range(N)]

    # Problem definition and solving
    prob = cp.Problem(objective, constraints)
    prob.solve()

    return [zi.value for zi in z], zg.value

# Generar datos de prueba
np.random.seed(43)
barz = [np.random.randn(2) for _ in range(5)]  # 5 puntos aleatorios en 2D
print(barz)
barz[-2] += np.array([0, 5])  # Modificar un punto para que sea diferente
barzg = np.random.randn(2)  # Punto de referencia global
epsilon = .5  # Radio de restricción

# Ejecutar la función
z_values, zg_value = crossLearningProject(barz, barzg, epsilon)

# Graficar resultados
plt.figure(figsize=(6,6))
barz = np.array(barz)
z_values = np.array(z_values)

plt.scatter(barz[:,0], barz[:,1], color='blue', label='barz (originales)')
plt.scatter(z_values[:,0], z_values[:,1], color='red', label='z (optimizados)', marker='x')
plt.scatter(barzg[0], barzg[1], color='green', marker='x', label='barzg (original)')
plt.scatter(zg_value[0], zg_value[1], color='black', marker='x', label='zg (optimizado)')

# Dibujar círculos de restricción
for i in range(len(z_values)):
    circle = plt.Circle(zg_value, epsilon, color='gray', fill=False, linestyle='dashed', alpha=0.5)
    plt.gca().add_patch(circle)

plt.legend()
plt.grid()
plt.xlabel('X')
plt.ylabel('z')
plt.title('Visualización de los puntos optimizados')
plt.show()