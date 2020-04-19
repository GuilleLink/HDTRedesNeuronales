#Paquetes a importar
import mnist_reader
import numpy as np
from training import *
from scipy import optimize as op

#Lectura de datos
X_train, y_train = mnist_reader.load_mnist('', kind='train')
X_test, y_test = mnist_reader.load_mnist('', kind='t10k')

#La matriz de entrenamiento se divide entre 1000 por consejo del profesor Samuel
X = X_train/1000.0
m,n = X.shape
y = y_train.reshape(m,1)
Y = (y == np.array(range(10))).astype(int)


#El n me dice numero de entradas y el 397 las neuronas del medio
#Solo se uso una capa por sugerencia del profesor
NeuronalNetwork = np.array([
        n,
        397,
        10
    ])
theta_shapes = np.hstack((
    NeuronalNetwork[1:].reshape(len(NeuronalNetwork)-1, 1),
    (NeuronalNetwork[:-1]+1).reshape(len(NeuronalNetwork)-1,1)
))


flat_thetas = flat_thetas_function([
    np.random.rand(*theta_shape)*0.01
    for theta_shape in theta_shapes
])

#Se debe correr en un python de 64-bits
print("\n---------------- OPTIMIZING ----------------\n")
print("SHAPE DE X:", X.shape)
print("SHAPE DE Y:", Y.shape)
result = op.minimize(
    fun = cost_function,
    x0 = flat_thetas,
    args = (theta_shapes, X, Y),
    method = 'L-BFGS-B',
    jac = backPropagation,
    options = {'disp': True, 'maxiter': 1500}
)
print("\n---------------- OPTIMIZED ----------------\n")

np.savetxt('thetas2.0.txt', result.x)
thetaResult = np.loadtxt('thetas2.0.txt')
thetaResult.size
