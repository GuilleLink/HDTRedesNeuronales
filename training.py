#Universidad del Valle de Guatemala
#AI
#Redes Neuronales
#Juan Guillermo Sandoval - 17577
import numpy as np
import pandas 
import math
from functools import reduce

#delta sup(l) != a sup(l)
#g es la sigmoide
#2.1 Set a(i) = x(i)
#2.2 Forward Prop z y a
#2.3 Delta, diferencia entre el valor real y el que obtuve
#2.4 Backward Propagation (Jacobian)
#2.5 
#3
#Lista de mnist del significado del label de la columna 1 para todos los elementos del csv
mnist = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

#Se hace una matriz de randoms y se divide entre 1000
#matRand = np.random.rand(10,10)
#matRand = matRand/1000

#Funcion para aplanar matrices
flat_thetas_function = lambda list_of_arrays: reduce(
    lambda acc, v: np.array([*acc.flatten(), *v.flatten()]),
    list_of_arrays
)

#Funcion sigmoide que se le va a aplicar a todos los valores de la matriz
def sigmoide(matrix):
    size = matrix.shape
    for i in range(0, size[0]):
        for j in range(0, size[1]):
            matrix[i][j] = 1/( 1 + math.exp(-matrix[i][j]))
    size = size[0] * size[1]
    return matrix, size

#Convierte el array de thetas en matriz
def inflate_Thetas(thetas, shape):
    layers = len(shape) + 1
    sizes = [shape[0]*shape[1] for shape in shape]
    steps = np.zeros(layers, dtype=int)
    
    for i in range(layers-1):
        steps[i+1]=steps[i]+sizes[i]
        
    return[
        thetas[steps[i]: steps[i+1]].reshape(*shape[i])
        for i in range(layers-1)
    ]

#Forward propagation (2.2) (vista en clase)
def feed_forward(thetas, x):
    a = [x]
    for i in range(len(thetas)):
        a.append(
            sigmoide(
                np.matmul(
                    np.hstack((
                        np.ones(len(x)).reshape(len(x),1),
                        a[i]
                    )), thetas[i].T
                )
            )
        )
    return a

#Calcula el costo (vista en clase)
def cost_function(flat_thetas, shapes, X, Y):
    a = feed_forward(
        inflate_Thetas(flat_thetas, shapes), 
        X
    )

    return -(Y * np.log(a[-1]) + (1 - Y) * np.log(1 - a[-1])).sum() / len(X)

#Back propagation aka cost_function_jacobian
def backPropagation(flat_thetas, shapes, X, Y):
    m, layers = len(X), len(shapes)+1
    thetas = inflate_Thetas(flat_thetas,shapes)
    a = feed_forward(thetas,X)    
    deltas = [*range(layers-1), a[-1]-Y]
    Deltas = []
    for i in range(layers-2,0,-1):
        deltas[i] = np.matmul(deltas[i+1],(np.delete(thetas[i],0,1)))*(a[i]*(1-a[i]))
        
    for i in range(layers-1):
        Deltas.append(
            (
                np.matmul(
                    deltas[i+1].T,
                    np.hstack((
                        np.ones(len(a[i])).reshape(len(a[i]),1),
                        a[i])))
            )/m
        )
    Deltas = np.asarray(Deltas)
        
    return flat_thetas_function(Deltas)
    
