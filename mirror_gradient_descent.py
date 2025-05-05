import numpy as np
#import numdifftools as nd

def negativa_entropia(x): # função de entropia negativa
    n = len(x)
    sum_ = 0

    for i in range(n):
        sum_ += x[i] * np.log(x[i]) - x[i]

    return  sum_


def grad_negativa_entropia(x): # gradiente da entropia negativa (vetor de ln(x_i))
    return np.log(x)

def inv_grad_negativa_entropia(x): # inversa do gradiente da entropia negativa
    return np.exp(x)

def norma_p(x, p = 2): # norma ao quadrado sobre 2
    return (1/p) * np.linalg.norm(x) ** p

def grad_norma_p(x, p = 2): # o gradiente dá o próprio x
    return np.sign(x) * (np.abs(x) ** (p - 1))

def inv_grad_norma_p(x, p = 2): # a inversa é o próprio x
    return np.sign(x) * (np.abs(x) ** (1 / (p-1)))


def mirror_gradient(grad_f, inv_grad_h, grad_h, x0, eta = 1e-1, epsilon = 1e-4, maxIter = 100000, verbose = False):

    xt = x0

    i = 0

   

    while (np.linalg.norm(grad_f(xt)) > epsilon):

        x_before = xt

        xt = inv_grad_h(grad_h(xt) - eta * grad_f(xt))

        if (np.linalg.norm(x_before -xt) <epsilon): # caso o algoritmo não esteja progredindo
            return xt

        if verbose:

            print(np.linalg.norm(grad_f(xt)))
        i += 1
        if (i == maxIter):
            break
    
    return xt