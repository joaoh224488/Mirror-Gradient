import numpy as np

def h(x): # função de entropia negativa
    n = len(x)
    sum_ = 0

    for i in range(n):
        sum_ += x[i] * np.log(x[i]) - x[i]

    return  sum_


def grad_h(x): # gradiente da entropia negativa (vetor de ln(x_i))
    return np.log(x)

def inv_grad_h(x): # inversa do gradiente da entropia negativa
    return np.exp(x)

def h_2(x): # norma ao quadrado sobre 2
    return (1/2) * np.linalg.norm(x) ** 2

def grad_h_2(x): # o gradiente dá o próprio x
    return x

def inv_grad_h_2(x): # a inversa é o próprio x
    return x


def mirror_gradient(f, grad_f, inv_grad_h, grad_h, x0):

    xt = x0

    eta = 1e-1

    total = 10000
    i = 0

    while (np.linalg.norm(grad_f(xt)) > 1e-3):

        xt = inv_grad_h(grad_h(xt) - eta * grad_f(xt))

        #print(xt)

        print(np.linalg.norm(grad_f(xt)))
        i += 1
        if (i == total):
            break
    
    return xt