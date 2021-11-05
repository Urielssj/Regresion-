# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 12:13:15 2021

@author: uriel
"""

import numpy as np

def gradiente_descendente(w_, b_, alpha, x, y):
    '''Algoritmo del gradiente descendente para minimizar el error
       cuadrático medio'''
    N = x.shape[0]      # Cantidad de datos

    # Gradientes: derivadas de la función de error con respecto
    # a los parámetros "w" y "b"
    dw = -(2/N)*np.sum(x*(y-w_*x-b_))
    db = -(2/N)*np.sum(y-w_*x-b_)

    # Actualizar los pesos usando la fórmula del gradiente descendente
    w = w_ - alpha*dw
    b = b_ - alpha*db


    return w, b
def get_abc(x,y):

    # def sistem(x,y):
    n=len(x)
    x_2= (x*x.reshape(1,n)).reshape(n,1)
    x_3=( x*x_2.reshape(1,n)).reshape(n,1)
    x_4=( x*x_3.reshape(1,n)).reshape(n,1)
    
    x_2y=np.sum(y*x_2.reshape(1,n))
    xy=np.sum(y*x.reshape(1,n))
    x=np.sum(x)
    x_2=np.sum(x_2)
    x_3=np.sum(x_3)
    x_4=np.sum(x_4)
    y=np.sum(y)
    
    sistem=np.array([[x_2, x, n],
            [x_3, x_2, x],
            [x_4, x_3, x_2]])
    r= np.array([y,
                 xy,
                 x_2y])
    
    inver=np.linalg.inv(sistem)
    
    abc=np.dot(inver,r)
    a=float(abc[0])
    b=float(abc[1])
    c=float(abc[2])
    return a,b,c

def gradienteConjugado(Ab, x0, verbose = False):
  """
  Esta función resuelve un sistema lineal Ax = b mediante el método
  de Gradiente Conjugado

  Args:
    Ab: Array bidimensional de numpy (Matriz ampliada del sistema)
    x0: Array unidimensional (Valor inicial)
    verbose: Booleano para mostrar o no los resultados relevantes

  Returns:
    x: Array unidimensional (Solución del sistema)
  """

  n = Ab.shape[0]
  x = x0.copy().reshape((n, 1))
  
  # Obtenemos matriz de coeficientes y vector de términos independientes
  A = Ab[:, :-1].copy()
  b = Ab[:, -1].copy()
  b = b.reshape((n, 1))

  r = b - A.dot(x)
  v = r.copy()
  t = float((r.transpose().dot(r)) / (v.transpose().dot(A.dot(v)))[0])
  if verbose:
    print("x^(0) = ", x.reshape((n)))
    print("r^(0) = ", r.reshape((n)))
    
    print("\nv^(1) = ", v.reshape((n)))
    print("t^(1) =", t)
  
  x = x + t * v
  r1 = b - A.dot(x)
  if verbose:
    print("x^(1) = {}".format(x.reshape((n))))
    print("r^(1) = {}".format(r1.reshape((n))))

  for k in range(2, n + 1):
    s = float((r1.transpose().dot(r1)) / (r.transpose().dot(r))[0])
    v = r1 + s * v
    t = float((r1.transpose().dot(r1)) / (v.transpose().dot(A.dot(v)))[0])
    x = x + t * v
    
    r = r1.copy()
    r1 = b - A.dot(x)

    if verbose:
      print("\ns^({}) = {}".format(k - 1, s))
      print("v^({}) = {}".format(k, v.reshape((n))))
      print("t^({}) = {}".format(k, t))
      print("x^({}) = {}".format(k, x.reshape((n))))
      print("r^({}) = {}".format(k, r1.reshape((n))))

  x = x.reshape((n))    
  if verbose:
    print("\nx =", x)
  return x

derivar = lambda f, x, h: (f(x+h)-f(x-h))/(2*h)

def Horner_method(coefs, z):
    n = len(coefs) - 1
    b = coefs[0]
    c = b
    for i in range(1,n):
        b = coefs[i] + z*b
        c = b + z*c
    b = coefs[n] + z*b
    
    return b, c

def Newton_method(f, x_start, umbral, max_iter):
    x_sol = x_start
    if isinstance(f, list):
        for i in range(max_iter):
            f_x, f_prima_x= Horner_method(f, x_start)
            x_sol-= (f_x/f_prima_x)
            if abs(f_x)<umbral:
                break; 
    else:
        for i in range(max_iter):
            f_x = f(x_sol)
            f_prima_x = derivar(f, x_sol, 0.0001)
            x_sol -= f_x/f_prima_x
            if abs(f(x_sol)) < umbral:
                break;
    return x_sol