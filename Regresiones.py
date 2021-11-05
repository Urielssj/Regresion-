# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 09:40:34 2021

@author: uriel
"""
import all_funciones as fun
import matplotlib.pyplot as plt 
import numpy as np 

# from 'C:/Users/uriel/OneDrive/Documentos/Universidad/3er Semestre/Programación phyton/Gradiente descendente.py' import gradiente_descendente(w_, b_, alpha, x, y)

def calcular (m,A,B):
    return A+m*B

def calcular_error(y,y_):
    '''Calcula el error cuadrático medio entre el dato original (y)
       y el dato generado por el modelo (y_)'''
    N = y.shape[0]
    error = np.sum((y -y_)**2)/N
    return error


x = []
y = []
grid  = np.arange(0,35,0.1)
with open('C:/Users/uriel/OneDrive/Documentos/Universidad/3er Semestre/Programación phyton/mtcars.csv','r') as archivo:
    lineas=archivo.read().splitlines()
    lineas.pop(0)
    for l in lineas:
        linea=l.split(',')
        x.append(float(linea[1]))
        y.append(float(linea[4]))
        
x=np.array(sorted(x))
y=np.array(sorted(y))
w = np.random.randn(1)[0]
b = np.random.randn(1)[0]

alpha = 0.0004
nits = 40000

# Entrenamiento
error = np.zeros((nits,1))
for i in range(nits):
    # Actualizar valor de los pesos usando el gradiente descendente
    w, b = fun.gradiente_descendente(w,b,alpha,x,y)
    # Calcular el valor de la predicción
    y_ = calcular(w,b,x)

    # Actualizar el valor del error
    error[i] = calcular_error(y,y_)


    




y_regr = calcular(w,b,x)
plt.plot(1,2,2)
plt.scatter(x,y)
plt.plot(x,y_regr,'r')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

a,b,c=fun.get_abc(x, y)
f= lambda _x:a*_x**2+b*_x+c

# plt.plot(1,2,2)
plt.scatter(x,y)
plt.plot(x_,f(x_),'r')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

