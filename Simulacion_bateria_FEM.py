# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 23:58:08 2020

@author: Sebastián & Jose
"""

# %% CREACIÓN DE LA MALLA Y PARÁMETROS DEL SISTEMA

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import meshio

# Dimensiones de la batería (mm)
Ly = 65
Lx = 18

# Parámetros físicos
kx = 1.09e-3
ky = 3.82e-3
cv = 1.83e-3
fuente = 0.41e-3

# Tamaño de los elementos (mm)
dx = 0.75
dy = 1.25

# Parámetros temporales
dt = 0.1
t_final = 120
t = np.arange(0, t_final, dt)

# Número de nodos
nx = int(Lx/dx) + 1
ny = int(Ly/dy) + 1

NumElem = (nx-1)*(ny-1)

# Puntos de integración por elemento (nxn)
n = 2

# Malla 2D para graficar
x = np.linspace(0, Lx, nx)  
y = np.linspace(0, Ly, ny)  
X, Y = np.meshgrid(x, y)
temperatura_graf = np.zeros((ny, nx))

# Vectores para coordenadas de los puntos
cordx = np.zeros((nx*ny,1))
cordy = np.zeros((nx*ny,1))
cordz = np.zeros((nx*ny,1))

a = 0
for i in range(1, (nx*ny) + 1):
    cordx[i-1] = x[a]
    if (i % ny) == 0 and i != 0:
        a = a + 1
        
for b in range(0, ny):       
    for i in range(b, nx*ny, ny):
        cordy[i] = y[b]
        
cordz[:,0] = 0

# Arreglo de puntos
points = np.array([cordx, cordy, cordz])
points = points.transpose()
points = points[0]

# Arreglo de conectividades
cells = np.zeros((NumElem + nx - 1, n*n)) 

for i in range(0, NumElem + nx - 1):
    cells[i] =  np.array([i, ny + i, ny + i + 1, i + 1])
    
# Eliminación de los elementos falsos producidos por el cambio de columna
indice = []

for j in range(1, nx):
    indice.append(ny*j - 1)
    
cells = np.delete(cells, indice, 0)

cells = cells.astype(int)

# Condiciones iniciales
T0 = 20
temperatura = np.ones((nx*ny, 1))*T0

# Parámetros de las condiciones de frontera
nodos_bc_izq = np.arange(0, ny)
val_bc_izq = 25

nodos_bc_der = np.arange((nx - 1)*ny, nx*ny)
val_bc_der = 25

nodos_bc_inf = np.arange(0, NumElem + nx, ny)
val_bc_inf = 35 

indice.append(max(cells[:,2]))
nodos_bc_sup = np.array(indice)
val_bc_sup = 35

# %% CÁLCULO DE INTEGRALES POR CUADRATURA GAUSSIANA

# Ubicación de los puntos de integración y pesos asociados
def Gauss2DQuad(n):
    ptGaussRef = np.zeros((4,2))
    if n==2:
        ptGaussRef[0,0] = -0.577350
        ptGaussRef[1,0] = -0.577350
        ptGaussRef[2,0] = 0.577350
        ptGaussRef[3,0] = 0.577350
        ptGaussRef[0,1] = -0.577350
        ptGaussRef[1,1] = 0.577350
        ptGaussRef[2,1] = -0.577350
        ptGaussRef[3,1] = 0.577350
        w = np.ones((1,4))    
    return w, ptGaussRef

def matrices(vertices, n, kx, ky, cv, fuente):
    [w, ptGaussRef] = Gauss2DQuad(n)
    
    r = ptGaussRef[:,0]  
    s = ptGaussRef[:,1]
    
    # Interpoladores canónicos
    def Psi0(r, s):
        return (1-r)*(1-s)/4
    def Psi1(r, s):
        return (1+r)*(1-s)/4
    def Psi2(r, s):
        return (1+r)*(1+s)/4
    def Psi3(r, s):
        return (1-r)*(1+s)/4
    
    def dPsi00(r, s):
        return -(1-s)/4
    def dPsi10(r, s):
        return (1-s)/4
    def dPsi20(r, s):
        return (1+s)/4
    def dPsi30(r, s):
        return -(1+s)/4
    def dPsi01(r, s):
        return -(1-r)/4
    def dPsi11(r, s):
        return -(1+r)/4
    def dPsi21(r, s):
        return (1+r)/4
    def dPsi31(r, s):
        return (1-r)/4
    
    # Matriz de los interpoladores canónicos
    def N(r, s):
        N = np.zeros((1, 4))
        N[0,0] = Psi0(r, s)
        N[0,1] = Psi1(r, s)
        N[0,2] = Psi2(r, s)
        N[0,3] = Psi3(r, s)
        return N
    
    def Ntrans(r, s):
        Ntrans = np.zeros((4, 1))
        Ntrans[0,0] = Psi0(r, s)
        Ntrans[1,0] = Psi1(r, s)
        Ntrans[2,0] = Psi2(r, s)
        Ntrans[3,0] = Psi3(r, s)
        return Ntrans
        
    # Matriz de derivadas de los interpoladores canónicos
    def D(r, s):
        D = np.zeros((2,4))
        D[0,0] = dPsi00(r, s)
        D[0,1] = dPsi10(r, s)
        D[0,2] = dPsi20(r, s)
        D[0,3] = dPsi30(r, s)
        D[1,0] = dPsi01(r, s)
        D[1,1] = dPsi11(r, s)
        D[1,2] = dPsi21(r, s)
        D[1,3] = dPsi31(r, s)
        return D
    
    def Dtrans(r, s):
        Dtrans = np.zeros((4,2))
        Dtrans[0,0] = dPsi00(r, s)
        Dtrans[1,0] = dPsi10(r, s)
        Dtrans[2,0] = dPsi20(r, s)
        Dtrans[3,0] = dPsi30(r, s)
        Dtrans[0,1] = dPsi01(r, s)
        Dtrans[1,1] = dPsi11(r, s)
        Dtrans[2,1] = dPsi21(r, s)
        Dtrans[3,1] = dPsi31(r, s)
        return Dtrans
    
    # Matriz de conductividad
    def K(kx, ky):
        K = np.zeros((2, 2))
        K[0, 0] = kx
        K[1, 1] = ky
        return K
    
    # Cálculo del determinante del jacobiano
    evalDetJacb = np.zeros((np.size(r), 1))
    for i in range(0, np.size(r)):
        evalDetJacb[i] = abs(LA.det(D(r[i],s[i]).dot(vertices)))
    
    # Integración por cuadratura gaussiana
    def integQuad(f, w, evalDetJacb):
        integral = 0
        for i in range(0, np.size(r)):
            integral = integral + f*w[:,i]*evalDetJacb[i,:]
            return integral
            
    # Matrices elementales
    K_elem = integQuad(Dtrans(r[i],s[i]).dot(K(kx,ky)) \
             .dot(D(r[i],s[i])), w, evalDetJacb)
        
    M_elem = integQuad(Ntrans(r[i],s[i]).dot(N(r[i],s[i])), w, evalDetJacb)
    M_elem = cv*M_elem
        
    F_elem = integQuad(N(r[i],s[i])*fuente, w, evalDetJacb) 
    F_elem = np.reshape(F_elem, (np.size(r), 1))

    return K_elem, M_elem, F_elem

# %% RECORRIDO POR ELEMENTO Y ENSAMBLE DE MATRICES GLOBALES
   
# Matrices globales
K_global = np.zeros((nx*ny, nx*ny))
M_global = np.zeros((nx*ny, nx*ny))
F_global = np.zeros((nx*ny, 1))

vertices = np.zeros((4,2))
     
# Recorrido por cada elemento de la malla 
for i in range(0, NumElem):
    
    # Coordenadas de los vértices del elemento
    for j in range(0, np.size(cells, 1)):
        vertices[j,0] = points[cells[i,j], 0]
        vertices[j,1] = points[cells[i,j], 1]
        
    K_elem, M_elem, F_elem = matrices(vertices,n,kx,ky,cv,fuente)
    
    # Ensamble de matrices globales
    for j in range(0, np.size(cells, 1)):
        K_global[cells[i,j], cells[i,:]] = K_global[cells[i,j], cells[i,:]] + K_elem[j,:]
        M_global[cells[i,j], cells[i,:]] = M_global[cells[i,j], cells[i,:]] + M_elem[j,:]    
        F_global[cells[i,j]] = F_global[cells[i,j]] + F_elem[j]

    R_global = M_global/dt
    
    # Matriz de rigidez
    L_global = R_global + K_global
    
# %% CÁLCULO DE TEMPERATURA

def condFront(nodos, valor):
    for i in range(0, len(nodos)):
        L_global[nodos[i], :] = 0
        L_global[nodos[i], nodos[i]] = 1
        b[nodos[i]] = valor
    return L_global, b

for k in range(0, len(t)):
    
    plt.clf()
    
    # Vector de cargas
    b = R_global.dot(temperatura) + F_global
    
    # Condiciones de frontera
    L_global, b = condFront(nodos_bc_inf, val_bc_inf)
    L_global, b = condFront(nodos_bc_sup, val_bc_sup)
    L_global, b = condFront(nodos_bc_izq, val_bc_izq)
    L_global, b = condFront(nodos_bc_der, val_bc_der)
    
    temperatura = LA.solve(L_global, b)
    
    # Gráfica de temperatura en función de la posición
    for i in range(1, nx + 1):
        temperatura_graf[:, i-1] = temperatura[ny*(i-1):ny*i, 0]
    
    fig = plt.figure(1)
    plt.contourf(X, Y, temperatura_graf, 100, cmap='inferno')  
    plt.colorbar()
    plt.axis('image')
    plt.xlabel('Posición en x (mm)') 
    plt.ylabel('Posición en y (mm)')  
    plt.title('Temperatura (°C)')          
    plt.show() 
    plt.pause(0.01)
    
# %% EXPORTACIÓN A PARAVIEW

cells = [("quad", np.array(cells))]
    
temperatura = list(temperatura)
temperatura = dict(temperatura=np.array(temperatura))

meshio.write_points_cells(
    "bateriaFEM.vtk",
    points,
    cells,
    point_data=temperatura
    )