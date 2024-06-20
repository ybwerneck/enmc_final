# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 22:28:36 2023

@author: yanbw
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 13:03:24 2020

@author: Mohammad Asif Zaman

Code for testng differentiation matrix operators

Solves 𝛁²u(x,y) = g(x,y) 

May 2, 20200: v1:
    - Dirichlet boundary condition implementation 
    
            


"""


import time
import math
import numpy as np
import pylab as py
import scipy.sparse as sp                 # import sparse matrix library
from scipy.sparse.linalg import spsolve

py.rcParams.update({'font.size': 14})

# import the file where the differentiation matrix operators are defined



def Diff_mat_1D(Nx):
    
    # First derivative
    D_1d = sp.diags([-1, 1], [-1, 1], shape = (Nx,Nx)) # A division by (2*dx) is required later.
    D_1d = sp.lil_matrix(D_1d)
    D_1d[0,[0,1,2]] = [-3, 4, -1]               # this is 2nd order forward difference (2*dx division is required)
    D_1d[Nx-1,[Nx-3, Nx-2, Nx-1]] = [1, -4, 3]  # this is 2nd order backward difference (2*dx division is required)
    
    # Second derivative
    D2_1d =  sp.diags([1, -2, 1], [-1,0,1], shape = (Nx, Nx)) # division by dx^2 required
    D2_1d = sp.lil_matrix(D2_1d)                  
    D2_1d[0,[0,1,2,3]] = [2, -5, 4, -1]                    # this is 2nd order forward difference. division by dx^2 required. 
    D2_1d[Nx-1,[Nx-4, Nx-3, Nx-2, Nx-1]] = [-1, 4, -5, 2]  # this is 2nd order backward difference. division by dx^2 required.
    
    return D_1d, D2_1d




def Diff_mat_2D(Nx,Ny):
    # 1D differentiation matrices
    Dx_1d, D2x_1d = Diff_mat_1D(Nx)
    Dy_1d, D2y_1d = Diff_mat_1D(Ny)


    # Sparse identity matrices
    Ix = sp.eye(Nx)
    Iy = sp.eye(Ny)


    
    # 2D matrix operators from 1D operators using kronecker product
    # First partial derivatives
    Dx_2d = sp.kron(Iy,Dx_1d)
    Dy_2d = sp.kron(Dy_1d,Ix)
    
    # Second partial derivatives
    D2x_2d = sp.kron(Iy,D2x_1d)
    D2y_2d = sp.kron(D2y_1d,Ix)
    
   
    
    # Return compressed Sparse Row format of the sparse matrices
    return Dx_2d.tocsr(), Dy_2d.tocsr(), D2x_2d.tocsr(), D2y_2d.tocsr()


# Defining custom plotting functions
def my_contourf(x,y,F,ttl):
    cnt = py.contourf(x,y,F,41,cmap = 'inferno')
    py.colorbar()
    py.xlabel('x'); py.ylabel('y'); py.title(ttl)
    return 0
    

def my_scatter(x,y,clr,ttl):
    py.plot(x,y,'.',markersize=2,color=clr)
    py.xlabel('x'); py.ylabel('y'); py.title(ttl)
    return 0
    



def solve_poisson(dl, L, q, mi_w, K):
    a = q*mi_w/K;
    
    #==============================================================================
    # Dirichlet/Neumann boundary conditions at outerwalls (boundary condition type is defined through boundary operators)
    uL = 0
    uR = 0
    uT = 0
    uB = 0
    #==============================================================================
    
    
    
    #==============================================================================
    # Dirichlet boundary conditions at an inner rectangular region
    ub2 =-a/2    
    ub3= -ub2        # boundary value
     
    xb2 = [0.0,0.02]        # lower and upper limits of x defining the inner boundary region
    yb2 = [L-0.02,L]        # lower and upper limits of y defining the inner boundary region
    
    print( xb2,yb2)
    xb3 = [L-0.02,L]        # lower and upper limits of x defining the inner boundary region
    yb3 = [0.0,0.02]    
    #==============================================================================
    
    
    n=int(L/dl)
    #==============================================================================
    # Define independent variables
    Nx = n                      # No. of grid points along x direction
    Ny = n                         # No. of grid points along y direction
    x = np.linspace(0,L,Nx)        # x variables in 1D
    y = np.linspace(0,L,Ny)        # y variable in 1D
    #==============================================================================
    
    #//////////////////////////////////////////////////////////////////////////////
    #//////////////////////////////////////////////////////////////////////////////
    #//////////////////////////////////////////////////////////////////////////////
    
    #==============================================================================
    
    
    dx = x[1] - x[0]                # grid spacing along x direction
    dy = y[1] - y[0]                # grid spacing along y direction
    
    X,Y = np.meshgrid(x,y)          # 2D meshgrid
    
    # 1D indexing
    Xu = X.ravel()                  # Unravel 2D meshgrid to 1D array
    Yu = Y.ravel()
    
    
    # Source function (right hand side vector)
    g = np.zeros(Nx*Ny) 
    
    print(Xu)
     
    
    # Loading finite difference matrix operators
    
    Dx_2d, Dy_2d, D2x_2d, D2y_2d = Diff_mat_2D(Nx,Ny)   # Calling 2D matrix operators from funciton
    
    
    
    #//////////////////////////////////////////////////////////////////////////////
    #//////////////////////////////////////////////////////////////////////////////
    #//////////////////////////////////////////////////////////////////////////////
    
    
    #==============================================================================
    # Boundary indices
    start_time = time.time()
    ind_unravel_L = np.squeeze(np.where(Xu==x[0]))          # Left boundary
    ind_unravel_R = np.squeeze(np.where(Xu==x[Nx-1]))     # Right boundary
    ind_unravel_B = np.squeeze(np.where(Yu==y[0]))          # Bottom boundary
    ind_unravel_T = np.squeeze(np.where(Yu==y[Ny-1]))       # Top boundary

    
    ind_boundary2_unravel = np.squeeze(np.where((Xu>xb2[0]) & (Xu<xb2[1]) & (Yu>yb2[0]) & (Yu<yb2[1])))  # inner boundaries defined by xb2 and yb2
    ind_boundary2 = np.where((X>=xb2[0]) & (X<=xb2[1]) & (Y>=yb2[0]) & (Y<=yb2[1]))    #  inner boundaries
    
    ind_boundary3_unravel = np.squeeze(np.where((Xu>xb3[0]) & (Xu<xb3[1]) & (Yu>yb3[0]) & (Yu<yb3[1])))  # inner boundaries defined by xb2 and yb2
    ind_boundary3 = np.where((X>=xb3[0]) & (X<=xb3[1]) & (Y>=yb3[0]) & (Y<=yb3[1]))    #  inner boundaries
    isin = lambda b,c:np.array([(b!=a) for a in c]).all()
        
    ind_boundary_unravel = np.squeeze(np.where((Xu==x[0]) | (Xu==x[Nx-1]) | (Yu==y[0]) | (Yu==y[Ny-1])))  # outer boundaries 1D unravel indices
    ind_boundary = np.where( ((X == x[0]) | (X == x[Nx-1]) | (Y == y[0]) | (Y == y[Ny-1])) )
   # outer boundary
    
    print("Boundary search time = %1.6s" % (time.time()-start_time))
    
    # Plot solution domain (with boundary)
    
    py.close('all')
    my_scatter(X,Y,'g','Solution grid')
    my_scatter(X[ind_boundary], Y[ind_boundary],'r','Solutiohn grid with boundary')
    my_scatter(X[ind_boundary2], Y[ind_boundary2],'b','')
    my_scatter(X[ind_boundary3], Y[ind_boundary3],'b','')
    #==============================================================================
    
    #//////////////////////////////////////////////////////////////////////////////
    #//////////////////////////////////////////////////////////////////////////////
    #//////////////////////////////////////////////////////////////////////////////
    
    
    #==============================================================================
    # Construction of the system matrix
    start_time = time.time()
    I_sp = sp.eye(Nx*Ny).tocsr()
    L_sys = D2x_2d/dx**2 + D2y_2d/dy**2     # system matrix without boundary conditions
    
    # Boundary operators
    BD = I_sp       # Dirichlet boundary operator
    BNx = Dx_2d     # Neumann boundary operator for x component
    BNy = Dy_2d     # Neumann boundary operator for y component    # Neumann boundary operator for y component
    
    # Selectively replace the rows of the system matrix that correspond to boundary value points. We replace these rows with 
    # those of the boundary operator
    
    # L_sys[ind_boundary_unravel,:] = BD[ind_boundary_unravel,:]    # Boundaries at the four edges
    
    L_sys[ind_boundary2_unravel,:] = BD[ind_boundary2_unravel,:]  # Boundaries defined by xb2 and yb2 (inside boundaries)
    L_sys[ind_boundary3_unravel,:] = BD[ind_boundary3_unravel,:]  # Boundaries defined by xb2 and yb2 (inside boundaries)
    
    
    L_sys[ind_unravel_T,:] = BNy[ind_unravel_T,:]    # Boundaries at the top layer
    L_sys[ind_unravel_B,:] = BD[ind_unravel_B,:]    # Boundaries at the bottom layer
    L_sys[ind_unravel_L,:] = BNx[ind_unravel_L,:]    # Boundaries at the left layer
    L_sys[ind_unravel_R,:] = BNx[ind_unravel_R,:]    # Boundaries at the right edges
    
    #==============================================================================
    
    #//////////////////////////////////////////////////////////////////////////////
    #//////////////////////////////////////////////////////////////////////////////
    #//////////////////////////////////////////////////////////////////////////////
    
    
    #==============================================================================
    # Construction of right hand vector (function of x and y)
    b = g
    # Insert boundary values at the boundary points
    b[ind_unravel_L] = uL
    b[ind_unravel_R] = uR
    b[ind_unravel_T] = uT
    b[ind_unravel_B] = uB
    
    b[ind_boundary2_unravel] = ub2
    b[ind_boundary3_unravel] = ub3
    #==============================================================================
    
    
    print("System matrix and right hand vector computation time = %1.6s" % (time.time()-start_time)) 
    
    
    
    #==============================================================================
    # solve
    start_time = time.time()
    u = spsolve(L_sys,b).reshape(Ny,Nx)
    print("spsolve() time = %1.6s" % (time.time()-start_time))
    #==============================================================================
    
    #//////////////////////////////////////////////////////////////////////////////
    #//////////////////////////////////////////////////////////////////////////////
    #//////////////////////////////////////////////////////////////////////////////
    
    #==============================================================================
    # Plot solution
    py.figure(figsize = (14,7))
    my_contourf(x,y,u,r'$\nabla^2 u = 0$')
    
    return u
#======================================
