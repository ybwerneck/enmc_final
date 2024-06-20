
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from sys import argv
import numpy as np
import matplotlib.pyplot as plt
import Poisson


def poisson(dl, L,q=10000,K=1):
    # Define the problem parameters
    Lx = L  # Length of the domain in x-direction
    Ly = L  # Length of the domain in y-direction
    dx = dy = dl
    Nx = Ny = int(L / dl)
    PRE=q
    # Define the boundary conditions
    u_top = 0.0
    u_bottom = 0.0
    u_left = 0.0
    u_right = 0.0

    # Create the grid
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y)

    # Initialize the u matrix
    u = np.zeros((Ny, Nx))
    

    # Apply the boundary conditions
    u[0, :] = u[1, :] - dx * u_top  # Top boundary (Neumann condition)
    u[-1, :] = u[-2, :] + dx * u_bottom  # Bottom boundary (Neumann condition)
    u[:, 0] = u[:, 1] - dy * u_left  # Left boundary (Neumann condition)
    u[:, -1] = u[:, -2] + dy * u_right  # Right boundary (Neumann condition)


    # Source boundary condition
    u[Nx - 2 : , 0:2] = -q
    u[Nx - 2 : , 0:2] = -q

    u[0: 2, -2:] = q
    u[0 : 2, -2:] = q
    u_new = np.copy(u)
    # Solve the Poisson equation
    max_iter = 1000
    tolerance = 1e-4
    for it in range(max_iter):
        u_new[1:-1, 1:-1] = (u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:]) / 4

        # Update the boundary conditions at each iteration
        u_new[0, :] = u_new[1, :] - dx * u_top
        u_new[-1, :] = u_new[-2, :] + dx * u_bottom
        u_new[:, 0] = u_new[:, 1] - dy * u_left
        u_new[:, -1] = u_new[:, -2] + dy * u_right

        # Check for convergence
        if np.max(np.abs(u_new - u)) < tolerance:
            break

        u = np.copy(u_new)

    # Calculate the gradients
    grad_y, grad_x= np.gradient(u)

        
    
    # Plot the solution and gradients
    PRE=1
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    im1 = ax[0, 0].imshow(u, cmap='coolwarm', origin='lower', extent=[0, L, 0, L], vmin=-PRE, vmax=PRE)
    ax[0, 0].set_title('Scalar Field')
    ax[0, 0].set_xlabel('x')
    ax[0, 0].set_ylabel('y')
    plt.colorbar(im1, ax=ax[0, 0])

    im2 = ax[0, 1].imshow(grad_x, cmap='coolwarm', origin='lower', extent=[0, L, 0, L], vmin=-PRE, vmax= PRE)
    ax[0, 1].set_title('Gradient in X-direction')
    ax[0, 1].set_xlabel('x')
    ax[0, 1].set_ylabel('y')
    plt.colorbar(im2, ax=ax[0, 1])

    im3 = ax[1, 0].imshow(grad_y, cmap='coolwarm', origin='lower', extent=[0, L, 0, L], vmin=-PRE, vmax= PRE)
    ax[1, 0].set_title('Gradient in Y-direction')
    ax[1, 0].set_xlabel('x')
    ax[1, 0].set_ylabel('y')
    plt.colorbar(im3, ax=ax[1, 0])

    ax[1, 1].quiver(X, Y, grad_x, grad_y)
    ax[1, 1].set_title('Gradient Vectors')
    ax[1, 1].set_xlabel('x')
    ax[1, 1].set_ylabel('y')

    plt.tight_layout()
    plt.show()
    
    plt.quiver(X,Y,grad_x,grad_y)
    plt.show()
    return grad_x, grad_y

