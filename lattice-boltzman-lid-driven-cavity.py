import numpy as np
import matplotlib.pyplot as plt

# Set up simulation parameters
nx = 51   # number of lattice points in x-direction
ny = 51   # number of lattice points in y-direction
maxit = 5000   # maximum number of iterations
delta = 1   # lattice spacing
tau = 0.6   # relaxation time
rho0 = 1.0   # reference density
u0 = 0.1   # reference velocity
Re = 100.0   # Reynolds number
nu = u0 * (ny-1) / Re   # kinematic viscosity
omega = 1.0 / tau

# Define the lattice velocity vectors
c = np.array([(x,y) for x in [-1,0,1] for y in [-1,0,1]])

# Define the equilibrium distribution function
def feq(rho,u):
    cu = np.dot(c,u)
    usqr = np.dot(u,u)
    feq = np.zeros((len(c)))
    for i in range(len(c)):
        cu = np.dot(c[i],u)
        feq[i] = rho * 4.0/9.0 * (1.0 - 1.5*usqr + 3.0*cu + 4.5*cu**2 - 1.5*usqr*cu)
    return feq

# Initialize the density, velocity, and distribution functions
f = np.zeros((nx,ny,len(c)))
rho = np.ones((nx,ny)) * rho0
u = np.zeros((nx,ny,2))

# Set the boundary conditions
u[-1,:,:] = u0   # top wall
u[:,0,:] = 0.0   # bottom wall
u[:,-1,:] = 0.0   # right wall
rho[:,0] = rho0   # bottom wall
rho[:,-1] = rho0   # right wall

# Main loop
for it in range(maxit):
    # Collision step
    for i in range(nx):
        for j in range(ny):
            feqval = feq(rho[i,j],u[i,j])
            f[i,j,:] = omega * feqval + (1.0 - omega) * f[i,j,:]
    
    # Streaming step
    for i in range(nx):
        for j in range(ny):
            for k in range(len(c)):
                ip,jp = i + c[k,0], j + c[k,1]
                # Check for boundary nodes
                if ip < 0 or ip >= nx or jp < 0 or jp >= ny:
                    continue
                f[ip,jp,k] = f[i,j,k]
    
    # Update the density and velocity
    rho = np.sum(f,axis=2)
    u[:,:,0] = np.sum(f*c[:,0],axis=2) / rho
    u[:,:,1] = np.sum(f*c[:,1],axis=2) / rho
    
    # Apply the boundary conditions
    # Apply the boundary conditions
    rho[0,:] = rho[1,:]   # left wall
    u[0,:,:] = u0   # left wall
    f[0,:,:] = feq(rho[0,:],u[0,:,:]) + f[1,:,:] - feq(rho[1,:],u[1,:,:])
    
    rho[-1,:] = rho0   # top wall
    f[-1,:,:] = feq(rho[-1,:],u[-1,:,:])
    
    rho[:,0,] = rho0   # bottom wall
    f[:,0,:] = feq(rho[:,0],u[:,0,:])
    
    rho[:,-1] = rho[:,-2,:]   # right wall
    u[:,-1,:] = u[:,-2,:]   # right wall
    f[:,-1,:] = feq(rho[:,-1],u[:,-1,:]) + f[:,-2,:] - feq(rho[:,-2],u[:,-2,:])
    
    # Compute the velocity magnitude
    vel = np.sqrt(u[:,:,0]**2 + u[:,:,1]**2)
    
    # Plot the velocity field
    if it % 100 == 0:
        plt.clf()
        plt.imshow(vel.T,origin='lower')
        plt.colorbar()
        plt.pause(0.001)
    
# Plot the final velocity field
plt.clf()
plt.imshow(vel.T,origin='lower')
plt.colorbar()
plt.show()
