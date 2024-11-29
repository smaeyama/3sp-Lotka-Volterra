import numpy as np
from numba import jit

# matA[0,:] =  a0, -a1, -a2, -a3
# matA[1,:] = -b0,  b1, -b2, -b3
# matA[2,:] =  c0, -c1, -c2, -c3

@jit(nopython=True)
def calc_derivative(var,matA,nvar):
    dvardt = np.zeros_like(var)
    for i in range(nvar):
        dvardt[i] = var[i] * (matA[i,0] + np.sum(matA[i,1:]*var[:]))
    return dvardt
    
@jit(nopython=True)
def kernel_solve_Lotka_Volterra(var_list,matA,dt,nt,nvar):
    var = var_list[0,:].copy()
    for it in range(nt-1):
        wvar = var.copy()
        k1 = calc_derivative(wvar,matA,nvar)
        wvar = var + 0.5*dt*k1
        k2 = calc_derivative(wvar,matA,nvar)
        wvar = var + 0.5*dt*k2
        k3 = calc_derivative(wvar,matA,nvar)
        wvar = var + dt*k3
        k4 = calc_derivative(wvar,matA,nvar)
        var = var + (dt/6.0)*(k1+2*k2+2*k3+k4)
        var_list[it+1,:] = var[:].copy()
    return var_list
    
def solve_Lotka_Volterra(matA,dt,nt,var0):
    nvar, _ = matA.shape
    var_list = np.zeros([nt,nvar])
    # Initial values
    var_list[0,:] = var0[:].copy()
    # Time integration
    var_list = kernel_solve_Lotka_Volterra(var_list,matA,dt,nt,nvar)
    # Output
    time = dt * np.arange(nt)
    return time, var_list




def sat_I_with_given_E(matA,E):
    _, a1, a2, a3 = -matA[0,:]
    b0, _, b2, b3 = -matA[1,:]
    _, c1, c2, c3 = -matA[2,:]
    a0 = matA[0,0]
    b1 = matA[1,1]
    c0 = matA[2,0]
    I = ((a0-a3*E)*b2+a2*(b0+b3*E))/(a1*b2+a2*b1)
    if (a0-a3*E)*b1/a1-(b0+b3*E)<0: # Test zonal flow quench
        I = (a0-a3*E)/a1
    if a0-a3*E<0: # Test ion-scale turbulence quench
        I = 0.0
    return I

def sat_Z_with_given_E(matA,E):
    _, a1, a2, a3 = -matA[0,:]
    b0, _, b2, b3 = -matA[1,:]
    _, c1, c2, c3 = -matA[2,:]
    a0 = matA[0,0]
    b1 = matA[1,1]
    c0 = matA[2,0]
    Z = ((a0-a3*E)*b1-a1*(b0+b3*E))/(a1*b2+a2*b1)
    if (a0-a3*E)*b1/a1-(b0+b3*E)<0: # Test zonal flow quench
        Z = 0.0
    if a0-a3*E<0: # Test ion-scale turbulence quench
        Z = 0.0
    return Z

def sat_I_with_given_a3E_b3E(matA,a3E,b3E):
    _, a1, a2, a3 = -matA[0,:]
    b0, _, b2, b3 = -matA[1,:]
    _, c1, c2, c3 = -matA[2,:]
    a0 = matA[0,0]
    b1 = matA[1,1]
    c0 = matA[2,0]
    I = ((a0-a3E)*b2+a2*(b0+b3E))/(a1*b2+a2*b1)
    # Test zonal flow quench
    I[(a0-a3E)*b1/a1-(b0+b3E)<0] = (a0-a3E[(a0-a3E)*b1/a1-(b0+b3E)<0])/a1
    # Test ion-scale turbulence quench
    I[a0-a3E<0] = 0.0
    return I
    
def sat_Z_with_given_a3E_b3E(matA,a3E,b3E):
    _, a1, a2, a3 = -matA[0,:]
    b0, _, b2, b3 = -matA[1,:]
    _, c1, c2, c3 = -matA[2,:]
    a0 = matA[0,0]
    b1 = matA[1,1]
    c0 = matA[2,0]
    Z = ((a0-a3E)*b1-a1*(b0+b3E))/(a1*b2+a2*b1)
    # Test zonal flow quench
    Z[(a0-a3E)*b1/a1-(b0+b3E)<0] = 0.0
    # Test ion-scale turbulence quench
    Z[a0-a3E<0] = 0.0
    return Z

def sat_I_ratio_with_given_a3E_b3E(matA,a3E,b3E):
    I_multi = sat_I_with_given_a3E_b3E(matA,a3E,b3E)
    I_single = sat_I_with_given_a3E_b3E(matA,0*a3E,0*b3E)
    return I_multi/I_single
    
def sat_Z_ratio_with_given_a3E_b3E(matA,a3E,b3E):
    Z_multi = sat_Z_with_given_a3E_b3E(matA,a3E,b3E)
    Z_single = sat_Z_with_given_a3E_b3E(matA,0*a3E,0*b3E)
    return Z_multi/Z_single

def sat_E_with_given_I_Z(matA,I,Z):
    _, a1, a2, a3 = -matA[0,:]
    b0, _, b2, b3 = -matA[1,:]
    _, c1, c2, c3 = -matA[2,:]
    a0 = matA[0,0]
    b1 = matA[1,1]
    c0 = matA[2,0]
    E = (c0-c1*I-c2*Z)/c3
    E[E<0] = 0.0
    return E

def equilibrium_IZE(matA):
    _, a1, a2, a3 = -matA[0,:]
    b0, _, b2, b3 = -matA[1,:]
    _, c1, c2, c3 = -matA[2,:]
    a0 = matA[0,0]
    b1 = matA[1,1]
    c0 = matA[2,0]

    IZE_list = []
    # i
    I=0; Z=0; E=0
    IZE_list.append([I,Z,E])
    # ii
    I=0; Z=0; E=c0/c3
    IZE_list.append([I,Z,E])
    # iii
    Z=0; E=0; I=a0/a1
    IZE_list.append([I,Z,E])
    # iv
    Z=0
    I=(a0*c3-a3*c0)/(a1*c3-a3*c1)
    E=(-a0*c1+a1*c0)/(a1*c3-a3*c1)
    if I>=0 and E>=0:
        IZE_list.append([I,Z,E])
    # v
    E=0
    I=(a0*b2+a2*b0)/(a1*b2+a2*b1)
    Z=(a0*b1-a1*b0)/(a1*b2+a2*b1)
    if I>=0 and Z>=0:
        IZE_list.append([I,Z,E])
    # vi
    A=-matA[:,1:]
    b=matA[:,0]
    I, Z, E = np.dot(np.linalg.inv(A),b)
    if I>=0 and Z>=0 and E>=0:
        IZE_list.append([I,Z,E])
    
    IZE_list = np.array(IZE_list)
    return IZE_list


def dIdt_dZdt_with_sat_E(matA,I,Z):
    nvar, _ = matA.shape
    _, a1, a2, a3 = -matA[0,:]
    b0, _, b2, b3 = -matA[1,:]
    _, c1, c2, c3 = -matA[2,:]
    a0 = matA[0,0]
    b1 = matA[1,1]
    c0 = matA[2,0]
    sat_E = sat_E_with_given_I_Z(matA,I,Z)
    dIdt = I * (matA[0,0] + matA[0,1]*I + matA[0,2]*Z + matA[0,3]*sat_E)
    dZdt = Z * (matA[1,0] + matA[1,1]*I + matA[1,2]*Z + matA[1,3]*sat_E)
    return dIdt, dZdt
    
def dIdt_dZdt_wo_E(matA,I,Z):
    nvar, _ = matA.shape
    _, a1, a2, a3 = -matA[0,:]
    b0, _, b2, b3 = -matA[1,:]
    _, c1, c2, c3 = -matA[2,:]
    a0 = matA[0,0]
    b1 = matA[1,1]
    c0 = matA[2,0]
    dIdt = I * (matA[0,0] + matA[0,1]*I + matA[0,2]*Z)
    dZdt = Z * (matA[1,0] + matA[1,1]*I + matA[1,2]*Z)
    return dIdt, dZdt