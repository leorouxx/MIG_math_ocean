import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.integrate as itg

from profil_celerite import ProfilCelerite
c = ProfilCelerite.from_csv("celerite_c2.csv")

ds = 1e-5
dx = 1e-8
dy = 1e-8
dz = 1e-8
w  = 6e5
epsilon = 5e-5
S_SPAN = 7
NB_RAYON = 49
NB_FRONT_D_ONDE = 3
NB_RAYON_AFFICHE = 25
PAS_AFFICHAGE = 500

def c_2(x, y, z):
    #print(c(-z)**-2)
    return c(-z)**-2

# df/dx (x,z)
def gradx(f, x, y, z):
    return (f(x+dx, y, z) - f(x, y, z))/dx

def grady(f, x,y,z):
    return (f(x, y+dy, z) - f(x, y, z))/dy

def gradz(f, x,y,z):
    return (f(x, y, z+dz) - f(x, y, z))/dz



def etape(X, Y, Z, c_2, i):
    X[i] = 2 * X[i-1] - X[i-2] + 2 * w**2 * gradx(c_2, X[i-2], Y[i-2], Z[i-2]) * ds**2
    Y[i] = 2 * Y[i-1] - Y[i-2] + 2 * w**2 * grady(c_2, X[i-2], Y[i-2], Z[i-2]) * ds**2
    Z[i] = 2 * Z[i-1] - Z[i-2] + 2 * w**2 * gradz(c_2, X[i-2], Y[i-2], Z[i-2]) * ds**2

# renvoie l'étape d'après
def Runge_Kutta_etape(s, R_n, func):
    k1 = func(s, R_n)
    k2 = func(s + ds/2, R_n + k1 * 0.5* ds)
    k3 = func(s + ds/2, R_n + 0.5 * ds * k2)
    k4 = func(s + ds,R_n + 0.5 * ds * k3)
    return R_n + (k1 + 2*k2 + 2*k3 + k4) * ds / 6

def temps(X,Y,Z,c_2):
    T = np.zeros(np.shape(X))
    for i in tqdm(range(len(X)-1)):
        T[i+1] = T[i] + 2 * w * c_2(X[i+1],Y[i+1],Z[i+1]) * ds      #*((X[i+1]-X[i])**2 + (Y[i+1]-Y[i])**2 + (Z[i+1]-Z[i])**2)**0.5)
    return np.array(T)


def sigma(X,Y,Z):
    sig = [np.zeros(np.shape(X[0]))]
    for i in tqdm(range(len(X)-1)):
        sig.append(sig[-1]+ ((X[i+1]-X[i])**2 + (Y[i+1]-Y[i])**2 + (Z[i+1]-Z[i])**2)**0.5)
    return np.array(sig)

def theta(X,Y,Z, tau):
    th = [np.zeros(np.shape(X[0]))] # \theta_0 = 0
    for i in tqdm(range(len(X)-3)): # tau est de taille -3
        th.append(th[-1]+ tau[i]*((X[i+1]-X[i])**2 + (Y[i+1]-Y[i])**2 + (Z[i+1]-Z[i])**2)**0.5) # d(sigma) = ((X[i+1]-X[i])**2 + (Y[i+1]-Y[i])**2 + (Z[i+1]-Z[i])**2)**0.5
    return np.array(th)

def repere_frenet(X,Y,Z):
    T_x = (X[1:]-X[:-1]) / ((X[1:]-X[:-1])**2 + (Y[1:]-Y[:-1])**2 + (Z[1:]-Z[:-1])**2)**0.5 # T ~ X[:-1] 
    T_y = (Y[1:]-Y[:-1]) / ((X[1:]-X[:-1])**2 + (Y[1:]-Y[:-1])**2 + (Z[1:]-Z[:-1])**2)**0.5 # T ~ X[:-1]
    T_z = (Z[1:]-Z[:-1]) / ((X[1:]-X[:-1])**2 + (Y[1:]-Y[:-1])**2 + (Z[1:]-Z[:-1])**2)**0.5 # T ~ X[:-1]

    KN_x = (T_x[1:]-T_x[:-1]) / ((X[1:-1]-X[:-2])**2 + (Y[1:-1]-Y[:-2])**2 + (Z[1:-1]-Z[:-2])**2)**0.5 # KN ~ X[:-2]
    KN_y = (T_y[1:]-T_y[:-1]) / ((X[1:-1]-X[:-2])**2 + (Y[1:-1]-Y[:-2])**2 + (Z[1:-1]-Z[:-2])**2)**0.5 # KN ~ X[:-2]
    KN_z = (T_z[1:]-T_z[:-1]) / ((X[1:-1]-X[:-2])**2 + (Y[1:-1]-Y[:-2])**2 + (Z[1:-1]-Z[:-2])**2)**0.5 # KN ~ X[:-2]

    K = (KN_x**2 + KN_y**2 + KN_z**2)**0.5

    N_x = KN_x / K # N ~ X[:-2]
    N_y = KN_y / K # N ~ X[:-2]
    N_z = KN_z / K # N ~ X[:-2]


    B_x = T_y[:-1]*N_z - T_z[:-1]*N_y # B ~ X[:-2]
    B_y = T_z[:-1]*N_x - T_x[:-1]*N_z # B ~ X[:-2]
    B_z = T_x[:-1]*N_y - T_y[:-1]*N_x # B ~ X[:-2]

    tauN_x = (B_x[1:]-B_x[:-1]) / ((X[1:-2]-X[:-3])**2 + (Y[1:-2]-Y[:-3])**2 + (Z[1:-2]-Z[:-3])**2)**0.5 # tau ~ X[:-3]
    tauN_y = (B_y[1:]-B_y[:-1]) / ((X[1:-2]-X[:-3])**2 + (Y[1:-2]-Y[:-3])**2 + (Z[1:-2]-Z[:-3])**2)**0.5 # tau ~ X[:-3]
    tauN_z = (B_z[1:]-B_z[:-1]) / ((X[1:-2]-X[:-3])**2 + (Y[1:-2]-Y[:-3])**2 + (Z[1:-2]-Z[:-3])**2)**0.5 # tau ~ X[:-3]
    tau = (tauN_x**2 + tauN_y**2 + tauN_z**2)**0.5


    return (T_x,T_y,T_z), (N_x,N_y,N_z), (B_x,B_y,B_z), K, tau

def base_centree_rayon(X,Y,Z,T,N,B,K,tau):
    Theta = theta(X,Y,Z,tau)
    E1 = np.cos(Theta)*N + np.sin(Theta)*B
    E2 = np.cos(Theta)*B * np.sin(Theta)*N

    return E1, E2

# renvoie le front t=t
def front_d_onde(X,Y,Z,Temps, t):
    indices = np.zeros(len(Temps[0]),dtype=int)
    plus_proche = abs(Temps[0]-t)
    
    for i in range(len(Temps)):
        indices[abs(Temps[i]-t) < plus_proche] = i
        plus_proche[abs(Temps[i]-t) < plus_proche] = abs(Temps[i]-t)[abs(Temps[i]-t) < plus_proche]
        

    return (np.array([X[indices[i], i] for i in range(len(indices))]),
            np.array([Y[indices[i], i] for i in range(len(indices))]), 
            np.array([Z[indices[i], i] for i in range(len(indices))]))

# dR/ds = equation(R)
def equation(s,R):
    dR=[R[3],R[4],R[5]]
    dR.append(2 * w**2 * gradx(c_2, R[0], R[1], R[2]))
    dR.append(2 * w**2 * grady(c_2, R[0], R[1], R[2]))
    dR.append(2 * w**2 * gradz(c_2, R[0], R[1], R[2]))
    dR.append(2 * w * c_2(R[0], R[1], R[2]))
    return np.array(dR)

def sphere(center = (0, 0, -2000),radius = 10, Nphi=10, Ntheta=20, phi0 = 0, phi1=np.pi, theta=2*np.pi):
    phi = np.linspace(phi0, phi1, Nphi)
    theta = np.linspace(0, theta, Ntheta)
    phi, theta = np.meshgrid(phi, theta)
    phi = phi.reshape(Nphi*Ntheta)
    theta = theta.reshape(Nphi*Ntheta)
    x = center[0] + radius * np.sin(phi) * np.cos(theta)
    y = center[1] + radius * np.sin(phi) * np.sin(theta)
    z = center[2] + radius * np.cos(phi)


    x2 = center[0] + radius * np.sin(phi) * np.cos(theta)   +  np.sin(phi) * np.cos(theta) * w*(c_2(x,y,z))**0.5 * ds
    y2 = center[1] + radius * np.sin(phi) * np.sin(theta)   +  np.sin(phi) * np.sin(theta) * w*(c_2(x,y,z))**0.5 * ds
    z2 = center[2] + radius * np.cos(phi)   +  np.cos(phi)* w*(c_2(x,y,z))**0.5 * ds

    return (x,y,z, x2,y2,z2) 

def plan(x_largeur=10,y_largeur=10, centre=(0,0), z_0=-2000, N=10):
    x_0 = np.linspace(centre[0]-x_largeur/2, centre[0]+x_largeur/2, N)
    y_0 = np.linspace(centre[1]-y_largeur/2, centre[1]+y_largeur/2, N)
    x,y = np.meshgrid(x_0,y_0)
    z = z_0 + np.zeros(N*N)
    return x.reshape(N*N), y.reshape(N*N), z


# Conditions initiales
Nphi = round(NB_RAYON**0.5)
Ntheta = NB_RAYON // Nphi
NB_RAYON = Nphi*Ntheta


x,y,z,x2,y2,z2 = sphere(radius=10, Nphi=Nphi, Ntheta=Ntheta, phi0=3*np.pi/7,  phi1=7*np.pi/10, theta=2, center=(0,0,-1000))

NB_POINTS = round(S_SPAN/ds)//PAS_AFFICHAGE


R = np.zeros((NB_POINTS, 7, NB_RAYON))
R[0] = np.array([x,y,z,(x2-x)/ds, (y2-y)/ds, (z2-z)/ds, np.zeros(NB_RAYON)])
R_n = R[0]

s = np.linspace(0, S_SPAN, NB_POINTS) 
print(s.shape)
print("Calcul des rayons :")
for i in tqdm(range(NB_POINTS*PAS_AFFICHAGE-1)):
    R_n = Runge_Kutta_etape(s, R_n, equation)
    if i%PAS_AFFICHAGE == 0:
        R[i//PAS_AFFICHAGE] = R_n

X = R[:, 0, :]
Y = R[:, 1, :]
Z = R[:, 2, :]
Temps = R[:, 6, :]
print(f'{Temps = }')
print(f'{X = }')


# T,N,B,K,tau = repere_frenet(X,Y,Z)
# E1, E2 = base_centree_rayon(X,Y,Z,T,N,B,K,tau)


t_max = np.min(Temps[-1])

print("Génération du plot :")

ax = plt.figure().add_subplot(projection='3d')
for i in tqdm(range(0, len(X[0]), NB_RAYON//NB_RAYON_AFFICHE)):
    ax.plot(X[Z[:,i]<=0,i], 
            Y[Z[:,i]<=0,i], 
            Z[Z[:,i]<=0,i], color="black")
for i in tqdm(range(1, NB_FRONT_D_ONDE+1)):
    x,y,z=front_d_onde(X, Y, Z, Temps, t_max*(i/(NB_FRONT_D_ONDE)))
    ax.plot(x,y,z)
ax.set_aspect('equal', adjustable='box')

plt.show()
