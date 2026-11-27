import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.integrate as itg

from profil_celerite import ProfilCelerite
c = ProfilCelerite.from_csv("celerite_c2.csv")

ds = 1e-4
dx = 1e-8
dy = 1e-8
dz = 1e-8
w  = 6e5
epsilon = 5e-5
S_SPAN = 3e-4
NB_RAYON = 1
NB_FRONT_D_ONDE = 1
NB_RAYON_AFFICHE = 4
PAS_AFFICHAGE = 1
norm_p_min=1e-10

def c_2(x, y, z):
    #print(c(-z)**-2)
    return c(-z)**(-2)

def c_0(x, y, z):
    return c(-z)

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
def Runge_Kutta_etape(R_n, func):
    k1 = func(R_n.copy())
    k2 = func(R_n.copy() + k1 * ds/2)
    k3 = func(R_n.copy() + k2 * ds/2)
    k4 = func(R_n.copy() + k3 * ds)
    return R_n.copy() + (k1 + 2*k2 + 2*k3 + k4) * ds / 6

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
def equation(R): # R = [x, y, z, px, py, pz, t]
    dR=[2 * R[3], 2 * R[4], 2 * R[5]] #2*p = q_dot

    x, y, z = R[0], R[1], R[2]
    dR.append(w**2 * gradx(c_2, x, y, z))# p_dot
    dR.append(w**2 * grady(c_2, x, y, z))
    dR.append(w**2 * gradz(c_2, x, y, z))

    dR.append(2 * w * c_2(x, y, z)) # t_dot
    return np.array(dR)

def sphere(center = (0, 0, -2000), radius = 10, Nphi=10, Ntheta=20, phi0 = 0, phi1=np.pi, theta=2*np.pi):
    phi = np.linspace(phi0, phi1, Nphi)
    theta = np.linspace(0, theta, Ntheta)

    phi, theta = np.meshgrid(phi, theta)

    phi = phi.reshape(Nphi*Ntheta)
    theta = theta.reshape(Nphi*Ntheta)

    x = center[0] + radius * np.sin(phi) * np.cos(theta)
    y = center[1] + radius * np.sin(phi) * np.sin(theta)
    z = center[2] + radius * np.cos(phi)

    px = w * (x - center[0])/(radius * c_0(x, y,z))
    py = w * (y - center[1])/(radius * c_0(x, y,z))
    pz = w * (z - center[2])/(radius * c_0(x, y,z))

    return (x,y,z, px, py, pz) 

def plan(x_largeur=10,y_largeur=10, centre=(0,0), z_0=-2000, N=10):
    x_0 = np.linspace(centre[0]-x_largeur/2, centre[0]+x_largeur/2, N)
    y_0 = np.linspace(centre[1]-y_largeur/2, centre[1]+y_largeur/2, N)
    x,y = np.meshgrid(x_0,y_0)
    z = z_0 + np.zeros(N*N)
    return x.reshape(N*N), y.reshape(N*N), z

def derivee_alpha0(var_alpha): # x1 = x, x2 = y
    x, y, z, p1, p2, p3 = var_alpha[0], var_alpha[1], var_alpha[2], var_alpha[3], var_alpha[4], var_alpha[5]
    d_alpha = [2*p1, 2*p2, 2*p3, w**2 * gradx(c_2, x, y, z), w**2 * grady(c_2, x, y, z), w**2 * gradz(c_2, x, y ,z)]
    d_alpha.append( (2 * c_2(x, y, z)) * p3 * (p2 * gradx(c_0, x, y, z) - p1 * grady(c_0, x, y, z))
                    / (p1**2 + p2**2)) 
    return np.array(d_alpha)

def derivee_alpha1(var_alpha): # x1 = y, x2 = z
    x, y, z, p3, p1, p2 = var_alpha[0],var_alpha[1],var_alpha[2],var_alpha[3],var_alpha[4],var_alpha[5]
    d_alpha = [2*p2, 2*p3, 2*p1, w**2 * grady(c_2, x, y, z), w**2 * gradz(c_2, x, y, z), w**2 * gradx(c_2, x, y ,z)]
    d_alpha.append( (2 * c_2(x, y, z)) * p3 * (p2 * grady(c_0, x, y, z) - p1 * gradz(c_0, x, y, z))
                    / (p1**2 + p2**2))     
    return np.array(d_alpha)

def derivee_alpha2(var_alpha): #x1 = z, x2 = x
    x,y,z,p2,p3,p1=var_alpha[0],var_alpha[1],var_alpha[2],var_alpha[3],var_alpha[4],var_alpha[5]
    d_alpha = [2*p3, 2*p1, 2*p2, w**2 * gradz(c_2, x, y, z), w**2 * gradx(c_2, x, y, z), w**2 * grady(c_2, x, y ,z)]
    d_alpha.append( (2 * c_2(x, y, z)) * p3 * (p2 * gradz(c_0, x, y, z) - p1 * gradx(c_0, x, y, z))
                    / (p1**2 + p2**2))     
    return np.array(d_alpha)

def alpha(base_nb, der_alpha, sum_impulse, coord_ray, ind, BCR_new, BCR_n_1): # base_nb choisit la permutation de la base cartésienne de départ
    if  sum_impulse < norm_p_min: # cas limite
        BCR_new[base_nb,ind]= BCR_n_1[base_nb,ind]
        raise SystemError 
    else :
        ray_bcr = np.array([coord_ray[0], 
                            coord_ray[1], 
                            coord_ray[2], 
                            coord_ray[3], 
                            coord_ray[4],
                            coord_ray[5], 
                            BCR_n_1[base_nb, ind]])
        
        BCR_new[base_nb, ind] = Runge_Kutta_etape(ray_bcr, der_alpha)[6]
    
def impulse_for_new_base(num_sum_p2_max, p): 

    return (p[(0 + num_sum_p2_max)%3], 
            p[(1 + num_sum_p2_max)%3], 
            p[(2 + num_sum_p2_max)%3])

def calcul_new_base(x, y, z, p1, p2, p3, angle_alpha):

    racine = np.linalg.norm(np.array([p1, p2]))
    eps1 = (c_0(x, y, z)/w) * np.array([ p1 * p3 / racine, 
                                         p2 * p3 / racine, 
                                         -racine ])

    eps2 = np.array([ -p2 / racine, p1 / racine, 0])

    print("YAY")
    print(w/c_0(x, y, z)
    print(np.linalg.norm(eps1))
    print("YOY")
     

    e1 = np.cos(angle_alpha)*eps1 - np.sin(angle_alpha)*eps2 #e1
    e2 = np.sin(angle_alpha)*eps1 + np.cos(angle_alpha)*eps2 #e2
    t = np.cross(e1, e2) #t

    return t, e1, e2



def base_centree_rayon(R_n, BCR_n_1):

    BCR_new = np.zeros_like(BCR_n_1) #12, NB_RAYON

    for ind in range(len(R_n[0, :])):  #ind est l'indice du rayon
        coord_ray = R_n[:, ind]  #PREND LES COORDONNÉES DU RAYON D'INDICE IND
        x, y, z = coord_ray[0], coord_ray[1], coord_ray[2]

        sum_p2=[np.linalg.norm(coord_ray[[3, 4]]) , #p1**2 + p2**2
                np.linalg.norm(coord_ray[[4, 5]]) , #p2**2 + p3**2
                np.linalg.norm(coord_ray[[5, 3]])   #p3**2 + p1**2
        ]       

        alpha(0, derivee_alpha0, sum_p2[0], coord_ray, ind, BCR_new, BCR_n_1) 
        alpha(1, derivee_alpha1, sum_p2[1], coord_ray, ind, BCR_new, BCR_n_1)
        alpha(2, derivee_alpha2, sum_p2[2], coord_ray, ind, BCR_new, BCR_n_1)


        # Repr Choice
        nb_sum_p2_max = sum_p2.index(np.max(sum_p2))  # (x, y, z)
        
        p1, p2, p3 = impulse_for_new_base(nb_sum_p2_max, coord_ray)

        alpha_star = BCR_new[nb_sum_p2_max, ind]

        t, e1, e2 = calcul_new_base(x, y, z, p1, p2, p3, alpha_star)

        print(np.linalg.norm(t), np.linalg.norm(e1), np.linalg.norm(e2))

        print(np.dot(t, e1), np.dot(t, e2), np.dot(e1, e2))
        p = np.array([p1, p2, p3])
        print(np.cross(t, p))

        BCR_new[3:6, ind] = t
        BCR_new[6:9, ind] = e1
        BCR_new[9:12, ind] = e2


    return BCR_new
        

  
# Conditions initiales
Nphi = round(NB_RAYON**0.5)
Ntheta = NB_RAYON // Nphi
NB_RAYON = Nphi*Ntheta

X0,Y0,Z0, Px0,Py0,Pz0 = sphere(radius=10, Nphi=Nphi, Ntheta=Ntheta, phi0=3*np.pi/7,  phi1=7*np.pi/10, theta=2, center=(0,0,-2000))


NB_POINTS = round(S_SPAN/ds)//PAS_AFFICHAGE


R = np.zeros((NB_POINTS, 7, NB_RAYON))
BCR = np.zeros((NB_POINTS, 12, NB_RAYON)) # 0-2 angles / 3-11 les axes

R[0, :, :] = np.array([X0, Y0, Z0, Px0, Py0, Pz0, np.zeros(NB_RAYON)])

#INIT
for ind in range(NB_RAYON):  #ind est l'indice du rayon
    x, y, z = X0[ind], Y0[ind], Z0[ind]
    px, py, pz = Px0[ind], Py0[ind], Pz0[ind]
    p = np.array([px, py, pz])

    sum_p2=[ px**2 + py**2 ,
             py**2 + pz**2 ,
             pz**2 + px**2 ]

    nb_sum_p2_max = sum_p2.index(np.max(sum_p2))  # (x, y, z) #should work
    
    #BCR
    p1, p2, p3 = impulse_for_new_base(nb_sum_p2_max, p)

    t, e1, e2 = calcul_new_base(x, y, z, p1, p2, p3, 0) 
    BCR[0, 3:6, ind] = t
    BCR[0, 6:9, ind] = e1
    BCR[0, 9:12, ind] = e2

R_n = R[0]
BCR_n = BCR[0, :, :] 

#COMPUTE
for i in tqdm(range(1, NB_POINTS * PAS_AFFICHAGE-1)): #! WEIRD

    res = Runge_Kutta_etape(R_n, equation)
    BCR_n = base_centree_rayon(res, BCR_n)

    if i%PAS_AFFICHAGE == 0:
        R[i//PAS_AFFICHAGE] = res
        BCR[i//PAS_AFFICHAGE] = BCR_n

    R_n = res.copy()

X = R[:, 0, :]
Y = R[:, 1, :]
Z = R[:, 2, :]

Temps = R[:, 6, :]
# print(f'{Temps = }')
# print(f'{X = }')

t_max = np.min(Temps[-1])

print("Génération du plot :")

import plotly.graph_objects as go

fig = go.Figure()

# Storage for vector coordinates
vectors = {
    'T': {'x': [], 'y': [], 'z': []},
    'E1': {'x': [], 'y': [], 'z': []},
    'E2': {'x': [], 'y': [], 'z': []}
}


def normalize_vectors(V):
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    return V / np.maximum(norms, 1e-12)


def add_vectors(storage, X, Y, Z, V, length=5):

    for j in range(len(X)):

        x, y, z = X[j], Y[j], Z[j]

        u, v, w = V[j] * length

        storage['x'].extend([x, x + u, None])
        storage['y'].extend([y, y + v, None])
        storage['z'].extend([z, z + w, None])


for i in tqdm(range(NB_RAYON)):

    # Mask
    mask = Z[:, i] <= 0

    # Positions
    X_filtered = X[mask, i]
    Y_filtered = Y[mask, i]
    Z_filtered = Z[mask, i]

    # Plot points
    fig.add_trace(go.Scatter3d(
        x=X_filtered,
        y=Y_filtered,
        z=Z_filtered,
        mode='markers',
        marker=dict(size=2, color='black'),
        showlegend=False
    ))

    # Extract basis vectors
    #print(BCR[:, 3:6, i])

    T = normalize_vectors(BCR[mask, 3:6, i])
    E1 = normalize_vectors(BCR[mask, 6:9, i])
    E2 = normalize_vectors(BCR[mask, 9:12, i])

    # Add vectors at EVERY filtered point
    add_vectors(vectors['T'], X_filtered, Y_filtered, Z_filtered, T)
    add_vectors(vectors['E1'], X_filtered, Y_filtered, Z_filtered, E1)
    add_vectors(vectors['E2'], X_filtered, Y_filtered, Z_filtered, E2)


# Plot the three complete vector fields
for name, color in [('T', 'red'), ('E1', 'blue'), ('E2', 'green')]:

    fig.add_trace(go.Scatter3d(
        x=vectors[name]['x'],
        y=vectors[name]['y'],
        z=vectors[name]['z'],
        mode='lines',
        line=dict(color=color, width=3),
        showlegend=False
    ))


# Layout
fig.update_layout(
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        aspectmode='manual',
        aspectratio=dict(x=1, y=1, z=1)
    )
)

fig.show()
