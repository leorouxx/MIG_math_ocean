import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.integrate as itg
import math




#############################################################
###         Calcul numérique de la propagation            ###
#############################################################


ds = 1e-5
dx = 1e-8
dy = 1e-8
dz = 1e-8
w  = 6e5
epsilon = 5e-5

S_SPAN = 7
NB_RAYON = 10
NB_FRONT_D_ONDE = 3
NB_RAYON_AFFICHE = 25
PAS_AFFICHAGE = 500


def c_2(x, y, z):
    #return c(-z)**-2
    return (1-epsilon*z)

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
    E2 = np.cos(Theta)*B - np.sin(Theta)*N

    return E1, E2

def plan(x_largeur=10,y_largeur=10, centre=(0,0), z_0=-200, N=10):
    x_0 = np.linspace(centre[0]-x_largeur/2, centre[0]+x_largeur/2, N)
    y_0 = np.linspace(centre[1]-y_largeur/2, centre[1]+y_largeur/2, N)
    x,y = np.meshgrid(x_0,y_0)
    z = z_0 + np.zeros(N*N)
    return x.reshape(N*N), y.reshape(N*N), z


###################################################################################
### Profil de célérité à gradient constant vers le bas, et plan initial à -pi/4 ###
###################################################################################


def position_impulsion(x_0,s) :
    x = x_0 + np.sqrt(2)*w*np.sqrt(1+epsilon*x_0)*s
    z = -x_0 + np.sqrt(2)*w*np.sqrt(1+epsilon*x_0)*s - (w**2)*epsilon*s**2
    ksi = w/(np.sqrt(2))*np.sqrt(1+epsilon*x_0)
    zeta = w/(np.sqrt(2))*np.sqrt(1+epsilon*x_0) -(w**2)*epsilon*s
    return [x,z,ksi,zeta]

def Mat_M(x_0,s) :
    m11 = 2*(w**2)*epsilon*((w/np.sqrt(2))*np.sqrt(1+epsilon*x_0)- (w**2)*epsilon*s)
    m12 = (w**2)*epsilon*(-1 +(w*epsilon/np.sqrt(2))*(1/np.sqrt(1+epsilon*x_0))*s)
    m22 = -(((w**2)*(epsilon**2)/2)/(1+epsilon*x_0))*s

def position_caustique_analytique(x_0) : 
    s_prim = (1/(np.sqrt(2)*w*epsilon)*(-1 + np.sqrt(5))*np.sqrt(1+ epsilon*x_0))
    return position_impulsion(x_0,s_prim)[0],position_impulsion(x_0,s_prim)[1]


def droite(x_longueur=1/epsilon, alpha=np.pi/4, z_0=0, N=100):
    x = np.linspace(0, x_longueur, N)
    y = np.zeros(N)
    z = z_0 + np.tan(-alpha) * x
    return x, y, z
# x + np.sin(alpha) * w * c_2(x,y,z)**0.5 * ds, y, z + np.cos(alpha) * w * c_2(x,y,z)**0.5 * ds


nb_ray=1000
Abs_curv = np.linspace(0,0.05, nb_ray)
rays = []
for x in droite()[0] :
    X = []
    Z = []
    for s in Abs_curv :
        X.append(position_impulsion(x,s)[0])
        Z.append(position_impulsion(x,s)[1])
    rays.append([X,Z])

def front_onde(s) :
    Front = []
    for r in rays :
        Front.append(r[s*nb_ray])

#############################################################################
###                 Calcul de courbure 2D                                 ###
#############################################################################

#tu dois changer determiner position indice, qui te donne les points pour calculer ta courbure


def distance_N2(p, q):
    """Distance euclidienne entre deux points [x, y]."""
    return math.hypot(p[0] - q[0], p[1] - q[1])

def derr(p1,p2):
    d = distance_N2(p1,p2)
    return [(p2[0]-p1[0])/d,(p2[1]-p1[1])/d]

def courbure(rayg,rayd,ray,t):
    """donne la courbure k sur le rayon ray à l'aide des valeurs de rayg et rayd (resp gauche et droite).
    Il ne faut que deux rayons dans rayg et rayd ET qu'ils soient dans le bon ordre !
    g: plus loin, plus proche, d: plus proche, plus loin
    - se règle au tir du rayon - ne gère pas les problèmes de retournement du front d'onde / de caustiques"""

    #pos_ref,b,ind_ref = determiner_position_indice(ray,t) #rayon de référence, sur lequel on calcule la courbure
    posg = []
    posd = []

    if not b:
        return False,0,0 #en dehors de la mer !
    for r in rayg:
   #     pos,b,indice = determiner_position_indice(r,t)
        posg.append(pos)

    for r in rayd:
    #    pos,b,indice = determiner_position_indice(r,t)
        #print("indice",indice)
        posd.append(pos)

    d = []
    for (x1,x2) in [posg,posd]:
        x0 = derr(x1,x2)
        d.append(x0)

    k = derr(d[0],d[1])
    return k



##############################################################################
###                            AFFICHAGE                                   ###
##############################################################################

## Tracé de l'enveloppe
kostik_X = [position_caustique_analytique(x)[0] for x in droite([0])]
kostik_Z = [position_caustique_analytique(x)[1] for x in droite([0])]

plt.scatter(
    kostik_X,
    kostik_Z,
    color="tab:red",
    s=50,
    zorder=3,
    label=" Caustique : enveloppe des rayons ")


## Tracé des rayons
for i in range(len(rays)) : 
    plt.plot(rays[i][0],rays[i][1], alpha = 0.6)

## Tracé de la condition initiale
X_0 = droite()[0]
Z_0 = droite()[2]

plt.plot(X_0,
         Z_0,
        label = "Plan initial d'émission")

## Plottage 
plt.ylabel(' Profondeur ')
plt.title('Tracé de rayons dans le cas du profil de c à gradient constant' )
plt.legend()
plt.show()


    


"""
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
    dR= np.empty_like(R) 
    dR[0] = R[3]
    dR[1] = R[4]
    dR[2] = R[5]
    dR[3] = 2 * w**2 * gradx(c_2, R[0], R[1], R[2])
    dR[4] = 2 * w**2 * grady(c_2, R[0], R[1], R[2])
    dR[5] = 2 * w**2 * gradz(c_2, R[0], R[1], R[2])
    dR[6] = 2 * w * c_2(R[0], R[1], R[2])
    return dR

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


# Conditions initiales
Nphi = round(NB_RAYON**0.5)
Ntheta = NB_RAYON // Nphi
NB_RAYON = Nphi*Ntheta


# x,y,z,x2,y2,z2 = sphere(radius=10, Nphi=Nphi, Ntheta=Ntheta, phi0=3*np.pi/7,  phi1=7*np.pi/10, theta=2, center=(0,0,-1000))
x,y,z,x2,y2,z2 = droite(N = NB_RAYON)

NB_POINTS = round(S_SPAN/ds)



R_n = np.array([x,y,z,(x2-x)/ds, (y2-y)/ds, (z2-z)/ds, np.zeros(NB_RAYON)])

s = np.linspace(0, S_SPAN, NB_POINTS) 
print(s.shape)
print("Calcul des rayons :")
with open("log", "x") as donnees:
    donnees.write(f'{NB_POINTS//PAS_AFFICHAGE} {NB_RAYON}\n')
    np.savetxt(donnees, R_n[[0,1,2,6]], delimiter=" ")
    for i in tqdm(range(1, NB_POINTS)):
        R_n = Runge_Kutta_etape(s, R_n, equation)
        if i%PAS_AFFICHAGE == 0:
            np.savetxt(donnees, R_n[[0,1,2,6], :], delimiter=" ")




"""








# X = R[:, 0, :]
# Y = R[:, 1, :]
# Z = R[:, 2, :]
# Temps = R[:, 6, :]
# print(f'{Temps = }')
# print(f'{X = }')


# # T,N,B,K,tau = repere_frenet(X,Y,Z)
# # E1, E2 = base_centree_rayon(X,Y,Z,T,N,B,K,tau)


# t_max = np.min(Temps[-1])

# print("Génération du plot :")

# ax = plt.figure().add_subplot(projection='3d')
# for i in tqdm(range(0, len(X[0]), NB_RAYON//NB_RAYON_AFFICHE)):
#     ax.plot(X[Z[:,i]<=0,i], 
#             Y[Z[:,i]<=0,i], 
#             Z[Z[:,i]<=0,i], color="black")
# for i in tqdm(range(1, NB_FRONT_D_ONDE+1)):
#     x,y,z=front_d_onde(X, Y, Z, Temps, t_max*(i/(NB_FRONT_D_ONDE)))
#     ax.plot(x,y,z)
# ax.set_aspect('equal', adjustable='box')

# plt.show()