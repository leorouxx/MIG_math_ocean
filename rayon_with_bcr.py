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
norm_p_min=1e-(10)

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

 
def base_centree_rayon(R_n_1,BCR,s,indice):
    ## calcul alphas
    for ind in range(len(var_R_n_1)) :
        coord_ray=_R_n_1[:,ind]

        sum_1=(coord_ray[3]*2)**2+(coord_ray[4]*2)**2
        sum_2=(coord_ray[5]*2)**2+(coord_ray[4]*2)**2 
        sum_3=(coord_ray[3]*2)**2+(coord_ray[5]*2)**2
        if  sum_1< norm_p_min:
            BCR[indice,0,ind]=BCR[indice-1,0,ind]
        else :
            ray_bcr=coord_ray
            ray_bcr.append(BCR[indice-1,0,ind])
            BCR[indice,0,ind]=BCR[indice-1,0,ind]+ Runge_Kutta_etape(ds,ray_bcr , derivee_alpha1)[6]

        if sum_2 <  norm_p_min:
            BCR[indice,1,ind]=BCR[indice-1,1,ind]
        else :
            ray_bcr=coord_ray
            ray_bcr.append(BCR[indice-1,1,ind])
            BCR[indice,1,ind]=BCR[indice-1,1,ind]+Runge_Kutta_etape(ds,ray_bcr , derivee_alpha2)[6]
        if sum_3<norm_p_min:
            BCR[indice,2,ind]=BCR[indice-1,2,ind]
        else :
            ray_bcr=coord_ray
            ray_bcr.append(BCR[indice-1,2,ind])
            BCR[indice,2,ind]=BCR[indice-1,2,ind]+Runge_Kutta_etape(ds,ray_bcr , derivee_alpha3)[6]

        sum_max = sum_1 # (x, y, z)
        x, y ,z = coord_ray[0], coord_ray[1], coord_ray[3]
        p1, p2, p3 = coord_ray[3], coord_ray[4], coord_ray[5]

        if sum_2 > sum_max: # (y, z, x)
            sum_max=sum_2
            y, z , x = coord_ray[0], coord_ray[1], coord_ray[3]
            p2, p3, p1 = coord_ray[3], coord_ray[4], coord_ray[5]

        if sum_3 > sum_max: # (z, x, y)
            sum_max = sum_3
            z, x , y = coord_ray[0], coord_ray[1], coord_ray[3]
            p3, p1, p2 = coord_ray[3], coord_ray[4], coord_ray[5]

        racine = np.sqrt(p1**2 + p2**2)

        BCR[indice,4,ind]=[( c(x, y, z)*p1*p3 ) / racine, ( c(x, y, z)*p2*p3 ) / racine, -c(x, y, z)*racine]
        BCR[indice,5,ind]=[ -p2 / racine, p1 / racine, 0]
        BCR[indice,3,ind]=np.cross(BCR[indice,4,ind], BCR[indice,5,ind])


def derivee_alpha1(var_alpha):
    d_alpha=[0 for k in range(len(var_alpha)-1)]
    x,y,z,p1,p2,p3=var_alpha[0],var_alpha[1],var_alpha[2],var_alpha[3],var_alpha[4],var_alpha[5]
    d_alpha.append(( p3*(p2*grady(c, x, y, z)) - p1*gradz(c, x, y, z) ) / (p1**2 + p2**2))
    return np.array(d_alpha)

def derivee_alpha2(var_alpha):
    d_alpha=[0 for k in range(len(var_alpha)-1)]
    x,y,z,p3,p1,p2=var_alpha[0],var_alpha[1],var_alpha[2],var_alpha[3],var_alpha[4],var_alpha[5]
    d_alpha.append(( p3*(p2[4]*grady(c, x, y, z)) - p1*gradz(c, x, y, z) ) / (p1**2 + p2**2))
    return np.array(d_alpha)

def derivee_alpha3(var_alpha):
    d_alpha=[0 for k in range(len(var_alpha)-1)]
    x,y,z,p2,p3,p1=var_alpha[0],var_alpha[1],var_alpha[2],var_alpha[3],var_alpha[4],var_alpha[5]
    d_alpha.append(( p3*(p2[4]*gradz(c, x, y, z)) - p1*gradx(c, x, y, z) ) / (p1**2 + p2**2))
    return np.array(d_alpha)




  
# Conditions initiales
Nphi = round(NB_RAYON**0.5)
Ntheta = NB_RAYON // Nphi
NB_RAYON = Nphi*Ntheta


x,y,z,x2,y2,z2 = sphere(radius=10, Nphi=Nphi, Ntheta=Ntheta, phi0=3*np.pi/7,  phi1=7*np.pi/10, theta=2, center=(0,0,-1000))

NB_POINTS = round(S_SPAN/ds)//PAS_AFFICHAGE


R = np.zeros((NB_POINTS, 7, NB_RAYON))
BCR=np.zeros((NB_POINTS,6,NB_RAYON))
R[0] = np.array([x,y,z,(x2-x)/ds, (y2-y)/ds, (z2-z)/ds, np.zeros(NB_RAYON)])
R_n = R[0]

s = np.linspace(0, S_SPAN, NB_POINTS) 
print(s.shape)
print("Calcul des rayons :")
for i in tqdm(range(NB_POINTS*PAS_AFFICHAGE-1)):
    R_n = Runge_Kutta_etape(s, R_n, equation)
    if i%PAS_AFFICHAGE == 0:
        R[i//PAS_AFFICHAGE] = R_n
    R_n_1=R_n

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

