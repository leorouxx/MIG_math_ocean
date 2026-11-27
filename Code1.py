import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

##########################################################
## Calcul de la Jacobienne associée au flot hamiltonien ##
##########################################################

Positions =  sigma(X,Y,Z) #liste des abcisses curvilignes sigma des points d'un rayon
Matrices_M= Matrice#matrice M donnée par l'équation de ricatti et fournie par le groupe 
Profondeurs=Z #liste des profondeurs des célérités de la liste Célérité
sigma_0=Positions[0]

def integrale_eq67(sigma_0, sigma, profondeurs, positions, matrices_M):
    z = profondeurs
    s = positions
    v = c(-z)
    f = (v*2)(matrices_M[:, 0, 0] + matrices_M[:, 1, 1])
    masque = (s>= sigma_0) & (s <= sigma)
    return np.trapz(f[masque], s[masque])

Intégrales=[]
for abcisse_s in Positions :
    Intégrales.append(integrale_eq67(sigma_0,abcisse_s,Profondeurs,Positions,Matrices_M))
plt.plot(Positions,Intégrales)
plt.show()

############################################################
###        Création d'un tube de rayons à la Jensen      ###
############################################################

######################################################################################
# Détection de zones suspectes de caustiques à partir des listes M(sigma), sigma, z. 
#
# Données attendues (une entrée par rayon i, indexées de la même façon) :
#     sigma_list[i] : array (n_i,)       abscisse curviligne le long du rayon i
#     z_list[i]     : array (n_i,)       profondeur z le long du rayon i
#     M_list[i]     : array (n_i, 2, 2)  matrice M(sigma) le long du rayon i
#
# Étape 1 : tracer la valeur propre moyenne tr(M)/2 en fonction de sigma,
#           repérer les rayons suspects (lam_moy très négative ou non finie).
# Étape 2 : refaire le même tracé en fonction de z, uniquement sur ces rayons,
#           et estimer la profondeur d'apparition de la caustique.
#######################################################################################



# ---------------------------------------------------------------- calculs
def valeur_propre_moyenne(M):
    """Moyenne des deux valeurs propres = trace/2"""
    M = np.asarray(M, dtype=float)
    return 0.5 * (M[:, 0, 0] + M[:, 1, 1])


def valeurs_propres(M):
    """lam_min, lam_max d'une matrice 2x2 symétrique, formule fermée."""
    M = np.asarray(M, dtype=float)
    a, b, d = M[:, 0, 0], 0.5 * (M[:, 0, 1] + M[:, 1, 0]), M[:, 1, 1]
    m = 0.5 * (a + d)
    r = np.sqrt(((a - d) / 2.0) ** 2 + b ** 2)
    return m - r, m + r


def seuil_auto(lam_list, facteur=10.0):
    """Seuil par défaut, régler le facteur""" 
    tout = np.concatenate([np.abs(l[np.isfinite(l)]) for l in lam_list])
    return facteur * np.median(tout)


def analyser_rayon(sigma, z, M, seuil):
    """Renvoie un dictionnaire de diagnostics pour un rayon."""
    lam = valeur_propre_moyenne(M)
    lam_min, _ = valeurs_propres(M)
    non_fini = ~np.isfinite(lam)
    # point suspect : moyenne très négative, ou valeur propre min très négative,
    # ou valeur non finie (explosion de l'intégrateur)
    suspect = non_fini | (lam < -seuil) | (lam_min < -seuil)
    idx = np.flatnonzero(suspect)
    res = {"lam": lam, "suspect": bool(idx.size), "k_debut": None,
           "sigma_debut": None, "z_debut": None, "sigma_star": None}
    if idx.size:
        k = int(idx[0])
        res.update(k_debut=k, sigma_debut=sigma[k], z_debut=z[k])
        # extrapolation de sigma* : mu = 1/lam_min est linéaire (pente c)
        ok = np.isfinite(lam_min[:k + 1]) & (lam_min[:k + 1] < 0)
        ids = np.flatnonzero(ok)[-5:]
        if ids.size >= 2:
            mu = 1.0 / lam_min[ids]
            pente, ord0 = np.polyfit(sigma[ids], mu, 1)
            if pente > 0 :
                res["sigma_star"] = -ord0 / pente
    return res


# ---------------------------------------------------------------- tracés
def tracer(abscisses, lam_list, indices, xlabel, titre, fichier, seuil=None,
           marques=None):
    fig, ax = plt.subplots(figsize=(9, 5))
    for i in indices:
        l = lam_list[i]
        ax.plot(abscisses[i], l, lw=1, label=f"rayon {i}" if len(indices) <= 12 else None)
        if marques is not None and marques.get(i) is not None:
            k = marques[i]
            ax.plot(abscisses[i][k], l[k], "ro", ms=4)
    if seuil is not None:
        ax.axhline(-seuil, color="k", ls="--", lw=1, label="seuil")
    # échelle symlog : l'explosion reste lisible sans écraser le reste
    ax.set_yscale("symlog", linthresh=max(seuil or 1.0, 1e-12) / 10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("valeur propre moyenne  tr(M)/2")
    ax.set_title(titre)
    ax.grid(alpha=0.3)
    if len(indices) <= 12:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fichier, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- pipeline
def detecter_caustiques(sigma_list, z_list, M_list, seuil=None,
                        prefixe="caustiques"):
    n = len(M_list)
    assert len(sigma_list) == n and len(z_list) == n, "listes de tailles différentes"
    lam_list = [valeur_propre_moyenne(M) for M in M_list]
    if seuil is None:
        seuil = seuil_auto(lam_list)
    print(f"Seuil utilisé : {seuil:.3g}  (à régler sur vos cas tests)")

    # Étape 1 : tous les rayons, lam_moy en fonction de sigma
    diag = [analyser_rayon(np.asarray(sigma_list[i], float),
                           np.asarray(z_list[i], float),
                           np.asarray(M_list[i], float), seuil) for i in range(n)]
    tracer(sigma_list, lam_list, range(n), "sigma",
           "Étape 1 : valeur propre moyenne vs sigma (tous les rayons)",
           f"{prefixe}_etape1_sigma.png", seuil,
           {i: d["k_debut"] for i, d in enumerate(diag)})

    # Sélection des rayons suspects
    suspects = [i for i, d in enumerate(diag) if d["suspect"]]
    print(f"{len(suspects)} rayon(s) suspect(s) sur {n} : {suspects}")
    if not suspects:
        return diag, suspects

    # Étape 2 : mêmes rayons, lam_moy en fonction de z
    tracer(z_list, lam_list, suspects, "z (profondeur)",
           "Étape 2 : valeur propre moyenne vs z (rayons suspects)",
           f"{prefixe}_etape2_z.png", seuil,
           {i: diag[i]["k_debut"] for i in suspects})

    # Profondeur d'apparition
    print("\nrayon | sigma_debut | z_debut | sigma* extrapolé")
    zs = []
    for i in suspects :
        d = diag[i]
        ss = "n/a" if d["sigma_star"] is None else f"{d['sigma_star']:.4g}"
        print(f"{i:5d} | {d['sigma_debut']:11.4g} | {d['z_debut']:7.4g} | {ss}")
        zs.append(d["z_debut"])
    print(f"\nProfondeur d'apparition (min sur les rayons suspects) : z = {min(zs):.4g}")
    return diag, suspects


# ---------------------------------------------------------------- démo
def _donnees_test(n_rayons=9, n_pas=300):
    """Données synthétiques : rayons qui focalisent à des sigma* différents.
    Hypothèse de test : M = I/(c (sigma - sigma*)) sur une valeur propre."""
    c = 1.0
    sig, zz, MM = [], [], []
    for i in range(n_rayons):
        s = np.linspace(0.05, 10, n_pas)
        s_star = 6.0 + 0.6 * (i - n_rayons // 2) ** 2 / 4 + 3 * (i % 2)
        lam1 = 1.0 / (c * (s - s_star + 1e-9))
        lam1 = np.where(s < s_star - 0.05, lam1, np.nan)  # arrêt avant explosion
        lam1 = np.where(s < s_star - 0.05, lam1, np.nan)
        lam2 = 1.0 / (c * (s + 0.5))
        M = np.zeros((n_pas, 2, 2))
        M[:, 0, 0], M[:, 1, 1] = lam1, lam2
        sig.append(s)
        zz.append(0.8 * s)
        MM.append(M)
    return sig, zz, MM


if name == "main":
    sigma_list, z_list, M_list = _donnees_test()
    detecter_caustiques(sigma_list, z_list, M_list)