"""
Extraction d'une fonction c(z) de classe C2, vectorisée, à partir du GP déjà ajusté.

Ne modifie pas l'interpolation : on tabule gp.predict sur une grille fine et on
la représente par une spline quintique (classe C4 >= C2) qui passe exactement
par les valeurs tabulées. Aucune hypothèse physique n'est ajoutée.

Utilisation, à la fin du script d'interpolation (après gp.fit) :

    from profil_celerite import ProfilCelerite
    prof = ProfilCelerite.depuis_gp(gp, df3['profondeur'].min(), df3['profondeur'].max())
    prof(z)              # c(z)        (z scalaire ou tableau numpy de forme quelconque)
    prof.d1(z)           # dc/dz
    prof.d2(z)           # d2c/dz2
    prof.to_csv("celerite_c2.csv")

Puis dans un autre script (sans sklearn ni refaire le krigeage) :

    prof = ProfilCelerite.from_csv("celerite_c2.csv")
"""

import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline


class ProfilCelerite:
    """c(z) de classe C2, construite sur une grille (z_i, c_i).

    Hors de [z_min, z_max] (paramètre `extrapolation`) :
      - "taylor" : développement de Taylor d'ordre 2 au bord -> c reste C2 partout
      - "nan"    : renvoie NaN (pour repérer toute sortie du domaine)
      - "bord"   : c, c' et c'' figés à leur valeur au bord (c n'est alors que C0)
    """

    def __init__(self, z, c, sigma=None, extrapolation="taylor"):
        z = np.asarray(z, float).ravel()
        c = np.asarray(c, float).ravel()
        if np.any(np.diff(z) <= 0):
            raise ValueError("z doit être strictement croissant")
        if extrapolation not in ("taylor", "nan", "bord"):
            raise ValueError("extrapolation : 'taylor', 'nan' ou 'bord'")
        self.z, self.c = z, c
        self.sigma = None if sigma is None else np.asarray(sigma, float).ravel()
        self.extrapolation = extrapolation
        self.z_min, self.z_max = z[0], z[-1]

        self._s0 = make_interp_spline(z, c, k=5)   # quintique, conditions "not-a-knot"
        self._s1 = self._s0.derivative(1)
        self._s2 = self._s0.derivative(2)
        # valeurs aux bords pour le prolongement
        self._bord = {zb: (float(self._s0(zb)), float(self._s1(zb)), float(self._s2(zb)))
                      for zb in (self.z_min, self.z_max)}

    # ------------------------------------------------------------------ #
    @classmethod
    def depuis_gp(cls, gp, z_min, z_max, pas=0.5, **kw):
        """Tabule gp.predict sur [z_min, z_max] avec un pas `pas` (en m)."""
        n = int(np.ceil((z_max - z_min) / pas)) + 1
        z = np.linspace(z_min, z_max, n)
        c, sigma = gp.predict(z.reshape(-1, 1), return_std=True)
        return cls(z, c, sigma, **kw)

    @classmethod
    def from_csv(cls, chemin, **kw):
        from pathlib import Path

        chemin = Path(chemin)

        if not chemin.is_absolute():
            chemin = Path(__file__).parent / chemin

        df = pd.read_csv(chemin)

        return cls(df["z"].values, df["c"].values,
                   df["sigma"].values if "sigma" in df else None, **kw)

    def to_csv(self, chemin):
        df = pd.DataFrame({"z": self.z, "c": self.c,
                           "dcdz": self.d1(self.z), "d2cdz2": self.d2(self.z)})
        if self.sigma is not None:
            df["sigma"] = self.sigma
        df.to_csv(chemin, index=False, float_format="%.17g")   # précision double exacte

    # ------------------------------------------------------------------ #
    def _eval(self, z, n):
        """Dérivée n-ième (n = 0, 1, 2), vectorisée."""
        z = np.asarray(z, float)
        spl = (self._s0, self._s1, self._s2)[n]
        out = spl(np.clip(z, self.z_min, self.z_max))

        for zb, dehors in ((self.z_min, z < self.z_min), (self.z_max, z > self.z_max)):
            if not np.any(dehors):
                continue
            c0, c1, c2 = self._bord[zb]
            if self.extrapolation == "nan":
                val = np.nan
            elif self.extrapolation == "bord":
                val = (c0, c1, c2)[n]
            else:  # Taylor d'ordre 2
                h = z - zb
                val = (c0 + c1 * h + 0.5 * c2 * h**2, c1 + c2 * h, c2 + 0 * h)[n]
            out = np.where(dehors, val, out)

        return float(out) if out.ndim == 0 else out

    def __call__(self, z):
        return self._eval(z, 0)

    def d1(self, z):
        return self._eval(z, 1)

    def d2(self, z):
        return self._eval(z, 2)
