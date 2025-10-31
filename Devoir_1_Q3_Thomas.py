# ------------------------------------------------------------
# MATH 60231 - Devoir 1, Problème 3a
# Simulation de rendements : normale vs Student-t (df=3)
# ------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Pour la reproductibilité
np.random.seed(42)

# 1. Simulation des données
n = 10_000  # longueur des séries
mu = 0      # moyenne centrée

# (a) Rendements gaussiens ~ N(0,1)
r_norm = np.random.normal(mu, 1, n)

# (b) Rendements Student-t (df=3)
r_t = stats.t(df=3).rvs(size=n)

# 2. Normalisation des variances

var_target = np.var(r_norm)
r_t_scaled = r_t * np.sqrt(var_target / np.var(r_t))

# 3. Calcul des statistiques sommaires
def stats_summary(series):
    return pd.Series({
        "Moyenne": np.mean(series),
        "Écart-type": np.std(series, ddof=1),
        "Asymétrie": stats.skew(series),
        "Aplatissement (kurtosis)": stats.kurtosis(series, fisher=False)
    })

summary_norm = stats_summary(r_norm)
summary_t = stats_summary(r_t_scaled)

summary = pd.DataFrame({
    "Normale": summary_norm,
    "Student-t (df=3)": summary_t
})

pd.set_option('display.float_format', '{:.4f}'.format)
print("=== Statistiques sommaires ===")
print(summary)

# 4. Visualisation
plt.figure(figsize=(10,6))
plt.hist(r_norm, bins=60, alpha=0.6, color='steelblue', density=True, label='Normale')
plt.hist(r_t_scaled, bins=60, alpha=0.6, color='darkorange', density=True, label='Student-t (df=3)')
plt.axvline(np.mean(r_norm), color='blue', linestyle='--', linewidth=1)
plt.axvline(np.mean(r_t_scaled), color='orange', linestyle='--', linewidth=1)
plt.title("Comparaison des distributions simulées")
plt.xlabel("Rendement simulé")
plt.ylabel("Densité")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Estimation par noyau de densité (KDE)
plt.figure(figsize=(10,6))
sns.kdeplot(r_norm, color='steelblue', label='Normale')
sns.kdeplot(r_t_scaled, color='darkorange', label='Student-t (df=3)')
plt.axvline(np.mean(r_norm), color='blue', linestyle='--', linewidth=1)
plt.axvline(np.mean(r_t_scaled), color='orange', linestyle='--', linewidth=1)
plt.title("Estimation de densité par noyau (KDE)")
plt.xlabel("Rendement simulé")
plt.ylabel("Densité estimée")
plt.legend()
plt.grid(alpha=0.3)
plt.show()





# ------------------------------------------------------------
# MATH 60231 - Devoir 1, Problème 3b
# Estimation de la VaR (5%) et du déficit attendu (ES) par 3 méthodes
# (i) quantile empirique
# (ii) approche paramétrique gaussienne
# (iii) approche paramétrique Student-t (df connu = 3)
# Ce code traite les deux séries déjà simulées : r_norm et r_t_scaled
# ------------------------------------------------------------

import numpy as np
import pandas as pd
from scipy import stats

# Reproductibilité
np.random.seed(42)

# (Re)générer ou supposer dispos : r_norm et r_t_scaled
# Si vous exécutez ce bloc indépendamment, (re)créez les deux séries :
n = 10_000
r_norm = np.random.normal(0, 1, n)
r_t = stats.t(df=3).rvs(size=n)
# Normaliser la t pour qu'elle ait la même variance que la normale
var_target = np.var(r_norm, ddof=0)
r_t_scaled = r_t * np.sqrt(var_target / np.var(r_t, ddof=0))

# Paramètres
alpha = 0.05  # niveau VaR 5%
v = 3         # degrés de liberté connus pour la t

# Fonction pour VaR empirique et ES empirique
def empirical_var_es(series, alpha):
    q = np.quantile(series, alpha)            # quantile des rendements
    var = -q                                  # VaR définie positive
    es = -np.mean(series[series <= q])        # ES (Expected Shortfall / déficit attendu)
    return var, es

# Fonction pour VaR et ES paramétrique gaussien (analytique)
def gaussian_var_es(series, alpha):
    mu = np.mean(series)
    sigma = np.std(series, ddof=1)
    z = stats.norm.ppf(alpha)                 # quantile standard normal
    var = -(mu + sigma * z)                   # VaR positive
    # ES pour la normale : - (mu + sigma * phi(z) / alpha)
    phi_z = stats.norm.pdf(z)
    es = -(mu + sigma * (phi_z / alpha))
    return var, es

# Fonction pour VaR et ES paramétrique Student-t (df connu)
# On utilise une approche "location-scale" : r ≈ mu + s * T_v
# Estimation : mu = sample mean, s = sample_std * sqrt((v-2)/v)
# VaR analytique uses t.ppf; ES computed by numeric formula via t-pdf analytic or Monte-Carlo.
def student_t_var_es(series, alpha, v, mc_draws=200_000):
    mu = np.mean(series)
    sample_std = np.std(series, ddof=1)
    # s such that Var(mu + s*T_v) = sample_var
    # Var(T_v) = v/(v-2) for v>2  => sample_var = s^2 * v/(v-2) => s = sample_std * sqrt((v-2)/v)
    s = sample_std * np.sqrt((v - 2) / v)
    t_q = stats.t.ppf(alpha, v)               # quantile of standard Student-t
    var = -(mu + s * t_q)

    # ES : nous utilisons une approximation Monte-Carlo sur la t centrée/échelle pour plus de robustesse
    # (cette approche est simple et précise si mc_draws assez grand)
    T = stats.t(df=v).rvs(size=mc_draws)
    X = mu + s * T
    threshold = mu + s * t_q
    tail = X[X <= threshold]
    if tail.size == 0:
        es = np.nan  # improbable mais pour sécurité
    else:
        es = -np.mean(tail)
    return var, es

# Appliquer aux deux séries
results = []

for name, series in [("Normale (simulée)", r_norm), ("Student-t (df=3) - scaled", r_t_scaled)]:
    var_emp, es_emp = empirical_var_es(series, alpha)
    var_gauss, es_gauss = gaussian_var_es(series, alpha)
    var_t, es_t = student_t_var_es(series, alpha, v, mc_draws=200_000)
    results.append({
        "Série": name,
        "VaR_empirique_5%": var_emp,
        "ES_empirique_5%": es_emp,
        "VaR_gauss_5%": var_gauss,
        "ES_gauss_5%": es_gauss,
        "VaR_tparam_5%": var_t,
        "ES_tparam_5%": es_t
    })

df_results = pd.DataFrame(results).set_index("Série")
pd.options.display.float_format = "{:.6f}".format
print(df_results)





# ------------------------------------------------------------
# MATH 60231 - Devoir 1, Problème 3c
# Bootstrap de la VaR empirique à 5 % pour la série Student-t simulée
# ------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Reproductibilité
np.random.seed(42)

# Données : série Student-t (df=3) normalisée
n = 10_000
r_t = stats.t(df=3).rvs(size=n)
var_target = 1
r_t_scaled = r_t * np.sqrt(var_target / np.var(r_t, ddof=0))

# Paramètres du bootstrap
alpha = 0.05
B = 2000  # nombre de rééchantillonnages bootstrap

# Fonction VaR empirique
def empirical_var(series, alpha):
    return -np.quantile(series, alpha)  # VaR positive

# Bootstrap
boot_var = np.empty(B)
for b in range(B):
    sample = np.random.choice(r_t_scaled, size=n, replace=True)
    boot_var[b] = empirical_var(sample, alpha)

# Résultats
var_hat = empirical_var(r_t_scaled, alpha)
ci_lower, ci_upper = np.percentile(boot_var, [2.5, 97.5])
interval_width = ci_upper - ci_lower

print(f"Estimation ponctuelle de la VaR(5%) : {var_hat:.4f}")
print(f"IC 95% bootstrap : [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"Largeur de l'intervalle : {interval_width:.4f}")

# Visualisation
plt.figure(figsize=(10,6))
plt.hist(boot_var, bins=40, color='skyblue', edgecolor='black', alpha=0.7, density=True)
plt.axvline(var_hat, color='red', linestyle='--', linewidth=2, label='VaR estimée')
plt.axvline(ci_lower, color='orange', linestyle='--', linewidth=1.5, label='Borne inf. IC 95%')
plt.axvline(ci_upper, color='orange', linestyle='--', linewidth=1.5, label='Borne sup. IC 95%')
plt.title("Distribution bootstrap de la VaR empirique à 5% (Student-t, df=3)")
plt.xlabel("VaR(5%) bootstrap")
plt.ylabel("Densité")
plt.legend()
plt.grid(alpha=0.3)
plt.show()





# ------------------------------------------------------------
# MATH 60231 - Devoir 1, Problème 3d
# Simulation d'un environnement "krachs rares mais graves"
# et estimation de la VaR(5%) par trois méthodes :
# (1) quantile empirique, (2) paramétrique gaussienne,
# (3) méthode récursive fondée sur une volatilité EWMA (RiskMetrics).
# ------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Reproductibilité
np.random.seed(42)

# 1) Simulation des rendements
n = 10_000

# paramètres :
p_normal = 0.95
mu_normal = 0.0        # moyenne 0
sigma_normal = 0.01    # écart-type 1% -> 0.01 en décimal

p_crash = 0.05
mu_crash = -0.10       # moyenne -10%
sigma_crash = 0.04     # écart-type 4%

# tirages
u = np.random.rand(n)
r = np.where(
    u < p_normal,
    np.random.normal(mu_normal, sigma_normal, size=n),
    np.random.normal(mu_crash, sigma_crash, size=n)
)

# quelques statistiques descriptives
summary = pd.Series({
    "N": n,
    "Moyenne": r.mean(),
    "Écart-type": r.std(ddof=1),
    "Asymétrie": stats.skew(r),
    "Kurtosis (pearson)": stats.kurtosis(r, fisher=False)
})
print("Statistiques descriptives de la série simulée :")
print(summary.round(6))

# histogramme rapide
plt.figure(figsize=(10,5))
plt.hist(r, bins=200, density=True, alpha=0.7, edgecolor='k')
plt.title("Histogramme des rendements simulés (krachs rares)")
plt.xlabel("Rendement")
plt.ylabel("Densité")
plt.xlim(-0.25, 0.05)
plt.grid(alpha=0.3)
plt.show()

# 2) Calcul des VaR à 5%
alpha = 0.05
z_alpha = stats.norm.ppf(alpha)  # quantile normal (négatif)

# (1) VaR empirique (historical)
VaR_emp = -np.quantile(r, alpha)

# (2) VaR paramétrique gaussienne (moyenne et écart-type de l'échantillon)
mu_hat = r.mean()
sigma_hat = r.std(ddof=1)
VaR_gauss = -(mu_hat + sigma_hat * z_alpha)

# (3) Méthode récursive fondée sur la volatilité (EWMA / RiskMetrics)
# EWMA: sigma2_t = lambda * sigma2_{t-1} + (1-lambda) * r_{t-1}^2
lam = 0.94
sigma2 = np.empty(n)
# initialiser sigma2[0] par la variance empirique de la fenêtre initiale
sigma2[0] = r[:250].var(ddof=1) if n >= 250 else r.var(ddof=1)
for t in range(1, n):
    sigma2[t] = lam * sigma2[t-1] + (1 - lam) * r[t-1]**2
sigma_ewma_last = np.sqrt(sigma2[-1])
# on utilise la moyenne empirique globale pour le terme de drift ou 0
mu_for_ewma = mu_hat
VaR_ewma = -(mu_for_ewma + sigma_ewma_last * z_alpha)

# Affichage des résultats
results = pd.Series({
    "VaR_empirique_5%": VaR_emp,
    "VaR_gauss_param_5%": VaR_gauss,
    "VaR_EWMA_5% (1-step ahead)": VaR_ewma,
    "sigma_hat (sample)": sigma_hat,
    "sigma_ewma_last": sigma_ewma_last
})
print("\nEstimation des VaR(5%) par méthode :")
print(results.round(6))

# VaR EWMA en série (pour voir la dynamique)
VaR_ewma_series = -(mu_for_ewma + np.sqrt(sigma2) * z_alpha)

plt.figure(figsize=(12,5))
plt.plot(VaR_ewma_series, label='VaR EWMA (série)', alpha=0.7)
plt.hlines(VaR_emp, 0, n-1, colors='red', linestyles='--', label='VaR empirique (const.)')
plt.hlines(VaR_gauss, 0, n-1, colors='green', linestyles=':', label='VaR gaussienne (const.)')
plt.title("Comparaison VaR(5%) : série EWMA vs constantes empiriques/gaussiennes")
plt.xlabel("Index")
plt.ylabel("VaR(5%)")
plt.xlim(0, n-1)
plt.ylim(0, max(VaR_ewma_series.max()*1.1, VaR_emp*1.1))
plt.legend()
plt.grid(alpha=0.3)
plt.show()





# ------------------------------------------------------------
# MATH 60231 - Devoir 1, Problème 3e
# Bootstrap comparant la VaR gaussienne et la VaR empirique
# sur données Student-t (df=3)
# ------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Reproductibilité
np.random.seed(42)

# 1) Simuler une seule série Student-t(df=3)
n = 10_000
v = 3  # degrés de liberté
alpha = 0.05  # niveau VaR 5%

# Série de rendements tirée de Student-t(df=3)
r = stats.t(df=v).rvs(size=n)

# (option) centrer si tu veux mean=0 — ici mean déjà proche de 0 for t but center anyway:
r = r - np.mean(r)

# 2) Paramètres bootstrap
B = 2000  # nombre de rééchantillonnages (augmenter si tu veux plus de précision)

# Préallouer
var_emp_boot = np.empty(B)
var_gauss_boot = np.empty(B)
violation_emp_on_orig = np.empty(B)
violation_gauss_on_orig = np.empty(B)

# Fonction utilitaire VaR empirique (retour positive)
def var_empirical(series, alpha):
    return -np.quantile(series, alpha)

def var_gaussian_from_sample(series, alpha):
    mu = np.mean(series)
    sigma = np.std(series, ddof=1)
    z = stats.norm.ppf(alpha)
    return -(mu + sigma * z)

# 3) Bootstrap
for b in range(B):
    # rééchantillon bootstrap avec remise depuis la série observée
    sample = np.random.choice(r, size=n, replace=True)

    # Estimations sur l'échantillon bootstrap
    v_emp = var_empirical(sample, alpha)
    v_gauss = var_gaussian_from_sample(sample, alpha)

    var_emp_boot[b] = v_emp
    var_gauss_boot[b] = v_gauss

    # Calcul du taux de violation sur la série originale r
    # Violation si retour <= -VaR_est  (puisque VaR est positif = perte)
    violation_emp_on_orig[b] = np.mean(r <= -v_emp)
    violation_gauss_on_orig[b] = np.mean(r <= -v_gauss)

# 4) Résultats récapitulatifs
# Statistiques des estimations de VaR
summary_vars = pd.DataFrame({
    "VaR_emp_boot": var_emp_boot,
    "VaR_gauss_boot": var_gauss_boot
})

print("Résumé des VaR (bootstrap) :")
print(summary_vars.describe().T[["mean","std","min","25%","50%","75%","max"]].round(6))

# Statistiques des taux de violation sur la série originale
summary_viol = pd.DataFrame({
    "violation_emp": violation_emp_on_orig,
    "violation_gauss": violation_gauss_on_orig
})

print("\nRésumé des taux de violation (appliqués à la série originale) :")
print(summary_viol.describe().T[["mean","std","min","25%","50%","75%","max"]].round(6))

# Distorsion de taille moyenne (mean(violation - alpha))
distortion_emp_mean = np.mean(violation_emp_on_orig - alpha)
distortion_gauss_mean = np.mean(violation_gauss_on_orig - alpha)

print("\nTaux nominal alpha =", alpha)
print(f"Distorsion moyenne (VaR empirique)    : {distortion_emp_mean:.6f}")
print(f"Distorsion moyenne (VaR gaussienne)   : {distortion_gauss_mean:.6f}")

# Proportion de bootstrap où la violation observée dépasse alpha (sous-couverture)
prop_undercoverage_gauss = np.mean(violation_gauss_on_orig > alpha)
prop_undercoverage_emp = np.mean(violation_emp_on_orig > alpha)

print(f"\nProportion de rééchantillons où violation > {alpha}:")
print(f" - VaR empirique : {prop_undercoverage_emp:.4f}")
print(f" - VaR gaussienne: {prop_undercoverage_gauss:.4f}")

# 5) Graphiques
plt.figure(figsize=(12,5))
plt.hist(var_emp_boot, bins=50, alpha=0.6, label='VaR empirique (bootstrap)')
plt.hist(var_gauss_boot, bins=50, alpha=0.6, label='VaR gaussienne (bootstrap)')
plt.legend()
plt.title("Histogramme des estimations bootstrap de la VaR(5%)")
plt.xlabel("VaR(5%)")
plt.ylabel("Fréquence")
plt.grid(alpha=0.3)
plt.show()

plt.figure(figsize=(12,4))
plt.boxplot([violation_emp_on_orig, violation_gauss_on_orig], labels=['Violation emp', 'Violation gauss'])
plt.axhline(alpha, color='red', linestyle='--', label=f'Alpha={alpha}')
plt.title("Distribution des taux de violation (appliqués à la série originale)")
plt.ylabel("Taux de violation observé")
plt.legend()
plt.grid(alpha=0.3)
plt.show()