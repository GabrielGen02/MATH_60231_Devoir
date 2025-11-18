
# importation des packages nécessaires pour le devoir 
import wrds
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import norm


# ──────────────────────────────────────────── # 
# ──────────────── problème 2 ──────────────── # 
# ──────────────────────────────────────────── # 

# Préparation des données pour répondre au problème 2 
# connexion a un compte wrds pour pouvoir télécharger les données requises pour le devoir 
db = wrds.Connection()
#db.rollback() # ferme et réinitialise la connexion a wrds 

# fonction permettant de créer un df contenant : date, permno, prix, rendement quotidien et le ticker
def get_prices(db, tickers, start_date='2012-01-03', end_date='2023-01-03'):
    """
    Extrait les prix quotidiens pour une liste de tickers entre deux dates.
    db : connexion WRDS active
    tickers : liste de tickers (ex: ['AAPL', 'JPM', 'XOM', 'SPY'])
    """
    try : 
        # Étape 1 : récupérer les permnos
        tickers_str = "', '".join(tickers)
        query_permno = f"""
            SELECT DISTINCT permno, ticker
            FROM crsp.stocknames
            WHERE ticker IN ('{tickers_str}')
        """
        df_permno = db.raw_sql(query_permno)
    
        # Étape 2 : extraire les prix
        permnos = df_permno['permno'].tolist()
        permnos_str = ", ".join(str(p) for p in permnos)
        query_prices = f"""
            SELECT date, permno, prc, ret
            FROM crsp.dsf
            WHERE date BETWEEN '{start_date}' AND '{end_date}'
            AND permno IN ({permnos_str})
        """
        df_prices = db.raw_sql(query_prices)
    
        df_prices = db.raw_sql(query_prices)

        # Merge pour ajouter les tickers au df final 
        df_prices = df_prices.merge(df_permno, on='permno', how='left')

        return df_prices
    except Exception as e:
            db.rollback()
            print("Erreur SQL, rollback effectué.")
            raise e


# exemple de df pouvant être créé : 
tickers_gabriel = ['SPY', 'GS', 'GOOGL', 'GNE']
df_gabriel = get_prices(db, tickers_gabriel)
print(df_gabriel.head()) # vision rapide du df 

# création d'un df avec des actions commencant par T et B 
tickers_bernabe = ["BAC", "BB", "BP"] # B
df_bernar = get_prices(db, tickers_bernabe)

tickers_tom = ["TD", "TXN", "TRP"] # T 
df_tom = get_prices(db, tickers_tom)

# validation que les dates du df sont conformes a l'énoncé
def filtrage_dates(df) : 
    # filtrage des dates du df 
    df = df[(df['date'] >= '2012-01-03') & (df['date'] <= '2023-01-03')]
    df['date'] = pd.to_datetime(df['date'], errors='coerce') # s'assure que les dates sont en datetime
    print(df['date'].min()) # s'assure que les dates sont correctements filtrés 
    print(df['date'].max())
    
    return df 

df_gabriel = filtrage_dates(df_gabriel)
df_ber = filtrage_dates(df_bernar)
df_tom = filtrage_dates(df_tom)

# --- Question 2 (a) ---

# fonction qui illustre le graphqiue des rendemments quotidien 
def tracer_rendements(df, x_col='date', y_col='ret', ticker_col='ticker'):
    """
    df : DataFrame contenant les colonnes 'date', 'ret', 'ticker'
    x_col : Date
    y_col : Rendemment 
    ticker_col : colonne contenant les tickers
    """
    tickers = df[ticker_col].unique()
    
    for t in tickers:
        df_t = df[df[ticker_col] == t].sort_values(by=x_col)
        
        plt.figure(figsize=(12, 5))
        plt.plot(df_t[x_col], df_t[y_col], color= '#FF9999', linewidth=1)
        
        # Formatage des dates sur l'axe des x
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.gcf().autofmt_xdate()  # rotation automatique
        
        plt.xlabel("Date")
        plt.ylabel(f"Rendemment de {t}")
        plt.title(f"Évolution du rendemment de {t} dans le temps")
        plt.grid(False)
        plt.tight_layout()
        plt.show()

tracer_rendements(df_gabriel) # graphique des rendement pour df_gabriel
tracer_rendements(df_ber)
tracer_rendements(df_tom)

# fonction qui calcule les statistiques descriptives des actions 
def stat_descriptive(df) : 
    """
    Affiche les stats descriptives des rendements pour un ticker donné.
    """
    tickers = df['ticker'].unique()
    
    for t in tickers:
        ret = df[df['ticker'] == t]['ret']
        print(f"Statistiques pour {t} :")
        print(ret.describe())

stat_descriptive(df_gabriel)
stat_descriptive(df_ber)
stat_descriptive(df_tom)

# --- Question 2 (b) ---

# préparation des données pour la question 2 (b) 
df_gabriel = df_gabriel[df_gabriel['ticker'] != 'GS'] # exclusion de l'action fesant parti du secteur bancaire 

# calcule du VaR historique 
def var_es_histo(df, alpha=0.05):
    """
    Calcule la VaR historique pour tous les tickers du Df
    """
    results = {}
    for t in df['ticker'].unique():
        ret = df[df['ticker'] == t]['ret'].sort_values() # classe en ordre croissant 
        var = ret.quantile(alpha)
        es = ret[ret <= var].mean()  # moyenne des pertes sous la VaR
        results[t] = {'VaR': var, 'Es': es}
        
        print(f"VaR à {int(alpha*100)}% pour {t} : {var:.4f}")
        print(f"Es à {int(alpha*100)}% pour {t} : {es:.4f}")
    return results 

var_es_histo(df_gabriel, alpha=0.05) # pour 5 %
var_es_histo(df_gabriel, alpha=0.01) # pour 1 % 

# --- Question 2 (c) ---

# calcule du VaR et Es paramétrique 
def var_es_para(df, alpha=0.05):
    """
    Calcule la VaR paramétriques pour tous les tickers du Df, on suppose que ret suit
    une loi normale 
    """
    results = {}
    if alpha == 0.05 : 
        z = -1.6449
    else : 
        z = -2.3263
    
    for t in df['ticker'].unique():
        ret_moy = df[df['ticker'] == t]['ret'].mean() # calcule la moyenne
        std = df[df['ticker'] == t]['ret'].std() # calcule la variance 
        var = ret_moy + (z*std)
        es = ret_moy - (std * norm.pdf(z) / alpha)
        results[t] = {'VaR': var, 'Es': es}
        
        print(f"VaR à {int(alpha*100)}% pour {t} : {var:.4f}")
        print(f"Es à {int(alpha*100)}% pour {t} : {es:.4f}")
    return results 
    
var_es_para(df_gabriel, alpha=0.05) # pour 5 %
var_es_para(df_gabriel, alpha=0.01) # pour 1 % 

# comparaison avec les résultats de la question précédente
para = var_es_para(df_gabriel, alpha=0.05)
histo = var_es_histo(df_gabriel, alpha=0.05)

# différence entre le modèle historique et paramétrique
for t in para:
    diff_var = abs(histo[t]['VaR']) - abs(para[t]['VaR'])
    diff_es = abs(histo[t]['Es']) - abs(para[t]['Es'])
    print(f"{t} différence dans le VaR: {diff_var:.4f}, différence dans le ES: {diff_es:.4f}") 

# --- Question 2 (d) ---

# Calcule du modèle VaR variant dans le temps et dépandent de la volatilité passé 
def var_ewma(df, alpha=0.05, lambda_=0.94):
    z = 1.6449 if alpha == 0.05 else 2.3263
    results = {}

    for tkr in df['ticker'].unique():
        ret = df[df['ticker'] == tkr].sort_values('date')['ret'].values
        dates = df[df['ticker'] == tkr].sort_values('date')['date'].values

        sigma2 = np.zeros_like(ret)
        sigma2[0] = np.var(ret[:20])  # initialisation avec les 20 premiers jours

        for t in range(1, len(ret)):
            sigma2[t] = lambda_ * sigma2[t-1] + (1 - lambda_) * ret[t-1]**2

        sigma = np.sqrt(sigma2)
        var_series = -z * sigma

        results[tkr] = pd.DataFrame({'date': dates, 'VaR': var_series, 'ret': ret})
    
    return results

results = var_ewma(df_gabriel, alpha=0.05)

# visualisation graphqiue de la VaR estimé plus haut 
for tkr, df_plot in results.items():
    plt.figure(figsize=(10, 4))
    plt.plot(df_plot['date'], df_plot['ret'], label='Rendements', alpha=0.6)
    plt.plot(df_plot['date'], df_plot['VaR'], label='VaR dynamique', color='red')
    plt.title(f"VaR dynamique vs rendements pour {tkr}")
    plt.xlabel("Date")
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- Question 2 (e) ---

# création de rendements négatifs simulants une crise 
tickers = ['GOOGL', 'SPY', 'GNE']
crash_dates = pd.date_range(start='2020-03-15', periods=6, freq='D')
crash_values = [-0.05, -0.04, -0.06, -0.06, -0.05, -0.07]

# remplace les valeurs du df original par ceux de crise 
for tkr in tickers:
    for date, value in zip(crash_dates, crash_values):
        mask = (df_gabriel['ticker'] == tkr) & (df_gabriel['date'] == date)
        df_gabriel.loc[mask, 'ret'] = value

df_gabriel[(df_gabriel['date'].isin(crash_dates)) & (df_gabriel['ticker'].isin(tickers))] # validation 

var_histo = var_es_histo(df_gabriel, alpha=0.05)
var_gauss = var_es_para(df_gabriel, alpha=0.05)
var_dyn = var_ewma(df_gabriel, alpha=0.05)

# comparaison graphique 
for tkr in ['GOOGL', 'SPY', 'GNE']:
    df_dyn = var_dyn[tkr]
    plt.figure(figsize=(10, 4))
    plt.plot(df_dyn['date'], df_dyn['ret'], label='Rendements', alpha=0.6)
    plt.plot(df_dyn['date'], df_dyn['VaR'], label='VaR récursive', color='red')

    # Ajouter les VaR fixes
    plt.axhline(var_gauss[tkr]['VaR'], color='blue', linestyle='--', label='VaR gaussienne')
    plt.axhline(var_histo[tkr]['VaR'], color='green', linestyle='--', label='VaR historique')

    plt.title(f"Comparaison des VaR pour {tkr}")
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- Question 2 (f) ---

# calcule du nombre de violations pour les 3 modèles (nb de jours ou le rend est inférieur à la VaR)

# Série de rendements utilisée pour les violations
ret_series = var_dyn[tkr]['ret']  # rendements quotidiens alignés dans le temps

# pour le modèle en simulation historique 
VaR_histo = var_histo[tkr]['VaR']
violation_histo = float((ret_series < VaR_histo).sum())

# pour le modèle paramétrique 
VaR_para = var_gauss[tkr]["VaR"]
violation_para = float((ret_series < VaR_para).sum())

# pour le modèle variant dans le temps 
violations_dyn = float((var_dyn[tkr]['ret'] < var_dyn[tkr]['VaR']).sum())

# affichage des résultats 
violations_dict = {
    "violation_histo": violation_histo,
    "violation_para": violation_para,
    "violation_dyn": violations_dyn
}

for v in violations_dict : 
    print(f"Le nombre de violations du modèle {v} est de {violations_dict[v]}")

# calcule de nb d'observations par ticker 
tickers = ['GOOGL', 'GNE', 'SPY']

for tkr in tickers:
    n_obs = len(df_gabriel[df_gabriel['ticker'] == tkr])
    print(f"{tkr} a {n_obs} observations")

# calcule de la statistique de test 
def stat_test_couverture(violations_dict, T=2769, alpha=0.05):
    stats = {}
    " calcule du numérateur puis de dénominateur puis de la stat de test"
    for v in violations_dict:
        nb_violations = violations_dict[v]
        nume = nb_violations - (alpha * T)
        denomi = (alpha * (1 - alpha) * T) ** 0.5
        S = nume / denomi
        stats[v] = round(S, 3)
    return stats

stats_test = stat_test_couverture(violations_dict, T=2769, alpha=0.05)
print(stats_test) # résultats 

# --- Question 2 (f) ---

# fonction donnant le nombres de switch entre violation et non violation 
def test_sequence(ret_series, VaR):   
    # Séquence binaire : 1 si violation, 0 sinon
    sequence = (ret_series < VaR).astype(int).values # créer une liste avec 1 = violation, 0 sinon

    # Comptage des états
    n0 = np.sum(sequence == 0)
    n1 = np.sum(sequence == 1)

    # Transitions entre états (0 à 1 ou 1 à 0)
    transitions = np.sum(sequence[1:] != sequence[:-1])

    # Nombre attendu de transitions
    expected = 2 * n0 * n1 / (n0 + n1)

    return {
        'n0': n0,
        'n1': n1,
        'transitions_observées': transitions,
        'transitions_attendues': round(expected, 2),
        'clustering': transitions < expected
    }

tickers = ['GOOGL', 'SPY', 'GNE']

# pour modèle historique
for tkr in tickers:
    result = test_sequence(var_dyn[tkr]['ret'], VaR_histo)
    print(f"{tkr} - transitions observées (historique) : {result['transitions_observées']}")
    print(f"{tkr} - transitions attendues (historique) : {result['transitions_attendues']}")

# pour modèle paramétrique
for tkr in tickers:
    result = test_sequence(var_dyn[tkr]['ret'], VaR_para)
    print(f"{tkr} - transitions observées (paramétrique) : {result['transitions_observées']}")
    print(f"{tkr} - transitions attendues (paramétrique) : {result['transitions_attendues']}")

# pour modèle variant dans le temps
for tkr in tickers:
    result = test_sequence(var_dyn[tkr]['ret'], var_dyn[tkr]['VaR'])
    print(f"{tkr} - transitions observées (variation dans le temps) : {result['transitions_observées']}")
    print(f"{tkr} - transitions attendues (variation dans le temps) : {result['transitions_attendues']}")


# ──────────────────────────────────────────── # 
# ──────────── fin du problème 2 ───────────── # 
# ──────────────────────────────────────────── # 

# ──────────────────────────────────────────── # 
# ──────────────── problème 3 ──────────────── # 
# ──────────────────────────────────────────── # 

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
plt.hist(r_t_scaled, bins=60, alpha=0.6, color='red', density=True, label='Student-t (df=3)')
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
sns.kdeplot(r_t_scaled, color='red', label='Student-t (df=3)')
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

# ──────────────────────────────────────────── # 
# ──────────── fin du problème 3 ───────────── # 
# ──────────────────────────────────────────── # 

# ──────────────────────────────────────────── # 
# ──────────────── problème 4 ──────────────── # 
# ──────────────────────────────────────────── # 

# df et action choisit pour la question 4 
tickers_4 = ["GOOGL"] # Bell
df_4 = get_prices(db, tickers_4, start_date='2012-01-03', end_date='2023-01-03') # extration des données 
df_4.head() # visualisation des données 

# --- Question (a) --- 

from scipy.stats import skew, kurtosis

# fonction calculant les statistisques descriptives de bases 
def stat_sommaire(df, prc='prc', ret='ret'):
    """
    Calcule et affiche les statistiques sommaires pour les prix et les rendements.

    Args:
        df (pd.DataFrame): DataFrame contenant les colonnes de prix et de rendements.
        col_prix (str): Nom de la colonne des prix.
        col_rendement (str): Nom de la colonne des rendements.
    """
    # Vérification des colonnes
    if prc in df.columns:
        prix = df[prc].dropna()
        print("\n Statistiques sur les PRIX")
        print(f"Moyenne: {prix.mean():.4f}")
        print(f"Écart-type: {prix.std():.4f}")
        print(f"Asymétrie: {skew(prix):.4f}")
        print(f"Aplatissement (excès de kurtosis): {kurtosis(prix, fisher=True):.4f}")
    else:
        print(f" Colonne '{prc}' introuvable.")

    if ret in df.columns:
        rendement = df[ret].dropna()
        print("\n Statistiques sur les RENDEMENTS")
        print(f"Moyenne: {rendement.mean():.4f}")
        print(f"Écart-type: {rendement.std():.4f}")
        print(f"Asymétrie: {skew(rendement):.4f}")
        print(f"Aplatissement (excès de kurtosis): {kurtosis(rendement, fisher=True):.4f}")
    else:
        print(f" Colonne '{ret}' introuvable.")

df_4 = filtrage_dates(df_4) # définission de la période entre 2012 et 2023 
res_sommaire = stat_sommaire(df_4, prc='prc', ret='ret')
tracer_rendements(df_4)

# --- Question (b) --- 

# calcule de la VaR avec une fenêtre de 250 jours 
wind = 250 # grosseur de la fenetre 
var_5 = df_4["ret"].rolling(window=wind).quantile(0.05)

# calcule le nombre de violation 
violations = df_4["ret"] < var_5
nb_violations_1 = violations.sum() # nb de vioaltions 
dates_violations = df_4["date"][violations] # extration des dates 

# test d'indépendance 
from statsmodels.stats.diagnostic import acorr_ljungbox

# Série booléenne des violations (rendement < VaR), convertie en 0/1
violations = (df_4["ret"] < var_5).astype(int)

# Supprimer les NaN (au début de la série, avant que la VaR soit définie)
violations_clean = violations.dropna()

# Test de Ljung-Box sur les 10 premiers lags
test_indep = acorr_ljungbox(violations_clean, lags=500, return_df=True)

print("\nTest d'indépendance des violations (Ljung-Box)")
print(test_indep)

# graphique illustrant les violations par rapport a la VaR et au rendement journalier 
plt.figure(figsize=(12,6))
plt.plot(df_4["date"], df_4["ret"], label="Rendements", color="blue")
plt.plot(df_4["date"], var_5, label="VaR 5%", color="red")
plt.scatter(df_4["date"][violations], df_4["ret"][violations], color="black", label="Violations", marker=".")
plt.legend()
plt.title("VaR historique à 5% vs Rendements")
plt.xlabel("Date")
plt.ylabel("Rendement")
plt.grid(False)
plt.show()

# --- Question (C) --- 

# Série de rendements propre
rets = df_4["ret"]

# Initialiser la liste
var_expansive = [np.nan] * len(rets)

# Boucle expansive
for t in range(250, len(rets)):
    window = rets[:t]  # Tous les rendements jusqu'à t-1
    var_expansive[t] = np.percentile(window, 5)

# calcule le nombre de violation 
violations = df_4["ret"] < var_expansive
nb_violations_2 = violations.sum() # nb de violations 
dates_violations = df_4["date"][violations] # extration des dates 

# graphique illustrant les violations par rapport a la VaR et au rendement journalier 
plt.figure(figsize=(12,6))
plt.plot(df_4["date"], df_4["ret"], label="Rendements", color="blue")
plt.plot(df_4["date"], var_expansive, label="VaR 5%", color="red")
plt.scatter(df_4["date"][violations], df_4["ret"][violations], color="black", label="Violations", marker=".")
plt.legend()
plt.title("VaR historique à 5% vs Rendements")
plt.xlabel("Date")
plt.ylabel("Rendement")
plt.grid(False)
plt.show()

# différence entre les deux modèles 
diff = (nb_violations_1 - nb_violations_2)
if diff < 0 : 
    print("modele 2 fait plus de violations")
else : 
    print("modele 1 fait plus de violations")

print(diff)
print(nb_violations_1)
print(nb_violations_2)

# ──────────────────────────────────────────── # 
# ──────────── fin du problème 4 ───────────── # 
# ──────────────────────────────────────────── #