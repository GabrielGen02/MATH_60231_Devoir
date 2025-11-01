
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
dates_violations = df_4["date"][violations] # extration des dates 

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


# ──────────────────────────────────────────── # 
# ──────────── fin du problème 4 ───────────── # 
# ──────────────────────────────────────────── #