
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

# validation que les dates du df sont conformes a l'énoncé
def filtrage_dates(df) : 
    # filtrage des dates du df 
    df = df[(df['date'] >= '2012-01-03') & (df['date'] <= '2023-01-03')]
    df['date'] = pd.to_datetime(df['date'], errors='coerce') # s'assure que les dates sont en datetime
    print(df['date'].min()) # s'assure que les dates sont correctements filtrés 
    print(df['date'].max())
    
    return df 
df_gabriel = filtrage_dates(df_gabriel)

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

# --- Question 2 (b) ---

# préparation des données pour la question 2 (b) 
df_gabriel = df_gabriel[df_gabriel['ticker'] != 'GM'] # exclusion de l'action fesant parti du secteur bancaire 

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

# différence entr ele modèle historique et paramétrique
for t in para:
    diff_var = histo[t]['VaR'] - para[t]['VaR']
    diff_es = histo[t]['Es'] - para[t]['Es']
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












