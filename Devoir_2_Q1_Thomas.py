###############################################################################
# Connexion WRDS

import wrds
import pandas as pd
import numpy as np

def connect_wrds():
    db = wrds.Connection()
    return db

# Sélection aléatoire de 50 entreprises du S&P 500 (via CRSP)

def get_sp500_sample(db, n=50, seed=123):
    # Liste officielle CRSP des membres S&P 500
    sp500 = db.get_table('crsp', 'msp500list')
    sp500 = sp500[['permno']].drop_duplicates()

    # Sélection aléatoire
    sample = sp500.sample(n=n, random_state=seed)

    # Récupérer les tickers CRSP
    stocknames = db.get_table('crsp', 'stocknames')[['permno', 'ticker']]
    sample = sample.merge(stocknames, on='permno', how='left')

    # Nettoyer : supprimer les lignes sans ticker
    sample = sample.dropna(subset=['ticker'])

    return sample

# Télécharger les prix (CRSP Daily)

def get_prices(db, sample, start='2010-01-01', end='2023-12-31'):
    tic_list = "', '".join(sample['ticker'].unique())

    query = f"""
        SELECT a.permno, a.date, a.prc, a.ret
        FROM crsp.dsf AS a
        JOIN crsp.stocknames AS b
            ON a.permno = b.permno
        WHERE b.ticker IN ('{tic_list}')
          AND a.date BETWEEN '{start}' AND '{end}'
    """

    prices = db.raw_sql(query)
    return prices

# Télécharger EPS (Compustat – trimestriel)

def get_eps(db, sample):
    # Récupération de gvkey via security table (Compustat)
    secm = db.get_table('comp', 'secm')[['gvkey', 'tic']]

    gvkey_map = sample.merge(secm, left_on='ticker', right_on='tic', how='left')

    gvkeys = gvkey_map['gvkey'].dropna().unique()
    gvkeys = "', '".join(gvkeys.astype(str))

    query = f"""
        SELECT gvkey, datadate, fqtr, epspx
        FROM comp.fundq
        WHERE gvkey IN ('{gvkeys}')
    """

    eps = db.raw_sql(query)
    return eps

# Calculer EPS glissant (TTM) et trailing PE

def compute_trailing_PE(prices, eps, link_table):
    link = link_table[['gvkey', 'lpermno']].dropna().rename(columns={'lpermno': 'permno'})
    eps = eps.merge(link, on='gvkey', how='left')

    eps = eps.sort_values(['permno', 'datadate'])
    eps['eps_ttm'] = eps.groupby('permno')['epspx'].rolling(4).sum().reset_index(0, drop=True)

    data = prices.merge(eps[['permno', 'datadate', 'eps_ttm']], on='permno', how='left')
    data = data.sort_values(['permno', 'date'])
    data['eps_ttm'] = data.groupby('permno')['eps_ttm'].ffill()

    data['trailing_pe'] = data['prc'] / data['eps_ttm']
    return data

# Pipeline complet

def run_pipeline():
    db = connect_wrds()

    sample = get_sp500_sample(db)
    prices = get_prices(db, sample)
    eps = get_eps(db, sample)

    link_table = db.get_table('crsp', 'ccmxpf_linktable')

    final_data = compute_trailing_PE(prices, eps, link_table)
    return final_data

# Exécution

if __name__ == "__main__":
    data = run_pipeline()
    print(data.head())




###############################################################################
# Chargement des librairies
###############################################################################

import pandas as pd
import numpy as np


###############################################################################
# Calcul des rendements quotidiens (si non fournis)
###############################################################################

def compute_returns(data):
    # Hypothèse : si CRSP.ret existe, on l’utilise. Sinon calcul manuel.
    if 'ret' in data.columns:
        data['return'] = data['ret']
    else:
        data = data.sort_values(['permno', 'date'])
        data['return'] = data.groupby('permno')['prc'].pct_change()
    return data


###############################################################################
# Estimation du BPA attendu (Expected EPS)
###############################################################################
# Hypothèse explicitement énoncée :
# L’article suggère que le BPA attendu peut être approché par une
# extrapolation linéaire du BPA TTM (Trailing Twelve Months).
#
# Nous utilisons :
#   Expected_EPS = eps_ttm_t−1 + (eps_ttm_t−1 − eps_ttm_t−2)
#
# => équivalent à projeter la croissance récente du BPA.
###############################################################################

def compute_expected_eps(data):
    data = data.sort_values(['permno', 'date'])

    # On prend la variation trimestrielle implicite du trailing EPS
    data['eps_ttm_lag1'] = data.groupby('permno')['eps_ttm'].shift(1)
    data['eps_ttm_lag2'] = data.groupby('permno')['eps_ttm'].shift(2)

    # Croissance récente
    data['eps_growth'] = data['eps_ttm_lag1'] - data['eps_ttm_lag2']

    # Expected EPS = EPS_{t−1} + croissance récente
    data['expected_eps'] = data['eps_ttm_lag1'] + data['eps_growth']

    return data


###############################################################################
# Calcul du ratio Prix / Expected EPS (Expected P/E ratio)
###############################################################################

def compute_expected_PE(data):
    data['expected_pe'] = data['prc'] / data['expected_eps']
    return data


###############################################################################
# Pipeline complet pour la partie (b)
###############################################################################

def run_part_b(data):
    # Calcul des rendements
    data = compute_returns(data)

    # Calcul du BPA attendu
    data = compute_expected_eps(data)

    # Calcul du ratio prix / BPA attendu
    data = compute_expected_PE(data)

    return data


###############################################################################
# Exécution (si tu veux voir les résultats)
###############################################################################

if __name__ == "__main__":
    # On suppose que 'data' provient de ton pipeline de la partie (a)
    # Exemple : data = run_pipeline()
    # Pour tester : mets ici ton fichier ou ton DataFrame

    try:
        data = run_pipeline()   # si tu exécutes depuis le script global
        result = run_part_b(data)
        print(result.head())
    except NameError:
        print("Attention : charge d'abord les données de la partie (a) ou importe ton DataFrame.")