# IMPORTS

import wrds
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# PARTIE (a)
# Construction des données : S&P 500 → Prix, BPA (EPS), Trailing P/E

def connect_wrds():
    print(">>> Connexion WRDS...")
    db = wrds.Connection()
    print(">>> Connexion WRDS OK.\n")
    return db

def get_sp500_sample(db, n=50, seed=123):
    print(">>> Récupération liste S&P 500...")
    sp500 = db.get_table('crsp', 'msp500list')
    sp500 = sp500[['permno']].drop_duplicates()

    sample = sp500.sample(n=n, random_state=seed)

    stocknames = db.get_table('crsp', 'stocknames')[['permno', 'ticker']]
    sample = sample.merge(stocknames, on='permno', how='left')

    sample = sample.dropna(subset=['ticker'])
    print(">>> Échantillon S&P 500 créé.\n")
    return sample

def get_prices(db, sample, start='2010-01-01', end='2023-12-31'):
    print(">>> Téléchargement des prix CRSP...")
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
    print(">>> Prix téléchargés.\n")
    return prices

def get_eps(db, sample):
    print(">>> Téléchargement EPS Compustat...")
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
    print(">>> EPS téléchargés.\n")
    return eps

def compute_trailing_PE(prices, eps, link_table):
    print(">>> Calcul du trailing EPS et trailing P/E...")
    link = link_table[['gvkey', 'lpermno']].dropna().rename(columns={'lpermno': 'permno'})
    eps = eps.merge(link, on='gvkey', how='left')

    eps = eps.sort_values(['permno', 'datadate'])
    eps['eps_ttm'] = eps.groupby('permno')['epspx'].rolling(4).sum().reset_index(0, drop=True)

    data = prices.merge(eps[['permno', 'datadate', 'eps_ttm']], on='permno', how='left')
    data = data.sort_values(['permno', 'date'])
    data['eps_ttm'] = data.groupby('permno')['eps_ttm'].ffill()

    data['trailing_pe'] = data['prc'] / data['eps_ttm']
    print(">>> Trailing P/E calculé.\n")
    return data

def run_part_a():
    db = connect_wrds()
    sample = get_sp500_sample(db)
    prices = get_prices(db, sample)
    eps = get_eps(db, sample)
    link_table = db.get_table('crsp', 'ccmxpf_linktable')
    data = compute_trailing_PE(prices, eps, link_table)
    return data

# PARTIE (b)
# Rendements, Expected EPS, Expected P/E

def compute_returns(data):
    print(">>> Calcul des rendements...")
    if 'ret' in data.columns:
        data['return'] = data['ret']
    else:
        data = data.sort_values(['permno', 'date'])
        data['return'] = data.groupby('permno')['prc'].pct_change()
    print(">>> Rendements OK.\n")
    return data

def compute_expected_eps(data):
    print(">>> Calcul du BPA attendu (Expected EPS)...")
    data = data.sort_values(['permno', 'date'])
    data['eps_ttm_lag1'] = data.groupby('permno')['eps_ttm'].shift(1)
    data['eps_ttm_lag2'] = data.groupby('permno')['eps_ttm'].shift(2)
    data['eps_growth'] = data['eps_ttm_lag1'] - data['eps_ttm_lag2']
    data['expected_eps'] = data['eps_ttm_lag1'] + data['eps_growth']
    print(">>> Expected EPS OK.\n")
    return data

def compute_expected_PE(data):
    print(">>> Calcul du Expected P/E...")
    data['expected_pe'] = data['prc'] / data['expected_eps']
    print(">>> Expected P/E OK.\n")
    return data

def run_part_b(data):
    data = compute_returns(data)
    data = compute_expected_eps(data)
    data = compute_expected_PE(data)
    return data

# PARTIE (c) — Statistiques + Graphiques

def descriptive_stats(data):
    print(">>> Statistiques descriptives :\n")
    stats = data[['prc', 'return', 'eps_ttm', 'trailing_pe', 'expected_pe']].describe()
    print(stats)
    return stats

def plot_timeseries(data):
    print("\n>>> Génération des graphiques...\n")

    variables = {
        'prc': "Prix de l'action",
        'return': "Rendements",
        'eps_ttm': "BPA glissant (TTM)",
        'trailing_pe': "P/B glissant",
        'expected_pe': "P/B basé sur BPA attendu"
    }

    for col, title in variables.items():
        plt.figure(figsize=(10,4))
        for firm in data['permno'].unique()[:5]:
            subset = data[data['permno'] == firm]
            plt.plot(subset['date'], subset[col])
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel(title)
        plt.tight_layout()
        plt.show()

# EXÉCUTION FINALE

if __name__ == "__main__":
    print(">>> Pipeline WRDS - Début\n")

    data = run_part_a()
    print(">>> Partie (a) terminée.\n", data.head())

    data = run_part_b(data)
    print("\n>>> Partie (b) terminée.\n", data.head())

    stats = descriptive_stats(data)
    plot_timeseries(data)

    print("\n>>> Pipeline COMPLET terminé.\n")