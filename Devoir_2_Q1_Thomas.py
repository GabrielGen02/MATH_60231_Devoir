import wrds
import pandas as pd
import numpy as np

# 1. Connexion
db = wrds.Connection()

# 2. Récupérer la composition actuelle du S&P 500 (CRSP)
sp500 = db.get_table('crsp', 'msp500')
last_date = sp500['date'].max()
sp500 = sp500[ sp500['date'] == last_date ]

# 3. Tirage aléatoire de 50 entreprises
sample_permnos = np.random.choice(sp500['permno'], size=50, replace=False)

# 4. Télécharger les prix quotidiens CRSP
prices = db.raw_sql(f"""
    SELECT permno, date, prc, ret
    FROM crsp.dsf
    WHERE permno IN ({','.join(map(str, sample_permnos))})
      AND date BETWEEN '2003-01-01' AND '2023-12-31'
""")

# 5. Télécharger les EPS trimestriels Compustat
eps = db.raw_sql(f"""
    SELECT gvkey, datadate, epspxq
    FROM comp.fundq
    WHERE gvkey IN (
        SELECT gvkey FROM crsp.ccmxpf_linktable
        WHERE permno IN ({','.join(map(str, sample_permnos))})
          AND linktype IN ('LU','LC')
    )
    AND datadate BETWEEN '2002-01-01' AND '2023-12-31'
""")

# 6. (Optionnel) Fusion CRSP–Compustat via CCM
links = db.get_table('crsp', 'ccmxpf_linktable')
links = links[ links['permno'].isin(sample_permnos) ]
merged = prices.merge(links[['gvkey','permno']], on='permno', how='left')
merged = merged.merge(eps, on='gvkey', how='left')

print("Données téléchargées :")
print("Prix CRSP :", prices.shape)
print("EPS Compustat :", eps.shape)