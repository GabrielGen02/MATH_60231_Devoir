
# importation des packages nécessaires pour le devoir 
import wrds
import pandas as pd

# connexion a un compte wrds pour pouvoir télécharger les données requises pour le devoir 
db = wrds.Connection()

# fonction permettant de créer un df contenant : date, permno, prix, rendement quotidien et le ticker
def get_prices(db, tickers, start_date='2012-01-03', end_date='2023-01-03'):
    """
    Extrait les prix quotidiens pour une liste de tickers entre deux dates.
    db : connexion WRDS active
    tickers : liste de tickers (ex: ['AAPL', 'JPM', 'XOM', 'SPY'])
    """
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

# exemple de df pouvant être créé : 
tickers_gabriel = ['SPY', 'JPM', 'AAPL', 'XOM']
df_gabriel = get_prices(db, tickers_gabriel)
print(df_gabriel.head())