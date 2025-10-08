import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from collections import Counter
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
os.makedirs('laby3_wykresy', exist_ok=True)

sns.set_style("whitegrid")
kolory = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def przygotuj_dane_wta(sciezka_pliku='wta_matches_combined.csv'):
    df = pd.read_csv(sciezka_pliku)
    gracze = {}
    for _, mecz in df.iterrows():
        id_zwyciezcy = mecz['winner_id']
        if id_zwyciezcy not in gracze:
            gracze[id_zwyciezcy] = {
                'id': id_zwyciezcy,
                'nazwa': mecz['winner_name'],
                'reka': mecz['winner_hand'],
                'wzrost': mecz['winner_ht'],
                'kraj': mecz['winner_ioc'],
                'mecze': 0,
                'wygrane': 0,
                'asy': 0,
                'df': 0,
                'mecze_asy': 0,
                'mecze_df': 0,
                'twarde_mecze': 0, 'twarde_wygrane': 0,
                'clay_mecze': 0, 'clay_wygrane': 0,
                'trawa_mecze': 0, 'trawa_wygrane': 0,
                'dywan_mecze': 0, 'dywan_wygrane': 0
            }
        
        gracze[id_zwyciezcy]['mecze'] += 1
        gracze[id_zwyciezcy]['wygrane'] += 1
        
        nawierzchnia = mecz['surface']
        if nawierzchnia == 'Hard':
            gracze[id_zwyciezcy]['twarde_mecze'] += 1
            gracze[id_zwyciezcy]['twarde_wygrane'] += 1
        elif nawierzchnia == 'Clay':
            gracze[id_zwyciezcy]['clay_mecze'] += 1
            gracze[id_zwyciezcy]['clay_wygrane'] += 1
        elif nawierzchnia == 'Grass':
            gracze[id_zwyciezcy]['trawa_mecze'] += 1
            gracze[id_zwyciezcy]['trawa_wygrane'] += 1
        elif nawierzchnia == 'Carpet':
            gracze[id_zwyciezcy]['dywan_mecze'] += 1
            gracze[id_zwyciezcy]['dywan_wygrane'] += 1
        
        if pd.notna(mecz['w_ace']):
            gracze[id_zwyciezcy]['asy'] += mecz['w_ace']
            gracze[id_zwyciezcy]['mecze_asy'] += 1
        
        if pd.notna(mecz['w_df']):
            gracze[id_zwyciezcy]['df'] += mecz['w_df']
            gracze[id_zwyciezcy]['mecze_df'] += 1
        
        id_przegranego = mecz['loser_id']
        if id_przegranego not in gracze:
            gracze[id_przegranego] = {
                'id': id_przegranego,
                'nazwa': mecz['loser_name'],
                'reka': mecz['loser_hand'],
                'wzrost': mecz['loser_ht'],
                'kraj': mecz['loser_ioc'],
                'mecze': 0,
                'wygrane': 0,
                'asy': 0,
                'df': 0,
                'mecze_asy': 0,
                'mecze_df': 0,
                'twarde_mecze': 0, 'twarde_wygrane': 0,
                'clay_mecze': 0, 'clay_wygrane': 0,
                'trawa_mecze': 0, 'trawa_wygrane': 0,
                'dywan_mecze': 0, 'dywan_wygrane': 0
            }
        
        gracze[id_przegranego]['mecze'] += 1
        if nawierzchnia == 'Hard':
            gracze[id_przegranego]['twarde_mecze'] += 1
        elif nawierzchnia == 'Clay':
            gracze[id_przegranego]['clay_mecze'] += 1
        elif nawierzchnia == 'Grass':
            gracze[id_przegranego]['trawa_mecze'] += 1
        elif nawierzchnia == 'Carpet':
            gracze[id_przegranego]['dywan_mecze'] += 1
        
        if pd.notna(mecz['l_ace']):
            gracze[id_przegranego]['asy'] += mecz['l_ace']
            gracze[id_przegranego]['mecze_asy'] += 1
        
        if pd.notna(mecz['l_df']):
            gracze[id_przegranego]['df'] += mecz['l_df']
            gracze[id_przegranego]['mecze_df'] += 1
    
    cechy_graczy = []
    MIN_MECZE = 20
    
    for gracz in gracze.values():
        if gracz['mecze'] < MIN_MECZE:
            continue
        
        wsp_wygranych = gracz['wygrane'] / gracz['mecze'] if gracz['mecze'] > 0 else 0
        sr_asy = gracz['asy'] / gracz['mecze_asy'] if gracz['mecze_asy'] > 0 else 0
        sr_df = gracz['df'] / gracz['mecze_df'] if gracz['mecze_df'] > 0 else 0
        
        twarde_wsp = gracz['twarde_wygrane'] / gracz['twarde_mecze'] if gracz['twarde_mecze'] > 0 else 0
        clay_wsp = gracz['clay_wygrane'] / gracz['clay_mecze'] if gracz['clay_mecze'] > 0 else 0
        trawa_wsp = gracz['trawa_wygrane'] / gracz['trawa_mecze'] if gracz['trawa_mecze'] > 0 else 0
        dywan_wsp = gracz['dywan_wygrane'] / gracz['dywan_mecze'] if gracz['dywan_mecze'] > 0 else 0
        
        cechy_graczy.append({
            'id': gracz['id'],
            'nazwa': gracz['nazwa'],
            'mecze': gracz['mecze'],
            'wsp_wygranych': wsp_wygranych,
            'sr_asy': sr_asy,
            'sr_df': sr_df,
            'wzrost': gracz['wzrost'] if pd.notna(gracz['wzrost']) else 0,
            'lewo_reczna': 1 if gracz['reka'] == 'L' else 0,
            'twarde_wsp': twarde_wsp,
            'clay_wsp': clay_wsp,
            'trawa_wsp': trawa_wsp,
            'dywan_wsp': dywan_wsp,
            'kraj': gracz['kraj']
        })
    
    df_graczy = pd.DataFrame(cechy_graczy)
    df_graczy = df_graczy.fillna(0)
    
    return df_graczy

def wybierz_cechy(df_graczy):
    cechy = [
        'wsp_wygranych', 'sr_asy', 'sr_df', 'wzrost', 'lewo_reczna',
        'twarde_wsp', 'clay_wsp', 'trawa_wsp', 'dywan_wsp'
    ]
    
    X = df_graczy[cechy].values
    
    skaler = StandardScaler()
    X_skalowane = skaler.fit_transform(X)
    
    return X_skalowane, cechy

def wykonaj_grupowanie(X, liczba_grup=4):
    kmeans = KMeans(n_clusters=liczba_grup, random_state=42)
    etykiety_kmeans = kmeans.fit_predict(X)
    
    hierarchiczny = AgglomerativeClustering(n_clusters=liczba_grup)
    etykiety_hierarchiczne = hierarchiczny.fit_predict(X)
    
    gmm = GaussianMixture(liczba_grup, random_state=42)
    etykiety_gmm = gmm.fit_predict(X)
    
    return {
        'kmeans': etykiety_kmeans,
        'hierarchiczny': etykiety_hierarchiczne,
        'gmm': etykiety_gmm
    }

def ocen_grupowanie(X, slownik_etykiet):
    wyniki = {}
    
    for nazwa, etykiety in slownik_etykiet.items():
        silhouette = silhouette_score(X, etykiety)
        calinski = calinski_harabasz_score(X, etykiety)
        davies = davies_bouldin_score(X, etykiety)
        
        wyniki[nazwa] = {
            'silhouette': silhouette,
            'calinski_harabasz': calinski,
            'davies_bouldin': davies
        }
    
    return wyniki

def wizualizuj_grupy_2d(X, slownik_etykiet, df_graczy):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (nazwa, etykiety) in enumerate(slownik_etykiet.items()):
        ax = axes[i]
        unikalne_etykiety = np.unique(etykiety)
        for etykieta in unikalne_etykiety:
            maska = etykiety == etykieta
            ax.scatter(X_pca[maska, 0], X_pca[maska, 1], 
                      c=kolory[etykieta % len(kolory)], 
                      label=f'Grupa {etykieta+1}',
                      alpha=0.7, s=50)
        
        if nazwa == 'kmeans':
            centroidy_pca = []
            for etykieta in unikalne_etykiety:
                maska = etykiety == etykieta
                centroid = X_pca[maska].mean(axis=0)
                centroidy_pca.append(centroid)
                ax.scatter(centroid[0], centroid[1], 
                          s=100, c='black', marker='x')
        
        if nazwa == 'kmeans':
            najlepsi_gracze = df_graczy.sort_values('mecze', ascending=False).head(5)
            for _, gracz in najlepsi_gracze.iterrows():
                idx = df_graczy[df_graczy['nazwa'] == gracz['nazwa']].index[0]
                ax.annotate(gracz['nazwa'], 
                           (X_pca[idx, 0], X_pca[idx, 1]),
                           fontsize=9, ha='right')
        
        ax.set_title(f'Grupowanie {nazwa.capitalize()}')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} wariancji)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} wariancji)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('laby3_wykresy/porownanie_grupowania_2d.png', dpi=300, bbox_inches='tight')
    plt.close()

def znajdz_optymalne_k(X, max_k=10):
    wsp_silhouette = []
    wsp_calinski = []
    wsp_davies = []
    
    for k in range(2, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        etykiety = kmeans.fit_predict(X)
        
        wsp_silhouette.append(silhouette_score(X, etykiety))
        wsp_calinski.append(calinski_harabasz_score(X, etykiety))
        wsp_davies.append(davies_bouldin_score(X, etykiety))
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(range(2, max_k+1), wsp_silhouette, 'o-')
    plt.xlabel('Liczba grup')
    plt.ylabel('Współczynnik Silhouette')
    plt.title('Współczynnik Silhouette vs. K')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(range(2, max_k+1), wsp_calinski, 'o-')
    plt.xlabel('Liczba grup')
    plt.ylabel('Współczynnik Calinski-Harabasz')
    plt.title('Współczynnik Calinski-Harabasz vs. K')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(range(2, max_k+1), wsp_davies, 'o-')
    plt.xlabel('Liczba grup')
    plt.ylabel('Współczynnik Davies-Bouldin')
    plt.title('Współczynnik Davies-Bouldin vs. K')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('laby3_wykresy/optymalne_k_metryki.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    optymalne_k = np.argmax(wsp_silhouette) + 2
    
    return optymalne_k, wsp_silhouette

def rozklad_wielkosci_grup(slownik_etykiet):
    rozklad = {}
    for nazwa, etykiety in slownik_etykiet.items():
        licznik = Counter(etykiety)
        rozklad[nazwa] = dict(sorted(licznik.items()))
    return rozklad

def oblicz_modulowosc(X, etykiety):
    liczba_grup = len(np.unique(etykiety))
    wewnetrzna_odl = 0
    for i in range(liczba_grup):
        maska = (etykiety == i)
        punkty_grupy = X[maska]
        if len(punkty_grupy) > 1:
            srodek_grupy = np.mean(punkty_grupy, axis=0)
            odl_punktow = np.sum((punkty_grupy - srodek_grupy) ** 2, axis=1)
            wewnetrzna_odl += np.sum(odl_punktow) / len(punkty_grupy)
    
    srodki_grup = []
    for i in range(liczba_grup):
        maska = (etykiety == i)
        punkty_grupy = X[maska]
        if len(punkty_grupy) > 0:
            srodki_grup.append(np.mean(punkty_grupy, axis=0))
    
    zewnetrzna_odl = 0
    liczba_par = 0
    for i in range(len(srodki_grup)):
        for j in range(i+1, len(srodki_grup)):
            zewnetrzna_odl += np.sum((srodki_grup[i] - srodki_grup[j]) ** 2)
            liczba_par += 1
    
    if liczba_par > 0:
        zewnetrzna_odl /= liczba_par
    
    modulowosc = 0
    if wewnetrzna_odl + zewnetrzna_odl > 0:
        modulowosc = (zewnetrzna_odl - wewnetrzna_odl) / (zewnetrzna_odl + wewnetrzna_odl)
    
    return modulowosc

def porownaj_zgodnosc_grup(slownik_etykiet):
    metody = list(slownik_etykiet.keys())
    liczba_metod = len(metody)
    
    macierz_nmi = np.zeros((liczba_metod, liczba_metod))
    macierz_ari = np.zeros((liczba_metod, liczba_metod))
    
    for i in range(liczba_metod):
        for j in range(liczba_metod):
            etykiety_i = slownik_etykiet[metody[i]]
            etykiety_j = slownik_etykiet[metody[j]]
            
            macierz_nmi[i, j] = normalized_mutual_info_score(etykiety_i, etykiety_j)
            macierz_ari[i, j] = adjusted_rand_score(etykiety_i, etykiety_j)
    
    return macierz_nmi, macierz_ari, metody

def wyswietl_listy_grup(df_graczy, slownik_etykiet):
    for nazwa_metody, etykiety in slownik_etykiet.items():
        print(f"\n=== LISTY GRUP - {nazwa_metody.upper()} ===")
        df_z_grupami = df_graczy.copy()
        df_z_grupami['grupa'] = etykiety
        
        liczba_grup = len(np.unique(etykiety))
        for i in range(liczba_grup):
            czlonkowie_grupy = df_z_grupami[df_z_grupami['grupa'] == i]
            najlepsi_czlonkowie = czlonkowie_grupy.sort_values('mecze', ascending=False)['nazwa'].head(10).tolist()
            
            print(f"Grupa {i+1} (Rozmiar: {len(czlonkowie_grupy)}): {', '.join(najlepsi_czlonkowie)}")

def main():
    df_graczy = przygotuj_dane_wta()
    print(f"Przetworzono {len(df_graczy)} graczy z wystarczającą liczbą meczów")
    X_skalowane, nazwy_cech = wybierz_cechy(df_graczy)
    
    optymalne_k, _ = znajdz_optymalne_k(X_skalowane)
    print(f"Optymalna liczba grup: {optymalne_k}")
    
    wyniki_grupowania = wykonaj_grupowanie(X_skalowane, liczba_grup=optymalne_k)
    wyswietl_listy_grup(df_graczy, wyniki_grupowania)
    
    rozklady = rozklad_wielkosci_grup(wyniki_grupowania)
    print("\n=== ROZKŁAD WIELKOŚCI GRUP ===")
    for metoda, rozklad in rozklady.items():
        print(f"{metoda}:")
        for grupa, rozmiar in rozklad.items():
            print(f"  Grupa {grupa+1}: {rozmiar} graczy")

    print("\n=== MODUŁOWOŚĆ ===")
    for metoda, etykiety in wyniki_grupowania.items():
        modulowosc = oblicz_modulowosc(X_skalowane, etykiety)
        print(f"{metoda}: {modulowosc:.4f}")
    
    macierz_nmi, macierz_ari, metody = porownaj_zgodnosc_grup(wyniki_grupowania)
    
    print("\n=== WSPÓŁCZYNNIKI GRUPOWANIA (NMI) ===")
    df_nmi = pd.DataFrame(macierz_nmi, index=metody, columns=metody)
    print(df_nmi)
    
    print("\n=== WSPÓŁCZYNNIKI GRUPOWANIA (ARI) ===")
    df_ari = pd.DataFrame(macierz_ari, index=metody, columns=metody)
    print(df_ari)

    metryki = ocen_grupowanie(X_skalowane, wyniki_grupowania)
    print("\n=== OCENA GRUPOWANIA ===")
    for nazwa, wartosci in metryki.items():
        print(f"{nazwa}:")
        print(f"Silhouette: {wartosci['silhouette']:.4f}")
        print(f"Calinski-Harabasz: {wartosci['calinski_harabasz']:.4f}")
        print(f"Davies-Bouldin: {wartosci['davies_bouldin']:.4f}")
    
if __name__ == "__main__":
    main()