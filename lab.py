import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import Counter
import os

def dane(file):
    df = pd.read_csv(file)
    df = df[['winner_name', 'loser_name']]
    return df

def graf(df):
    G = nx.Graph()
    players = set(df['winner_name'].unique()).union(set(df['loser_name'].unique()))
    G.add_nodes_from(players)
    wagi={}
    for _, row in df.iterrows():
        winner = row['winner_name']
        loser = row['loser_name']
        pair = tuple(sorted([winner, loser]))
        if pair in wagi:
            wagi[pair] += 1
        else:
            wagi[pair] = 1

    for (player1, player2), count in wagi.items():
        G.add_edge(player1, player2, weight=count, label=str(count))
    
    return G

def rozklad_stopni(G):
    """
    Rozkład stopni wierzchołka - ile zawodniczek ma określoną liczbę 
    przeciwniczek. 
    W danych WTA pozwala zidentyfikować, które tenisistki grały 
    przeciwko największej liczbie różnych rywalek. 
    """

    stopnie = [G.degree(n) for n in G.nodes()]
    c_stopnie = Counter(stopnie)

    wartosci = list(c_stopnie.keys()) # unikalne wartości stopni
    wezly = list(c_stopnie.values()) # liczba węzłów dla każdego stopnia

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(wartosci, wezly, color='blue')
    ax.set_title("Rozkład stopni wierzchołków")
    ax.set_xlabel("Stopnie")
    ax.set_ylabel("Wartości stopni")

    plt.tight_layout()
    return fig

def main():
    df = dane("wta_matches_combined.csv")
    G = graf(df)

    folder = 'laby2_wykresy'
    if not os.path.exists(folder):
        os.makedirs(folder)

    plt.figure(1)
    rozklad_stopni(G)
    plt.savefig(os.path.join(folder, 'rozklad_stopni.png'))


if __name__ == "__main__":
    main()