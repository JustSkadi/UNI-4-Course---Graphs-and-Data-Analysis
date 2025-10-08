import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

def load_data(filename):
    df = pd.read_csv(filename)
    return df

def create_graph(df):
    G = nx.Graph()
    all_players = set(df['winner_name'].unique()).union(set(df['loser_name'].unique()))
    G.add_nodes_from(all_players)
    m_count = {}
    for _, row in df.iterrows():
        winner = row['winner_name']
        loser = row['loser_name']
        pair = tuple(sorted([winner, loser]))
        if pair in m_count:
            m_count[pair] += 1
        else:
            m_count[pair] = 1
    
    for (player1, player2), count in m_count.items():
        G.add_edge(player1, player2, weight=count, label=str(count))

    return G

def display_graph_info(G):
    print("\n=== INFORMACJE O GRAFIE ===")
    print(f"Rodzaj grafu: Graf nieukierunkowany, ważony")
    print(f"Rząd grafu (liczba wierzchołków): {G.number_of_nodes()}")
    print(f"Rozmiar grafu (liczba krawędzi): {G.number_of_edges()}")
    print(f"Czy graf jest spójny: {nx.is_connected(G)}")
    
    if not nx.is_connected(G):
        comp = list(nx.connected_components(G))
        print(f"Liczba składowych spójnych: {len(comp)}")
        largest_cc = max(comp, key=len)
        print(f"Rozmiar max składowej spójnej: {len(largest_cc)}")

def visualize_graph(G, output_filename="wta_graph.png", top_players=100):
    
    player_match_count = {}
    for player in G.nodes():
        player_match_count[player] = sum(G[player][neighbor]['weight'] for neighbor in G[player])
    
    # Dobieramy top zawodniczki żeby było lepiej widoczne ( cokolwiek widoczne xD )
    top_players_list = sorted(player_match_count.items(), key=lambda x: x[1], reverse=True)[:top_players]
    top_players_set = set(player for player, _ in top_players_list)
    
    G_viz = G.subgraph(top_players_set)
    # G_viz = G # podmianka na pełny graf w razie czego
    
    plt.figure(figsize=(20, 20))
    plt.title("Graf meczów WTA - waga krawędzi to liczba meczów", fontsize=20)
    
    pos = nx.spring_layout(G_viz, k=0.3, iterations=50, seed=42)
    degree_centrality = nx.degree_centrality(G_viz)
    node_sizes = [degree_centrality[node] * 10000 + 10 for node in G_viz.nodes()]
    
    nodes = nx.draw_networkx_nodes(G_viz, pos, 
                                  node_size=node_sizes,
                                  node_color=list(degree_centrality.values()),
                                  cmap=plt.cm.viridis,
                                  alpha=0.8)
    
    plt.colorbar(nodes, label="Stopień centralności")
    
    edge_weights = [G_viz[u][v]['weight'] for u, v in G_viz.edges()]
    max_weight = max(edge_weights)
    edge_widths = [0.1 + (w / max_weight) * 2 for w in edge_weights]
    
    nx.draw_networkx_edges(G_viz, pos, 
                          width=edge_widths,
                          alpha=0.3,
                          edge_color='gray')
    
    labels = {player: player for player in G_viz.nodes() if player in top_players_set}
    nx.draw_networkx_labels(G_viz, pos, labels=labels, font_size=8)
    
    legend_elements = [
        Line2D([0], [0], color='gray', lw=1, label='1 mecz'),
        Line2D([0], [0], color='gray', lw=2, label='5 meczów'),
        Line2D([0], [0], color='gray', lw=3, label='10+ meczów'),
        Patch(facecolor='lightblue', edgecolor='black', label='Mniejsza centralność'),
        Patch(facecolor='darkblue', edgecolor='black', label='Większa centralność')
    ]
    
    plt.legend(handles=legend_elements, loc='upper right')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()

def calculate_node_centrality(G, top_n=20):
    print("\n=== MIARY CENTRALNOŚCI DLA WIERZCHOŁKÓW ===")
    degree_centrality = nx.degree_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G, k=100)  # używamy k=100 dla przybliżenia
    
    centrality_df = pd.DataFrame({
        'Zawodniczka': list(G.nodes()),
        'Stopień (Degree)': [degree_centrality[node] for node in G.nodes()],
        'Bliskość (Closeness)': [closeness_centrality[node] for node in G.nodes()],
        'Pośrednictwo (Betweenness)': [betweenness_centrality[node] for node in G.nodes()]
    })
    
    print(f"\nTop {top_n} zawodniczek według stopnia centralności (Degree Centrality):")
    degree_top = centrality_df.sort_values('Stopień (Degree)', ascending=False).head(top_n)
    print(degree_top[['Zawodniczka', 'Stopień (Degree)']].to_string(index=False))
    
    print(f"\nTop {top_n} zawodniczek według bliskości (Closeness Centrality):")
    closeness_top = centrality_df.sort_values('Bliskość (Closeness)', ascending=False).head(top_n)
    print(closeness_top[['Zawodniczka', 'Bliskość (Closeness)']].to_string(index=False))
    
    print(f"\nTop {top_n} zawodniczek według pośrednictwa (Betweenness Centrality):")
    betweenness_top = centrality_df.sort_values('Pośrednictwo (Betweenness)', ascending=False).head(top_n)
    print(betweenness_top[['Zawodniczka', 'Pośrednictwo (Betweenness)']].to_string(index=False))
    
    plt.figure(figsize=(15, 10))
    plt.subplot(3, 1, 1)
    sns.barplot(x='Stopień (Degree)', y='Zawodniczka', hue='Zawodniczka', data=degree_top.head(10), palette='viridis', legend=False)
    plt.title('Top 10 zawodniczek według stopnia centralności')

    plt.subplot(3, 1, 2)
    sns.barplot(x='Bliskość (Closeness)', y='Zawodniczka', hue='Zawodniczka', data=closeness_top.head(10), palette='viridis', legend=False)
    plt.title('Top 10 zawodniczek według bliskości')

    plt.subplot(3, 1, 3)
    sns.barplot(x='Pośrednictwo (Betweenness)', y='Zawodniczka', hue='Zawodniczka', data=betweenness_top.head(10), palette='viridis', legend=False)
    plt.title('Top 10 zawodniczek według pośrednictwa')
    
    plt.tight_layout()
    plt.savefig('centralnosc_wierzcholkow.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Zapisano wizualizację miar centralności do pliku centralnosc_wierzcholkow.png")
    
    return centrality_df

def calculate_edge_centrality(G, top_n=20):
    print("\n=== MIARY CENTRALNOŚCI DLA KRAWĘDZI ===")
    edge_betweenness = nx.edge_betweenness_centrality(G, k=100)  # przybliżenie
    
    edge_data = []
    for (u, v), centrality in edge_betweenness.items():
        weight = G[u][v]['weight']
        edge_data.append((u, v, weight, centrality))
    
    edge_df = pd.DataFrame(edge_data, columns=['Zawodniczka 1', 'Zawodniczka 2', 'Liczba meczów', 'Pośrednictwo (Betweenness)'])
    
    print(f"\nTop {top_n} krawędzi według pośrednictwa (Betweenness Centrality):")
    edge_top = edge_df.sort_values('Pośrednictwo (Betweenness)', ascending=False).head(top_n)
    print(edge_top.to_string(index=False))
    
    plt.figure(figsize=(15, 8))
    top_10_edges = edge_top.head(10)
    edge_labels = [f"{row['Zawodniczka 1']} vs {row['Zawodniczka 2']}" for _, row in top_10_edges.iterrows()]
    plot_df = top_10_edges.copy()
    plot_df['Para'] = edge_labels
    
    sns.barplot(x='Pośrednictwo (Betweenness)', y='Para', hue='Para', data=plot_df, palette='viridis', legend=False)
    plt.title('Top 10 krawędzi według pośrednictwa')
    plt.tight_layout()
    plt.savefig('centralnosc_krawedzi.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return edge_df

def generate_matrices(G, max_nodes=20):
    print("\n=== MACIERZE GRAFU ===")
    if G.number_of_nodes() > max_nodes:
        print(f"Graf jest zbyt duży do wizualizacji macierzy. Wybieram podgraf z top {max_nodes} zawodniczek.")

        degree = dict(G.degree(weight='weight'))
        top_nodes = sorted(degree.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
        top_nodes = [node for node, _ in top_nodes]
        
        subgraph = G.subgraph(top_nodes)
        print(f"Utworzono podgraf z {subgraph.number_of_nodes()} wierzchołkami i {subgraph.number_of_edges()} krawędziami.")
        G_matrix = subgraph
    else:
        G_matrix = G
    
    A = nx.adjacency_matrix(G_matrix, weight='weight').todense()
    
    nodes_list = list(G_matrix.nodes())
    adj_df = pd.DataFrame(A, index=nodes_list, columns=nodes_list)
    
    plt.figure(figsize=(12, 10))
    plt.title("Macierz sąsiedztwa", fontsize=16)
    sns.heatmap(adj_df, cmap="viridis", annot=True, fmt=".0f", linewidths=.5)
    plt.tight_layout()
    plt.savefig('macierz_sasiedztwa.png', dpi=300, bbox_inches='tight')
    plt.close()

    edges = list(G_matrix.edges())
    inc_matrix = np.zeros((len(nodes_list), len(edges)))

    for i, edge in enumerate(edges):
        source_idx = nodes_list.index(edge[0])
        target_idx = nodes_list.index(edge[1])
        weight = G_matrix[edge[0]][edge[1]]['weight']
        
        inc_matrix[source_idx, i] = weight
        inc_matrix[target_idx, i] = weight
    
    edge_labels = [f"{u}_{v}" for u, v in edges]
    inc_df = pd.DataFrame(inc_matrix, index=nodes_list, columns=edge_labels)
    
    plt.figure(figsize=(15, 10))
    plt.title("Macierz incydencji (uproszczona)", fontsize=16)
    sns.heatmap(inc_df, cmap="viridis", annot=True, fmt=".0f", linewidths=.5)
    plt.tight_layout()
    plt.savefig('macierz_incydencji.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return adj_df, inc_df

def main(filename='wta_matches_combined.csv'):
    df = load_data(filename)
    G = create_graph(df)
    display_graph_info(G)
    visualize_graph(G, output_filename="wta_graph_full.png", top_players=50)
    node_centrality = calculate_node_centrality(G)
    edge_centrality = calculate_edge_centrality(G)
    adj_matrix, inc_matrix = generate_matrices(G, max_nodes=20)

if __name__ == "__main__":
    main()