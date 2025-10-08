import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import Counter
import os

def load_data(filename):
    df = pd.read_csv(filename)
    df = df[['winner_name', 'loser_name']]
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

def plot_degree_distribution(G):
    degrees = [G.degree(n) for n in G.nodes()]
    degree_counts = Counter(degrees)
    
    x = list(degree_counts.keys())
    y = list(degree_counts.values())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, y, color='skyblue')
    ax.set_title('Degree Distribution')
    ax.set_xlabel('Degree')
    ax.set_ylabel('Count')
    
    axins = fig.add_axes([0.55, 0.55, 0.35, 0.35])
    axins.loglog(x, y, 'r.', markersize=5)
    axins.set_title('Log-Log Scale')
    axins.set_xlabel('Degree (log)')
    axins.set_ylabel('Count (log)')
    
    plt.tight_layout()
    return fig

def network_density(G):
    n = G.number_of_nodes()
    m = G.number_of_edges()
    possible_edges = n * (n - 1) / 2
    density = m / possible_edges
    return density

def network_diameter(G):
    try:
        return nx.diameter(G)
    except nx.NetworkXError:
        largest_cc = max(nx.connected_components(G), key=len)
        largest_component = G.subgraph(largest_cc)
        return nx.diameter(largest_component)

def average_path_length(G):
    try:
        return nx.average_shortest_path_length(G)
    except nx.NetworkXError:
        largest_cc = max(nx.connected_components(G), key=len)
        largest_component = G.subgraph(largest_cc)
        return nx.average_shortest_path_length(largest_component)

def shortest_path(G, source, target):
    try:
        path = nx.shortest_path(G, source=source, target=target)
        length = nx.shortest_path_length(G, source=source, target=target)
        return path, length
    except nx.NetworkXNoPath:
        return None, float('inf')
    except nx.NodeNotFound:
        return None, -1

def path_length_distribution(G):
    path_lengths = []
    
    largest_cc = max(nx.connected_components(G), key=len)
    largest_component = G.subgraph(largest_cc)
    
    sample_size = min(100, len(largest_component))
    sampled_nodes = np.random.choice(list(largest_component.nodes()), size=sample_size, replace=False)
    
    for i, source in enumerate(sampled_nodes):
        for target in list(sampled_nodes)[i+1:]:
            try:
                length = nx.shortest_path_length(largest_component, source=source, target=target)
                path_lengths.append(length)
            except:
                continue
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(path_lengths, kde=True, ax=ax)
    ax.set_title('Path Length Distribution')
    ax.set_xlabel('Path Length')
    ax.set_ylabel('Frequency')
    
    return fig, np.mean(path_lengths) if path_lengths else None

def centrality_measures(G):
    degree_cent = nx.degree_centrality(G)
    
    between_cent = nx.betweenness_centrality(G)
    
    close_cent = nx.closeness_centrality(G)
    
    edge_between = nx.edge_betweenness_centrality(G)
    
    try:
        eigen_cent = nx.eigenvector_centrality(G, max_iter=1000)
    except:
        eigen_cent = {}
    
    return {
        'degree': degree_cent,
        'betweenness': between_cent,
        'closeness': close_cent,
        'edge_betweenness': edge_between,
        'eigenvector': eigen_cent
    }

def plot_centrality_distributions(centrality_dict):
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    
    sns.histplot(list(centrality_dict['degree'].values()), kde=True, ax=axes[0, 0])
    axes[0, 0].set_title('Degree Centrality Distribution')
    
    sns.histplot(list(centrality_dict['betweenness'].values()), kde=True, ax=axes[0, 1])
    axes[0, 1].set_title('Betweenness Centrality Distribution')
    
    sns.histplot(list(centrality_dict['closeness'].values()), kde=True, ax=axes[1, 0])
    axes[1, 0].set_title('Closeness Centrality Distribution')
    
    if centrality_dict['eigenvector']:
        sns.histplot(list(centrality_dict['eigenvector'].values()), kde=True, ax=axes[1, 1])
        axes[1, 1].set_title('Eigenvector Centrality Distribution')
    else:
        axes[1, 1].text(0.5, 0.5, 'Eigenvector centrality computation failed', 
                        horizontalalignment='center', verticalalignment='center')
    
    sns.histplot(list(centrality_dict['edge_betweenness'].values()), kde=True, ax=axes[2, 0])
    axes[2, 0].set_title('Edge Betweenness Distribution')
    
    axes[2, 1].axis('off')
    
    plt.tight_layout()
    return fig

def calculate_pagerank(G):
    pagerank = nx.pagerank(G)
    
    top_players = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:20]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    names = [player[0] for player in top_players]
    values = [player[1] for player in top_players]
    
    y_pos = np.arange(len(names))
    ax.barh(y_pos, values, color='green')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_title('Top 20 Players by PageRank')
    ax.set_xlabel('PageRank Value')
    
    return fig, pagerank

def connected_components_analysis(G):
    components = list(nx.connected_components(G))
    component_sizes = [len(c) for c in components]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sorted_sizes = sorted(component_sizes, reverse=True)
    ax.bar(range(1, len(sorted_sizes) + 1), sorted_sizes, color='purple')
    ax.set_title('Connected Component Sizes')
    ax.set_xlabel('Component Rank')
    ax.set_ylabel('Size')
    ax.set_yscale('log')
    
    return fig, {
        'num_components': len(components),
        'largest_component_size': max(component_sizes),
        'smallest_component_size': min(component_sizes),
        'average_component_size': sum(component_sizes) / len(components) if components else 0
    }

def k_connectivity(G):
    largest_cc = max(nx.connected_components(G), key=len)
    largest_component = G.subgraph(largest_cc)
    
    if len(largest_component) > 100:
        node_connectivity = nx.approximation.node_connectivity(largest_component)
        edge_connectivity = nx.edge_connectivity(largest_component)
    else:
        node_connectivity = nx.node_connectivity(largest_component)
        edge_connectivity = nx.edge_connectivity(largest_component)
    
    return {
        'node_connectivity': node_connectivity,
        'edge_connectivity': edge_connectivity
    }

def node_categories(G):
    articulation_points = list(nx.articulation_points(G))
    
    bridges = list(nx.bridges(G))
    
    degrees = dict(G.degree())
    threshold = np.percentile(list(degrees.values()), 95)
    hubs = [node for node, degree in degrees.items() if degree > threshold]
    
    return {
        'articulation_points': articulation_points,
        'bridges': bridges,
        'hubs': hubs
    }

def find_cliques(G):
    maximal_cliques = list(nx.find_cliques(G))
    
    maximal_cliques = [c for c in maximal_cliques if len(c) >= 3]
    
    if maximal_cliques:
        largest_clique_size = max(len(c) for c in maximal_cliques)
    else:
        largest_clique_size = 0
    
    clique_sizes = [len(c) for c in maximal_cliques]
    clique_counts = Counter(clique_sizes)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sizes = list(clique_counts.keys())
    counts = list(clique_counts.values())
    ax.bar(sizes, counts, color='orange')
    ax.set_title('Clique Size Distribution')
    ax.set_xlabel('Clique Size')
    ax.set_ylabel('Count')
    
    return fig, {
        'num_maximal_cliques': len(maximal_cliques),
        'largest_clique_size': largest_clique_size,
        'clique_counts': dict(clique_counts),
        'example_cliques': maximal_cliques[:5] if maximal_cliques else []
    }

def main():
    df = load_data("wta_matches_combined.csv")
    G = create_graph(df)
    
    print("\n--- Basic Network Properties ---")
    density = network_density(G)
    diameter = network_diameter(G)
    avg_path = average_path_length(G)
    print(f"Network Density: {density:.6f}")
    print(f"Network Diameter: {diameter}")
    print(f"Average Path Length: {avg_path:.4f}")
    
    output_dir = "laby2_wykresy"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.figure(1)
    plot_degree_distribution(G)
    plt.savefig(os.path.join(output_dir, "degree_distribution.png"))
    
    centrality = centrality_measures(G)
    plt.figure(2)
    plot_centrality_distributions(centrality)
    plt.savefig(os.path.join(output_dir, "centrality_distributions.png"))
    
    plt.figure(3)
    pagerank_fig, pagerank = calculate_pagerank(G)
    plt.savefig(os.path.join(output_dir, "pagerank.png"))
    
    print("\n--- Top 10 Players by PageRank ---")
    top_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
    for i, (player, score) in enumerate(top_pagerank, 1):
        print(f"{i}. {player}: {score:.6f}")
    
    plt.figure(4)
    cc_fig, cc_stats = connected_components_analysis(G)
    plt.savefig(os.path.join(output_dir, "connected_components.png"))
    
    print(f"Number of connected components: {cc_stats['num_components']}")
    print(f"Largest component size: {cc_stats['largest_component_size']}")
    print(f"Smallest component size: {cc_stats['smallest_component_size']}")
    print(f"Average component size: {cc_stats['average_component_size']:.2f}")
    
    k_conn = k_connectivity(G)
    print(f"Node connectivity: {k_conn['node_connectivity']}")
    print(f"Edge connectivity: {k_conn['edge_connectivity']}")
    
    categories = node_categories(G)
    print(f"Number of articulation points: {len(categories['articulation_points'])}")
    print(f"Number of bridges: {len(categories['bridges'])}")
    print(f"Number of hubs: {len(categories['hubs'])}")
    
    print("\n--- Top 5 Hubs (high-degree nodes) ---")
    hub_degrees = [(hub, G.degree(hub)) for hub in categories['hubs'][:5]]
    for i, (hub, degree) in enumerate(sorted(hub_degrees, key=lambda x: x[1], reverse=True)[:5], 1):
        print(f"{i}. {hub}: {degree} connections")
    
    plt.figure(5)
    clique_fig, clique_stats = find_cliques(G)
    plt.savefig(os.path.join(output_dir, "clique_distribution.png"))
    
    print(f"Number of maximal cliques: {clique_stats['num_maximal_cliques']}")
    print(f"Largest clique size: {clique_stats['largest_clique_size']}")
    
    path_fig, avg_path_length = path_length_distribution(G)
    plt.savefig(os.path.join(output_dir, "path_length_distribution.png"))
    if avg_path_length:
        print(f"Average path length from sampling: {avg_path_length:.4f}")
    
    if len(G.nodes()) >= 2:
        sample_players = list(G.nodes())[:2]
        path, length = shortest_path(G, sample_players[0], sample_players[1])
        print(f"Shortest path from {sample_players[0]} to {sample_players[1]}: {path}")
        print(f"Path length: {length}")

if __name__ == "__main__":
    main()