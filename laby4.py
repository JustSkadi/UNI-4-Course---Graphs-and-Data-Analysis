import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import os

def load_data(file_path='wta_matches_combined.csv'):
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['winner_name', 'loser_name'])
    return df

def create_player_graph(df, min_matches=5):
    G = nx.DiGraph()
    player_counts = {}
    for _, row in df.iterrows():
        winner = row['winner_name']
        loser = row['loser_name']
        
        if winner not in player_counts:
            player_counts[winner] = 0
        if loser not in player_counts:
            player_counts[loser] = 0
            
        player_counts[winner] += 1
        player_counts[loser] += 1
    
    active_players = {player for player, count in player_counts.items() if count >= min_matches}
    print(f"Players with at least {min_matches} matches: {len(active_players)}")
    
    for player in active_players:
        G.add_node(player)
    
    edge_weights = {}
    
    for _, row in df.iterrows():
        winner = row['winner_name']
        loser = row['loser_name']
        
        if winner in active_players and loser in active_players:
            edge_key = (winner, loser)
            if edge_key not in edge_weights:
                edge_weights[edge_key] = 0
            edge_weights[edge_key] += 1
    
    for (winner, loser), weight in edge_weights.items():
        G.add_edge(winner, loser, weight=weight)
    
    print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G

def calculate_graph_metrics(G):
    metrics = {
        'degree': dict(G.degree()),
        'in_degree': dict(G.in_degree()),
        'out_degree': dict(G.out_degree())
    }
    
    largest_cc = max(nx.weakly_connected_components(G), key=len)
    subgraph = G.subgraph(largest_cc)
    
    print(f"Largest connected component has {subgraph.number_of_nodes()} nodes")
    
    metrics['betweenness'] = nx.betweenness_centrality(subgraph, k=100)
    metrics['pagerank'] = nx.pagerank(subgraph)
    
    return metrics

def draw_graph_spring_layout(G, metrics, title="Spring Layout Visualization", figsize=(12, 10), node_size_metric='degree', top_n=100):
    top_players = sorted(metrics[node_size_metric].items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_player_names = [player for player, _ in top_players]
    
    subgraph = G.subgraph(top_player_names)
    
    node_sizes = [metrics[node_size_metric][node] * 20 for node in subgraph.nodes()]
    
    edge_weights = [data['weight'] for _, _, data in subgraph.edges(data=True)]
    max_edge_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [0.5 + (3 * weight / max_edge_weight) for weight in edge_weights]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    pos = nx.spring_layout(subgraph, seed=42, k=0.15)
    
    nx.draw_networkx_edges(subgraph, pos, alpha=0.3, width=edge_widths, edge_color='gray', ax=ax)
    
    out_in_ratio = {}
    for node in subgraph.nodes():
        out_deg = subgraph.out_degree(node)
        in_deg = subgraph.in_degree(node) 
        out_in_ratio[node] = out_deg / max(in_deg, 1)
    
    node_colors = [out_in_ratio[node] for node in subgraph.nodes()]
    vmin, vmax = min(node_colors), max(node_colors)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.coolwarm
    
    nx.draw_networkx_nodes(subgraph, pos, node_size=node_sizes, 
                          node_color=node_colors, cmap=cmap, alpha=0.8, ax=ax)
    
    top_20_players = [player for player, _ in top_players[:20]]
    labels = {node: node for node in subgraph.nodes() if node in top_20_players}
    nx.draw_networkx_labels(subgraph, pos, labels=labels, font_size=8, font_weight='bold', ax=ax)
    
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='Wins/Losses Ratio')
    
    ax.set_title(f"{title}\nTop {top_n} players by {node_size_metric}")
    ax.axis('off')
    plt.tight_layout()
    
    return fig

def draw_graph_circular_layout(G, metrics, title="Circular Layout Visualization", figsize=(14, 14), node_size_metric='pagerank', top_n=50):
    top_players = sorted(metrics[node_size_metric].items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_player_names = [player for player, _ in top_players]
    
    subgraph = G.subgraph(top_player_names)
    
    max_metric = max(metrics[node_size_metric].values())
    node_sizes = [metrics[node_size_metric][node] * 3000 / max_metric for node in subgraph.nodes()]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    pos = nx.circular_layout(subgraph)
    
    nx.draw_networkx_edges(subgraph, pos, alpha=0.2, edge_color='gray', ax=ax)
    
    node_colors = [metrics[node_size_metric][node] for node in subgraph.nodes()]
    
    nx.draw_networkx_nodes(subgraph, pos, node_size=node_sizes, 
                          node_color=node_colors, cmap='viridis', alpha=0.8, ax=ax)
    
    labels = {node: node for node in subgraph.nodes()}
    nx.draw_networkx_labels(subgraph, pos, labels=labels, font_size=8, ax=ax)
    
    ax.set_title(f"{title}\nTop {top_n} players by {node_size_metric}")
    ax.axis('off')
    plt.tight_layout()
    
    return fig

def deepwalk_embedding(G, dimensions=64, walk_length=10, num_walks=80, window=5):
    walks = []
    nodes = list(G.nodes())
    
    for _ in tqdm(range(num_walks)):
        np.random.shuffle(nodes)
        for node in nodes:
            walk = [node]
            for _ in range(walk_length - 1):
                current = walk[-1]
                neighbors = list(G.neighbors(current))
                if not neighbors:
                    break
                walk.append(np.random.choice(neighbors))
            walks.append(walk)
    
    str_walks = [[str(n) for n in walk] for walk in walks]
    
    try:
        from gensim.models import Word2Vec
        model = Word2Vec(str_walks, vector_size=dimensions, window=window, min_count=0, sg=1, workers=4)
        
        embeddings = {node: model.wv[str(node)] for node in G.nodes() if str(node) in model.wv}
        
    except ImportError:
        node_to_index = {node: i for i, node in enumerate(G.nodes())}
        n_nodes = len(node_to_index)
        
        cooc_matrix = np.zeros((n_nodes, n_nodes))
        
        for walk in walks:
            for i in range(len(walk)):
                for j in range(max(0, i-window), min(len(walk), i+window+1)):
                    if i != j:
                        source_idx = node_to_index[walk[i]]
                        target_idx = node_to_index[walk[j]]
                        cooc_matrix[source_idx, target_idx] += 1
        
        u, _, _ = np.linalg.svd(cooc_matrix, full_matrices=False)
        embeddings = {node: u[node_to_index[node], :dimensions] for node in G.nodes()}
    
    return embeddings

def visualize_embeddings(embeddings, metrics, title="Graph Embeddings Visualization", figsize=(12, 10)):
    nodes = list(embeddings.keys())
    embedding_matrix = np.array([embeddings[node] for node in nodes])
    tsne = TSNE(n_components=2, random_state=42)
    node_pos = tsne.fit_transform(embedding_matrix)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    pageranks = [metrics['pagerank'][node] for node in nodes]
    
    degrees = [metrics['degree'][node] for node in nodes]
    
    scaler = MinMaxScaler(feature_range=(20, 500))
    node_sizes = scaler.fit_transform(np.array(degrees).reshape(-1, 1)).flatten()
    
    scatter = ax.scatter(node_pos[:, 0], node_pos[:, 1], 
                         c=pageranks, cmap='viridis', 
                         s=node_sizes, alpha=0.7)
    
    top_indices = np.argsort(pageranks)[-20:]
    for i in top_indices:
        ax.annotate(nodes[i], (node_pos[i, 0], node_pos[i, 1]), fontsize=8)
    
    fig.colorbar(scatter, ax=ax, label='PageRank')
    ax.set_title(f"{title}\nColored by PageRank, sized by Degree")
    plt.tight_layout()
    
    return fig

def analyze_by_surface(df, G, min_matches=5):
    surfaces = df['surface'].unique()
    
    surface_graphs = {}
    for surface in surfaces:
        surface_df = df[df['surface'] == surface]
        print(f"\nAnalyzing {surface} surface matches ({len(surface_df)} matches)")
        
        surface_graphs[surface] = create_player_graph(surface_df, min_matches=min_matches)
    
    return surface_graphs

def main():
    output_dir = "laby4_wykresy"
    os.makedirs(output_dir, exist_ok=True)
    
    df = load_data()
    G = create_player_graph(df, min_matches=10)
    metrics = calculate_graph_metrics(G)
    embeddings = deepwalk_embedding(G, dimensions=64)
    
    spring_fig = draw_graph_spring_layout(G, metrics, 
                                        title="WTA Player Network - Force-Directed Layout",
                                        node_size_metric='degree')
    spring_fig.savefig(os.path.join(output_dir, 'wta_network_spring_layout.png'), dpi=300)
    
    circular_fig = draw_graph_circular_layout(G, metrics, 
                                           title="WTA Player Network - Circular Layout",
                                           node_size_metric='pagerank')
    circular_fig.savefig(os.path.join(output_dir, 'wta_network_circular_layout.png'), dpi=300)
    
    embedding_fig = visualize_embeddings(embeddings, metrics, 
                                      title="WTA Player Network - Graph Embedding")
    embedding_fig.savefig(os.path.join(output_dir, 'wta_network_embedding.png'), dpi=300)
    
    surface_graphs = analyze_by_surface(df, G, min_matches=5)
    if 'Hard' in surface_graphs:
        hard_graph = surface_graphs['Hard']
        hard_metrics = calculate_graph_metrics(hard_graph)
        hard_fig = draw_graph_spring_layout(hard_graph, hard_metrics, 
                                         title="WTA Player Network on Hard Courts",
                                         node_size_metric='pagerank')
        hard_fig.savefig(os.path.join(output_dir, 'wta_network_hard_courts.png'), dpi=300)
    
    summary = {
        "total_players": G.number_of_nodes(),
        "total_matches": G.number_of_edges(),
        "top_by_pagerank": sorted(metrics['pagerank'].items(), key=lambda x: x[1], reverse=True)[:10],
        "top_by_degree": sorted(metrics['degree'].items(), key=lambda x: x[1], reverse=True)[:10]
    }
    
    return summary

if __name__ == "__main__":
    summary = main()
    print(f"\nTotal players: {summary['total_players']}")
    print(f"Total matches: {summary['total_matches']}")
    
    print("\nTop 10 players by PageRank:")
    for i, (player, score) in enumerate(summary['top_by_pagerank'], 1):
        print(f"{i}. {player}: {score:.4f}")
    
    print("\nTop 10 players by degree (total matches):")
    for i, (player, degree) in enumerate(summary['top_by_degree'], 1):
        print(f"{i}. {player}: {degree}")