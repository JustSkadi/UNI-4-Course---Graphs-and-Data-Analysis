import os
import re
import glob
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

output_dir = 'laby5_wykresy'
os.makedirs(output_dir, exist_ok=True)

def load_data(folder_path):
    """
    Wczytuje dane z plików i grupuje je według wielkości okna i przesunięcia.
    
    Parameters:
    folder_path (str): Ścieżka do folderu z plikami
    
    Returns:
    dict: Słownik z danymi pogrupowanymi według (wielkość_okna, przesunięcie)
    """
    data_groups = {}
    for file_path in glob.glob(os.path.join(folder_path, "*.txt")):
        filename = os.path.basename(file_path)
        match = re.match(r'o(\d+)p(\d+)_(\d+)\.txt', filename)
        if match:
            window_size, shift, number = map(int, match.groups())
            key = (window_size, shift)
            if key not in data_groups:
                data_groups[key] = []
            
            edges = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip() and ';' in line:
                        parts = line.strip().split(';')
                        source = int(parts[0])
                        target = int(parts[1])
                        weight = float(parts[2].replace(',', '.'))
                        edges.append((source, target, weight))
            
            data_groups[key].append({
                'file': filename,
                'edges': edges,
                'number': number
            })
    
    # Sortowanie danych według numeru w nazwie pliku
    for key in data_groups:
        data_groups[key].sort(key=lambda x: x['number'])
    
    return data_groups

def predict_and_evaluate(G, method='jaccard'):
    """
    Wykonuje predykcję połączeń wybraną metodą i ocenia jej skuteczność.
    
    Parameters:
    G (nx.Graph): Graf do analizy
    method (str): Metoda predykcji (jaccard, adamic_adar lub preferential_attachment)
    
    Returns:
    tuple: (metrics, pred_data) - metryki oceny predykcji i dane do wizualizacji
    """
    edges = list(G.edges())

    edge_train, edge_test = train_test_split(edges, test_size=0.2, random_state=42)
    G_train = G.copy()
    G_train.remove_edges_from(edge_test)
    
    # Predykcja
    if method == 'jaccard':
        preds = nx.jaccard_coefficient(G_train.to_undirected())
    elif method == 'adamic_adar':
        preds = nx.adamic_adar_index(G_train.to_undirected())
    elif method == 'preferential_attachment':
        preds = nx.preferential_attachment(G_train.to_undirected())
    else:
        return None, None
    
    scores = {}
    for u, v, p in preds:
        scores[(u, v)] = p
    
    y_true = []
    y_pred = []
    
    # Ocena predykcji
    for u, v in edge_test:
        score = max(scores.get((u, v), 0), scores.get((v, u), 0))
        y_true.append(1)
        y_pred.append(score)
    
    # Dodanie przykładów negatywnych
    non_edges = list(nx.non_edges(G_train))[:len(edge_test)]
    for u, v in non_edges:
        score = max(scores.get((u, v), 0), scores.get((v, u), 0))
        y_true.append(0)
        y_pred.append(score)
    
    # Normalizacja
    if len(y_pred) > 1:
        y_pred_norm = np.array(y_pred)
        if np.max(y_pred_norm) != np.min(y_pred_norm):
            y_pred_norm = (y_pred_norm - np.min(y_pred_norm)) / (np.max(y_pred_norm) - np.min(y_pred_norm))
        
        # Metryki
        roc_auc = roc_auc_score(y_true, y_pred_norm)
        threshold = np.median(y_pred_norm)
        y_binary = [1 if s >= threshold else 0 for s in y_pred_norm]
        precision = precision_score(y_true, y_binary, zero_division=0)
        recall = recall_score(y_true, y_binary, zero_division=0)
        
        return {
            'roc_auc': roc_auc,
            'precision': precision,
            'recall': recall
        }, (G_train, edge_test, scores)
    
    return None, None

def analyze_weekly_patterns(data_groups):
    """
    Analizuje wzorce komunikacji w ciągu tygodnia.
    
    Parameters:
    data_groups (dict): Słownik z danymi pogrupowanymi według (wielkość_okna, przesunięcie)
    
    Returns:
    list or None: Średnia liczba połączeń dla każdego dnia tygodnia lub None jeśli brak danych
    """
    if (1, 1) in data_groups and len(data_groups[(1, 1)]) >= 7:
        daily_data = data_groups[(1, 1)]
        days_of_week = ['Poniedziałek', 'Wtorek', 'Środa', 'Czwartek', 'Piątek', 'Sobota', 'Niedziela']
        weekly_pattern = [[] for _ in range(7)]

        for i, data in enumerate(daily_data):
            day_of_week = i % 7
            edge_count = len(data['edges'])
            weekly_pattern[day_of_week].append(edge_count)
        
        # Obliczanie średniej liczby połączeń dla każdego dnia tygodnia
        avg_connections = [np.mean(day_data) if day_data else 0 for day_data in weekly_pattern]
        
        # Wizualizacja wzorca tygodniowego
        plt.figure(figsize=(12, 6))
        bars = plt.bar(days_of_week, avg_connections, color='skyblue')
        
        # Dodanie etykiet z wartościami na słupkach
        for bar, value in zip(bars, avg_connections):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                     f'{value:.1f}', ha='center', fontsize=10)
        
        plt.xlabel('Dzień tygodnia')
        plt.ylabel('Średnia liczba połączeń')
        plt.title('Tygodniowy wzorzec komunikacji mailowej')
        plt.grid(axis='y', alpha=0.3)
        
        plt.axvspan(4.5, 6.5, color='lightgray', alpha=0.3, label='Weekend')
        plt.legend()  # Dodanie legendy
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'tygodniowy_wzorzec.jpg'))
        plt.close()
        
        return avg_connections
    
    return None

def analyze_temporal_trends(data_groups):
    """
    Analizuje trendy czasowe dla różnych wielkości okien.
    
    Parameters:
    data_groups (dict): Słownik z danymi pogrupowanymi według (wielkość_okna, przesunięcie)
    """
    plt.figure(figsize=(15, 8))
    
    # Analizuj trendy dla różnych parametrów okien
    for (window_size, shift), group_data in data_groups.items():
        if len(group_data) >= 5:  # Tylko jeśli mamy wystarczająco danych
            indices = list(range(len(group_data)))
            edge_counts = [len(data['edges']) for data in group_data]
            
            plt.plot(indices, edge_counts, 'o-', linewidth=2, 
                     label=f'Okno {window_size} dni, przesunięcie {shift} dni')
    
    plt.xlabel('Indeks okna czasowego')
    plt.ylabel('Liczba połączeń')
    plt.title('Trendy czasowe liczby połączeń dla różnych parametrów okien')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'trendy_czasowe.jpg'))
    plt.close()

def visualize_graph(G, window_info, filename):
    """
    Wizualizuje graf sieci komunikacji z kolorami oznaczającymi centralność.
    
    Parameters:
    G (nx.Graph): Graf do wizualizacji
    window_info (tuple): (rozmiar_okna, przesunięcie)
    filename (str): Nazwa pliku do zapisania wizualizacji
    """
    window_size, shift = window_info
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Obliczanie centralności wierzchołków
    degree_cent = nx.degree_centrality(G)
    node_sizes = [300 * degree_cent[n] + 50 for n in G.nodes()]
    node_colors = list(degree_cent.values())
    
    # Rysowanie grafu
    pos = nx.spring_layout(G, seed=42)
    nodes = nx.draw_networkx_nodes(G, pos, ax=ax, 
                                 node_size=node_sizes, 
                                 node_color=node_colors, 
                                 cmap=plt.cm.viridis)
    edges = nx.draw_networkx_edges(G, pos, ax=ax,
                                 edge_color='gray', alpha=0.7)
    
    # Etykiety tylko dla najważniejszych wierzchołków
    key_nodes = set(sorted(degree_cent, key=degree_cent.get, reverse=True)[:20])
    nx.draw_networkx_labels(G, pos, ax=ax,
                          labels={n: str(n) for n in key_nodes},
                          font_size=8, font_weight='bold')
    
    # Dodanie paska kolorów
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=max(node_colors)))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='Centralność stopnia')
    
    plt.title(f'Sieć komunikacji - okno {window_size} dni, przesunięcie {shift} dni')
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def visualize_prediction(G, pred_data, method, window_info, filename):
    """
    Wizualizuje wyniki predykcji połączeń.
    
    Parameters:
    G (nx.Graph): Graf oryginalny
    pred_data (tuple): (G_train, edge_test, scores) - dane predykcji
    method (str): Metoda predykcji
    window_info (tuple): (rozmiar_okna, przesunięcie)
    filename (str): Nazwa pliku do zapisania wizualizacji
    """
    if not pred_data:
        return
        
    G_train, test_edges, scores = pred_data
    window_size, shift = window_info
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    pos = nx.spring_layout(G, seed=42)
    nodes = nx.draw_networkx_nodes(G_train, pos, ax=ax, node_size=300, node_color='skyblue', alpha=0.8)
    edges = nx.draw_networkx_edges(G_train, pos, ax=ax, alpha=0.5, width=1.0, edge_color='gray')
    
    # Rysuj usunięte krawędzie (testowe)
    removed = nx.draw_networkx_edges(G, pos, ax=ax, edgelist=test_edges, width=2.0, 
                                   edge_color='red', style='dashed', 
                                   label='Usunięte (testowe)')
    
    # Znajdź top przewidywane krawędzie - zoptymalizowana wersja
    pred_edges = []
    
    # Używaj tylko par z wyników predykcji, co jest znacznie bardziej wydajne
    for (u, v), score in scores.items():
        if u != v and not G_train.has_edge(u, v) and (u, v) not in test_edges and (v, u) not in test_edges:
            if score > 0:
                pred_edges.append((u, v, score))
    
    # Sortuj i wybierz top 10 (lub mniej jeśli nie ma tylu)
    pred_edges.sort(key=lambda x: x[2], reverse=True)
    top_pred = pred_edges[:min(10, len(pred_edges))]
    
    # Rysuj przewidywane krawędzie
    if top_pred:
        predicted = nx.draw_networkx_edges(G, pos, ax=ax,
                                        edgelist=[(u, v) for u, v, _ in top_pred], 
                                        width=2.0, edge_color='green', style='dotted', 
                                        label='Przewidywane')
    
    # Dodaj etykiety tylko dla kluczowych wierzchołków
    key_nodes = set()
    for u, v in test_edges:
        key_nodes.add(u)
        key_nodes.add(v)
    for u, v, _ in top_pred[:5]:
        key_nodes.add(u)
        key_nodes.add(v)
    
    nx.draw_networkx_labels(G, pos, ax=ax,
                           labels={node: str(node) for node in key_nodes}, 
                           font_size=10, font_weight='bold')
    
    plt.title(f'Predykcja połączeń - {method} (Okno {window_size}, przesunięcie {shift})')
    ax.set_axis_off()
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='gray', lw=1, label='Istniejące połączenia'),
        Line2D([0], [0], color='red', lw=2, linestyle='--', label='Usunięte (testowe)'),
        Line2D([0], [0], color='green', lw=2, linestyle=':', label='Przewidywane')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def main(folder_path='maile_pwr'):
    """
    Główna funkcja analizy sieci komunikacji mailowej.
    
    Parameters:
    folder_path (str): Ścieżka do folderu z plikami
    """
    # Wczytanie danych
    data_groups = load_data(folder_path)
    if not data_groups:
        print("Brak danych do analizy.")
        return
    
    # Wybór parametrów okien do analizy
    window_params = list(data_groups.keys())
    selected_params = []
    
    # Priorytetyzacja konkretnych parametrów
    priority_params = [(1, 1), (7, 7)]
    for param in priority_params:
        if param in data_groups:
            selected_params.append(param)
    
    # Jeśli nie znaleziono priorytetowych parametrów, wybierz dostępne
    if not selected_params and window_params:
        selected_params = [window_params[0]]
        if len(window_params) > 1:
            selected_params.append(window_params[-1])
    
    print(f"Wybrane parametry okien: {selected_params}")
    
    # Analiza wzorców tygodniowych
    print("Analiza wzorców tygodniowych...")
    weekly_patterns = analyze_weekly_patterns(data_groups)
    
    # Analiza trendów czasowych
    print("Analiza trendów czasowych...")
    analyze_temporal_trends(data_groups)
    
    # Słownik na wyniki dla wszystkich metod i parametrów okien
    all_results = {}
    
    for window_size, shift in selected_params:
        print(f"Analiza dla okna {window_size} dni, przesunięcie {shift} dni")

        # Tworzenie grafu dla wybranego okna czasowego
        sample_data = data_groups[(window_size, shift)][0]
        G = nx.DiGraph()
        for source, target, weight in sample_data['edges']:
            G.add_edge(source, target, weight=weight)
        
        print(f"Graf: {G.number_of_nodes()} wierzchołków, {G.number_of_edges()} krawędzi")
        
        # Wizualizacja grafu - poprawiona wersja z użyciem dedykowanej funkcji
        visualize_graph(G, (window_size, shift), f'graf_okno{window_size}_przesuniecie{shift}.jpg')
        
        # Predykcja połączeń trzema różnymi metodami
        methods = ['jaccard', 'adamic_adar', 'preferential_attachment']
        results = {}
        
        for method in methods:
            print(f"Predykcja metodą {method}")
            metrics, pred_data = predict_and_evaluate(G, method)
            if metrics:
                results[method] = metrics
                all_results[(window_size, shift, method)] = metrics
                print(f"ROC AUC: {metrics['roc_auc']:.4f}")
                print(f"Precision: {metrics['precision']:.4f}")
                print(f"Recall: {metrics['recall']:.4f}")
                
                # Wizualizacja predykcji
                visualize_prediction(G, pred_data, method, (window_size, shift),
                                    f'predykcja_{method}_okno{window_size}_przesuniecie{shift}.jpg')
        
        # Porównanie metod predykcji
        if results:
            plt.figure(figsize=(12, 8))
            metrics = ['roc_auc', 'precision', 'recall']
            metric_names = ['ROC AUC', 'Precyzja', 'Czułość']
            x = np.arange(len(methods))
            width = 0.25
            
            for i, (metric, name) in enumerate(zip(metrics, metric_names)):
                values = [results[method][metric] for method in methods]
                plt.bar(x + i*width, values, width, label=name)
                
                for j, v in enumerate(values):
                    plt.text(x[j] + i*width, v + 0.02, f'{v:.3f}', 
                           ha='center', va='bottom', fontsize=10)
            
            plt.xlabel('Metoda predykcji')
            plt.ylabel('Wartość metryki')
            plt.title(f'Porównanie metod predykcji - okno {window_size} dni, przesunięcie {shift} dni')
            plt.xticks(x + width, methods)
            plt.ylim(0, 1.1)  # Ustawienie zakresu osi Y
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'porownanie_metod_okno{window_size}_przesuniecie{shift}.jpg'))
            plt.close()
    
    # Analiza wpływu parametrów okna na skuteczność predykcji
    if len(all_results) > 1:
        window_sizes = sorted(set(ws for ws, _, _ in all_results.keys()))
        
        plt.figure(figsize=(14, 8))
        methods = ['jaccard', 'adamic_adar', 'preferential_attachment']
        markers = ['o', 's', '^']
        
        for method, marker in zip(methods, markers):
            sizes = []
            roc_aucs = []
            
            for ws in window_sizes:
                for (window_size, shift, m), metrics in all_results.items():
                    if window_size == ws and m == method:
                        sizes.append(ws)
                        roc_aucs.append(metrics['roc_auc'])
            
            if sizes:
                plt.plot(sizes, roc_aucs, marker + '-', linewidth=2, markersize=8, label=f'Metoda {method}')
        
        plt.xlabel('Wielkość okna czasowego (dni)')
        plt.ylabel('ROC AUC')
        plt.title('Wpływ wielkości okna czasowego na skuteczność predykcji')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'wplyw_wielkosci_okna.jpg'))
        plt.close()
    
    print(f"Analiza zakończona. Wykresy zapisano w katalogu: {output_dir}")

if __name__ == "__main__":
    main()