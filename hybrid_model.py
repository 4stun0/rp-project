#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sistema Híbrido de Clusterização e Data Storytelling para EAD
Combina K-means, SOM e análise sequencial para gerar narrativas significativas sobre dados educacionais.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from minisom import MiniSom
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy import stats
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from mlxtend.frequent_patterns import apriori, association_rules
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import hdbscan
import umap
from tqdm import tqdm

# Ignorar avisos para visualização mais limpa
warnings.filterwarnings('ignore')

class HybridModel:
    """
    Implementação do Sistema Híbrido de Clusterização e Data Storytelling
    """
    def __init__(self, data_path):
        """
        Inicializa o modelo híbrido
        
        Args:
            data_path: Caminho para o diretório contendo os dados
        """
        self.data_path = data_path
        self.logs_df = None
        self.quiz_grades_df = None
        self.quiz_list_df = None
        self.resource_list_df = None
        self.event_mapping_df = None
        self.user_features = None
        self.sequence_features = None
        self.scaled_features = None
        self.kmeans_labels = None
        self.som_labels = None
        self.hierarchical_labels = None
        self.final_labels = None
        self.lstm_model = None
        self.feature_names = []
        self.pca_model = None
        self.pca_features = None
        self.cluster_evaluation = {}
        
    def load_data(self):
        """
        Carrega os dados dos arquivos CSV
        """
        print("Carregando dados...")
        self.logs_df = pd.read_csv(os.path.join(self.data_path, 'dataset/see_course2060_12-11_to_11-12_logs_filtered.csv'),
                                  encoding='utf-8', low_memory=False)
        self.quiz_grades_df = pd.read_csv(os.path.join(self.data_path, 'dataset/see_course2060_quiz_grades.csv'),
                                         encoding='utf-8', low_memory=False)
        self.quiz_list_df = pd.read_csv(os.path.join(self.data_path, 'dataset/see_course2060_quiz_list.csv'),
                                       encoding='utf-8', low_memory=False)
        try:
            self.resource_list_df = pd.read_csv(os.path.join(self.data_path, 'dataset/see_course2060_resource_list.csv'),
                                              encoding='utf-8', low_memory=False, error_bad_lines=False)
        except:
            print("Erro ao carregar resource_list_df, tentando formato alternativo...")
            try:
                self.resource_list_df = pd.read_csv(os.path.join(self.data_path, 'dataset/see_course2060_resource_list.csv'),
                                                  encoding='utf-8', sep='\t', low_memory=False)
            except Exception as e:
                print(f"Não foi possível carregar resource_list_df: {str(e)}")
                
        self.event_mapping_df = pd.read_csv(os.path.join(self.data_path, 'dataset/event_mapping.csv'),
                                           encoding='utf-8', low_memory=False)
        
        print(f"Logs de eventos: {self.logs_df.shape}")
        print(f"Notas dos quizzes: {self.quiz_grades_df.shape}")
        print(f"Lista de quizzes: {self.quiz_list_df.shape}")
        if self.resource_list_df is not None:
            print(f"Lista de recursos: {self.resource_list_df.shape}")
        print(f"Mapeamento de eventos: {self.event_mapping_df.shape}")
        
    def process_timestamp(self):
        """
        Converte o timestamp unix para formato datetime
        """
        self.logs_df['timestamp'] = pd.to_datetime(self.logs_df['t'], unit='s')
        self.logs_df['date'] = self.logs_df['timestamp'].dt.date
        self.logs_df['hour'] = self.logs_df['timestamp'].dt.hour
        self.logs_df['weekday'] = self.logs_df['timestamp'].dt.weekday
        
    def extract_features(self):
        """
        Extrai características dos dados para clusterização
        """
        print("Extraindo características dos dados...")
        
        # Processamos a coluna de timestamp se ainda não foi processada
        if 'timestamp' not in self.logs_df.columns:
            self.process_timestamp()
            
        # Listagem de usuários únicos
        unique_users = self.logs_df['userid'].unique()
        
        # Inicializa o dataframe de características
        features = []
        
        for user in tqdm(unique_users, desc="Processando usuários"):
            # Filtra os logs para o usuário atual
            user_logs = self.logs_df[self.logs_df['userid'] == user]
            
            # 1. Características de Engajamento
            total_events = len(user_logs)
            unique_dates = user_logs['date'].nunique()
            # Calcula a diferença em dias, evitando divisão por zero quando todas as entradas são do mesmo dia
            date_diff = (user_logs['date'].max() - user_logs['date'].min()).days
            active_days_ratio = unique_dates / date_diff if total_events > 1 and date_diff > 0 else 0
            
            # Distribuição de eventos por dia da semana
            weekday_counts = user_logs['weekday'].value_counts()
            weekday_features = [weekday_counts.get(i, 0) / total_events if total_events > 0 else 0 for i in range(7)]
            
            # Distribuição de eventos por hora
            hour_counts = user_logs['hour'].value_counts()
            morning_ratio = sum(hour_counts.get(i, 0) for i in range(5, 12)) / total_events if total_events > 0 else 0
            afternoon_ratio = sum(hour_counts.get(i, 0) for i in range(12, 18)) / total_events if total_events > 0 else 0
            evening_ratio = sum(hour_counts.get(i, 0) for i in range(18, 24)) / total_events if total_events > 0 else 0
            night_ratio = sum(hour_counts.get(i, 0) for i in range(0, 5)) / total_events if total_events > 0 else 0
            
            # 2. Características de Acesso a Recursos
            component_counts = user_logs['component'].value_counts()
            action_counts = user_logs['action'].value_counts()
            
            core_ratio = component_counts.get('core', 0) / total_events if total_events > 0 else 0
            quiz_ratio = component_counts.get('mod_quiz', 0) / total_events if total_events > 0 else 0
            forum_ratio = component_counts.get('mod_forum', 0) / total_events if total_events > 0 else 0
            resource_ratio = component_counts.get('mod_resource', 0) / total_events if total_events > 0 else 0
            
            viewed_ratio = action_counts.get('viewed', 0) / total_events if total_events > 0 else 0
            submitted_ratio = action_counts.get('submitted', 0) / total_events if total_events > 0 else 0
            
            # 3. Desempenho acadêmico
            user_grades = self.quiz_grades_df[self.quiz_grades_df['userid'] == user]
            avg_grade = user_grades['student_grade'].mean() if len(user_grades) > 0 else None
            max_grade = user_grades['student_grade'].max() if len(user_grades) > 0 else None
            
            # 4. Comportamento social 
            forum_actions = user_logs[user_logs['component'] == 'mod_forum'].shape[0]
            forum_posts = user_logs[(user_logs['component'] == 'mod_forum') & 
                                  (user_logs['action'] == 'created')].shape[0]
            
            # 5. Tempo entre eventos consecutivos
            if total_events > 1:
                user_logs_sorted = user_logs.sort_values('timestamp')
                diff_seconds = np.diff(user_logs_sorted['timestamp'].astype(int)) / 1e9  # Convertendo para segundos
                median_time_between = np.median(diff_seconds)
                mean_time_between = np.mean(diff_seconds)
            else:
                median_time_between = 0
                mean_time_between = 0
                
            # Reúne todas as características
            user_features = {
                'userid': user,
                'total_events': total_events,
                'unique_dates': unique_dates,
                'active_days_ratio': active_days_ratio,
                'weekday_mon': weekday_features[0],
                'weekday_tue': weekday_features[1],
                'weekday_wed': weekday_features[2],
                'weekday_thu': weekday_features[3],
                'weekday_fri': weekday_features[4],
                'weekday_sat': weekday_features[5],
                'weekday_sun': weekday_features[6],
                'morning_ratio': morning_ratio,
                'afternoon_ratio': afternoon_ratio,
                'evening_ratio': evening_ratio,
                'night_ratio': night_ratio,
                'core_ratio': core_ratio,
                'quiz_ratio': quiz_ratio,
                'forum_ratio': forum_ratio,
                'resource_ratio': resource_ratio,
                'viewed_ratio': viewed_ratio,
                'submitted_ratio': submitted_ratio,
                'avg_grade': avg_grade,
                'max_grade': max_grade,
                'forum_actions': forum_actions,
                'forum_posts': forum_posts,
                'median_time_between': median_time_between,
                'mean_time_between': mean_time_between
            }
            
            features.append(user_features)
            
        # Cria o DataFrame de características
        self.user_features = pd.DataFrame(features)
        
        # Preenche valores NaN com a média ou zero
        for col in self.user_features.columns:
            if col != 'userid':
                if self.user_features[col].dtype in [np.float64, np.int64]:
                    self.user_features[col].fillna(self.user_features[col].mean(), inplace=True)
        
        # Guarda os nomes das características para uso posterior
        self.feature_names = [col for col in self.user_features.columns if col != 'userid']
        
        # Substituímos inf e -inf por valores grandes/pequenos, se existirem
        self.user_features.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.user_features.fillna(self.user_features.mean(), inplace=True)
        
        print(f"Características extraídas: {self.user_features.shape}")
        print(f"Features disponíveis: {self.feature_names}")
        
        return self.user_features
    
    def extract_sequence_features(self):
        """
        Extrai características sequenciais dos logs
        """
        print("Extraindo características sequenciais...")
        
        unique_users = self.logs_df['userid'].unique()
        sequence_data = []
        
        for user in tqdm(unique_users, desc="Processando sequências de usuários"):
            user_logs = self.logs_df[self.logs_df['userid'] == user].sort_values('timestamp')
            
            if len(user_logs) < 2:
                continue
                
            # Criamos uma sequência de ações e componentes
            action_sequence = user_logs['action'].tolist()
            component_sequence = user_logs['component'].tolist()
            target_sequence = user_logs['target'].tolist()
            
            # Combinamos para criar uma sequência única
            combined_sequence = [f"{a}_{c}_{t}" for a, c, t in zip(action_sequence, component_sequence, target_sequence)]
            
            sequence_data.append({
                'userid': user,
                'sequence': combined_sequence
            })
            
        self.sequence_features = pd.DataFrame(sequence_data)
        return self.sequence_features
            
    def normalize_features(self):
        """
        Normaliza as características para uso nos algoritmos de clusterização
        """
        print("Normalizando características...")
        
        # Selecionamos apenas as colunas numéricas para normalização
        numeric_features = self.user_features.select_dtypes(include=['float64', 'int64']).columns.tolist()
        features_to_scale = [f for f in numeric_features if f != 'userid']
        
        # Aplicamos a normalização
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.user_features[features_to_scale])
        
        # Criamos um novo dataframe com os dados normalizados
        self.scaled_features = pd.DataFrame(scaled_data, columns=features_to_scale)
        self.scaled_features['userid'] = self.user_features['userid'].values
        
        print(f"Características normalizadas: {self.scaled_features.shape}")
        return self.scaled_features
        
    def apply_pca(self, n_components=5):
        """
        Aplica PCA para redução de dimensionalidade
        
        Args:
            n_components: Número de componentes principais a manter
        """
        print(f"Aplicando PCA para redução de dimensionalidade (n_components={n_components})...")
        
        # Selecionamos as features (excluindo userid)
        features_for_pca = self.scaled_features.drop('userid', axis=1)
        
        # Aplicamos PCA
        self.pca_model = PCA(n_components=n_components)
        pca_result = self.pca_model.fit_transform(features_for_pca)
        
        # Criamos um dataframe com o resultado do PCA
        self.pca_features = pd.DataFrame(
            pca_result, 
            columns=[f'PC{i+1}' for i in range(n_components)]
        )
        self.pca_features['userid'] = self.scaled_features['userid'].values
        
        # Armazena a variância explicada para análise
        explained_variance = self.pca_model.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        print(f"Variância explicada por componente: {explained_variance}")
        print(f"Variância explicada acumulada: {cumulative_variance}")
        
        # Visualizamos a variância explicada acumulada
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, marker='o')
        plt.xlabel('Número de Componentes')
        plt.ylabel('Variância Explicada Acumulada')
        plt.title('Variância Explicada por Número de Componentes PCA')
        plt.grid(True)
        plt.savefig(os.path.join(self.data_path, 'pca_variance_explained.png'))
        
        return self.pca_features
    
    def apply_kmeans_clustering(self, n_clusters=None):
        """
        Aplica K-means para clusterização inicial
        
        Args:
            n_clusters: Número de clusters para K-means (se None, busca o melhor valor)
        """
        print("Aplicando algoritmo K-means para clusterização...")
        
        # Se n_clusters não for fornecido, busca o melhor valor usando silhouette
        if n_clusters is None:
            print("Encontrando o melhor número de clusters...")
            
            silhouette_scores = []
            K_range = range(2, 11)
            
            for k in K_range:
                kmeans_model = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans_model.fit_predict(self.pca_features.drop('userid', axis=1))
                score = silhouette_score(self.pca_features.drop('userid', axis=1), cluster_labels)
                silhouette_scores.append(score)
                print(f"K={k}, Silhouette Score: {score:.4f}")
                
            best_k = K_range[np.argmax(silhouette_scores)]
            print(f"Melhor número de clusters (K-means): {best_k}")
            
                        # Visualizamos os scores para cada valor de K
            plt.figure(figsize=(10, 6))
            plt.plot(K_range, silhouette_scores, marker='o')
            plt.xlabel('Número de Clusters (K)')
            plt.ylabel('Silhouette Score')
            plt.title('Silhouette Score por Número de Clusters')
            plt.grid(True)
            plt.savefig(os.path.join(self.data_path, 'kmeans_silhouette.png'))
            
            # Usamos o melhor K encontrado
            n_clusters = best_k
        
        # Aplicamos K-means com o número ideal de clusters
        kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        data_for_clustering = self.pca_features.drop('userid', axis=1)
        self.kmeans_labels = kmeans_model.fit_predict(data_for_clustering)
        self.kmeans_model = kmeans_model
        
        # Adicionamos as labels de cluster ao dataframe
        self.pca_features['kmeans_cluster'] = self.kmeans_labels
        
        # Visualizamos os clusters (primeiras 2 componentes principais)
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            self.pca_features['PC1'],
            self.pca_features['PC2'],
            c=self.kmeans_labels,
            cmap='viridis',
            alpha=0.7,
            s=50
        )
        
        centers = kmeans_model.cluster_centers_[:, :2]  # Primeiras duas componentes principais
        plt.scatter(
            centers[:, 0],
            centers[:, 1],
            c='red',
            s=200,
            alpha=0.8,
            marker='X'
        )
        
        plt.title(f'Visualização dos Clusters K-means (K={n_clusters})')
        plt.xlabel('Componente Principal 1')
        plt.ylabel('Componente Principal 2')
        plt.colorbar(scatter, label='Cluster')
        plt.grid(True)
        plt.savefig(os.path.join(self.data_path, 'kmeans_clusters.png'))
        
        # Avaliamos a qualidade dos clusters
        silhouette_avg = silhouette_score(data_for_clustering, self.kmeans_labels)
        print(f"Silhouette Score para K={n_clusters}: {silhouette_avg:.4f}")
        
        return self.kmeans_labels

    def apply_som_clustering(self, map_size=None, max_clusters=15):
        """
        Aplica Self-Organizing Maps (SOM) para clusterização
        
        Args:
            map_size: Tamanho do mapa SOM (se None, calcula automaticamente)
            max_clusters: Número máximo de clusters SOM a considerar
        """
        print("Aplicando Self-Organizing Maps (SOM) para clusterização...")
        
        # Se o tamanho do mapa não for fornecido, calculamos automaticamente
        if map_size is None:
            # Heurística: mapsize = 5 * sqrt(num_samples)
            n_samples = len(self.pca_features)
            map_size_recommendation = int(5 * np.sqrt(n_samples))
            map_size = min(map_size_recommendation, 10)  # Limitamos a 10x10 para evitar mapas muito grandes
            print(f"Tamanho de mapa recomendado: {map_size}x{map_size}")
        
        # Preparamos os dados para o SOM (excluímos a coluna userid e kmeans_cluster se existir)
        cols_to_drop = ['userid']
        if 'kmeans_cluster' in self.pca_features.columns:
            cols_to_drop.append('kmeans_cluster')
            
        som_data = self.pca_features.drop(cols_to_drop, axis=1).values
        
        # Inicializamos e treinamos o SOM
        som_shape = (map_size, map_size)
        som = MiniSom(som_shape[0], som_shape[1], som_data.shape[1], sigma=1.0, learning_rate=0.5, random_seed=42)
        
        print("Inicializando SOM...")
        som.random_weights_init(som_data)
        
        print("Treinando SOM...")
        som.train_random(som_data, 10000, verbose=True)  # 10000 iterações
        
        # Obtemos as BMUs (Best Matching Units) para cada amostra
        print("Mapeando amostras para o SOM...")
        bmu_indices = np.array([som.winner(x) for x in som_data])
        
        # Convertemos coordenadas BMU em índices de cluster únicos
        som_labels = np.zeros(len(som_data), dtype=int)
        for i, (x, y) in enumerate(bmu_indices):
            som_labels[i] = x * som_shape[1] + y
        
        # Se temos mais clusters que o máximo permitido, aplicamos uma segunda clusterização
        unique_clusters = len(np.unique(som_labels))
        if unique_clusters > max_clusters:
            print(f"SOM gerou {unique_clusters} clusters, aplicando agrupamento adicional para reduzir para no máximo {max_clusters}...")
            
            # Extraímos os centróides de cada cluster SOM
            centroids = []
            unique_labels = np.unique(som_labels)
            for label in unique_labels:
                mask = som_labels == label
                if np.sum(mask) > 0:  # Garantimos que há pelo menos uma amostra
                    centroids.append(np.mean(som_data[mask], axis=0))
            
            # Aplicamos K-means nos centróides para agrupar clusters similares
            if len(centroids) > max_clusters:
                kmeans = KMeans(n_clusters=max_clusters, random_state=42)
                centroid_labels = kmeans.fit_predict(centroids)
                
                # Mapeamos os rótulos originais para os novos rótulos agrupados
                label_mapping = {old: new for old, new in zip(unique_labels, centroid_labels)}
                new_som_labels = np.array([label_mapping[label] for label in som_labels])
                som_labels = new_som_labels
        
        # Remapeamos os labels para valores consecutivos
        unique_labels = np.unique(som_labels)
        mapping = {label: i for i, label in enumerate(unique_labels)}
        som_labels = np.array([mapping[label] for label in som_labels])
        
        print(f"SOM gerou {len(unique_labels)} clusters efetivos.")
        
        # Adicionamos as labels de cluster ao dataframe
        self.pca_features['som_cluster'] = som_labels
        self.som_labels = som_labels
        self.som_model = som
        
        # Criamos uma visualização da matriz unificada do SOM
        plt.figure(figsize=(12, 10))
        plt.pcolor(som.distance_map().T, cmap='bone_r')  # Transposta para corresponder à visualização padrão
        plt.colorbar(label='Distância')
        plt.title('SOM Distance Map (U-Matrix)')
        plt.savefig(os.path.join(self.data_path, 'som_umatrix.png'))
        
        return self.som_labels

    def apply_hierarchical_clustering(self, n_clusters=None):
        """
        Aplica agrupamento hierárquico para refinar clusters
        
        Args:
            n_clusters: Número de clusters para o agrupamento hierárquico
        """
        print("Aplicando Agrupamento Hierárquico para refinamento dos clusters...")
        
        # Se não fornecido, usamos o mesmo número de clusters do K-means
        if n_clusters is None:
            if hasattr(self, 'kmeans_model') and self.kmeans_model is not None:
                n_clusters = self.kmeans_model.n_clusters
            else:
                n_clusters = 5  # Valor padrão se K-means não tiver sido executado
        
        # Preparamos os dados para clustering - combinamos os resultados de K-means e SOM
        cols_to_drop = ['userid']
        if 'kmeans_cluster' in self.pca_features.columns:
            self.pca_features['kmeans_cluster_scaled'] = self.pca_features['kmeans_cluster'] / self.pca_features['kmeans_cluster'].max()
        
        if 'som_cluster' in self.pca_features.columns:
            self.pca_features['som_cluster_scaled'] = self.pca_features['som_cluster'] / self.pca_features['som_cluster'].max()
        
        # Usamos todas as features disponíveis para o agrupamento hierárquico
        cols_for_clustering = [col for col in self.pca_features.columns 
                              if col not in cols_to_drop]
        
        hierarchical_data = self.pca_features[cols_for_clustering].values
        
        # Aplicamos o agrupamento hierárquico
        hierarchical_model = AgglomerativeClustering(
            n_clusters=n_clusters,
            affinity='euclidean',
            linkage='ward'
        )
        
        self.hierarchical_labels = hierarchical_model.fit_predict(hierarchical_data)
        self.hierarchical_model = hierarchical_model
        
        # Adicionamos as labels de cluster ao dataframe
        self.pca_features['hierarchical_cluster'] = self.hierarchical_labels
        
        # Visualizamos o dendrograma (limitado a uma amostra se o dataset for muito grande)
        max_samples_for_dendrogram = 100
        if len(self.pca_features) > max_samples_for_dendrogram:
            # Amostramos apenas um subconjunto dos dados para o dendrograma
            sample_indices = np.random.choice(len(self.pca_features), max_samples_for_dendrogram, replace=False)
            sample_data = hierarchical_data[sample_indices]
            
            print(f"Gerando dendrograma com amostra de {max_samples_for_dendrogram} pontos...")
            plt.figure(figsize=(16, 10))
            linked = linkage(sample_data, method='ward')
            dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
            plt.title('Dendrograma Hierárquico (Amostra)')
            plt.xlabel('Amostras')
            plt.ylabel('Distância')
            plt.savefig(os.path.join(self.data_path, 'hierarchical_dendrogram_sample.png'))
        else:
            print("Gerando dendrograma completo...")
            plt.figure(figsize=(16, 10))
            linked = linkage(hierarchical_data, method='ward')
            dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
            plt.title('Dendrograma Hierárquico')
            plt.xlabel('Amostras')
            plt.ylabel('Distância')
            plt.savefig(os.path.join(self.data_path, 'hierarchical_dendrogram.png'))
        
        # Avaliamos a qualidade dos clusters
        silhouette_avg = silhouette_score(hierarchical_data, self.hierarchical_labels)
        print(f"Silhouette Score para Agrupamento Hierárquico (clusters={n_clusters}): {silhouette_avg:.4f}")
        
        return self.hierarchical_labels
        
    def apply_hybrid_clustering(self, min_cluster_size=10):
        """
        Combina os resultados de K-means e SOM para criar clusters híbridos
        
        Args:
            min_cluster_size: Tamanho mínimo de cluster para considerar
        """
        print("Aplicando clusterização híbrida (K-means + SOM)...")
        
        # Criamos uma tabela de contingência para visualizar a relação entre K-means e SOM
        contingency_table = pd.crosstab(self.kmeans_labels, self.som_labels, rownames=['K-means'], colnames=['SOM'])
        print("Tabela de contingência dos clusters K-means vs. SOM:")
        print(contingency_table)
        
        # Combinamos os rótulos de K-means e SOM para criar clusters híbridos
        hybrid_labels = np.array([f"{k}_{s}" for k, s in zip(self.kmeans_labels, self.som_labels)])
        
        # Contamos a frequência de cada combinação
        unique_combinations, counts = np.unique(hybrid_labels, return_counts=True)
        
        # Filtramos combinações com menos de min_cluster_size amostras
        valid_combinations = unique_combinations[counts >= min_cluster_size]
        
        # Se não houver combinações válidas, reduzimos o tamanho mínimo
        if len(valid_combinations) == 0:
            print(f"Nenhuma combinação com pelo menos {min_cluster_size} amostras. Reduzindo o tamanho mínimo.")
            min_cluster_size = 5
            valid_combinations = unique_combinations[counts >= min_cluster_size]
        
        print(f"Número de combinações únicas: {len(unique_combinations)}")
        print(f"Número de combinações válidas (>= {min_cluster_size} amostras): {len(valid_combinations)}")
        
        # Remapeamos os rótulos híbridos para valores consecutivos
        # Combinações com menos de min_cluster_size amostras são agrupadas em um cluster "outros"
        mapping = {}
        for i, combo in enumerate(valid_combinations):
            mapping[combo] = i
        
        # Criamos um cluster "outros" para combinações pequenas
        other_cluster_id = len(valid_combinations)
        
        # Aplicamos o mapeamento
        hybrid_labels_mapped = np.array([mapping.get(label, other_cluster_id) for label in hybrid_labels])
        
        # Criamos um dataframe com os resultados
        hybrid_df = pd.DataFrame({
            'userid': self.pca_features['userid'],
            'kmeans_cluster': self.kmeans_labels,
            'som_cluster': self.som_labels,
            'hybrid_cluster': hybrid_labels_mapped
        })
        
        # Armazenamos os rótulos híbridos
        self.hybrid_labels = hybrid_labels_mapped
        
        return hybrid_df
    
    def combine_clustering_results(self):
        """
        Combina os resultados de K-means, SOM e Agrupamento Hierárquico
        para produzir um conjunto final de clusters, com etapa adicional para
        mesclar clusters com características muito semelhantes
        """
        print("Combinando resultados de clusterização...")
        
        # Verificamos quais métodos de clusterização foram aplicados
        has_kmeans = hasattr(self, 'kmeans_labels') and self.kmeans_labels is not None
        has_som = hasattr(self, 'som_labels') and self.som_labels is not None
        has_hierarchical = hasattr(self, 'hierarchical_labels') and self.hierarchical_labels is not None
        
        if not (has_kmeans or has_som or has_hierarchical):
            print("Erro: Nenhum método de clusterização foi aplicado ainda")
            return None
        
        # Inicializamos o dataframe com os IDs de usuário
        cluster_df = pd.DataFrame({'userid': self.pca_features['userid'].values})
        
        # Adicionamos os resultados de cada método
        if has_kmeans:
            cluster_df['kmeans_cluster'] = self.kmeans_labels
            
        if has_som:
            cluster_df['som_cluster'] = self.som_labels
            
        if has_hierarchical:
            cluster_df['hierarchical_cluster'] = self.hierarchical_labels
            
        # Criamos clusters finais com base nos votos majoritários
        if has_kmeans and has_som and has_hierarchical:
            # Usamos Random Forest para combinar os clusters
            # Treinamos 3 modelos, cada um prevendo um tipo de cluster a partir dos outros dois
            print("Usando Random Forest para combinar clusters...")
            
            # Preparamos os dados
            X_kmeans = pd.DataFrame({'som': self.som_labels, 'hier': self.hierarchical_labels})
            X_som = pd.DataFrame({'kmeans': self.kmeans_labels, 'hier': self.hierarchical_labels})
            X_hier = pd.DataFrame({'kmeans': self.kmeans_labels, 'som': self.som_labels})
            
            # Treinamos os modelos
            rf_kmeans = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_kmeans.fit(X_kmeans, self.kmeans_labels)
            
            rf_som = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_som.fit(X_som, self.som_labels)
            
            rf_hier = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_hier.fit(X_hier, self.hierarchical_labels)
            
            # Calculamos a importância das features (qual clusterização tem mais peso)
            importance_kmeans = np.mean(rf_kmeans.feature_importances_)
            importance_som = np.mean(rf_som.feature_importances_)
            importance_hier = np.mean(rf_hier.feature_importances_)
            
            total_importance = importance_kmeans + importance_som + importance_hier
            weight_kmeans = importance_kmeans / total_importance
            weight_som = importance_som / total_importance
            weight_hier = importance_hier / total_importance
            
            print(f"Pesos de cada método: K-means={weight_kmeans:.2f}, SOM={weight_som:.2f}, Hierárquico={weight_hier:.2f}")
            
            # Determinamos o número final de clusters (média ponderada arredondada)
            n_clusters_kmeans = len(np.unique(self.kmeans_labels))
            n_clusters_som = len(np.unique(self.som_labels))
            n_clusters_hier = len(np.unique(self.hierarchical_labels))
            
            n_clusters_final = int(np.round(
                weight_kmeans * n_clusters_kmeans + 
                weight_som * n_clusters_som + 
                weight_hier * n_clusters_hier
            ))
            
            # Criamos clusters finais com base nos votos majoritários
            final_labels = np.zeros(len(self.pca_features), dtype=int)
            
            for i in range(len(self.pca_features)):
                votes = {
                    'kmeans': self.kmeans_labels[i],
                    'som': self.som_labels[i],
                    'hierarchical': self.hierarchical_labels[i]
                }
                
                # Contamos os votos
                vote_counts = {}
                for method, label in votes.items():
                    if label not in vote_counts:
                        vote_counts[label] = 0
                    vote_counts[label] += 1
                
                # Escolhemos o cluster com mais votos
                final_label = max(vote_counts, key=vote_counts.get)
                final_labels[i] = final_label
            
            # Adicionamos os rótulos finais ao dataframe
            cluster_df['final_cluster'] = final_labels
            
            return cluster_df
        
        if self.kmeans_labels is None or self.som_labels is None:
            print("Execute both apply_kmeans_clustering() and apply_som_clustering() first")
            return None
        
        # Criamos uma tabela de contingência para ver a relação entre os clusters K-means e SOM
        contingency_table = pd.crosstab(
            pd.Series(self.kmeans_labels, name='K-means'), 
            pd.Series(self.som_labels, name='SOM')
        )
        
        print("Tabela de contingência dos clusters K-means vs. SOM:")
        print(contingency_table)
        
        # Criamos um novo rótulo combinado usando ambos os clusters
        hybrid_labels = np.array([f"K{k}_S{s}" for k, s in zip(self.kmeans_labels, self.som_labels)])
        
        # Contamos as combinações únicas
        unique_combinations = np.unique(hybrid_labels)
        print(f"Número de combinações únicas: {len(unique_combinations)}")
        
        # Mapeamos as combinações para novos IDs de cluster
        mapping = {combo: i for i, combo in enumerate(unique_combinations)}
        final_labels = np.array([mapping[label] for label in hybrid_labels])
        
        # NOVA ETAPA: Verificar e mesclar clusters semelhantes
        # Primeiro, calculamos os centroides de cada cluster no espaço PCA
        pca_features_array = self.pca_features.drop('userid', axis=1).values
        cluster_centroids = {}
        for cluster_id in np.unique(final_labels):
            mask = final_labels == cluster_id
            if np.sum(mask) > 0:  # Evitar clusters vazios
                cluster_centroids[cluster_id] = np.mean(pca_features_array[mask], axis=0)
        
        # Calculamos a similaridade entre os centroides dos clusters
        similarity_threshold = 0.85  # Limiar de similaridade para mesclar clusters
        clusters_to_merge = {}
        
        # Função para calcular similaridade de cosseno entre dois vetores
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        # Identificamos clusters para mesclar
        for cluster_i in cluster_centroids:
            for cluster_j in cluster_centroids:
                if cluster_i < cluster_j:  # Evitar comparações duplicadas
                    similarity = cosine_similarity(cluster_centroids[cluster_i], cluster_centroids[cluster_j])
                    if similarity > similarity_threshold:
                        print(f"Clusters {cluster_i} e {cluster_j} são muito semelhantes (similaridade = {similarity:.2f})")
                        if cluster_i not in clusters_to_merge:
                            clusters_to_merge[cluster_i] = []
                        clusters_to_merge[cluster_i].append(cluster_j)
        
        # Mesclamos os clusters semelhantes
        if clusters_to_merge:
            print("Mesclando clusters semelhantes...")
            new_labels = final_labels.copy()
            for main_cluster, similar_clusters in clusters_to_merge.items():
                for similar_cluster in similar_clusters:
                    new_labels[final_labels == similar_cluster] = main_cluster
            
            # Remapeamos os IDs para serem consecutivos
            unique_new_labels = np.unique(new_labels)
            remap = {old_id: new_id for new_id, old_id in enumerate(unique_new_labels)}
            final_labels = np.array([remap[label] for label in new_labels])
            
            print(f"Número de clusters após mesclagem: {len(np.unique(final_labels))}")
        
        self.hybrid_labels = final_labels
        
        # Adicionamos os rótulos híbridos ao DataFrame de características
        result_df = self.pca_features.copy()
        result_df['kmeans_cluster'] = self.kmeans_labels
        result_df['som_cluster'] = self.som_labels
        result_df['hybrid_cluster'] = final_labels
        
        # Visualizamos os clusters híbridos no espaço 2D do PCA
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            result_df['PC1'], 
            result_df['PC2'], 
            c=result_df['hybrid_cluster'], 
            cmap='tab20', 
            alpha=0.7,
            s=50
        )
        plt.colorbar(scatter, label='Cluster Híbrido')
        plt.title(f'Clusterização Híbrida (K-means + SOM)')
        plt.xlabel('Componente Principal 1')
        plt.ylabel('Componente Principal 2')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.data_path, 'hybrid_clusters.png'))
        
        return result_df
    
    def analyze_clusters(self, cluster_labels):
        """
        Analisa os clusters e extrai características distintivas
        
        Args:
            cluster_labels: Rótulos de cluster a serem analisados
        """
        print("Analisando características dos clusters...")
        
        # Combinamos os rótulos de cluster com as características originais
        analysis_df = self.user_features.copy()
        analysis_df['cluster'] = cluster_labels
        
        # Calculamos estatísticas por cluster
        cluster_stats = []
        
        for cluster_id in np.unique(cluster_labels):
            cluster_data = analysis_df[analysis_df['cluster'] == cluster_id]
            n_members = len(cluster_data)
            
            # Calculamos médias para cada característica
            means = cluster_data.drop(['userid', 'cluster'], axis=1).mean()
            
            # Adicionamos ao DataFrame de estatísticas
            stats = {'cluster_id': cluster_id, 'n_members': n_members}
            stats.update({col: means[col] for col in means.index})
            cluster_stats.append(stats)
            
        # Criamos DataFrame de estatísticas
        stats_df = pd.DataFrame(cluster_stats)
        
        # Identificamos as características mais distintivas para cada cluster
        global_means = analysis_df.drop(['userid', 'cluster'], axis=1).mean()
        
        distinctive_features = {}
        
        for cluster_id in np.unique(cluster_labels):
            cluster_row = stats_df[stats_df['cluster_id'] == cluster_id].iloc[0]
            
            # Calculamos o desvio da média global
            deviations = {}
            for feature in self.feature_names:
                if feature != 'userid':
                    deviation = (cluster_row[feature] - global_means[feature]) / global_means[feature] if global_means[feature] != 0 else 0
                    deviations[feature] = deviation
                    
            # Ordenamos as características por desvio absoluto
            sorted_deviations = sorted(deviations.items(), key=lambda x: abs(x[1]), reverse=True)
            
            # Pegamos as 5 características mais distintivas
            distinctive_features[cluster_id] = sorted_deviations[:5]
        
        return stats_df, distinctive_features

    def analyze_sequences(self, cluster_labels, max_length=10):
        """
        Analisa as sequências de interação por cluster
        
        Args:
            cluster_labels: Rótulos de cluster a analisar
            max_length: Comprimento máximo das sequências a considerar
        """
        print("Analisando sequências de interação por cluster...")
        
        # Preparamos os dados de sequência com informações de cluster
        user_clusters = pd.DataFrame({
            'userid': self.user_features['userid'],
            'cluster': cluster_labels
        })
        
        # Convertemos o timestamp se ainda não estiver convertido
        if 'timestamp' not in self.logs_df.columns:
            self.process_timestamp()
            
        # Identificamos sequências comuns por cluster
        cluster_sequences = {}
        
        for cluster_id in np.unique(cluster_labels):
            # Filtramos os usuários neste cluster
            users_in_cluster = user_clusters[user_clusters['cluster'] == cluster_id]['userid'].values
            
            # Filtramos os logs apenas para usuários neste cluster
            cluster_logs = self.logs_df[self.logs_df['userid'].isin(users_in_cluster)]
            
            # Agrupamos por usuário e ordenamos por timestamp
            sequence_patterns = []
            
            for userid in users_in_cluster:
                user_logs = cluster_logs[cluster_logs['userid'] == userid].sort_values('timestamp')
                
                if len(user_logs) < 2:
                    continue
                    
                # Criamos sequência de interações (component + action)
                interactions = []
                for _, row in user_logs.iterrows():
                    interaction = f"{row['component']}_{row['action']}"
                    interactions.append(interaction)
                
                # Limitamos ao tamanho máximo
                if len(interactions) > max_length:
                    interactions = interactions[:max_length]
                    
                sequence_patterns.append(tuple(interactions))
            
            # Contamos as sequências mais comuns
            if sequence_patterns:
                counter = Counter(sequence_patterns)
                common_sequences = counter.most_common(5)
                cluster_sequences[cluster_id] = common_sequences
            else:
                cluster_sequences[cluster_id] = []
        
        return cluster_sequences
    
    def extract_narratives(self, cluster_labels, cluster_stats, distinctive_features, cluster_sequences):
        """
        Extrai narrativas significativas de cada cluster
        
        Args:
            cluster_labels: Rótulos de cluster
            cluster_stats: Estatísticas dos clusters
            distinctive_features: Características distintivas por cluster
            cluster_sequences: Sequências comuns por cluster
        """
        print("Gerando narrativas para os clusters...")
        
        narratives = {}
        
        # Mapeamento de componentes e ações para descrições mais amigáveis
        component_mapping = {
            'core': 'atividade básica do curso',
            'mod_quiz': 'quiz',
            'mod_forum': 'fórum de discussão',
            'mod_resource': 'material didático',
            'mod_folder': 'pasta de recursos',
            'mod_page': 'página de conteúdo',
            'mod_chat': 'chat',
            'mod_url': 'link externo'
        }
        
        action_mapping = {
            'viewed': 'visualização',
            'started': 'início',
            'submitted': 'submissão',
            'created': 'criação',
            'downloaded': 'download',
            'updated': 'atualização',
            'sent': 'envio'
        }
        
        # Dicionário para armazenar perfis de cluster
        cluster_profiles = {}
        
        for cluster_id in np.unique(cluster_labels):
            # Obtemos as estatísticas do cluster (cluster_stats é um DataFrame)
            stats_row = cluster_stats[cluster_stats['cluster_id'] == cluster_id].iloc[0]
            n_members = stats_row['n_members']
            
            # Obtemos características distintivas
            distinctive = distinctive_features.get(cluster_id, [])
            
            # Obtemos sequências comuns
            sequences = cluster_sequences.get(cluster_id, [])
            
            # Determinamos o perfil e características principais do cluster
            profile_features = []
            
            for feature, deviation in distinctive:
                if abs(deviation) < 0.1:  # Ignoramos desvios muito pequenos
                    continue
                    
                description = ""
                
                if "total_events" in feature and deviation > 0:
                    description = "alta participação geral"
                elif "total_events" in feature and deviation < 0:
                    description = "baixa participação geral"
                elif "quiz_ratio" in feature and deviation > 0:
                    description = "foco em quizzes e avaliações"
                elif "forum_ratio" in feature and deviation > 0:
                    description = "participação ativa em fóruns"
                elif "forum_ratio" in feature and deviation < 0:
                    description = "pouca participação em fóruns"
                elif "resource_ratio" in feature and deviation > 0:
                    description = "alto consumo de materiais didáticos"
                elif "morning_ratio" in feature and deviation > 0:
                    description = "preferência por acessos matutinos"
                elif "evening_ratio" in feature and deviation > 0:
                    description = "preferência por acessos noturnos"
                elif "avg_grade" in feature and deviation > 0:
                    description = "desempenho acadêmico acima da média"
                elif "avg_grade" in feature and deviation < 0:
                    description = "desempenho acadêmico abaixo da média"
                elif "weekday_sat" in feature or "weekday_sun" in feature and deviation > 0:
                    description = "preferência por estudar nos finais de semana"
                elif feature in ["median_time_between", "mean_time_between"] and deviation < 0:
                    description = "acessos frequentes com pequenos intervalos"
                elif feature in ["median_time_between", "mean_time_between"] and deviation > 0:
                    description = "acessos espaçados com grandes intervalos"
                
                if description:
                    profile_features.append(description)
            
            # Determinamos o perfil principal do cluster
            if not profile_features:
                profile = "Perfil neutro sem características distintivas"
            else:
                profile = "Grupo de estudantes com " + ", ".join(profile_features[:3])
            
            # Análise de sequências
            sequence_descriptions = []
            
            for seq, count in sequences:
                if count < 3:  # Ignoramos sequências muito raras
                    continue
                    
                # Convertemos a sequência para descrição legível
                seq_steps = []
                for interaction in seq:
                    try:
                        component, action = interaction.split('_')
                        component_desc = component_mapping.get(component, component)
                        action_desc = action_mapping.get(action, action)
                        seq_steps.append(f"{action_desc} de {component_desc}")
                    except:
                        seq_steps.append(interaction)
                
                if seq_steps:
                    sequence_descriptions.append({
                        "sequence": " → ".join(seq_steps[:3]) + ("..." if len(seq_steps) > 3 else ""),
                        "count": count
                    })
            
            # Montamos a narrativa final
            narrative = {
                "cluster_id": cluster_id,
                "size": n_members,
                "size_percentage": n_members / len(cluster_labels) * 100,
                "profile": profile,
                "key_features": profile_features[:5],
                "common_sequences": sequence_descriptions[:3]
            }
            
            # Adicionamos uma descrição textual personalizada
            if "alta participação" in profile:
                if "desempenho acadêmico acima da média" in profile:
                    narrative["description"] = "Grupo de estudantes engajados com alto desempenho"
                elif "desempenho acadêmico abaixo da média" in profile:
                    narrative["description"] = "Grupo de estudantes ativos mas com dificuldades de aprendizado"
                else:
                    narrative["description"] = "Grupo de estudantes muito ativos na plataforma"
            elif "baixa participação" in profile:
                if "desempenho acadêmico acima da média" in profile:
                    narrative["description"] = "Grupo de estudantes eficientes (fazem menos mas com bons resultados)"
                else:
                    narrative["description"] = "Grupo de estudantes com baixo engajamento"
            elif "foco em quizzes" in profile:
                narrative["description"] = "Grupo de estudantes focados em avaliações"
            elif "participação ativa em fóruns" in profile:
                narrative["description"] = "Grupo de estudantes colaborativos"
            else:
                narrative["description"] = "Grupo com perfil misto de participação"
            
        
        # Limitamos ao tamanho máximo
        if len(interactions) > max_length:
            interactions = interactions[:max_length]
            
        sequence_patterns.append(tuple(interactions))
    
        # Contamos as sequências mais comuns
        if sequence_patterns:
            counter = Counter(sequence_patterns)
            common_sequences = counter.most_common(5)
            cluster_sequences[cluster_id] = common_sequences
        else:
            cluster_sequences[cluster_id] = []
        
        return cluster_sequences

    def extract_narratives(self, cluster_labels, cluster_stats, distinctive_features, cluster_sequences):
        """
        Extrai narrativas significativas de cada cluster
        
        Args:
            cluster_labels: Rótulos de cluster
            cluster_stats: Estatísticas dos clusters
            distinctive_features: Características distintivas por cluster
            cluster_sequences: Sequências comuns por cluster
        """
        print("Gerando narrativas para os clusters...")
        
        narratives = {}
        
        # Mapeamento de componentes e ações para descrições mais amigáveis
        component_mapping = {
            'core': 'atividade básica do curso',
            'mod_quiz': 'quiz',
            'mod_forum': 'fórum de discussão',
            'mod_resource': 'material didático',
            'mod_folder': 'pasta de recursos',
            'mod_page': 'página de conteúdo',
            'mod_chat': 'chat',
            'mod_url': 'link externo'
        }
        
        action_mapping = {
            'viewed': 'visualização',
            'started': 'início',
            'submitted': 'submissão',
            'created': 'criação',
            'downloaded': 'download',
            'updated': 'atualização',
            'sent': 'envio'
        }
        
        # Dicionário para armazenar perfis de cluster
        cluster_profiles = {}
        
        for cluster_id in np.unique(cluster_labels):
            # Obtemos as estatísticas do cluster (cluster_stats é um DataFrame)
            stats_row = cluster_stats[cluster_stats['cluster_id'] == cluster_id].iloc[0]
            n_members = stats_row['n_members']
            
            # Obtemos características distintivas
            distinctive = distinctive_features.get(cluster_id, [])
            
            # Obtemos sequências comuns
            sequences = cluster_sequences.get(cluster_id, [])
            
            # Determinamos o perfil e características principais do cluster
            profile_features = []
            
            for feature, deviation in distinctive:
                if abs(deviation) < 0.1:  # Ignoramos desvios muito pequenos
                    continue
                    
                description = ""
                
                if "total_events" in feature and deviation > 0:
                    description = "alta participação geral"
                elif "total_events" in feature and deviation < 0:
                    description = "baixa participação geral"
                elif "quiz_ratio" in feature and deviation > 0:
                    description = "foco em quizzes e avaliações"
                elif "forum_ratio" in feature and deviation > 0:
                    description = "participação ativa em fóruns"
                elif "forum_ratio" in feature and deviation < 0:
                    description = "pouca participação em fóruns"
                elif "resource_ratio" in feature and deviation > 0:
                    description = "alto consumo de materiais didáticos"
                elif "morning_ratio" in feature and deviation > 0:
                    description = "preferência por acessos matutinos"
                elif "evening_ratio" in feature and deviation > 0:
                    description = "preferência por acessos noturnos"
                elif "avg_grade" in feature and deviation > 0:
                    description = "desempenho acadêmico acima da média"
                elif "avg_grade" in feature and deviation < 0:
                    description = "desempenho acadêmico abaixo da média"
                elif "weekday_sat" in feature or "weekday_sun" in feature and deviation > 0:
                    description = "preferência por estudar nos finais de semana"
                elif feature in ["median_time_between", "mean_time_between"] and deviation < 0:
                    description = "acessos frequentes com pequenos intervalos"
                elif feature in ["median_time_between", "mean_time_between"] and deviation > 0:
                    description = "acessos espaçados com grandes intervalos"
                
                if description:
                    profile_features.append(description)
            
            # Determinamos o perfil principal do cluster
            if not profile_features:
                profile = "Perfil neutro sem características distintivas"
            else:
                profile = "Grupo de estudantes com " + ", ".join(profile_features[:3])
            
            # Análise de sequências
            sequence_descriptions = []
            
            for seq, count in sequences:
                if count < 3:  # Ignoramos sequências muito raras
                    continue
                    
                # Convertemos a sequência para descrição legível
                seq_steps = []
                for interaction in seq:
                    try:
                        component, action = interaction.split('_')
                        component_desc = component_mapping.get(component, component)
                        action_desc = action_mapping.get(action, action)
                        seq_steps.append(f"{action_desc} de {component_desc}")
                    except:
                        seq_steps.append(interaction)
                
                if seq_steps:
                    sequence_descriptions.append({
                        "sequence": " → ".join(seq_steps[:3]) + ("..." if len(seq_steps) > 3 else ""),
                        "count": count
                    })
            
            # Montamos a narrativa final
            narrative = {
                "cluster_id": cluster_id,
                "size": n_members,
                "size_percentage": n_members / len(cluster_labels) * 100,
                "profile": profile,
                "key_features": profile_features[:5],
                "common_sequences": sequence_descriptions[:3]
            }
            
            # Adicionamos uma descrição textual personalizada
            if "alta participação" in profile:
                if "desempenho acadêmico acima da média" in profile:
                    narrative["description"] = "Grupo de estudantes engajados com alto desempenho"
                elif "desempenho acadêmico abaixo da média" in profile:
                    narrative["description"] = "Grupo de estudantes ativos mas com dificuldades de aprendizado"
                else:
                    narrative["description"] = "Grupo de estudantes muito ativos na plataforma"
            elif "baixa participação" in profile:
                if "desempenho acadêmico acima da média" in profile:
                    narrative["description"] = "Grupo de estudantes eficientes (fazem menos mas com bons resultados)"
                else:
                    narrative["description"] = "Grupo de estudantes com baixo engajamento"
            elif "foco em quizzes" in profile:
                narrative["description"] = "Grupo de estudantes focados em avaliações"
            elif "participação ativa em fóruns" in profile:
                narrative["description"] = "Grupo de estudantes colaborativos"
            else:
                narrative["description"] = "Grupo com perfil misto de participação"
            
            # Armazenamos a narrativa
            narratives[cluster_id] = narrative
            
        return narratives

    def visualize_narratives(self, narratives, output_path=None, min_cluster_size_percent=1.0):
        """
        Visualiza as narrativas extraídas dos clusters
        
        Args:
            narratives: Dicionário de narrativas por cluster
            output_path: Caminho para salvar a visualização (se None, usa o data_path)
            min_cluster_size_percent: Tamanho mínimo do cluster em porcentagem para ser visualizado
        """
        if output_path is None:
            output_path = self.data_path
            
        print("Gerando visualizações das narrativas...")
        
        # Criamos um DataFrame a partir das narrativas
        narratives_df = pd.DataFrame([
            {
                "Cluster": k,
                "Tamanho": v["size"],
                "Percentual": v["size_percentage"],
                "Perfil": v["profile"],
                "Descrição": v["description"]
            }
            for k, v in narratives.items()
        ])
        
        # Ordenamos por tamanho do cluster
        narratives_df = narratives_df.sort_values("Tamanho", ascending=False).reset_index(drop=True)
        
        # Filtramos clusters pequenos se min_cluster_size_percent > 0
        if min_cluster_size_percent > 0:
            print(f"Filtrando clusters com menos de {min_cluster_size_percent:.1f}% dos dados...")
            narratives_df = narratives_df[narratives_df["Percentual"] >= min_cluster_size_percent].reset_index(drop=True)
            print(f"Mantidos {len(narratives_df)} clusters após filtragem.")
            
        # Visualizamos a distribuição dos clusters
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x="Cluster", y="Tamanho", data=narratives_df, palette="viridis")
        
        for i, row in narratives_df.iterrows():
            ax.text(i, row["Tamanho"] + 1, f"{row['Percentual']:.1f}%", 
                ha='center', fontsize=9)
            
        plt.title("Distribuição de Estudantes por Cluster")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "cluster_distribution.png"))
        
        # Geramos um relatório HTML com as narrativas
        html_content = """
        <html>
        <head>
            <title>Narrativas de Clusters de Estudantes</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .container { 
                    max-width: 1200px; 
                    margin: 0 auto; 
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
                    gap: 20px;
                }
                .cluster-card { 
                    border: 1px solid #ddd; 
                    border-radius: 8px; 
                    padding: 15px; 
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    height: 100%;
                }
                .cluster-header { 
                    display: flex; 
                    justify-content: space-between; 
                    border-bottom: 1px solid #eee;
                    padding-bottom: 10px;
                    margin-bottom: 10px;
                }
                .cluster-title { font-size: 18px; font-weight: bold; }
                .cluster-size { font-size: 16px; color: #666; }
                .cluster-description { font-size: 16px; margin: 10px 0; }
                .cluster-features { margin: 10px 0; }
                .feature-tag {
                    display: inline-block;
                    background: #f0f0f0;
                    padding: 5px 10px;
                    border-radius: 15px;
                    margin-right: 5px;
                    margin-bottom: 5px;
                    font-size: 12px;
                }
                .sequences-title { font-weight: bold; margin-top: 15px; }
                .sequence-item { margin: 5px 0; font-size: 14px; }
                .header-info {
                    margin-bottom: 20px;
                    padding: 10px;
                    background-color: #f8f9fa;
                    border-radius: 5px;
                }
            </style>
        </head>
        <body>
            <div class="header-info">
                <h1>Narrativas de Clusters de Estudantes</h1>
                <p>Análise baseada em algoritmo híbrido de clusterização (K-means + SOM)</p>
                <p>Mostrando {len(narratives_df)} clusters (mínimo de {min_cluster_size_percent:.1f}% de estudantes por cluster)</p>
            </div>
            <div class="container">
        """
        
        for _, row in narratives_df.iterrows():
            cluster_id = row["Cluster"]
            narrative = narratives[cluster_id]
            
            html_content += f"""
            <div class="cluster-card">
                <div class="cluster-header">
                    <div class="cluster-title">Cluster {cluster_id}</div>
                    <div class="cluster-size">{narrative['size']} estudantes ({narrative['size_percentage']:.1f}%)</div>
                </div>
                <div class="cluster-description">{narrative['description']}</div>
                <div class="cluster-profile">{narrative['profile']}</div>
                <div class="cluster-features">
            """
            
            # Adicionar as características principais
            for feature in narrative['key_features']:
                html_content += f'<span class="feature-tag">{feature}</span>'
                
            html_content += """
                </div>
            """
            
            # Adicionar as sequências comuns, se houver
            if narrative['common_sequences']:
                html_content += """
                <div class="sequences-title">Padrões de navegação comuns:</div>
                <ul>
                """
                for seq in narrative['common_sequences']:
                    html_content += f"""<li class="sequence-item">{seq['sequence']} (observado {seq['count']} vezes)</li>"""
                html_content += """
                </ul>
                """
                
            html_content += """
            </div>
            """
        
        # Fechamos o container e o HTML
        html_content += """
            </div>
        </body>
        </html>
        """
        
        # Salvamos o relatório HTML
        with open(os.path.join(output_path, "narrativas_clusters.html"), "w", encoding="utf-8") as f:
            f.write(html_content)
        
        # Também salvamos as narrativas em CSV para análise posterior
        narratives_df.to_csv(os.path.join(output_path, "narrativas_clusters.csv"), index=False, encoding="utf-8")
        
        print(f"Visualizações salvas em: {output_path}")
        return narratives_df

    def run_full_pipeline(self, n_clusters=None, map_size=None, max_som_clusters=20, min_cluster_size=10, min_cluster_size_percent=1.0):
        """
        Executa o pipeline completo de análise
        
        Args:
            n_clusters: Número de clusters para K-means (se None, busca o melhor)
            map_size: Tamanho do mapa SOM (se None, calcula automaticamente)
            max_som_clusters: Número máximo de clusters SOM a considerar
            min_cluster_size: Tamanho mínimo do cluster para ser considerado na clusterização híbrida
            min_cluster_size_percent: Tamanho mínimo do cluster em porcentagem para ser visualizado
        """
        print("Executando pipeline completo do sistema híbrido...")
        
        # 1. Carregamos os dados
        self.load_data()
        
        # 2. Extraímos características
        self.extract_features()
        
        # 3. Normalizamos as características
        self.normalize_features()
        
        # 4. Aplicamos PCA para redução de dimensionalidade
        self.apply_pca(n_components=5)
        
        # 5. Aplicamos K-means
        self.apply_kmeans_clustering(n_clusters=n_clusters)
        
        # 6. Aplicamos SOM
        self.apply_som_clustering(map_size=map_size, max_clusters=max_som_clusters)
        
        # 7. Combinamos os resultados para criar clusters híbridos
        hybrid_df = self.apply_hybrid_clustering(min_cluster_size=min_cluster_size)
        
        # 8. Analisamos os clusters
        stats_df, distinctive_features = self.analyze_clusters(hybrid_df['hybrid_cluster'].values)
        
        # 9. Analisamos sequências por cluster
        cluster_sequences = self.analyze_sequences(hybrid_df['hybrid_cluster'].values)
        
        # 10. Extraímos narrativas
        narratives = self.extract_narratives(
            hybrid_df['hybrid_cluster'].values,
            stats_df, 
            distinctive_features,
            cluster_sequences
        )
        
        # 11. Visualizamos as narrativas
        narratives_df = self.visualize_narratives(narratives, output_path=None, min_cluster_size_percent=min_cluster_size_percent)
        
        print("Pipeline completo executado com sucesso!")
        return hybrid_df, narratives

# Função principal para executar o código
if __name__ == "__main__":
    data_path = "."  # Diretório atual
    
    # Inicializamos o modelo híbrido
    model = HybridModel(data_path)
    
    # Executamos o pipeline completo
    hybrid_df, narratives = model.run_full_pipeline()
    
    print("Processo concluído. Verifique os arquivos gerados no diretório de dados.")

