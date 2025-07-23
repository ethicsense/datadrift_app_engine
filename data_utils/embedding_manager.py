import torch
import clip
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import DBSCAN, AgglomerativeClustering
import os
from PIL import Image
import hashlib


class EmbeddingManager:
    """임베딩 추출 및 분석을 관리하는 클래스"""
    
    def __init__(self, device=None):
        """
        EmbeddingManager 초기화
        
        Args:
            device: 사용할 디바이스 (None이면 자동 선택)
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model = None
        self.preprocess = None
        print(f"Using device: {self.device}")
    
    def load_model(self, model_name="ViT-B/16"):
        """
        CLIP 모델을 로드합니다.
        
        Args:
            model_name: 사용할 CLIP 모델명
        """
        print(f"Loading Embedding Model: {model_name}")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        print("Model loaded successfully")
    
    def extract_embedding(self, file_path):
        """
        단일 파일의 임베딩을 추출하고 해시와 함께 결과를 반환합니다.
        
        Args:
            file_path: 분석할 파일 경로
        
        Returns:
            dict: 임베딩 결과 (해시 포함) 또는 None
        """
        if self.model is None:
            self.load_model()
        
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} does not exist.")
            return None
        
        try:
            with Image.open(file_path) as img:
                # 임베딩 추출
                with torch.no_grad():
                    print(f"Extracting embedding for {file_path}...")
                    input_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
                    embedding = self.model.encode_image(input_tensor).cpu().numpy().flatten()
                
                # 해시 계산
                import hashlib
                hasher = hashlib.md5()
                with open(file_path, 'rb') as f:
                    buf = f.read()
                    hasher.update(buf)
                file_hash = hasher.hexdigest()
                
                return {
                    'hash': file_hash,
                    'path': os.path.abspath(file_path),
                    'embedding': embedding.tolist()
                }
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
    
    def perform_clustering(self, embeddings_data, file_names, file_paths, n_clusters=None, method='kmeans'):
        """
        임베딩을 기반으로 클러스터링 분석을 수행합니다.
        
        Args:
            embeddings_data: 임베딩 데이터 리스트
            file_names: 파일명 리스트
            file_paths: 파일 경로 리스트
            n_clusters: 클러스터 수 (None이면 자동 결정)
            method: 클러스터링 방법 ('kmeans', 'dbscan', 'hierarchical')
        
        Returns:
            dict: 클러스터링 결과
        """
        print(f"Starting clustering analysis with method: {method}")
        
        if len(embeddings_data) < 2:
            print("Not enough data for clustering")
            return None
        
        embeddings_array = np.array(embeddings_data)
        print(f"Loaded {len(embeddings_data)} embeddings with shape: {embeddings_array.shape}")
        
        # PCA를 사용하여 차원 축소 (시각화용)
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings_array)
        
        # 클러스터링 수행
        clustering_result = self._apply_clustering(embeddings_2d, method, n_clusters)
        
        # Centroid 계산 및 유사도 점수 계산
        centroids, centroid_similarities = self._calculate_centroids_and_similarities(
            embeddings_array, clustering_result['labels'], clustering_result['n_clusters']
        )
        
        # 클러스터별 통계 계산
        cluster_stats = self._calculate_cluster_stats(clustering_result['labels'], file_names, file_paths, centroid_similarities)
        
        # 결과 저장
        clustering_results = {
            'method': method,
            'n_clusters': clustering_result['n_clusters'],
            'cluster_labels': clustering_result['labels'].tolist(),
            'file_names': file_names,
            'file_paths': file_paths,
            'embeddings_2d': embeddings_2d.tolist(),
            'pca_components': pca.components_.tolist(),
            'pca_explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'cluster_stats': cluster_stats,
            'centroids': centroids.tolist(),
            'centroid_similarities': centroid_similarities
        }
        
        # 결과 출력
        print(f"\nClustering Results:")
        print(f"Method: {method}")
        print(f"Number of clusters: {clustering_result['n_clusters']}")
        print(f"Total samples: {len(embeddings_data)}")
        
        for i in range(clustering_result['n_clusters']):
            cluster_size = cluster_stats[f'cluster_{i}']['size']
            avg_similarity = np.mean(centroid_similarities[f'cluster_{i}'])
            print(f"Cluster {i}: {cluster_size} samples ({cluster_size/len(embeddings_data)*100:.1f}%) - Avg similarity: {avg_similarity:.4f}")
        
        return {
            'clustering_results': clustering_results,
            'method': method,
            'n_clusters': clustering_result['n_clusters'],
            'total_samples': len(embeddings_data),
            'cluster_stats': cluster_stats,
            'centroids': centroids,
            'centroid_similarities': centroid_similarities
        }
    
    def _apply_clustering(self, embeddings_2d, method, n_clusters):
        """클러스터링 알고리즘을 적용합니다."""
        if method == 'kmeans':
            if n_clusters is None:
                # Elbow method로 최적 클러스터 수 결정
                print("Determining optimal number of clusters using Elbow method...")
                model = KMeans(random_state=42)
                visualizer = KElbowVisualizer(model, k=(1, min(10, len(embeddings_2d))))
                visualizer.fit(embeddings_2d)
                n_clusters = visualizer.elbow_value_
                print(f"Optimal number of clusters: {n_clusters}")
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings_2d)
            
        elif method == 'dbscan':
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            cluster_labels = dbscan.fit_predict(embeddings_2d)
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            
        elif method == 'hierarchical':
            if n_clusters is None:
                n_clusters = min(5, len(embeddings_2d) // 2)
            
            hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
            cluster_labels = hierarchical.fit_predict(embeddings_2d)
        
        return {
            'labels': cluster_labels,
            'n_clusters': n_clusters
        }
    
    def _calculate_cluster_stats(self, cluster_labels, file_names, file_paths, centroid_similarities=None):
        """클러스터별 통계를 계산합니다."""
        cluster_stats = {}
        n_clusters = len(set(cluster_labels))
        
        for i in range(n_clusters):
            cluster_indices = np.where(cluster_labels == i)[0]
            
            # 기본 통계
            stats = {
                'size': len(cluster_indices),
                'files': [file_names[idx] for idx in cluster_indices],
                'paths': [file_paths[idx] for idx in cluster_indices]
            }
            
            # Centroid 유사도 정보 추가
            if centroid_similarities and f'cluster_{i}' in centroid_similarities:
                similarities = centroid_similarities[f'cluster_{i}']
                stats['centroid_similarities'] = similarities
                stats['avg_similarity'] = np.mean(similarities) if similarities else 0.0
                stats['min_similarity'] = np.min(similarities) if similarities else 0.0
                stats['max_similarity'] = np.max(similarities) if similarities else 0.0
                
                # 유사도가 높은 파일들 (상위 5개)
                if similarities:
                    sorted_indices = np.argsort(similarities)[::-1]  # 내림차순 정렬
                    top_indices = sorted_indices[:5]
                    stats['top_similar_files'] = [
                        {
                            'file': file_names[cluster_indices[idx]],
                            'similarity': similarities[idx]
                        }
                        for idx in top_indices
                    ]
                else:
                    stats['top_similar_files'] = []
            
            cluster_stats[f'cluster_{i}'] = stats
        
        return cluster_stats
    
    def _calculate_centroids_and_similarities(self, embeddings_array, cluster_labels, n_clusters):
        """
        각 클러스터의 centroid를 계산하고 centroid 기준 유사도 점수를 계산합니다.
        
        Args:
            embeddings_array: 임베딩 배열
            cluster_labels: 클러스터 레이블
            n_clusters: 클러스터 수
        
        Returns:
            tuple: (centroids, centroid_similarities)
        """
        centroids = np.zeros((n_clusters, embeddings_array.shape[1]))
        centroid_similarities = {}
        
        for i in range(n_clusters):
            # 클러스터 i에 속한 샘플들의 인덱스
            cluster_indices = np.where(cluster_labels == i)[0]
            
            if len(cluster_indices) > 0:
                # 클러스터 i의 centroid 계산 (평균)
                cluster_embeddings = embeddings_array[cluster_indices]
                centroids[i] = np.mean(cluster_embeddings, axis=0)
                
                # 각 샘플과 centroid 간의 유사도 계산
                similarities = []
                for idx in cluster_indices:
                    sample_embedding = embeddings_array[idx]
                    similarity = self._calculate_cosine_similarity(sample_embedding, centroids[i])
                    similarities.append(similarity)
                
                centroid_similarities[f'cluster_{i}'] = similarities
            else:
                # 빈 클러스터인 경우
                centroids[i] = np.zeros(embeddings_array.shape[1])
                centroid_similarities[f'cluster_{i}'] = []
        
        return centroids, centroid_similarities
    
    def _calculate_cosine_similarity(self, vec1, vec2):
        """
        두 벡터 간의 코사인 유사도를 계산합니다.
        
        Args:
            vec1: 첫 번째 벡터
            vec2: 두 번째 벡터
        
        Returns:
            float: 코사인 유사도 (0~1)
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)


def run_clustering_analysis(directories, formats, n_clusters=None, method='kmeans', device=None):
    """
    클러스터링 분석을 실행하는 편의 함수
    
    Args:
        directories: 분석할 디렉토리 리스트
        formats: 이미지 포맷 리스트
        n_clusters: 클러스터 수
        method: 클러스터링 방법
        device: 사용할 디바이스
    
    Returns:
        dict: 클러스터링 결과
    """
    manager = EmbeddingManager(device)
    results = {}
    
    for directory in directories:
        if not os.path.exists(directory):
            print(f"Error: Directory {directory} does not exist.")
            continue
        
        # 캐시에서 임베딩 데이터 로드
        from cache_utils.cache_manager import get_cached_analysis_data
        embedding_cache = get_cached_analysis_data(directory, "image_drift_content")
        
        if not embedding_cache:
            print(f"No embedding data found in cache for {directory}")
            print("Please run embedding extraction first")
            continue
        
        # 임베딩 데이터 수집
        embeddings = []
        file_paths = []
        file_names = []
        
        for file_name, data in embedding_cache.items():
            if 'embedding' in data:
                embeddings.append(data['embedding'])
                file_paths.append(data['path'])
                file_names.append(file_name)
        
        if len(embeddings) < 2:
            print(f"Not enough data for clustering in {directory}")
            continue
        
        # 클러스터링 분석 수행
        clustering_result = manager.perform_clustering(embeddings, file_names, file_paths, n_clusters, method)
        
        if clustering_result:
            results[directory] = clustering_result
    
    return results 