import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import base64
from io import BytesIO
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# 캐시 매니저 import
from cache_utils.cache_manager import get_cached_analysis_data

# XAI 가이드라인 생성기 import
try:
    from .xai_guideline_generator import create_xai_guideline
except ImportError:
    # 상대 import가 실패할 경우 절대 import 시도
    try:
        from xai_guideline_generator import create_xai_guideline
    except ImportError:
        create_xai_guideline = None

'''
📊 {dataset_name} 통합 분석 리포트
├── 📊 Dataset Information & Statistics (데이터베이스 분석 결과)
├── 🖼️ Image Analysis Results (이미지 분석 결과)
│   ├── 📈 Summary Statistics
│   ├── 📋 Format Distribution
│   ├── 📊 Visualizations
│   ├── 🖼️ Sample Images
│   ├── 📊 Detailed Statistics
│   ├── 🧠 Embedding Information
│   └── 📐 Resolution Information
└── 🔍 Data Drift Analysis Results (드리프트 분석 결과)
'''

class ImageAnalysisReport:
    def __init__(self, directory):
        self.directory = directory
        # 데이터를 컨텐츠별로 분리하여 저장
        self.attr_data = self.load_attribute_data()
        self.embed_data = self.load_embedding_data()
        self.xai_data = self.load_xai_data()
        self.clustering_data = self.load_clustering_data()
        
        # 기존 호환성을 위한 병합 데이터 (필요시에만 사용)
        self.data = self._merge_data_for_compatibility()
        
    def load_attribute_data(self):
        """속성 분석 데이터를 로드합니다."""
        cached_data = get_cached_analysis_data(self.directory, "attribute_analysis")
        if cached_data is not None:
            print(f"📊 Loaded attribute data: {len(cached_data)} files")
            return cached_data
            
        print("ℹ️  No attribute analysis data found in cache")
        return {}
    
    def load_embedding_data(self):
        """임베딩 분석 데이터를 로드합니다."""
        cached_data = get_cached_analysis_data(self.directory, "embedding_analysis")
        if cached_data is not None:
            print(f"📊 Loaded embedding data: {len(cached_data)} files")
            return cached_data
            
        print("ℹ️  No embedding analysis data found in cache")
        return {}
    
    def load_clustering_data(self):
        """클러스터링 데이터를 로드합니다."""
        # 1. 전용 클러스터링 분석 데이터 확인
        cached_data = get_cached_analysis_data(self.directory, "clustering_analysis")
        if cached_data is not None:
            print(f"📊 Loaded clustering data : ")
            return cached_data
        
        # 2. 임베딩 분석 결과에서 클러스터링 정보 추출
        cached_data = get_cached_analysis_data(self.directory, "embedding_analysis")
        if cached_data is not None:
            # 클러스터링 정보 추출
            clustering_data = {}
            for filename, data in cached_data.items():
                if isinstance(data, dict) and 'cluster' in data:
                    clustering_data[filename] = {
                        'cluster': data['cluster'],
                        'embedding': data.get('embedding', None)
                    }
            
            if clustering_data:
                print(f"📊 Loaded clustering data from embedding analysis: {len(clustering_data)} files")
                return clustering_data
            
        print("ℹ️  No clustering data found in cache")
        return {}
    
    def load_xai_data(self):
        """XAI 분석 데이터를 로드합니다."""
        cached_data = get_cached_analysis_data(self.directory, "xai_analysis")
        if cached_data is not None:
            print(f"📊 Loaded XAI data: {len(cached_data)} files")
            return cached_data
        print("ℹ️  No XAI analysis data found in cache")
        return {}
    
    def _merge_data_for_compatibility(self):
        """기존 호환성을 위해 데이터를 병합합니다."""
        merged_data = self.attr_data.copy()
        
        # 임베딩 데이터 병합
        for filename, embed_item in self.embed_data.items():
            if filename in merged_data:
                merged_data[filename].update(embed_item)
            else:
                merged_data[filename] = embed_item
        
        return merged_data
    
    def get_combined_data(self):
        """모든 분석 데이터를 병합하여 반환합니다."""
        combined_data = {}
        
        # 속성 데이터 추가
        for filename, attr_item in self.attr_data.items():
            combined_data[filename] = attr_item.copy()
        
        # 임베딩 데이터 추가
        for filename, embed_item in self.embed_data.items():
            if filename in combined_data:
                combined_data[filename].update(embed_item)
            else:
                combined_data[filename] = embed_item
        
        # XAI 데이터 추가
        for filename, xai_item in self.xai_data.items():
            if filename in combined_data:
                combined_data[filename]['xai_analysis'] = xai_item
            else:
                combined_data[filename] = {'xai_analysis': xai_item}
        
        return combined_data
    
    def create_summary_stats(self):
        """기본 통계 정보를 생성합니다."""
        # 속성 데이터가 있는지 확인
        if not self.attr_data:
            return {}
        
        total_images = len(self.attr_data)
        total_size = sum(item['size'] for item in self.attr_data.values())
        
        # 형식별 통계
        formats = {}
        resolutions = {}
        sizes = []
        noise_levels = []
        sharpness_values = []
        
        for item in self.attr_data.values():
            # 형식별 카운트
            fmt = item['format']
            formats[fmt] = formats.get(fmt, 0) + 1
            
            # 해상도별 카운트
            res = item['resolution']
            resolutions[res] = resolutions.get(res, 0) + 1
            
            # 수치 데이터
            sizes.append(item['size'])
            noise_levels.append(item['noise_level'])
            sharpness_values.append(item['sharpness'])
        
        return {
            'total_images': total_images,
            'total_size_mb': total_size,
            'avg_size_mb': np.mean(sizes),
            'formats': formats,
            'resolutions': resolutions,
            'size_stats': {
                'min': np.min(sizes),
                'max': np.max(sizes),
                'mean': np.mean(sizes),
                'std': np.std(sizes)
            },
            'noise_stats': {
                'min': np.min(noise_levels),
                'max': np.max(noise_levels),
                'mean': np.mean(noise_levels),
                'std': np.std(noise_levels)
            },
            'sharpness_stats': {
                'min': np.min(sharpness_values),
                'max': np.max(sharpness_values),
                'mean': np.mean(sharpness_values),
                'std': np.std(sharpness_values)
            }
        }
    
    def create_basic_attribute_charts(self):
        """기본 속성 관련 차트들을 생성합니다."""
        charts = {}
        
        # 1. 파일 크기 분포 히스토그램
        sizes = [item['size'] for item in self.data.values()]
        plt.figure(figsize=(10, 6))
        plt.hist(sizes, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('File Size Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('File Size (MB)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        charts['size_distribution'] = self.fig_to_base64()
        plt.close()
        
        # 2. 형식별 분포 파이 차트
        formats = {}
        for item in self.data.values():
            fmt = item['format']
            formats[fmt] = formats.get(fmt, 0) + 1
        
        if formats:
            plt.figure(figsize=(8, 8))
            plt.pie(formats.values(), labels=formats.keys(), autopct='%1.1f%%', startangle=90)
            plt.title('Image Format Distribution', fontsize=14, fontweight='bold')
            charts['format_distribution'] = self.fig_to_base64()
            plt.close()
        
        # 3. 해상도별 분포 (상위 10개)
        resolutions = {}
        for item in self.data.values():
            res = item['resolution']
            resolutions[res] = resolutions.get(res, 0) + 1
        
        top_resolutions = dict(sorted(resolutions.items(), key=lambda x: x[1], reverse=True)[:10])
        
        if top_resolutions:
            plt.figure(figsize=(12, 6))
            plt.bar(range(len(top_resolutions)), list(top_resolutions.values()), color='lightgreen')
            plt.title('Top 10 Resolution Distribution', fontsize=14, fontweight='bold')
            plt.xlabel('Resolution')
            plt.ylabel('Count')
            plt.xticks(range(len(top_resolutions)), list(top_resolutions.keys()), rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
            charts['resolution_distribution'] = self.fig_to_base64()
            plt.close()
        
        return charts
    
    def create_quality_attribute_charts(self):
        """이미지 품질 속성 관련 차트들을 생성합니다."""
        charts = {}
        
        # 노이즈 vs 선명도 산점도
        noise_levels = [item['noise_level'] for item in self.data.values()]
        sharpness_values = [item['sharpness'] for item in self.data.values()]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(noise_levels, sharpness_values, alpha=0.6, color='coral')
        plt.title('Noise Level vs Sharpness', fontsize=14, fontweight='bold')
        plt.xlabel('Noise Level')
        plt.ylabel('Sharpness')
        plt.grid(True, alpha=0.3)
        charts['noise_vs_sharpness'] = self.fig_to_base64()
        plt.close()
        
        return charts
    
    def create_embedding_charts(self):
        """임베딩 분석 관련 차트들을 생성합니다."""
        charts = {}
        
        # 임베딩 공간 시각화 (PCA)
        if self.embed_data and len(self.embed_data) > 1:
            embeddings = np.array([item['embedding'] for item in self.embed_data.values()])
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(embeddings)
            
            plt.figure(figsize=(10, 8))
            plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6, color='purple')
            plt.title('Image Embeddings (PCA 2D)', fontsize=14, fontweight='bold')
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            plt.grid(True, alpha=0.3)
            charts['embeddings_pca'] = self.fig_to_base64()
            plt.close()
        
        return charts
    
    def create_clustering_charts(self):
        """클러스터링 분석 관련 차트들을 생성합니다."""
        charts = {}
        
        if self.clustering_data and 'embeddings_2d' in self.clustering_data:
            embeddings_2d = np.array(self.clustering_data['embeddings_2d'])
            cluster_labels = np.array(self.clustering_data['cluster_labels'])
            method = self.clustering_data.get('method', 'Unknown')
            n_clusters = self.clustering_data.get('n_clusters', 0)
            
            # 클러스터링 결과 시각화
            plt.figure(figsize=(12, 8))
            scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                c=cluster_labels, cmap='viridis', alpha=0.6)
            
            # 클러스터 중심점 표시 (클러스터별 색상으로 구분)
            if 'centroids' in self.clustering_data and self.clustering_data['centroids']:
                centroids_2d = np.array(self.clustering_data['centroids'])
                
                # 클러스터별 색상 정의 (viridis 컬러맵 사용)
                cluster_colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))
                
                # 2D 센트로이드 표시 (클러스터별 색상, X 마커)
                for i in range(n_clusters):
                    plt.scatter(centroids_2d[i, 0], centroids_2d[i, 1], 
                               c=[cluster_colors[i]], marker='x', s=200, linewidths=3, 
                               label='2D Centroid' if i == 0 else "")
                
                # 고차원 센트로이드를 PCA로 축소한 좌표 표시 (클러스터별 색상, O 마커)
                if 'centroids_high_dim' in self.clustering_data and 'pca_components' in self.clustering_data:
                    try:
                        # 고차원 센트로이드 가져오기
                        centroids_high_dim = np.array(self.clustering_data['centroids_high_dim'])
                        pca_components = np.array(self.clustering_data['pca_components'])
                        
                        # 고차원 센트로이드를 2D로 축소
                        centroids_high_dim_2d = np.dot(centroids_high_dim, pca_components.T)
                        
                        # 고차원 센트로이드 PCA 축소 좌표 표시 (클러스터별 색상, O 마커)
                        for i in range(n_clusters):
                            plt.scatter(centroids_high_dim_2d[i, 0], centroids_high_dim_2d[i, 1], 
                                       c=[cluster_colors[i]], marker='o', s=150, linewidths=2, alpha=0.8,
                                       label='High-Dim Centroid' if i == 0 else "")
                        
                        # 클러스터 번호와 거리 정보를 클러스터 외곽에 표시
                        for i in range(n_clusters):
                            # 2D 센트로이드와 고차원 센트로이드 PCA 축소 좌표 간의 거리 계산
                            distance = np.linalg.norm(centroids_2d[i] - centroids_high_dim_2d[i])
                            
                            # 거리에 따른 색상 변경
                            if distance > 0.5:
                                text_color = 'red'  # 큰 차이
                                alpha = 0.9
                            elif distance > 0.2:
                                text_color = 'orange'  # 중간 차이
                                alpha = 0.8
                            else:
                                text_color = 'green'  # 작은 차이
                                alpha = 0.7
                            
                            # 클러스터 외곽에 레이블 배치를 위한 위치 계산
                            cluster_indices = np.where(cluster_labels == i)[0]
                            cluster_points = embeddings_2d[cluster_indices]
                            
                            if len(cluster_points) > 0:
                                # 클러스터의 중심점 계산
                                cluster_center = np.mean(cluster_points, axis=0)
                                
                                # 클러스터의 반지름 계산 (중심에서 가장 먼 점까지의 거리)
                                distances_from_center = np.linalg.norm(cluster_points - cluster_center, axis=1)
                                cluster_radius = np.max(distances_from_center)
                                
                                # 레이블을 클러스터 외곽에 배치 (반지름의 1.3배 거리)
                                label_distance = cluster_radius * 1.3
                                
                                # 중심점에서 가장 멀리 있는 방향으로 레이블 배치
                                if cluster_radius > 0:
                                    # 가장 멀리 있는 점의 방향 계산
                                    max_dist_idx = np.argmax(distances_from_center)
                                    direction = cluster_points[max_dist_idx] - cluster_center
                                    direction = direction / np.linalg.norm(direction)
                                    
                                    # 레이블 위치 계산
                                    label_x = cluster_center[0] + direction[0] * label_distance
                                    label_y = cluster_center[1] + direction[1] * label_distance
                                else:
                                    # 클러스터가 한 점인 경우 센트로이드 근처에 배치
                                    label_x = centroids_2d[i, 0] + 0.1
                                    label_y = centroids_2d[i, 1] + 0.1
                                
                                # 통합된 레이블 표시 (클러스터 번호 + 거리)
                                plt.annotate(f'C{i} (d={distance:.3f})', 
                                           (label_x, label_y),
                                           xytext=(0, 0), textcoords='offset points',
                                           fontsize=10, color=text_color, fontweight='bold',
                                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=alpha, edgecolor=cluster_colors[i], linewidth=1.5))
                            
                            # 두 센트로이드를 연결하는 선 그리기 (클러스터별 색상)
                            plt.plot([centroids_2d[i, 0], centroids_high_dim_2d[i, 0]], 
                                   [centroids_2d[i, 1], centroids_high_dim_2d[i, 1]], 
                                   color=cluster_colors[i], linestyle='--', alpha=0.6, linewidth=1.5)
                        
                        title_text = f'Clustering Results - {method.upper()} ({n_clusters} clusters)\nX: 2D Centroids, O: High-Dim Centroids (PCA) - Same color = Same cluster'
                        
                    except Exception as e:
                        print(f"⚠️ Error projecting high-dimensional centroids: {e}")
                        # 고차원 센트로이드 투영 실패 시 기본 2D 센트로이드만 표시
                        for i in range(n_clusters):
                            plt.scatter(centroids_2d[i, 0], centroids_2d[i, 1], 
                                       c=[cluster_colors[i]], marker='x', s=200, linewidths=3, 
                                       label=f'C{i} Centroid' if i == 0 else "")
                        
                        # 클러스터 번호를 클러스터 외곽에 표시
                        for i in range(n_clusters):
                            cluster_indices = np.where(cluster_labels == i)[0]
                            cluster_points = embeddings_2d[cluster_indices]
                            
                            if len(cluster_points) > 0:
                                cluster_center = np.mean(cluster_points, axis=0)
                                distances_from_center = np.linalg.norm(cluster_points - cluster_center, axis=1)
                                cluster_radius = np.max(distances_from_center)
                                label_distance = cluster_radius * 1.3
                                
                                if cluster_radius > 0:
                                    max_dist_idx = np.argmax(distances_from_center)
                                    direction = cluster_points[max_dist_idx] - cluster_center
                                    direction = direction / np.linalg.norm(direction)
                                    label_x = cluster_center[0] + direction[0] * label_distance
                                    label_y = cluster_center[1] + direction[1] * label_distance
                                else:
                                    label_x = centroids_2d[i, 0] + 0.1
                                    label_y = centroids_2d[i, 1] + 0.1
                                
                                plt.annotate(f'C{i}', 
                                           (label_x, label_y),
                                           xytext=(0, 0), textcoords='offset points',
                                           fontsize=10, fontweight='bold', color=cluster_colors[i],
                                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor=cluster_colors[i], linewidth=1.5))
                        
                        title_text = f'Clustering Results - {method.upper()} ({n_clusters} clusters)\n2D Density-Weighted Centroids (X markers)'
                else:
                    # 고차원 센트로이드 데이터가 없는 경우 기본 표시
                    cluster_colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))
                    
                    for i in range(n_clusters):
                        plt.scatter(centroids_2d[i, 0], centroids_2d[i, 1], 
                                   c=[cluster_colors[i]], marker='x', s=200, linewidths=3, 
                                   label=f'C{i} Centroid' if i == 0 else "")
                    
                    # 클러스터 번호를 클러스터 외곽에 표시
                    for i in range(n_clusters):
                        cluster_indices = np.where(cluster_labels == i)[0]
                        cluster_points = embeddings_2d[cluster_indices]
                        
                        if len(cluster_points) > 0:
                            cluster_center = np.mean(cluster_points, axis=0)
                            distances_from_center = np.linalg.norm(cluster_points - cluster_center, axis=1)
                            cluster_radius = np.max(distances_from_center)
                            label_distance = cluster_radius * 1.3
                            
                            if cluster_radius > 0:
                                max_dist_idx = np.argmax(distances_from_center)
                                direction = cluster_points[max_dist_idx] - cluster_center
                                direction = direction / np.linalg.norm(direction)
                                label_x = cluster_center[0] + direction[0] * label_distance
                                label_y = cluster_center[1] + direction[1] * label_distance
                            else:
                                label_x = centroids_2d[i, 0] + 0.1
                                label_y = centroids_2d[i, 1] + 0.1
                            
                            plt.annotate(f'C{i}', 
                                       (label_x, label_y),
                                       xytext=(0, 0), textcoords='offset points',
                                       fontsize=10, fontweight='bold', color=cluster_colors[i],
                                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor=cluster_colors[i], linewidth=1.5))
                    
                    title_text = f'Clustering Results - {method.upper()} ({n_clusters} clusters)\n2D Density-Weighted Centroids (X markers)'
            
            plt.title(title_text, fontsize=14, fontweight='bold')
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.colorbar(scatter, label='Cluster')
            plt.legend()
            plt.grid(True, alpha=0.3)
            charts['clustering_results'] = self.fig_to_base64()
            plt.close()
            
            # 클러스터 크기 분포
            if 'cluster_stats' in self.clustering_data:
                cluster_sizes = []
                cluster_names = []
                for i in range(n_clusters):
                    cluster_key = f'cluster_{i}'
                    if cluster_key in self.clustering_data['cluster_stats']:
                        cluster_sizes.append(self.clustering_data['cluster_stats'][cluster_key]['size'])
                        cluster_names.append(f'Cluster {i}')
                
                plt.figure(figsize=(10, 6))
                plt.bar(cluster_names, cluster_sizes, color='lightcoral', alpha=0.7)
                plt.title('Cluster Size Distribution', fontsize=14, fontweight='bold')
                plt.xlabel('Cluster')
                plt.ylabel('Number of Images')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                
                # 값 표시
                for i, v in enumerate(cluster_sizes):
                    plt.text(i, v + max(cluster_sizes) * 0.01, str(v), ha='center', va='bottom')
                
                charts['cluster_size_distribution'] = self.fig_to_base64()
                plt.close()
        
        return charts
    
    def create_clustering_summary(self):
        """클러스터링 분석 결과 요약을 생성합니다."""
        if not self.clustering_data:
            return {}
        
        method = self.clustering_data.get('method', 'Unknown')
        n_clusters = self.clustering_data.get('n_clusters', 0)
        total_samples = len(self.clustering_data.get('file_names', []))
        
        cluster_stats = self.clustering_data.get('cluster_stats', {})
        cluster_summary = []
        
        for i in range(n_clusters):
            cluster_key = f'cluster_{i}'
            if cluster_key in cluster_stats:
                size = cluster_stats[cluster_key]['size']
                percentage = (size / total_samples * 100) if total_samples > 0 else 0
                cluster_summary.append({
                    'cluster_id': i,
                    'size': size,
                    'percentage': percentage,
                    'sample_files': cluster_stats[cluster_key]['files'][:5]  # 상위 5개 파일만
                })
        
        return {
            'method': method,
            'n_clusters': n_clusters,
            'total_samples': total_samples,
            'cluster_summary': cluster_summary
        }
    
    def create_xai_visualizations(self):
        """XAI 분석 결과를 시각화합니다. (클러스터별 대표 이미지만) - 최적화됨"""
        if not self.xai_data:
            print("ℹ️  No XAI data found. XAI visualizations skipped.")
            return {}
        
        try:
            from data_utils.xai_visualizer import XAIVisualizer
            
            visualizer = XAIVisualizer()
            
            # 클러스터별 대표 이미지 선택
            representative_images = self._select_representative_images()
            
            if not representative_images:
                print("ℹ️  No representative images found. Skipping XAI visualizations.")
                return {}
            
            print(f"🔍 Processing {len(representative_images)} representative XAI analysis results...")
            
            # 배치 처리를 위한 데이터 준비
            xai_batch = []
            for filename in representative_images:
                if filename in self.xai_data:
                    xai_result = self.xai_data[filename]
                    if isinstance(xai_result, dict):
                        xai_batch.append((filename, xai_result))
                        print(f"  📋 Added {filename} to batch processing")
                    else:
                        print(f"  ⚠️  Skipping {filename}: not a dictionary")
                else:
                    print(f"  ⚠️  Representative image {filename} not found in XAI data")
            
            if not xai_batch:
                print("ℹ️  No valid XAI data for batch processing.")
                return {}
            
            # 배치 시각화 생성 (병렬 처리)
            print(f"🔄 Starting batch visualization for {len(xai_batch)} images...")
            visualizations = visualizer.create_comprehensive_visualization_batch(xai_batch)
            
            print(f"🎨 Generated {len(visualizations)} XAI visualizations from {len(representative_images)} representative image")
            return visualizations
            
        except ImportError:
            print("Warning: xai_visualizer module not found. XAI visualization skipped.")
            return {}
        except Exception as e:
            print(f"Error creating XAI visualizations: {e}")
            return {}
    
    def _select_representative_images(self):
        """전체 샘플 중 하나의 대표 이미지를 선택합니다."""
        representative_images = []
        
        print(f"  🔍 Selecting single representative image from all samples...")
        
        # XAI 데이터가 있는 파일들만 필터링
        if self.xai_data:
            xai_available_files = list(self.xai_data.keys())
            print(f"    - Found {len(xai_available_files)} files with XAI data")
            
            if xai_available_files:
                # 무작위로 하나의 대표 이미지 선택
                import random
                selected_file = random.choice(xai_available_files)
                representative_images.append(selected_file)
                print(f"  📊 Selected single representative image: {selected_file}")
                print(f"  📊 Total XAI files available: {len(xai_available_files)}")
            else:
                print(f"  ⚠️  No XAI data available for representative selection")
        else:
            print(f"  ⚠️  No XAI data available for representative selection")
        
        return representative_images
    
    def create_xai_summary_stats(self):
        """XAI 분석 결과 요약 통계를 생성합니다."""
        if not self.xai_data:
            return {}
        
        total_xai_files = len(self.xai_data)
        
        # 분석 품질 지표
        quality_metrics = {
            'high_quality_analyses': 0,  # IoU > 0.5
            'medium_quality_analyses': 0,  # 0.3 < IoU <= 0.5
            'low_quality_analyses': 0,  # IoU <= 0.3
            'no_detection_analyses': 0,  # IoU = 0
            'high_entropy_analyses': 0,  # Shannon entropy > 2.0
            'low_entropy_analyses': 0,  # Shannon entropy <= 1.0
            'complex_components': 0,  # > 5 connected components
            'simple_components': 0,  # <= 2 connected components
        }
        
        # 검출된 객체 정보
        detected_classes = {}
        model_info = {}
        
        # 대표 이미지 정보 (단일 대표 이미지)
        representative_info = {
            'total_samples': total_xai_files,
            'representative_images': 1,
            'sample_coverage': 1.0 / total_xai_files if total_xai_files > 0 else 0.0
        }
        
        for filename, xai_result in self.xai_data.items():
            if isinstance(xai_result, dict):
                # 모델 정보 수집
                if 'model_name' in xai_result:
                    model_name = xai_result['model_name']
                    model_info[model_name] = model_info.get(model_name, 0) + 1
                
                # IoU 품질 분석
                if 'overlap_analysis' in xai_result:
                    overlap = xai_result['overlap_analysis']
                    iou = overlap.get('iou', 0)
                    
                    if iou > 0.5:
                        quality_metrics['high_quality_analyses'] += 1
                    elif iou > 0.3:
                        quality_metrics['medium_quality_analyses'] += 1
                    elif iou > 0:
                        quality_metrics['low_quality_analyses'] += 1
                    else:
                        quality_metrics['no_detection_analyses'] += 1
                    
                    # 검출된 클래스 정보
                    class_name = overlap.get('largest_class_name', 'Unknown')
                    detected_classes[class_name] = detected_classes.get(class_name, 0) + 1
                
                # 엔트로피 품질 분석
                if 'entropy_results' in xai_result:
                    entropy = xai_result['entropy_results']
                    shannon_entropy = entropy.get('shannon', 0)
                    
                    if shannon_entropy > 2.0:
                        quality_metrics['high_entropy_analyses'] += 1
                    elif shannon_entropy <= 1.0:
                        quality_metrics['low_entropy_analyses'] += 1
                
                # Connected Components 복잡도 분석
                if 'components_analysis' in xai_result:
                    components = xai_result['components_analysis']
                    num_components = components.get('num_components', 0)
                    
                    if num_components > 5:
                        quality_metrics['complex_components'] += 1
                    elif num_components <= 2:
                        quality_metrics['simple_components'] += 1
        
        # 단일 대표 이미지 정보 (클러스터링과 무관하게)
        # representative_info는 이미 위에서 설정됨
        
        # 품질 요약
        quality_summary = {
            'excellent': quality_metrics['high_quality_analyses'],
            'good': quality_metrics['medium_quality_analyses'],
            'poor': quality_metrics['low_quality_analyses'] + quality_metrics['no_detection_analyses']
        }
        
        return {
            'total_files': total_xai_files,
            'quality_summary': quality_summary,
            'quality_metrics': quality_metrics,
            'detected_classes': detected_classes,
            'model_info': model_info,
            'representative_info': representative_info,
            'analysis_coverage': {
                'with_detections': quality_metrics['high_quality_analyses'] + quality_metrics['medium_quality_analyses'] + quality_metrics['low_quality_analyses'],
                'without_detections': quality_metrics['no_detection_analyses'],
                'high_entropy': quality_metrics['high_entropy_analyses'],
                'complex_patterns': quality_metrics['complex_components']
            }
        }
    
    def fig_to_base64(self):
        """matplotlib figure를 base64 인코딩된 이미지로 변환합니다."""
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode()
        buf.close()
        return img_str
    
    def create_sample_images_table(self, max_samples=10):
        """샘플 이미지들의 정보를 테이블로 생성합니다."""
        if not self.attr_data:
            return []
        
        # 파일 크기 순으로 정렬하여 샘플 선택
        sorted_items = sorted(self.attr_data.items(), key=lambda x: x[1]['size'], reverse=True)
        samples = sorted_items[:max_samples]
        
        sample_data = []
        for filename, item in samples:
            sample_data.append({
                'filename': filename,
                'size_mb': f"{item['size']:.2f}",
                'format': item['format'],
                'resolution': item['resolution'],
                'noise_level': f"{item['noise_level']:.4f}",
                'sharpness': f"{item['sharpness']:.4f}",
                'path': item['path']
            })
        
        return sample_data
    
    def generate_html_body(self):
        """report_layout.py에 맞는 HTML 본문만 생성합니다."""
        if not self.attr_data and not self.embed_data and not self.xai_data:
            print("❌ No analysis data found. Please run analysis first.")
            return None
        
        # 데이터 준비
        summary = self.create_summary_stats()
        samples = self.create_sample_images_table()
        
        # 각 섹션별 차트 생성
        basic_charts = self.create_basic_attribute_charts()
        quality_charts = self.create_quality_attribute_charts()
        embedding_charts = self.create_embedding_charts()
        clustering_charts = self.create_clustering_charts()
        
        # XAI 시각화 및 요약 통계 생성 (대표 이미지 선택은 한 번만)
        xai_charts = self.create_xai_visualizations()
        xai_summary = self.create_xai_summary_stats()
        
        print(f"📊 Report data summary:")
        print(f"  - Summary stats: {'✅' if summary else '❌'}")
        print(f"  - Basic charts: {len(basic_charts)} charts")
        print(f"  - Quality charts: {len(quality_charts)} charts")
        print(f"  - Embedding charts: {len(embedding_charts)} charts")
        print(f"  - Clustering charts: {len(clustering_charts)} charts")
        print(f"  - Samples: {len(samples)} sample images")
        print(f"  - XAI charts: {len(xai_charts)} XAI visualizations")
        print(f"  - XAI summary: {'✅' if xai_summary else '❌'}")
        
        # report_layout 모듈의 함수들을 사용하여 HTML 생성
        try:
            from report_generator.report_layout import (
                generate_summary_statistics_section,
                generate_format_distribution_section,
                generate_visualizations_section,
                generate_sample_images_section,
                generate_detailed_statistics_section,
                generate_embedding_info_section,
                generate_resolution_info_section,
                generate_clustering_summary_section,
                generate_xai_analysis_section
            )
            
            # HTML 파트들을 동적으로 생성
            html_parts = []
            
            # ===== 속성 및 임베딩 분석 섹션 시작 =====
            html_parts.append("""
            <div style="margin-bottom: 40px; padding: 25px; background: #f8f9fa; border-radius: 12px; border: 2px solid #e9ecef;">
                <h2 style="color: #495057; margin-bottom: 20px; padding-bottom: 12px; border-bottom: 3px solid #007bff; font-size: 1.6em;">
                    🖼️ Image Analysis Results (속성 및 임베딩 분석)
                </h2>
            """)
            
            # ===== 1. 기본 속성 분석 =====
            # 1-1. 요약 통계 섹션 (파일 크기 차트 포함)
            html_parts.append(generate_summary_statistics_section(summary, basic_charts.get('size_distribution')))
            
            # 1-2. 샘플 이미지 테이블 섹션 (파일 크기 차트 바로 아래)
            html_parts.append(generate_sample_images_section(samples))
            
            # 1-3. 형식별 분포 섹션 (형식별 분포 차트 포함)
            html_parts.append(generate_format_distribution_section(summary, basic_charts.get('format_distribution')))
            
            # 1-4. 해상도 정보 섹션 (해상도 분포 차트 포함)
            html_parts.append(generate_resolution_info_section(summary, basic_charts.get('resolution_distribution')))
            
            # ===== 2. 이미지 품질 속성 =====
            # 2-1. 상세 통계 섹션 (품질 속성 차트 포함)
            html_parts.append(generate_detailed_statistics_section(summary, quality_charts.get('noise_vs_sharpness')))
            
            # ===== 3. 임베딩 분석 =====
            # 3-1. 임베딩 정보 섹션 (임베딩 차트 포함)
            html_parts.append(generate_embedding_info_section(self.embed_data, embedding_charts.get('embeddings_pca')))
            
            # ===== 4. 클러스터링 분석 =====
            # 4-1. 클러스터링 요약 섹션 (클러스터링 차트 포함)
            clustering_summary = self.create_clustering_summary()
            html_parts.append(generate_clustering_summary_section(clustering_summary, clustering_charts))
            
            # ===== 속성 및 임베딩 분석 섹션 종료 =====
            html_parts.append("""
            </div>
            """)
            
            # ===== XAI 분석 섹션 시작 =====
            # 9. XAI 분석 결과 섹션 (report_layout 모듈 사용)
            if xai_summary or xai_charts:
                print(f"🎨 Adding XAI section with {len(xai_charts)} visualizations and summary stats")
            else:
                print("ℹ️  No XAI data available, skipping XAI section")
                
            if xai_summary or xai_charts:
                html_parts.append("""
            <div style="margin-bottom: 40px; padding: 25px; background: #fff3cd; border-radius: 12px; border: 2px solid #ffc107;">
                <h2 style="color: #495057; margin-bottom: 20px; padding-bottom: 12px; border-bottom: 3px solid #ffc107; font-size: 1.6em;">
                    🧠 XAI (Explainable AI) Analysis Results
                </h2>
            """)
                    
                # XAI 분석 결과 섹션 (report_layout 모듈 사용)
                html_parts.append(generate_xai_analysis_section(xai_summary, xai_charts))
                    
                html_parts.append("""
            </div>
            """)
                
            # ===== XAI 분석 섹션 종료 =====
            
            return ''.join(html_parts)
                
        except ImportError as e:
            print(f"❌ Error importing report_layout functions: {e}")
            return None
        except Exception as e:
            print(f"❌ Error generating HTML: {e}")
            return None

def create_report_body(directory):
    """report_layout.py에 맞는 HTML 본문만 생성합니다."""
    try:
        report = ImageAnalysisReport(directory)
        return report.generate_html_body()
    except Exception as e:
        # 에러가 발생하면 None을 반환하여 보고서 생성을 중단
        print(f"❌ Error in create_report_body: {e}")
        return None

def create_xai_guideline_report(output_dir="."):
    """XAI 가이드라인 보고서를 생성합니다."""
    if create_xai_guideline is None:
        print("⚠️  XAI guideline generator not available. Skipping guideline generation.")
        return None
    
    try:
        output_path = os.path.join(output_dir, "xai_guideline.html")
        create_xai_guideline(output_path)
        print(f"📚 XAI 가이드라인이 생성되었습니다: {output_path}")
        return output_path
    except Exception as e:
        print(f"❌ Error creating XAI guideline: {e}")
        return None

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python create_report.py <directory> [--guideline]")
        sys.exit(1)
    
    directory = sys.argv[1]
    
    # 가이드라인 생성 옵션 확인
    generate_guideline = "--guideline" in sys.argv
    
    # 메인 보고서 생성
    body_content = create_report_body(directory)
    print("Generated HTML body content:")
    print(body_content)
    
    # 가이드라인 생성 (옵션)
    if generate_guideline:
        create_xai_guideline_report()
    
    print(body_content)
