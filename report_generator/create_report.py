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
    
    def create_visualizations(self):
        """시각화 차트들을 생성합니다."""
        if not self.attr_data:
            return {}
        
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
        
        # 3. 노이즈 vs 선명도 산점도
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
        
        # 4. 해상도별 분포 (상위 10개)
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
        
        # 5. 임베딩 공간 시각화 (PCA)
        # 임베딩 데이터가 있는 경우
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
        
        # 6. 클러스터링 결과 시각화 (클러스터링 데이터가 있는 경우)
        if self.clustering_data and 'embeddings_2d' in self.clustering_data:
            embeddings_2d = np.array(self.clustering_data['embeddings_2d'])
            cluster_labels = np.array(self.clustering_data['cluster_labels'])
            method = self.clustering_data.get('method', 'Unknown')
            n_clusters = self.clustering_data.get('n_clusters', 0)
            
            plt.figure(figsize=(12, 8))
            scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                c=cluster_labels, cmap='viridis', alpha=0.6)
            
            # 클러스터 중심점 표시 (centroids 사용)
            if 'centroids' in self.clustering_data and self.clustering_data['centroids']:
                centroids = np.array(self.clustering_data['centroids'])
                # PCA를 사용하여 2D로 변환
                pca_components = np.array(self.clustering_data['pca_components'])
                centroids_2d = centroids @ pca_components.T
                plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], 
                           c='red', marker='x', s=200, linewidths=3, label='Cluster Centroids')
            
            plt.title(f'Clustering Results - {method.upper()} ({n_clusters} clusters)', fontsize=14, fontweight='bold')
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.colorbar(scatter, label='Cluster')
            plt.legend()
            plt.grid(True, alpha=0.3)
            charts['clustering_results'] = self.fig_to_base64()
            plt.close()
            
            # 7. 클러스터 크기 분포
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
        """XAI 분석 결과를 시각화합니다. (클러스터별 대표 이미지만)"""
        if not self.xai_data:
            print("ℹ️  No XAI data found. XAI visualizations skipped.")
            return {}
        
        try:
            from data_utils.xai_visualizer import XAIVisualizer
            
            visualizer = XAIVisualizer()
            visualizations = {}
            
            # 클러스터별 대표 이미지 선택
            representative_images = self._select_representative_images()
            
            if not representative_images:
                print("ℹ️  No representative images found. Skipping XAI visualizations.")
                return {}
            
            print(f"🔍 Processing {len(representative_images)} representative XAI analysis results...")
            
            # 대표 이미지별 XAI 분석 결과 시각화
            for filename in representative_images:
                if filename in self.xai_data:
                    xai_result = self.xai_data[filename]
                    if isinstance(xai_result, dict):
                        print()
                        print(f"  🔍 Processing XAI data for representative image: {filename}")
                        # print(f"    - Available keys: {list(xai_result.keys())}")
                        
                        # 시각화 생성 (create_comprehensive_visualization 사용)
                        try:
                            img_visualizations = visualizer.create_comprehensive_visualization(xai_result)
                            
                            # 파일명을 키로 사용하여 저장 (확장자 포함)
                            for viz_type, viz_data in img_visualizations.items():
                                key = f"{filename}_{viz_type}"
                                visualizations[key] = viz_data
                            
                            print(f"  ✅ Processed XAI visualization for {filename} ({len(img_visualizations)} visualizations)")
                        except Exception as viz_error:
                            print(f"  ❌ Failed to create visualizations for {filename}: {viz_error}")
                    else:
                        print(f"  ⚠️  Skipping {filename}: not a dictionary")
                else:
                    print(f"  ⚠️  Representative image {filename} not found in XAI data")
            
            print(f"🎨 Generated {len(visualizations)} XAI visualizations from {len(representative_images)} representative images")
            return visualizations
            
        except ImportError:
            print("Warning: xai_visualizer module not found. XAI visualization skipped.")
            return {}
        except Exception as e:
            print(f"Error creating XAI visualizations: {e}")
            return {}
    
    def _select_representative_images(self):
        """클러스터별 대표 이미지를 선택합니다."""
        representative_images = []
        
        print(f"  🔍 Checking clustering data...")
        print(f"    - clustering_data exists: {hasattr(self, 'clustering_data')}")
        print(f"    - clustering_data content: {len(self.clustering_data) if hasattr(self, 'clustering_data') and self.clustering_data else 0} items")
        
        # 클러스터링 데이터가 있는 경우
        if hasattr(self, 'clustering_data') and self.clustering_data:
            try:
                # 클러스터별로 대표 이미지 선택
                cluster_groups = {}
                
                # 클러스터링 데이터 구조 확인
                # print(f"    - Clustering data type: {type(self.clustering_data)}")
                # print(f"    - Clustering data keys: {list(self.clustering_data.keys()) if isinstance(self.clustering_data, dict) else 'Not a dict'}")
                
                # 클러스터 정보 추출 (세 가지 구조 지원)
                if isinstance(self.clustering_data, dict):
                    # 구조 1: 클러스터링 분석 결과 구조
                    if 'cluster_labels' in self.clustering_data and 'file_names' in self.clustering_data:
                        print(f"    - Found clustering analysis structure with {len(self.clustering_data['file_names'])} files")
                        cluster_labels = self.clustering_data['cluster_labels']
                        file_names = self.clustering_data['file_names']
                        
                        for i, (label, filename) in enumerate(zip(cluster_labels, file_names)):
                            cluster_id = label
                            if cluster_id not in cluster_groups:
                                cluster_groups[cluster_id] = []
                            cluster_groups[cluster_id].append(filename)
                    
                    # 구조 2: {filename: {cluster: id, embedding: [...]}}
                    elif any(isinstance(v, dict) and 'cluster' in v for v in self.clustering_data.values()):
                        for filename, cluster_info in self.clustering_data.items():
                            if isinstance(cluster_info, dict) and 'cluster' in cluster_info:
                                cluster_id = cluster_info['cluster']
                                if cluster_id not in cluster_groups:
                                    cluster_groups[cluster_id] = []
                                cluster_groups[cluster_id].append(filename)
                    
                    # 구조 3: {filename: cluster_id}
                    elif any(isinstance(v, int) for v in self.clustering_data.values()):
                        for filename, cluster_id in self.clustering_data.items():
                            if isinstance(cluster_id, int):
                                if cluster_id not in cluster_groups:
                                    cluster_groups[cluster_id] = []
                                cluster_groups[cluster_id].append(filename)
                
                print(f"    - Found {len(cluster_groups)} clusters: {list(cluster_groups.keys())}")
                
                # 각 클러스터에서 무작위 대표 이미지 선택
                import random
                print()
                for cluster_id, filenames in cluster_groups.items():
                    # XAI 데이터가 있는 파일들만 필터링
                    xai_available_files = [f for f in filenames if f in self.xai_data]
                    
                    if xai_available_files:
                        # 무작위로 대표 이미지 선택
                        selected_file = random.choice(xai_available_files)
                        representative_images.append(selected_file)
                        print(f"  📊 Selected random representative for cluster {cluster_id}: {selected_file}")
                    else:
                        print(f"  ⚠️  No XAI data available for cluster {cluster_id}")
                
                print(f"  📊 Selected {len(representative_images)} representative images from {len(cluster_groups)} clusters")
                print()
                
            except Exception as e:
                print(f"  ⚠️  Error selecting representative images: {e}")
                # 오류 발생 시 처음 3개 이미지만 선택
                if self.xai_data:
                    representative_images = list(self.xai_data.keys())[:3]
                    print(f"  📊 Fallback: Selected first 3 images as representatives")
        else:
            # 클러스터링 데이터가 없는 경우 처음 3개 이미지만 선택
            if self.xai_data:
                representative_images = list(self.xai_data.keys())[:3]
                print(f"  📊 No clustering data found. Selected first 3 images as representatives")
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
        
        # 대표 이미지 정보 (클러스터링 기반)
        representative_info = {
            'total_clusters': 0,
            'representative_images': 0,
            'cluster_coverage': 0.0
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
        
        # 클러스터링 정보 (대표 이미지 선택 기준)
        if hasattr(self, 'clustering_data') and self.clustering_data:
            if 'cluster_labels' in self.clustering_data:
                unique_clusters = len(set(self.clustering_data['cluster_labels']))
                representative_info['total_clusters'] = unique_clusters
                # 대표 이미지 수는 클러스터 수와 동일 (이미 선택된 대표 이미지 수 사용)
                representative_info['representative_images'] = unique_clusters
                representative_info['cluster_coverage'] = 100.0  # 모든 클러스터에서 대표 이미지 선택
        
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
        charts = self.create_visualizations()
        samples = self.create_sample_images_table()
        
        # XAI 시각화 및 요약 통계 생성 (대표 이미지 선택은 한 번만)
        xai_charts = self.create_xai_visualizations()
        xai_summary = self.create_xai_summary_stats()
        
        print(f"📊 Report data summary:")
        print(f"  - Summary stats: {'✅' if summary else '❌'}")
        print(f"  - Charts: {len(charts)} visualizations")
        print(f"  - Samples: {len(samples)} sample images")
        print(f"  - XAI charts: {len(xai_charts)} XAI visualizations")
        print(f"  - XAI summary: {'✅' if xai_summary else '❌'}")
        
        # HTML 파트들을 동적으로 생성
        html_parts = []
        
        # 1. 요약 통계 섹션 (항상 존재)
        html_parts.append(f"""
        <div style="margin-bottom: 20px;">
            <h3 style="color: #495057; margin-bottom: 15px; padding-bottom: 8px; border-bottom: 2px solid #dee2e6;">📈 Summary Statistics</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px;">
                <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #e9ecef;">
                    <h4 style="color: #6c757d; margin: 0 0 8px 0; font-size: 0.9em;">Total Images</h4>
                    <div style="font-size: 1.8em; font-weight: bold; color: #495057;">{summary.get('total_images', 0):,}</div>
                </div>
                <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #e9ecef;">
                    <h4 style="color: #6c757d; margin: 0 0 8px 0; font-size: 0.9em;">Total Size</h4>
                    <div style="font-size: 1.8em; font-weight: bold; color: #495057;">{summary.get('total_size_mb', 0):.2f} MB</div>
                </div>
                <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #e9ecef;">
                    <h4 style="color: #6c757d; margin: 0 0 8px 0; font-size: 0.9em;">Average Size</h4>
                    <div style="font-size: 1.8em; font-weight: bold; color: #495057;">{summary.get('avg_size_mb', 0):.2f} MB</div>
                </div>
                <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #e9ecef;">
                    <h4 style="color: #6c757d; margin: 0 0 8px 0; font-size: 0.9em;">Unique Formats</h4>
                    <div style="font-size: 1.8em; font-weight: bold; color: #495057;">{len(summary.get('formats', {}))}</div>
                </div>
            </div>
        </div>
        """)
        
        # 2. 형식별 분포 섹션 (형식 데이터가 있는 경우)
        if summary.get('formats'):
            html_parts.append("""
        <div style="margin-bottom: 20px;">
            <h3 style="color: #495057; margin-bottom: 15px; padding-bottom: 8px; border-bottom: 2px solid #dee2e6;">📋 Format Distribution</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px;">
            """)
            
            for fmt, count in summary.get('formats', {}).items():
                percentage = (count / summary['total_images']) * 100
                html_parts.append(f"""
                <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #e9ecef;">
                    <h4 style="color: #6c757d; margin: 0 0 8px 0; font-size: 0.9em;">
                        <span style="background: #e74c3c; color: white; padding: 3px 6px; border-radius: 8px; font-size: 0.7em; font-weight: bold;">{fmt.upper()}</span>
                    </h4>
                    <div style="font-size: 1.5em; font-weight: bold; color: #495057;">{count:,}</div>
                    <p style="margin: 5px 0 0 0; color: #6c757d; font-size: 0.9em;">({percentage:.1f}%)</p>
                </div>
                """)
            
            html_parts.append("""
            </div>
        </div>
            """)
        
        # 3. 시각화 섹션 (차트가 있는 경우)
        if charts:
            html_parts.append("""
        <div style="margin-bottom: 20px;">
            <h3 style="color: #495057; margin-bottom: 15px; padding-bottom: 8px; border-bottom: 2px solid #dee2e6;">📊 Visualizations</h3>
        """)
            
            # 파일 크기 분포
            if 'size_distribution' in charts:
                html_parts.append(f"""
            <div style="text-align: center; margin: 20px 0;">
                <img src="data:image/png;base64,{charts['size_distribution']}" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
            </div>
                """)
            
            # 형식별 분포
            if 'format_distribution' in charts:
                html_parts.append(f"""
            <div style="text-align: center; margin: 20px 0;">
                <img src="data:image/png;base64,{charts['format_distribution']}" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
            </div>
                """)
            
            # 노이즈 vs 선명도
            if 'noise_vs_sharpness' in charts:
                html_parts.append(f"""
            <div style="text-align: center; margin: 20px 0;">
                <img src="data:image/png;base64,{charts['noise_vs_sharpness']}" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
            </div>
                """)
            
            # 해상도별 분포
            if 'resolution_distribution' in charts:
                html_parts.append(f"""
            <div style="text-align: center; margin: 20px 0;">
                <img src="data:image/png;base64,{charts['resolution_distribution']}" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
            </div>
                """)
            
            # 임베딩 PCA
            if 'embeddings_pca' in charts:
                html_parts.append(f"""
            <div style="text-align: center; margin: 20px 0;">
                <img src="data:image/png;base64,{charts['embeddings_pca']}" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
            </div>
                """)
            
            # 클러스터링 결과
            if 'clustering_results' in charts:
                html_parts.append(f"""
            <div style="text-align: center; margin: 20px 0;">
                <img src="data:image/png;base64,{charts['clustering_results']}" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
            </div>
                """)
            
            # 클러스터 크기 분포
            if 'cluster_size_distribution' in charts:
                html_parts.append(f"""
            <div style="text-align: center; margin: 20px 0;">
                <img src="data:image/png;base64,{charts['cluster_size_distribution']}" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
            </div>
                """)
            
            html_parts.append("""
        </div>
            """)
        
        # 4. 샘플 이미지 테이블 섹션 (샘플이 있는 경우)
        if samples:
            html_parts.append("""
        <div style="margin-bottom: 20px;">
            <h3 style="color: #495057; margin-bottom: 15px; padding-bottom: 8px; border-bottom: 2px solid #dee2e6;">🖼️ Sample Images (Top 10 by Size)</h3>
            <table style="width: 100%; border-collapse: collapse; margin: 15px 0; border-radius: 5px; overflow: hidden; background: white;">
                <thead>
                    <tr>
                        <th style="background: #6c757d; color: white; padding: 10px; text-align: left;">Filename</th>
                        <th style="background: #6c757d; color: white; padding: 10px; text-align: left;">Size (MB)</th>
                        <th style="background: #6c757d; color: white; padding: 10px; text-align: left;">Format</th>
                        <th style="background: #6c757d; color: white; padding: 10px; text-align: left;">Resolution</th>
                        <th style="background: #6c757d; color: white; padding: 10px; text-align: left;">Noise Level</th>
                        <th style="background: #6c757d; color: white; padding: 10px; text-align: left;">Sharpness</th>
                    </tr>
                </thead>
                <tbody>
            """)
            
            for sample in samples:
                html_parts.append(f"""
                    <tr>
                        <td style="padding: 8px; border-bottom: 1px solid #dee2e6;">{sample['filename']}</td>
                        <td style="padding: 8px; border-bottom: 1px solid #dee2e6;">{sample['size_mb']}</td>
                        <td style="padding: 8px; border-bottom: 1px solid #dee2e6;">
                            <span style="background: #e74c3c; color: white; padding: 2px 6px; border-radius: 8px; font-size: 0.8em; font-weight: bold;">{sample['format'].upper()}</span>
                        </td>
                        <td style="padding: 8px; border-bottom: 1px solid #dee2e6;">{sample['resolution']}</td>
                        <td style="padding: 8px; border-bottom: 1px solid #dee2e6;">{sample['noise_level']}</td>
                        <td style="padding: 8px; border-bottom: 1px solid #dee2e6;">{sample['sharpness']}</td>
                    </tr>
                """)
            
            html_parts.append("""
                </tbody>
            </table>
        </div>
            """)
        
        # 5. 상세 통계 섹션 (통계 데이터가 있는 경우)
        if summary.get('size_stats') or summary.get('noise_stats') or summary.get('sharpness_stats'):
            html_parts.append("""
        <div style="margin-bottom: 20px;">
            <h3 style="color: #495057; margin-bottom: 15px; padding-bottom: 8px; border-bottom: 2px solid #dee2e6;">📊 Detailed Statistics</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px;">
            """)
            
            if summary.get('size_stats'):
                html_parts.append(f"""
                <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #e9ecef;">
                    <h4 style="color: #495057; margin: 0 0 10px 0;">File Size Statistics</h4>
                    <p style="margin: 5px 0; color: #6c757d;">Min: {summary.get('size_stats', {}).get('min', 0):.2f} MB</p>
                    <p style="margin: 5px 0; color: #6c757d;">Max: {summary.get('size_stats', {}).get('max', 0):.2f} MB</p>
                    <p style="margin: 5px 0; color: #6c757d;">Mean: {summary.get('size_stats', {}).get('mean', 0):.2f} MB</p>
                    <p style="margin: 5px 0; color: #6c757d;">Std: {summary.get('size_stats', {}).get('std', 0):.2f} MB</p>
                </div>
                """)
            
            if summary.get('noise_stats'):
                html_parts.append(f"""
                <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #e9ecef;">
                    <h4 style="color: #495057; margin: 0 0 10px 0;">Noise Level Statistics</h4>
                    <p style="margin: 5px 0; color: #6c757d;">Min: {summary.get('noise_stats', {}).get('min', 0):.4f}</p>
                    <p style="margin: 5px 0; color: #6c757d;">Max: {summary.get('noise_stats', {}).get('max', 0):.4f}</p>
                    <p style="margin: 5px 0; color: #6c757d;">Mean: {summary.get('noise_stats', {}).get('mean', 0):.4f}</p>
                    <p style="margin: 5px 0; color: #6c757d;">Std: {summary.get('noise_stats', {}).get('std', 0):.4f}</p>
                </div>
                """)
            
            if summary.get('sharpness_stats'):
                html_parts.append(f"""
                <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #e9ecef;">
                    <h4 style="color: #495057; margin: 0 0 10px 0;">Sharpness Statistics</h4>
                    <p style="margin: 5px 0; color: #6c757d;">Min: {summary.get('sharpness_stats', {}).get('min', 0):.4f}</p>
                    <p style="margin: 5px 0; color: #6c757d;">Max: {summary.get('sharpness_stats', {}).get('max', 0):.4f}</p>
                    <p style="margin: 5px 0; color: #6c757d;">Mean: {summary.get('sharpness_stats', {}).get('mean', 0):.4f}</p>
                    <p style="margin: 5px 0; color: #6c757d;">Std: {summary.get('sharpness_stats', {}).get('std', 0):.4f}</p>
                </div>
                """)
            
            html_parts.append("""
            </div>
        </div>
            """)
        
        # 6. 임베딩 정보 섹션 (임베딩 데이터가 있는 경우)
        if self.embed_data:
            embeddings = [item['embedding'] for item in self.embed_data.values()]
            if embeddings:
                embedding_dim = len(embeddings[0])
                html_parts.append(f"""
        <div style="margin-bottom: 20px;">
            <h3 style="color: #495057; margin-bottom: 15px; padding-bottom: 8px; border-bottom: 2px solid #dee2e6;">🪐 Embedding Informations</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
                <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #e9ecef;">
                    <h4 style="color: #6c757d; margin: 0 0 8px 0; font-size: 0.9em;">Embedding Dimension</h4>
                    <div style="font-size: 1.8em; font-weight: bold; color: #495057;">{embedding_dim}</div>
                </div>
                <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #e9ecef;">
                    <h4 style="color: #6c757d; margin: 0 8px 0; font-size: 0.9em;">Total Embeddings</h4>
                    <div style="font-size: 1.8em; font-weight: bold; color: #495057;">{len(embeddings):,}</div>
                </div>
            </div>
        </div>
                """)
        
        # 7. 해상도 정보 섹션 (해상도 데이터가 있는 경우)
        if summary.get('resolutions'):
            top_resolutions = dict(sorted(summary['resolutions'].items(), key=lambda x: x[1], reverse=True)[:5])
            html_parts.append("""
        <div style="margin-bottom: 20px;">
            <h3 style="color: #495057; margin-bottom: 15px; padding-bottom: 8px; border-bottom: 2px solid #dee2e6;">📐 Resolution Information</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px;">
            """)
            
            for res, count in top_resolutions.items():
                percentage = (count / summary['total_images']) * 100
                html_parts.append(f"""
                <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #e9ecef;">
                    <h4 style="color: #6c757d; margin: 0 0 8px 0; font-size: 0.9em;">{res}</h4>
                    <div style="font-size: 1.5em; font-weight: bold; color: #495057;">{count:,}</div>
                    <p style="margin: 5px 0 0 0; color: #6c757d; font-size: 0.9em;">({percentage:.1f}%)</p>
                </div>
                """)
            
            html_parts.append("""
            </div>
        </div>
            """)
        
        # 8. 클러스터링 요약 섹션 (클러스터링 데이터가 있는 경우)
        clustering_summary = self.create_clustering_summary()
        if clustering_summary:
            html_parts.append(f"""
        <div style="margin-bottom: 20px;">
            <h3 style="color: #495057; margin-bottom: 15px; padding-bottom: 8px; border-bottom: 2px solid #dee2e6;">🧠 Clustering Summary</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
                <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #e9ecef;">
                    <h4 style="color: #6c757d; margin: 0 0 8px 0; font-size: 0.9em;">Method</h4>
                    <div style="font-size: 1.8em; font-weight: bold; color: #495057;">{clustering_summary.get('method', 'N/A').upper()}</div>
                </div>
                <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #e9ecef;">
                    <h4 style="color: #6c757d; margin: 0 0 8px 0; font-size: 0.9em;">Total Samples</h4>
                    <div style="font-size: 1.8em; font-weight: bold; color: #495057;">{clustering_summary.get('total_samples', 0):,}</div>
                </div>
                <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #e9ecef;">
                    <h4 style="color: #6c757d; margin: 0 0 8px 0; font-size: 0.9em;">Clusters</h4>
                    <div style="font-size: 1.8em; font-weight: bold; color: #495057;">{clustering_summary.get('n_clusters', 0)}</div>
                </div>
            </div>
        </div>
            """)
            
            # 클러스터 상세 테이블
            if clustering_summary.get('cluster_summary'):
                html_parts.append("""
            <div style="margin-top: 20px;">
                <h4 style="color: #495057; margin-bottom: 10px;">Cluster Details</h4>
                <table style="width: 100%; border-collapse: collapse; margin: 15px 0; border-radius: 5px; overflow: hidden; background: white;">
                    <thead>
                        <tr>
                            <th style="background: #6c757d; color: white; padding: 10px; text-align: left;">Cluster ID</th>
                            <th style="background: #6c757d; color: white; padding: 10px; text-align: left;">Size</th>
                            <th style="background: #6c757d; color: white; padding: 10px; text-align: left;">Percentage</th>
                            <th style="background: #6c757d; color: white; padding: 10px; text-align: left;">Sample Files</th>
                        </tr>
                    </thead>
                    <tbody>
                """)
                
                for cluster in clustering_summary['cluster_summary']:
                    html_parts.append(f"""
                        <tr>
                            <td style="padding: 8px; border-bottom: 1px solid #dee2e6;">{cluster['cluster_id']}</td>
                            <td style="padding: 8px; border-bottom: 1px solid #dee2e6;">{cluster['size']:,}</td>
                            <td style="padding: 8px; border-bottom: 1px solid #dee2e6;">{cluster['percentage']:.1f}%</td>
                            <td style="padding: 8px; border-bottom: 1px solid #dee2e6;">
                                <ul style="margin: 0; padding-left: 20px; list-style-type: none;">
                                    {', '.join([f'<li>{f}' for f in cluster['sample_files']])}
                                </ul>
                            </td>
                        </tr>
                    """)
                
                html_parts.append("""
                    </tbody>
                </table>
            </div>
                """)
        
        # 9. XAI 분석 결과 섹션 (XAI 데이터가 있는 경우)
        if xai_summary or xai_charts:
            print(f"🎨 Adding XAI section with {len(xai_charts)} visualizations and summary stats")
        else:
            print("ℹ️  No XAI data available, skipping XAI section")
            
        if xai_summary or xai_charts:
            html_parts.append("""
        <div style="margin-bottom: 30px;">
            <h3 style="color: #495057; margin-bottom: 15px; padding-bottom: 8px; border-bottom: 2px solid #dee2e6;">🧠 XAI (Explainable AI) Analysis Results</h3>
        """)
            
            # XAI 요약 통계 추가
            if xai_summary:
                html_parts.append(f"""
            <div style="margin-bottom: 20px;">
                <h4 style="color: #495057; margin-bottom: 15px; padding-bottom: 8px; border-bottom: 1px solid #dee2e6;">📊 XAI Analysis Summary</h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px;">
                    <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #e9ecef;">
                        <h5 style="color: #6c757d; margin: 0 0 8px 0; font-size: 0.9em;">Total Files Analyzed</h5>
                        <div style="font-size: 1.8em; font-weight: bold; color: #495057;">{xai_summary.get('total_files', 0):,}</div>
                    </div>
                    <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #e9ecef;">
                        <h5 style="color: #6c757d; margin: 0 0 8px 0; font-size: 0.9em;">High Quality Analyses</h5>
                        <div style="font-size: 1.8em; font-weight: bold; color: #28a745;">{xai_summary.get('quality_summary', {}).get('excellent', 0):,}</div>
                        <small style="color: #6c757d;">IoU > 0.5</small>
                    </div>
                    <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #e9ecef;">
                        <h5 style="color: #6c757d; margin: 0 0 8px 0; font-size: 0.9em;">Complex Patterns</h5>
                        <div style="font-size: 1.8em; font-weight: bold; color: #ffc107;">{xai_summary.get('analysis_coverage', {}).get('complex_patterns', 0):,}</div>
                        <small style="color: #6c757d;">> 5 components</small>
                    </div>
                    <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #e9ecef;">
                        <h5 style="color: #6c757d; margin: 0 0 8px 0; font-size: 0.9em;">Representative Images</h5>
                        <div style="font-size: 1.8em; font-weight: bold; color: #17a2b8;">{xai_summary.get('representative_info', {}).get('representative_images', 0):,}</div>
                        <small style="color: #6c757d;">from {xai_summary.get('representative_info', {}).get('total_clusters', 0)} clusters</small>
                    </div>
                </div>
            </div>
                """)
                

            
            # XAI 시각화들을 파일별로 그룹화
            file_groups = {}
            for key, viz_data in xai_charts.items():
                # 키 형식: "filename_viztype" (파일명에 언더스코어가 포함될 수 있음)
                # 예: "carnorm_2_cam_heatmap" -> filename="carnorm_2", viz_type="cam_heatmap"
                
                # 알려진 시각화 타입들을 찾아서 파일명과 분리
                known_viz_types = [
                    'cam_heatmap', 'cam_threshold_analysis', 'cam_distribution_analysis',
                    'cam_statistics', 'connected_components', 'entropy_analysis',
                    'centroid_analysis', 'overlap_analysis', 'overlap_statistics'
                ]
                
                filename = key
                viz_type = None
                
                # 알려진 시각화 타입으로 끝나는지 확인
                for viz_type_name in known_viz_types:
                    if key.endswith(f'_{viz_type_name}'):
                        # 마지막 언더스코어를 기준으로 분리
                        filename = key[:-len(f'_{viz_type_name}')]
                        viz_type = viz_type_name
                        break
                
                # 알려진 타입이 없으면 기본 분리 방식 사용
                if viz_type is None:
                    parts = key.split('_', 1)
                    if len(parts) == 2:
                        filename, viz_type = parts
                    else:
                        # 분리할 수 없는 경우 전체를 파일명으로 사용
                        filename = key
                        viz_type = 'unknown'
                
                if filename not in file_groups:
                    file_groups[filename] = {}
                file_groups[filename][viz_type] = viz_data
            
            # 각 파일별로 XAI 결과 표시
            for filename, viz_types in file_groups.items():
                # 파일명을 더 명확하게 표시 (확장자 포함)
                display_filename = filename
                if '.' in filename:
                    # 확장자가 있는 경우 파일 타입 아이콘 추가
                    ext = filename.split('.')[-1].lower()
                    if ext in ['jpg', 'jpeg']:
                        icon = "🖼️"
                    elif ext == 'png':
                        icon = "🖼️"
                    elif ext in ['bmp', 'tiff', 'tif']:
                        icon = "🖼️"
                    else:
                        icon = "📄"
                else:
                    icon = "📄"
                
                html_parts.append(f"""
            <div style="margin-bottom: 25px; padding: 20px; background: #f8f9fa; border-radius: 10px; border: 1px solid #e9ecef;">
                <h4 style="color: #495057; margin-bottom: 15px; border-bottom: 1px solid #dee2e6; padding-bottom: 8px;">
                    {icon} {display_filename}
                </h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px;">
                """)
                
                # 시각화 타입별 제목 매핑
                viz_titles = {
                    'cam_heatmap': '🔥 CAM Heatmap',
                    'cam_threshold_analysis': '🎯 Threshold Analysis',
                    'cam_distribution_analysis': '📊 Distribution Analysis',
                    'cam_statistics': '📈 CAM Statistics',
                    'connected_components': '🔗 Connected Components Analysis',
                    'entropy_analysis': '📊 Entropy Analysis',
                    'centroid_analysis': '🎯 Centroid Analysis',
                    'overlap_analysis': '🔍 Overlap Analysis',
                    'overlap_statistics': '📊 Overlap Statistics'
                }
                
                # 원하는 순서로 시각화 타입 정렬
                desired_order = [
                    'cam_heatmap',
                    'cam_statistics',
                    'cam_threshold_analysis',
                    'connected_components',
                    'entropy_analysis',
                    'centroid_analysis',
                    'overlap_analysis',
                    'overlap_statistics',
                    'cam_distribution_analysis'  # 기타 타입들은 마지막에
                ]
                
                # 정렬된 순서로 시각화 출력
                for viz_type in desired_order:
                    if viz_type in viz_types:
                        viz_data = viz_types[viz_type]
                        title = viz_titles.get(viz_type, viz_type.replace('_', ' ').title())
                        html_parts.append(f"""
                    <div style="background: white; padding: 15px; border-radius: 8px; border: 1px solid #dee2e6;">
                        <h5 style="color: #495057; margin-bottom: 10px; font-size: 1.1em;">{title}</h5>
                        <div style="text-align: center;">
                            <img src="data:image/png;base64,{viz_data}" 
                                 style="max-width: 100%; height: auto; border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                        </div>
                    </div>
                        """)
                
                # 정렬되지 않은 기타 시각화 타입들도 출력
                for viz_type, viz_data in viz_types.items():
                    if viz_type not in desired_order:
                        title = viz_titles.get(viz_type, viz_type.replace('_', ' ').title())
                        html_parts.append(f"""
                    <div style="background: white; padding: 15px; border-radius: 8px; border: 1px solid #dee2e6;">
                        <h5 style="color: #495057; margin-bottom: 10px; font-size: 1.1em;">{title}</h5>
                        <div style="text-align: center;">
                            <img src="data:image/png;base64,{viz_data}" 
                                 style="max-width: 100%; height: auto; border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                        </div>
                    </div>
                        """)
                
                html_parts.append("""
                </div>
            </div>
                """)
            
            html_parts.append("""
        </div>
            """)
        
        return ''.join(html_parts)

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
