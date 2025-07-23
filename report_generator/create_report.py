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
        self.data = self.load_data()
        self.clustering_data = self.load_clustering_data()
        
    def load_data(self):
        """캐시된 분석 데이터를 로드합니다."""
        # 캐시 매니저에서 데이터 로드
        cached_data = get_cached_analysis_data(self.directory, "image_analysis")
        if cached_data is not None:
            return cached_data
        
        # 캐시에 데이터가 없으면 빈 딕셔너리 반환
        return {}
    
    def load_clustering_data(self):
        """캐시된 클러스터링 분석 데이터를 로드합니다."""
        # 캐시 매니저에서 클러스터링 데이터 로드
        cached_data = get_cached_analysis_data(self.directory, "clustering_analysis")
        if cached_data is not None:
            return cached_data
        
        # 캐시에 데이터가 없으면 빈 딕셔너리 반환
        return {}
    
    def create_summary_stats(self):
        """기본 통계 정보를 생성합니다."""
        if not self.data:
            return {}
        
        total_images = len(self.data)
        total_size = sum(item['size'] for item in self.data.values())
        
        # 형식별 통계
        formats = {}
        resolutions = {}
        sizes = []
        noise_levels = []
        sharpness_values = []
        
        for item in self.data.values():
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
        if not self.data:
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
        embeddings = np.array([item['embedding'] for item in self.data.values()])
        if len(embeddings) > 1:
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
        if not self.data:
            return []
        
        # 파일 크기 순으로 정렬하여 샘플 선택
        sorted_items = sorted(self.data.items(), key=lambda x: x[1]['size'], reverse=True)
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
        if not self.data:
            return '<p>분석 데이터가 없습니다.</p>'
        
        # 데이터 준비
        summary = self.create_summary_stats()
        charts = self.create_visualizations()
        samples = self.create_sample_images_table()
        
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
                <h4 style="color: #495057; margin-bottom: 10px;">File Size Distribution</h4>
                <img src="data:image/png;base64,{charts['size_distribution']}" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
            </div>
                """)
            
            # 형식별 분포
            if 'format_distribution' in charts:
                html_parts.append(f"""
            <div style="text-align: center; margin: 20px 0;">
                <h4 style="color: #495057; margin-bottom: 10px;">Image Format Distribution</h4>
                <img src="data:image/png;base64,{charts['format_distribution']}" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
            </div>
                """)
            
            # 노이즈 vs 선명도
            if 'noise_vs_sharpness' in charts:
                html_parts.append(f"""
            <div style="text-align: center; margin: 20px 0;">
                <h4 style="color: #495057; margin-bottom: 10px;">Noise Level vs Sharpness</h4>
                <img src="data:image/png;base64,{charts['noise_vs_sharpness']}" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
            </div>
                """)
            
            # 해상도별 분포
            if 'resolution_distribution' in charts:
                html_parts.append(f"""
            <div style="text-align: center; margin: 20px 0;">
                <h4 style="color: #495057; margin-bottom: 10px;">Top 10 Resolution Distribution</h4>
                <img src="data:image/png;base64,{charts['resolution_distribution']}" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
            </div>
                """)
            
            # 임베딩 PCA
            if 'embeddings_pca' in charts:
                html_parts.append(f"""
            <div style="text-align: center; margin: 20px 0;">
                <h4 style="color: #495057; margin-bottom: 10px;">Image Embeddings (PCA 2D)</h4>
                <img src="data:image/png;base64,{charts['embeddings_pca']}" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
            </div>
                """)
            
            # 클러스터링 결과
            if 'clustering_results' in charts:
                html_parts.append(f"""
            <div style="text-align: center; margin: 20px 0;">
                <h4 style="color: #495057; margin-bottom: 10px;">Clustering Results</h4>
                <img src="data:image/png;base64,{charts['clustering_results']}" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
            </div>
                """)
            
            # 클러스터 크기 분포
            if 'cluster_size_distribution' in charts:
                html_parts.append(f"""
            <div style="text-align: center; margin: 20px 0;">
                <h4 style="color: #495057; margin-bottom: 10px;">Cluster Size Distribution</h4>
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
        if self.data and any('embedding' in item for item in self.data.values()):
            embeddings = [item['embedding'] for item in self.data.values() if 'embedding' in item]
            if embeddings:
                embedding_dim = len(embeddings[0])
                html_parts.append(f"""
        <div style="margin-bottom: 20px;">
            <h3 style="color: #495057; margin-bottom: 15px; padding-bottom: 8px; border-bottom: 2px solid #dee2e6;">🧠 Embedding Information</h3>
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
            html_parts.append("""
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
        
        return ''.join(html_parts)

def create_report_body(directory):
    """report_layout.py에 맞는 HTML 본문만 생성합니다."""
    try:
        report = ImageAnalysisReport(directory)
        return report.generate_html_body()
    except Exception as e:
        return f'<div>오류: {e}</div>'

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python create_report.py <directory>")
        sys.exit(1)
    
    directory = sys.argv[1]
    body_content = create_report_body(directory)
    print("Generated HTML body content:")
    print(body_content)

    print(body_content)
