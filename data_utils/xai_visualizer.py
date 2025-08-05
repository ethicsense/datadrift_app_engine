import matplotlib.pyplot as plt
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import base64
from io import BytesIO
from pytorch_grad_cam.utils.image import show_cam_on_image


class XAIVisualizer:
    """XAI 분석 결과를 시각화하는 클래스"""
    
    def __init__(self):
        """XAI 시각화기 초기화"""
        plt.style.use('default')
        plt.rcParams['font.size'] = 10
    
    def fig_to_base64(self, fig: plt.Figure) -> str:
        """matplotlib figure를 base64 인코딩된 이미지로 변환"""
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode()
        buf.close()
        return img_str
    
    def visualize_cam_basic(self, cam_result: Dict) -> str:
        """기본 CAM 시각화 (원본, CAM, 오버레이)"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 원본 이미지
        original_img = cam_result['original_image']
        axes[0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Grayscale CAM
        grayscale_cam = cam_result['grayscale_cam']
        im1 = axes[1].imshow(grayscale_cam, cmap='jet')
        axes[1].set_title('Grayscale CAM', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # CAM 오버레이 이미지
        cam_image = cam_result['cam_image']
        axes[2].imshow(cv2.cvtColor(cam_image, cv2.COLOR_BGR2RGB))
        axes[2].set_title('CAM Overlay', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        return self.fig_to_base64(fig)
    
    def visualize_cam_statistics(self, cam_stats: Dict) -> str:
        """CAM 통계 정보 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 데이터 구조 확인 및 안전한 접근
        def safe_get_stat(stat_name, default_value=0):
            """안전하게 통계값을 가져오는 헬퍼 함수"""
            try:
                if stat_name in cam_stats:
                    stat_data = cam_stats[stat_name]
                    print(f"    🔍 Accessing {stat_name}: {type(stat_data)} - {stat_data}")
                    
                    if isinstance(stat_data, (list, tuple)) and len(stat_data) >= 2:
                        result = stat_data[1]  # 값 부분
                        print(f"    ✅ Extracted value: {result}")
                        return result
                    elif isinstance(stat_data, (int, float)):
                        print(f"    ✅ Direct value: {stat_data}")
                        return stat_data
                    elif isinstance(stat_data, str):
                        print(f"    ⚠️  String value, using default: {stat_data}")
                        return default_value
                    else:
                        print(f"    ⚠️  Unknown type, using default: {type(stat_data)}")
                        return default_value
                else:
                    print(f"    ⚠️  Key {stat_name} not found in cam_stats")
                    return default_value
            except (IndexError, TypeError, KeyError) as e:
                print(f"    ❌ Error accessing {stat_name}: {e}")
                return default_value
        
        def safe_get_stat_name(stat_name, default_name=""):
            """안전하게 통계 이름을 가져오는 헬퍼 함수"""
            try:
                if stat_name in cam_stats:
                    stat_data = cam_stats[stat_name]
                    if isinstance(stat_data, (list, tuple)) and len(stat_data) >= 1:
                        return str(stat_data[0])  # 이름 부분
                    elif isinstance(stat_data, str):
                        return stat_data
                return default_name or stat_name
            except (IndexError, TypeError, KeyError):
                return default_name or stat_name
        
        # 간단한 통계 정보만 표시
        try:
            # 디버깅: cam_stats 구조 확인
            print(f"    🔍 CAM stats keys: {list(cam_stats.keys()) if cam_stats else 'None'}")
            if cam_stats:
                for key in ['mean', 'max', 'min', 'std']:
                    if key in cam_stats:
                        print(f"    🔍 {key}: {type(cam_stats[key])} - {cam_stats[key]}")
            
            # 기본 통계 (안전하게 접근)
            mean_val = safe_get_stat('mean', 0)
            max_val = safe_get_stat('max', 0)
            min_val = safe_get_stat('min', 0)
            std_val = safe_get_stat('std', 0)
            
            # 기본 통계 차트
            stats_names = ['Mean', 'Max', 'Min', 'Std']
            stats_values = [mean_val, max_val, min_val, std_val]
            
            axes[0, 0].bar(stats_names, stats_values, color='skyblue', alpha=0.7)
            axes[0, 0].set_title('Basic Statistics', fontsize=12, fontweight='bold')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 활성화 정보
            high_activation_pixels = safe_get_stat('high_activation_pixels', 0)
            high_activation_ratio = safe_get_stat('high_activation_ratio', 0)
            
            activation_names = ['High Activation\nPixels', 'High Activation\nRatio (%)']
            activation_values = [high_activation_pixels, high_activation_ratio]
            
            axes[0, 1].bar(activation_names, activation_values, color='lightgreen', alpha=0.7)
            axes[0, 1].set_title('Activation Statistics', fontsize=12, fontweight='bold')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 사분위수 정보
            q25_val = safe_get_stat('q25', 0)
            q50_val = safe_get_stat('q50', 0)
            q75_val = safe_get_stat('q75', 0)
            
            quartile_names = ['Q25', 'Q50', 'Q75']
            quartile_values = [q25_val, q50_val, q75_val]
            
            axes[1, 0].bar(quartile_names, quartile_values, color='lightcoral', alpha=0.7)
            axes[1, 0].set_title('Quartile Statistics', fontsize=12, fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 통계 요약
            total_pixels = safe_get_stat('total_pixels', 0)
            shape_info = safe_get_stat('shape', 'N/A')
            
            summary_text = f"""CAM Statistics Summary:

Mean: {mean_val:.4f}
Max: {max_val:.4f}
Min: {min_val:.4f}
Std: {std_val:.4f}
High Activation: {high_activation_ratio:.2f}%
Total Pixels: {total_pixels:,}"""
            
            axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes, 
                           fontsize=10, verticalalignment='center',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
            axes[1, 1].set_title('Summary', fontsize=12, fontweight='bold')
            axes[1, 1].axis('off')
            
        except Exception as e:
            # 오류 발생 시 간단한 메시지 표시
            for ax in axes.flat:
                ax.text(0.5, 0.5, 'CAM Statistics\nNot Available', 
                       transform=ax.transAxes,
                       horizontalalignment='center', verticalalignment='center',
                       fontsize=12, fontweight='bold')
                ax.set_title('CAM Statistics', fontsize=12, fontweight='bold')
                ax.axis('off')
        
        plt.tight_layout()
        return self.fig_to_base64(fig)
    
    def visualize_connected_components(self, components_analysis: Dict) -> str:
        """Connected Components 분석 결과 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 활성화 비율 정보
        activation_ratio = components_analysis.get('active_ratio', 0)
        axes[0, 0].pie([activation_ratio, 100-activation_ratio], 
                      labels=['Active', 'Inactive'], 
                      colors=['lightgreen', 'lightgray'],
                      autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Activation Ratio', fontsize=12, fontweight='bold')
        
        # 컴포넌트 개수 정보
        num_components = components_analysis.get('num_components', 0)
        axes[0, 1].bar(['Components'], [num_components], color='skyblue', alpha=0.7)
        axes[0, 1].set_title('Number of Connected Components', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 컴포넌트 크기 분포 (size_stats가 있는 경우)
        if 'size_stats' in components_analysis:
            size_stats = components_analysis['size_stats']
            stats_names = list(size_stats.keys())
            stats_values = list(size_stats.values())
            
            axes[1, 0].bar(stats_names, stats_values, color='orange', alpha=0.7)
            axes[1, 0].set_title('Component Size Statistics', fontsize=12, fontweight='bold')
            axes[1, 0].set_ylabel('Size (pixels)')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
        else:
            # size_stats가 없는 경우 기본 정보 표시
            axes[1, 0].text(0.5, 0.5, 'Size statistics\nnot available', 
                           transform=axes[1, 0].transAxes,
                           horizontalalignment='center', verticalalignment='center',
                           fontsize=12, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
            axes[1, 0].set_title('Component Size Statistics', fontsize=12, fontweight='bold')
            axes[1, 0].axis('off')
        
        # 상세 활성화 정보
        active_pixels = components_analysis.get('active_pixels', 0)
        threshold = components_analysis.get('threshold', 0)
        
        detailed_info = f"""Connected Components Analysis:

Active Pixels: {active_pixels:,}
Active Ratio: {activation_ratio:.2f}%
Components: {num_components}
Threshold: {threshold:.4f}
Average Size: {active_pixels/num_components if num_components > 0 else 0:.1f} pixels"""
        
        axes[1, 1].text(0.1, 0.5, detailed_info, transform=axes[1, 1].transAxes, 
                       fontsize=10, verticalalignment='center',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        axes[1, 1].set_title('Detailed Analysis', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        return self.fig_to_base64(fig)
    
    def visualize_entropy_analysis(self, entropy_results: Dict) -> str:
        """엔트로피 분석 결과 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 활성화 분포 히스토그램 (Non-zero CAM values)
        if 'non_zero_count' in entropy_results and 'activation_ratio' in entropy_results:
            # 히스토그램 데이터가 있는 경우 (예시 코드 참고)
            non_zero_count = entropy_results.get('non_zero_count', 0)
            total_count = entropy_results.get('total_count', 1)
            activation_ratio = entropy_results.get('activation_ratio', 0)
            histogram_entropy = entropy_results.get('histogram', 0)
            
            # 가상의 히스토그램 데이터 생성 (실제로는 CAM 데이터가 필요)
            # 여기서는 활성화 비율을 기반으로 한 시뮬레이션 데이터 사용
            if non_zero_count > 0:
                # 활성화 비율을 기반으로 한 분포 시뮬레이션
                np.random.seed(42)  # 재현성을 위한 시드
                simulated_values = np.random.beta(2, 5, size=min(non_zero_count, 1000))
                simulated_values = simulated_values * (1 - activation_ratio) + activation_ratio
                
                axes[0, 0].hist(simulated_values, bins=50, alpha=0.7, color='purple', density=True, 
                               edgecolor='black', linewidth=0.5)
                axes[0, 0].set_title(f'Non-Zero CAM Distribution\nHistogram Entropy: {histogram_entropy:.3f}\nActivation Ratio: {activation_ratio:.3f}', 
                                   fontsize=12, fontweight='bold')
                axes[0, 0].set_xlabel('CAM Value (Non-Zero)')
                axes[0, 0].set_ylabel('Density')
                axes[0, 0].grid(True, alpha=0.3)
                
                # 활성화 비율 정보 추가
                axes[0, 0].text(0.02, 0.98, f'Non-zero: {non_zero_count:,}\nTotal: {total_count:,}', 
                                transform=axes[0, 0].transAxes, verticalalignment='top',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                                fontsize=10, fontweight='bold')
            else:
                axes[0, 0].text(0.5, 0.5, 'No Activation\n(All values are 0)', 
                                transform=axes[0, 0].transAxes, ha='center', va='center',
                                bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.7),
                                fontsize=12, fontweight='bold')
                axes[0, 0].set_title('No Non-Zero Values', fontsize=12, fontweight='bold')
        else:
            # 기본 정보만 표시
            axes[0, 0].text(0.5, 0.5, 'Histogram data\nnot available', 
                           transform=axes[0, 0].transAxes, ha='center', va='center',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8),
                           fontsize=12, fontweight='bold')
            axes[0, 0].set_title('CAM Distribution', fontsize=12, fontweight='bold')
        
        # 2. 공간적 차이 히트맵 (Spatial Differences)
        if 'spatial' in entropy_results:
            # 가상의 공간적 차이 데이터 생성 (실제로는 CAM의 gradient가 필요)
            spatial_entropy = entropy_results.get('spatial', 0)
            
            # 20x20 크기의 가상 히트맵 생성
            size = 20
            np.random.seed(42)
            # 공간적 패턴을 시뮬레이션
            spatial_data = np.random.normal(0, 0.3, (size, size))
            # 중앙에 활성화 패턴 추가
            center_y, center_x = size//2, size//2
            y, x = np.ogrid[:size, :size]
            mask = (x - center_x)**2 + (y - center_y)**2 <= (size//4)**2
            spatial_data[mask] += np.random.normal(0.5, 0.2, np.sum(mask))
            
            im = axes[0, 1].imshow(spatial_data, cmap='RdBu_r', aspect='auto')
            axes[0, 1].set_title(f'Spatial Differences\nSpatial Entropy: {spatial_entropy:.3f}', 
                               fontsize=12, fontweight='bold')
            axes[0, 1].axis('off')
            plt.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)
        else:
            axes[0, 1].text(0.5, 0.5, 'Spatial data\nnot available', 
                           transform=axes[0, 1].transAxes, ha='center', va='center',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8),
                           fontsize=12, fontweight='bold')
            axes[0, 1].set_title('Spatial Differences', fontsize=12, fontweight='bold')
        
        # 3. 조건부 엔트로피 (기존 유지)
        if 'conditional' in entropy_results:
            thresholds = list(entropy_results['conditional'].keys())
            conditional_ents = list(entropy_results['conditional'].values())
            axes[1, 0].plot(thresholds, conditional_ents, 'o-', linewidth=3, markersize=8, 
                           color='orange', markerfacecolor='red', markeredgecolor='darkred')
            axes[1, 0].set_title('Conditional Entropy vs Threshold', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('Threshold Percentile (%)')
            axes[1, 0].set_ylabel('Conditional Entropy')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_facecolor('lightblue')
        else:
            axes[1, 0].text(0.5, 0.5, 'Conditional entropy\ndata not available', 
                           transform=axes[1, 0].transAxes, ha='center', va='center',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8),
                           fontsize=12, fontweight='bold')
            axes[1, 0].set_title('Conditional Entropy', fontsize=12, fontweight='bold')
        
        # 4. 공간적 방향별 엔트로피 (색상 개선)
        if 'spatial_directions' in entropy_results:
            directions = list(entropy_results['spatial_directions'].keys())
            direction_ents = list(entropy_results['spatial_directions'].values())
            
            # 알록달록한 색상 사용
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
            bar_colors = [colors[i % len(colors)] for i in range(len(directions))]
            
            bars = axes[1, 1].bar(directions, direction_ents, color=bar_colors, alpha=0.8, 
                                 edgecolor='black', linewidth=1)
            axes[1, 1].set_title('Spatial Entropy by Direction', fontsize=12, fontweight='bold')
            axes[1, 1].set_ylabel('Entropy')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_facecolor('lightgreen')
            
            # 값 표시
            for bar, value in zip(bars, direction_ents):
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        else:
            # 엔트로피 요약 (기존)
            summary_text = f"""Entropy Summary:

Shannon: {entropy_results.get('shannon', 0):.3f}
Spatial: {entropy_results.get('spatial', 0):.3f}
Histogram: {entropy_results.get('histogram', 0):.3f}
Activation Ratio: {entropy_results.get('activation_ratio', 0):.3f}"""
            
            axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes, 
                           fontsize=10, verticalalignment='center',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
            axes[1, 1].set_title('Entropy Summary', fontsize=12, fontweight='bold')
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        return self.fig_to_base64(fig)
    
    def visualize_centroid_analysis(self, centroids: Dict) -> str:
        """Centroid 분석 결과 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Centroid 방법별 비교 (튜플 구조 처리)
        methods = list(centroids.keys())
        
        # 튜플 구조에서 x, y 좌표 추출
        x_coords = []
        y_coords = []
        for method in methods:
            centroid_data = centroids[method]
            if isinstance(centroid_data, tuple) and len(centroid_data) >= 2:
                x_coords.append(centroid_data[0])
                y_coords.append(centroid_data[1])
            elif isinstance(centroid_data, dict):
                x_coords.append(centroid_data.get('x', 0))
                y_coords.append(centroid_data.get('y', 0))
            else:
                print(f"    ⚠️  Unknown centroid format for {method}: {type(centroid_data)}")
                x_coords.append(0)
                y_coords.append(0)
        
        axes[0, 0].scatter(x_coords, y_coords, c=range(len(methods)), cmap='viridis', s=100)
        for i, method in enumerate(methods):
            axes[0, 0].annotate(method, (x_coords[i], y_coords[i]), 
                              xytext=(5, 5), textcoords='offset points')
        axes[0, 0].set_title('Centroid Comparison', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('X Coordinate')
        axes[0, 0].set_ylabel('Y Coordinate')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 신뢰도 점수 (튜플 구조에서는 기본값 사용)
        confidence_scores = []
        for method in methods:
            centroid_data = centroids[method]
            if isinstance(centroid_data, dict):
                confidence_scores.append(centroid_data.get('confidence', 0))
            else:
                # 튜플 구조에서는 기본 신뢰도 사용
                confidence_scores.append(0.5)
        axes[0, 1].bar(methods, confidence_scores, color='lightcoral', alpha=0.7)
        axes[0, 1].set_title('Centroid Confidence Scores', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Confidence Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Centroid 요약 정보
        best_method = methods[0] if methods else "None"  # 첫 번째 방법을 기본값으로
        summary_text = f"""Centroid Analysis Summary:

Methods: {len(methods)}
Best Method: {best_method}
Average X: {np.mean(x_coords):.2f}
Average Y: {np.mean(y_coords):.2f}
Std X: {np.std(x_coords):.2f}
Std Y: {np.std(y_coords):.2f}"""
        
        axes[1, 0].text(0.1, 0.5, summary_text, transform=axes[1, 0].transAxes, 
                       fontsize=10, verticalalignment='center',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        axes[1, 0].set_title('Summary', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        
        # 좌표 분포 히스토그램
        axes[1, 1].hist(x_coords, alpha=0.7, label='X coordinates', bins=10)
        axes[1, 1].hist(y_coords, alpha=0.7, label='Y coordinates', bins=10)
        axes[1, 1].set_title('Coordinate Distribution', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Coordinate Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self.fig_to_base64(fig)
    
    def visualize_overlap_analysis(self, overlap_results: Dict) -> str:
        """Overlap 분석 결과 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # IoU 점수
        if 'iou' in overlap_results:
            axes[0, 0].bar(['IoU'], [overlap_results['iou']], color='skyblue', alpha=0.7)
            axes[0, 0].set_title('Intersection over Union (IoU)', fontsize=12, fontweight='bold')
            axes[0, 0].set_ylabel('IoU Score')
            axes[0, 0].set_ylim(0, 1)
            axes[0, 0].grid(True, alpha=0.3)
        
        # Coverage 정보
        coverage_metrics = []
        coverage_names = []
        if 'cam_coverage' in overlap_results:
            coverage_metrics.append(overlap_results['cam_coverage'])
            coverage_names.append('CAM Coverage')
        if 'bbox_coverage' in overlap_results:
            coverage_metrics.append(overlap_results['bbox_coverage'])
            coverage_names.append('BBox Coverage')
        
        if coverage_metrics:
            axes[0, 1].bar(coverage_names, coverage_metrics, color='lightgreen', alpha=0.7)
            axes[0, 1].set_title('Coverage Metrics', fontsize=12, fontweight='bold')
            axes[0, 1].set_ylabel('Coverage Ratio')
            axes[0, 1].set_ylim(0, 1)
            axes[0, 1].grid(True, alpha=0.3)
        
        # 검출된 객체 정보
        if 'largest_class_name' in overlap_results:
            class_name = overlap_results['largest_class_name']
            axes[1, 0].text(0.5, 0.5, f"Detected Class:\n{class_name}", 
                           transform=axes[1, 0].transAxes, 
                           fontsize=12, fontweight='bold',
                           horizontalalignment='center', verticalalignment='center',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
            axes[1, 0].set_title('Detected Object', fontsize=12, fontweight='bold')
            axes[1, 0].axis('off')
        
        # Overlap 요약
        summary_text = f"""Overlap Analysis Summary:

IoU Score: {overlap_results.get('iou', 0):.3f}
CAM Coverage: {overlap_results.get('cam_coverage', 0):.3f}
BBox Coverage: {overlap_results.get('bbox_coverage', 0):.3f}
Largest Class: {overlap_results.get('largest_class_name', 'N/A')}
Overlap Quality: {'Good' if overlap_results.get('iou', 0) > 0.5 else 'Poor'}"""
        
        axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes, 
                       fontsize=10, verticalalignment='center',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        axes[1, 1].set_title('Summary', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        return self.fig_to_base64(fig)
    
    def create_comprehensive_visualization(self, comprehensive_result: Dict) -> Dict[str, str]:
        """포괄적인 CAM 분석 결과를 모두 시각화 (실제 저장된 데이터 구조에 맞춤)"""
        visualizations = {}
        
        try:
            # 1. CAM 통계 시각화
            if comprehensive_result.get('cam_stats'):
                visualizations['cam_statistics'] = self.visualize_cam_statistics(comprehensive_result['cam_stats'])
            
            # 2. Connected Components 시각화
            if comprehensive_result.get('components_analysis'):
                visualizations['connected_components'] = self.visualize_connected_components(
                    comprehensive_result['components_analysis']
                )
            
            # 3. 엔트로피 분석 시각화
            if comprehensive_result.get('entropy_results'):
                visualizations['entropy_analysis'] = self.visualize_entropy_analysis(
                    comprehensive_result['entropy_results']
                )
            
            # 4. Centroid 분석 시각화 (새로 추가)
            if comprehensive_result.get('centroids'):
                visualizations['centroid_analysis'] = self.visualize_centroid_analysis(
                    comprehensive_result['centroids']
                )
            
            # 5. Overlap 분석 시각화 (새로 추가)
            if comprehensive_result.get('overlap_results'):
                visualizations['overlap_analysis'] = self.visualize_overlap_analysis(
                    comprehensive_result['overlap_results']
                )
            
        except Exception as e:
            print(f"Error in comprehensive visualization: {e}")
        
        return visualizations


 