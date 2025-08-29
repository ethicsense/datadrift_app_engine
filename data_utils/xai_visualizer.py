import matplotlib
matplotlib.use('Agg')  # GUI 백엔드 비활성화로 메모리 사용량 최적화
import matplotlib.pyplot as plt
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import base64
from io import BytesIO
from pytorch_grad_cam.utils.image import show_cam_on_image
from yolo_cam.utils.image import show_cam_on_image as show_yolocam_on_image
import os
import multiprocessing as mp
from functools import partial


# 전역 함수로 정의 (병렬 처리용)
def process_single_visualization_wrapper(filename_xai_tuple):
    """단일 XAI 결과를 처리하는 전역 함수 (병렬 처리용)"""
    filename, xai_result = filename_xai_tuple
    try:
        # 각 프로세스에서 새로운 XAIVisualizer 인스턴스 생성
        visualizer = XAIVisualizer()
        # 개별 시각화 생성
        visualizations = visualizer.create_comprehensive_visualization(xai_result)
        return filename, visualizations
    except Exception as e:
        print(f"    ❌ Error processing {filename}: {e}")
        return filename, {}


class XAIVisualizer:
    """XAI 분석 결과를 시각화하는 클래스"""
    
    def __init__(self):
        """XAI 시각화기 초기화"""
        plt.style.use('default')
        plt.rcParams['font.size'] = 14
        # 메모리 최적화를 위한 설정
        plt.rcParams['figure.max_open_warning'] = 0
        plt.rcParams['figure.dpi'] = 100  # DPI 낮춰서 메모리 절약
        
        # 경고 메시지 억제
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        warnings.filterwarnings('ignore', message='.*Glyph.*missing from font.*')
        warnings.filterwarnings('ignore', message='.*This figure includes Axes.*not compatible.*')
        
        # 이모지 폰트 문제 해결을 위한 설정
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'AppleGothic', 'sans-serif']
    

    
    def _safe_tight_layout(self, fig):
        """안전한 tight_layout 적용"""
        try:
            fig.tight_layout()
        except:
            # tight_layout이 실패하면 수동으로 조정
            fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.3, wspace=0.3)
    
    def fig_to_base64(self, fig: plt.Figure) -> str:
        """matplotlib figure를 base64 인코딩된 이미지로 변환 (최적화됨)"""
        try:
            buf = BytesIO()
            # DPI를 낮춰서 메모리 사용량 절약 (300 -> 150)
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            buf.seek(0)
            img_str = base64.b64encode(buf.getvalue()).decode()
            buf.close()
            # 즉시 figure 닫기로 메모리 해제
            plt.close(fig)
            return img_str
        except Exception as e:
            plt.close(fig)
            raise e
    
    def create_comprehensive_visualization_batch(self, xai_results: List[Tuple[str, Dict]]) -> Dict[str, str]:
        """여러 XAI 결과를 배치로 처리하여 시각화 생성 (병렬 처리)"""
        if not xai_results:
            return {}
        
        # CPU 코어 수에 따라 워커 수 결정 (최대 4개)
        num_workers = min(4, mp.cpu_count())
        
        print(f"    🔄 Starting batch visualization with {num_workers} workers...")
        
        # 병렬 처리 실행 (전역 함수 사용)
        with mp.Pool(processes=num_workers) as pool:
            results = pool.map(process_single_visualization_wrapper, xai_results)
        
        # 결과 병합
        all_visualizations = {}
        for filename, visualizations in results:
            if visualizations:
                for viz_type, viz_data in visualizations.items():
                    key = f"{filename}_{viz_type}"
                    all_visualizations[key] = viz_data
        
        print(f"    ✅ Batch visualization completed: {len(all_visualizations)} visualizations")
        return all_visualizations
    
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
        
        # Labeled Mask 시각화 (바 그래프 대신)
        num_components = components_analysis.get('num_components', 0)
        optimal_threshold_info = components_analysis.get('optimal_threshold_info', {})
        
        # CAM 데이터에서 labeled_mask 재생성 (이미 로드된 CAM 데이터 사용)
        labeled_mask = None
        if num_components > 0 and optimal_threshold_info:
            try:
                # 이미 로드된 CAM 데이터 사용
                if hasattr(self, '_current_cam_data') and self._current_cam_data is not None:
                    grayscale_cam = self._current_cam_data
                else:
                    print(f"    ⚠️  CAM data not available for labeled_mask generation")
                    return
                    
                # 최적 임계값으로 labeled_mask 생성
                from scipy import ndimage
                optimal_threshold = optimal_threshold_info.get('optimal_threshold', 0)
                binary_mask = grayscale_cam > optimal_threshold
                labeled_mask, _ = ndimage.label(binary_mask)
                
                print(f"    ✅ Generated labeled_mask with optimal threshold: {optimal_threshold:.4f}")

            except Exception as e:
                print(f"    ❌ Failed to generate labeled_mask: {e}")
        
        # Labeled mask 시각화
        if labeled_mask is not None and num_components > 0:
            # 알록달록한 색상 팔레트 생성
            colors = plt.cm.Set3(np.linspace(0, 1, num_components + 1))  # +1 for background
            colors[0] = [0.9, 0.9, 0.9, 1.0]  # 배경색을 연한 회색으로
            
            # Labeled mask 표시
            im = axes[0, 1].imshow(labeled_mask, cmap='Set3', interpolation='nearest')
            optimal_percentile = optimal_threshold_info.get('optimal_percentile', 0)
            optimal_threshold = optimal_threshold_info.get('optimal_threshold', 0)
            axes[0, 1].set_title(f'Labeled Components\n({num_components} components)\nOptimal: {optimal_percentile}% ({optimal_threshold:.3f})', 
                               fontsize=12, fontweight='bold')
            axes[0, 1].axis('off')
            
            # 컬러바 추가
            from matplotlib.patches import Patch
            legend_elements = []
            for i in range(1, min(num_components + 1, 8)):  # 최대 7개 컴포넌트만 표시
                legend_elements.append(Patch(facecolor=colors[i], label=f'Component {i}'))
            
            if legend_elements:
                axes[0, 1].legend(handles=legend_elements, loc='upper right', fontsize=8)
        else:
            # Labeled mask가 없는 경우 기본 정보 표시
            axes[0, 1].text(0.5, 0.5, f'Labeled Mask\nNot Available\n\nComponents: {num_components}', 
                           transform=axes[0, 1].transAxes,
                           horizontalalignment='center', verticalalignment='center',
                           fontsize=14, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
            axes[0, 1].set_title('Labeled Components', fontsize=12, fontweight='bold')
            axes[0, 1].axis('off')
        
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
                           fontsize=16, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
            axes[1, 0].set_title('Component Size Statistics', fontsize=16, fontweight='bold')
            axes[1, 0].axis('off')
        
        # 상세 활성화 정보
        active_pixels = components_analysis.get('active_pixels', 0)
        threshold = components_analysis.get('threshold', 0)
        optimal_percentile = optimal_threshold_info.get('optimal_percentile', 0)
        optimal_threshold = optimal_threshold_info.get('optimal_threshold', 0)
        
        detailed_info = f"""Connected Components Analysis:

Active Pixels: {active_pixels:,}
Active Ratio: {activation_ratio:.2f}%
Components: {num_components}
Current Threshold: {threshold:.4f}
Optimal Threshold: {optimal_threshold:.4f} ({optimal_percentile}%)
Average Size: {active_pixels/num_components if num_components > 0 else 0:.1f} pixels"""
        
        axes[1, 1].text(0.1, 0.5, detailed_info, transform=axes[1, 1].transAxes, 
                       fontsize=16, verticalalignment='center',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        axes[1, 1].set_title('Detailed Analysis', fontsize=18, fontweight='bold')
        axes[1, 1].axis('off')
        
        self._safe_tight_layout(fig)
        return self.fig_to_base64(fig)
    
    def visualize_entropy_analysis(self, entropy_results: Dict, cam_data: np.ndarray = None) -> str:
        """엔트로피 분석 결과 시각화 - 실제 CAM 데이터 사용"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 활성화 분포 히스토그램 (Non-zero CAM values)
        if 'non_zero_count' in entropy_results and 'activation_ratio' in entropy_results:
            non_zero_count = entropy_results.get('non_zero_count', 0)
            total_count = entropy_results.get('total_count', 1)
            activation_ratio = entropy_results.get('activation_ratio', 0)
            histogram_entropy = entropy_results.get('histogram', 0)
            
            # 실제 CAM 데이터 사용
            if cam_data is not None:
                # 실제 CAM 데이터에서 non-zero 값들 추출
                non_zero_cam = cam_data.flatten()[cam_data.flatten() > 0]
                
                if len(non_zero_cam) > 0:
                    # 실제 데이터로 히스토그램 생성
                    axes[0, 0].hist(non_zero_cam, bins=50, alpha=0.7, color='purple', density=True, 
                                   edgecolor='black', linewidth=0.5)
                    axes[0, 0].set_title(f'Non-Zero CAM Distribution\nHistogram Entropy: {histogram_entropy:.3f}\nActivation Ratio: {activation_ratio:.3f}', 
                                       fontsize=12, fontweight='bold')
                    axes[0, 0].set_xlabel('CAM Value (Non-Zero)')
                    axes[0, 0].set_ylabel('Density')
                    axes[0, 0].grid(True, alpha=0.3)
                    
                    # 실제 데이터 정보 추가
                    axes[0, 0].text(0.02, 0.98, f'Non-zero: {len(non_zero_cam):,}\nTotal: {cam_data.size:,}\nMin: {non_zero_cam.min():.3f}\nMax: {non_zero_cam.max():.3f}', 
                                    transform=axes[0, 0].transAxes, verticalalignment='top',
                                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                                    fontsize=12, fontweight='bold')
                else:
                    axes[0, 0].text(0.5, 0.5, 'No Non-Zero Values\nin CAM Data', 
                                    transform=axes[0, 0].transAxes, ha='center', va='center',
                                    bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.7),
                                    fontsize=16, fontweight='bold')
                    axes[0, 0].set_title('No Non-Zero Values', fontsize=16, fontweight='bold')
            else:
                # CAM 데이터가 없는 경우 기존 방식 사용 (하지만 경고 표시)
                if non_zero_count > 0:
                    # 시뮬레이션 데이터 사용 (경고와 함께)
                    np.random.seed(42)
                    simulated_values = np.random.beta(2, 5, size=min(non_zero_count, 1000))
                    simulated_values = simulated_values * (1 - activation_ratio) + activation_ratio
                    
                    axes[0, 0].hist(simulated_values, bins=50, alpha=0.7, color='purple', density=True, 
                                   edgecolor='black', linewidth=0.5)
                    axes[0, 0].set_title(f'Non-Zero CAM Distribution (SIMULATED)\nHistogram Entropy: {histogram_entropy:.3f}\nActivation Ratio: {activation_ratio:.3f}', 
                                       fontsize=12, fontweight='bold')
                    axes[0, 0].set_xlabel('CAM Value (Non-Zero)')
                    axes[0, 0].set_ylabel('Density')
                    axes[0, 0].grid(True, alpha=0.3)
                    
                    # 시뮬레이션 경고 추가
                    axes[0, 0].text(0.02, 0.98, f'⚠️ SIMULATED DATA\nNon-zero: {non_zero_count:,}\nTotal: {total_count:,}', 
                                    transform=axes[0, 0].transAxes, verticalalignment='top',
                                    bbox=dict(boxstyle='round,pad=0.3', facecolor='orange', alpha=0.7),
                                    fontsize=12, fontweight='bold')
                else:
                    axes[0, 0].text(0.5, 0.5, 'No Activation\n(All values are 0)', 
                                    transform=axes[0, 0].transAxes, ha='center', va='center',
                                    bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.7),
                                    fontsize=16, fontweight='bold')
                    axes[0, 0].set_title('No Non-Zero Values', fontsize=16, fontweight='bold')
        else:
            # 기본 정보만 표시
            axes[0, 0].text(0.5, 0.5, 'Histogram data\nnot available', 
                           transform=axes[0, 0].transAxes, ha='center', va='center',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8),
                           fontsize=16, fontweight='bold')
            axes[0, 0].set_title('CAM Distribution', fontsize=16, fontweight='bold')
        
        # 2. 공간적 차이 히트맵 (Spatial Differences)
        if 'spatial' in entropy_results:
            spatial_entropy = entropy_results.get('spatial', 0)
            
            # 실제 CAM 데이터 사용
            if cam_data is not None:
                # CAM 데이터의 gradient 계산
                from scipy import ndimage
                
                # Sobel 필터를 사용한 gradient 계산
                grad_x = ndimage.sobel(cam_data, axis=1)
                grad_y = ndimage.sobel(cam_data, axis=0)
                gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                
                # Gradient magnitude를 히트맵으로 표시
                im = axes[0, 1].imshow(gradient_magnitude, cmap='RdBu_r', aspect='auto')
                axes[0, 1].set_title(f'Spatial Differences (Gradient)\nSpatial Entropy: {spatial_entropy:.3f}', 
                                   fontsize=12, fontweight='bold')
                axes[0, 1].axis('off')
                plt.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)
            else:
                # CAM 데이터가 없는 경우 기존 시뮬레이션 사용 (경고와 함께)
                size = 20
                np.random.seed(42)
                spatial_data = np.random.normal(0, 0.3, (size, size))
                center_y, center_x = size//2, size//2
                y, x = np.ogrid[:size, :size]
                mask = (x - center_x)**2 + (y - center_y)**2 <= (size//4)**2
                spatial_data[mask] += np.random.normal(0.5, 0.2, np.sum(mask))
                
                im = axes[0, 1].imshow(spatial_data, cmap='RdBu_r', aspect='auto')
                axes[0, 1].set_title(f'Spatial Differences (SIMULATED)\nSpatial Entropy: {spatial_entropy:.3f}', 
                                   fontsize=12, fontweight='bold')
                axes[0, 1].axis('off')
                plt.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)
                    
                # 시뮬레이션 경고 추가
                axes[0, 1].text(0.02, 0.98, '⚠️ SIMULATED DATA', 
                               transform=axes[0, 1].transAxes, verticalalignment='top',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='orange', alpha=0.7),
                               fontsize=12, fontweight='bold')
        else:
            axes[0, 1].text(0.5, 0.5, 'Spatial data\nnot available', 
                           transform=axes[0, 1].transAxes, ha='center', va='center',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8),
                           fontsize=16, fontweight='bold')
            axes[0, 1].set_title('Spatial Differences', fontsize=16, fontweight='bold')
        
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
                           fontsize=16, fontweight='bold')
            axes[1, 0].set_title('Conditional Entropy', fontsize=16, fontweight='bold')
        
        # 4. 공간적 방향별 엔트로피 (색상 개선)
        if 'spatial_directions' in entropy_results:
            directions = list(entropy_results['spatial_directions'].keys())
            direction_ents = list(entropy_results['spatial_directions'].values())
            
            # 알록달록한 색상 사용
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
            bar_colors = [colors[i % len(colors)] for i in range(len(directions))]
            
            # 0이 아닌 값들만 바 차트로 표시
            non_zero_directions = []
            non_zero_ents = []
            non_zero_colors = []
            
            for i, (direction, value) in enumerate(zip(directions, direction_ents)):
                if value > 0:
                    non_zero_directions.append(direction)
                    non_zero_ents.append(value)
                    non_zero_colors.append(bar_colors[i])
            
            if non_zero_directions:
                bars = axes[1, 1].bar(non_zero_directions, non_zero_ents, color=non_zero_colors, alpha=0.8, 
                                     edgecolor='black', linewidth=1)
                axes[1, 1].set_title('Spatial Entropy by Direction', fontsize=12, fontweight='bold')
                axes[1, 1].set_ylabel('Entropy')
                axes[1, 1].grid(True, alpha=0.3)
                axes[1, 1].set_facecolor('lightgreen')
                
                # 값 표시 - 바 높이에 따라 위치 조정
                for bar, value in zip(bars, non_zero_ents):
                    height = bar.get_height()
                    # 바가 너무 작으면 위에, 크면 안쪽에 표시
                    if height < 0.01:
                        text_y = height + 0.002
                        va = 'bottom'
                    else:
                        text_y = height * 0.8
                        va = 'center'
                    
                    axes[1, 1].text(bar.get_x() + bar.get_width()/2., text_y,
                                   f'{value:.3f}', ha='center', va=va, fontsize=12, fontweight='bold',
                                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
                
                # 0인 값들은 별도로 표시
                zero_directions = [d for d, v in zip(directions, direction_ents) if v == 0]
                if zero_directions:
                    zero_text = f"Zero entropy: {', '.join(zero_directions)}"
                    axes[1, 1].text(0.02, 0.98, zero_text, transform=axes[1, 1].transAxes,
                                   fontsize=10, verticalalignment='top',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.7))
            else:
                # 모든 값이 0인 경우
                axes[1, 1].text(0.5, 0.5, 'All directions have\nzero entropy', 
                               transform=axes[1, 1].transAxes, ha='center', va='center',
                               fontsize=14, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
                axes[1, 1].set_title('Spatial Entropy by Direction', fontsize=12, fontweight='bold')
                axes[1, 1].set_ylabel('Entropy')
                axes[1, 1].grid(True, alpha=0.3)
                axes[1, 1].set_facecolor('lightgreen')
        else:
            # 엔트로피 요약 (기존)
            summary_text = f"""Entropy Summary:

Shannon: {entropy_results.get('shannon', 0):.3f}
Spatial: {entropy_results.get('spatial', 0):.3f}
Histogram: {entropy_results.get('histogram', 0):.3f}
Activation Ratio: {entropy_results.get('activation_ratio', 0):.3f}"""
            
            axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes, 
                           fontsize=16, verticalalignment='center',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
            axes[1, 1].set_title('Entropy Summary', fontsize=16, fontweight='bold')
            axes[1, 1].axis('off')
        
        self._safe_tight_layout(fig)
        return self.fig_to_base64(fig)
    
    def visualize_centroid_analysis(self, centroids: Dict, cam_result: Dict = None) -> str:
        """Centroid 분석 결과 시각화 - CAM overlay 위에 센트로이드 표시"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Centroid 방법별 비교 (새로운 딕셔너리 구조 처리)
        methods = list(centroids.keys())
        
        # 새로운 딕셔너리 구조에서 x, y 좌표와 confidence 추출
        x_coords = []
        y_coords = []
        confidence_scores = []
        descriptions = []
        
        for method in methods:
            centroid_data = centroids[method]
            if isinstance(centroid_data, dict):
                # 새로운 딕셔너리 구조
                x_coords.append(centroid_data.get('x', 0))
                y_coords.append(centroid_data.get('y', 0))
                confidence_scores.append(centroid_data.get('confidence', 0))
                descriptions.append(centroid_data.get('description', f'{method} centroid'))
            elif isinstance(centroid_data, tuple) and len(centroid_data) >= 2:
                # 기존 튜플 구조 (하위 호환성)
                x_coords.append(centroid_data[0])
                y_coords.append(centroid_data[1])
                confidence_scores.append(0.5)  # 기본값
                descriptions.append(f'{method} centroid (legacy)')
            else:
                print(f"    ⚠️  Unknown centroid format for {method}: {type(centroid_data)}")
                x_coords.append(0)
                y_coords.append(0)
                confidence_scores.append(0)
                descriptions.append(f'{method} centroid (error)')
        
        # 1. CAM Overlay 위에 센트로이드 표시 (메인 시각화)
        try:
            # cam_result를 사용하여 CAM overlay 생성
            if cam_result and cam_result.get('image_path') and cam_result.get('grayscale_cam') is not None:
                # 원본 이미지 로드
                original_img = cv2.imread(cam_result['image_path'])
                if original_img is not None:
                    # RGB 변환
                    rgb_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                    img = np.float32(rgb_img) / 255
                    
                    # CAM overlay 생성
                    grayscale_cam = cam_result['grayscale_cam']
                    cam_overlay = show_yolocam_on_image(img, grayscale_cam, use_rgb=True)
                    
                    # CAM overlay 표시
                    axes[0, 0].imshow(cam_overlay)
                    axes[0, 0].set_title('CAM Overlay with Centroids', fontsize=14, fontweight='bold')
                    
                    # 센트로이드들을 CAM overlay 위에 표시
                    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta']
                    for i, method in enumerate(methods):
                        if i < len(x_coords) and i < len(y_coords):
                            x, y = x_coords[i], y_coords[i]
                            confidence = confidence_scores[i]
                            color = colors[i % len(colors)]
                            
                            # 신뢰도에 따른 점 크기 조정
                            marker_size = 50 + int(confidence * 100)
                            
                            # 센트로이드 점 표시
                            axes[0, 0].scatter(x, y, c=color, s=marker_size, marker='o', 
                                             edgecolors='white', linewidth=2, alpha=0.8)
                            
                            # 레이블 표시 (알고리즘 + 좌표 + 신뢰도)
                            label = f"{method}\n({x:.1f}, {y:.1f})\nConf: {confidence:.3f}"
                            axes[0, 0].annotate(label, (x, y), 
                                              xytext=(10, 10), textcoords='offset points',
                                              fontsize=9, fontweight='bold',
                                              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                                              arrowprops=dict(arrowstyle='->', color=color, lw=2))
                    
                    axes[0, 0].axis('off')
                else:
                    print(f"    ❌ Failed to load image: {cam_result['image_path']}")
                    axes[0, 0].text(0.5, 0.5, 'Image Load Failed', 
                                   transform=axes[0, 0].transAxes,
                                   horizontalalignment='center', verticalalignment='center',
                                   fontsize=16, fontweight='bold')
                    axes[0, 0].set_title('CAM Overlay with Centroids', fontsize=14, fontweight='bold')
                    axes[0, 0].axis('off')
            else:
                print(f"    ❌ CAM data not available for centroid visualization")
                axes[0, 0].text(0.5, 0.5, 'CAM Data Not Available', 
                               transform=axes[0, 0].transAxes,
                               horizontalalignment='center', verticalalignment='center',
                               fontsize=16, fontweight='bold')
                axes[0, 0].set_title('CAM Overlay with Centroids', fontsize=14, fontweight='bold')
                axes[0, 0].axis('off')
        except Exception as e:
            print(f"    ❌ Failed to generate CAM overlay with centroids: {e}")
            axes[0, 0].text(0.5, 0.5, 'Visualization Error', 
                           transform=axes[0, 0].transAxes,
                           horizontalalignment='center', verticalalignment='center',
                           fontsize=16, fontweight='bold')
            axes[0, 0].set_title('CAM Overlay with Centroids', fontsize=14, fontweight='bold')
            axes[0, 0].axis('off')
        
        # 2. 신뢰도 점수 (실제 계산된 값 사용)
        if confidence_scores:
            bars = axes[0, 1].bar(methods, confidence_scores, color='lightcoral', alpha=0.7)
            axes[0, 1].set_title('Centroid Confidence Scores', fontsize=12, fontweight='bold')
            axes[0, 1].set_ylabel('Confidence Score')
            axes[0, 1].set_ylim(0, 1)
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3)
            
            # 값 표시
            for bar, value in zip(bars, confidence_scores):
                height = bar.get_height()
                axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                               f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        else:
            axes[0, 1].text(0.5, 0.5, 'Confidence Scores\nNot Available', 
                           transform=axes[0, 1].transAxes,
                           horizontalalignment='center', verticalalignment='center',
                           fontsize=16, fontweight='bold')
            axes[0, 1].set_title('Centroid Confidence Scores', fontsize=12, fontweight='bold')
        
        # 3. Centroid 요약 정보 (개선된 버전)
        if methods:
            best_method_idx = np.argmax(confidence_scores) if confidence_scores else 0
            best_method = methods[best_method_idx]
            best_confidence = confidence_scores[best_method_idx] if confidence_scores else 0
            
            summary_text = f"""Centroid Analysis Summary:

Methods: {len(methods)}
Best Method: {best_method}
   Confidence: {best_confidence:.3f}

Statistics:
• Average X: {np.mean(x_coords):.2f}
• Average Y: {np.mean(y_coords):.2f}
• Std X: {np.std(x_coords):.2f}
• Std Y: {np.std(y_coords):.2f}

Method Descriptions:"""
            
            # 각 방법의 설명 추가
            for i, (method, desc) in enumerate(zip(methods, descriptions)):
                conf = confidence_scores[i] if i < len(confidence_scores) else 0
                summary_text += f"\n• {method}: {conf:.3f}"
                if len(desc) > 50:
                    desc = desc[:47] + "..."
                summary_text += f"\n  {desc}"
            
            axes[1, 0].text(0.05, 0.95, summary_text, transform=axes[1, 0].transAxes, 
                           fontsize=10, verticalalignment='top',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
            axes[1, 0].set_title('Summary & Descriptions', fontsize=16, fontweight='bold')
            axes[1, 0].axis('off')
        else:
            axes[1, 0].text(0.5, 0.5, 'No Centroid Data\nAvailable', 
                           transform=axes[1, 0].transAxes,
                           horizontalalignment='center', verticalalignment='center',
                           fontsize=16, fontweight='bold')
            axes[1, 0].set_title('Summary', fontsize=16, fontweight='bold')
            axes[1, 0].axis('off')
        
        # 4. 좌표 분포 히스토그램 (개선된 버전)
        if x_coords and y_coords:
            # X, Y 좌표를 별도로 히스토그램
            axes[1, 1].hist(x_coords, alpha=0.7, label='X coordinates', bins=min(10, len(x_coords)), 
                           color='skyblue', edgecolor='black')
            axes[1, 1].hist(y_coords, alpha=0.7, label='Y coordinates', bins=min(10, len(y_coords)), 
                           color='lightcoral', edgecolor='black')
            axes[1, 1].set_title('Coordinate Distribution', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Coordinate Value')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            # 통계 정보 추가
            stats_text = f'X: μ={np.mean(x_coords):.1f}, σ={np.std(x_coords):.1f}\nY: μ={np.mean(y_coords):.1f}, σ={np.std(y_coords):.1f}'
            axes[1, 1].text(0.02, 0.98, stats_text, transform=axes[1, 1].transAxes, 
                           verticalalignment='top', fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        else:
            axes[1, 1].text(0.5, 0.5, 'Coordinate Data\nNot Available', 
                           transform=axes[1, 1].transAxes,
                           horizontalalignment='center', verticalalignment='center',
                           fontsize=16, fontweight='bold')
            axes[1, 1].set_title('Coordinate Distribution', fontsize=12, fontweight='bold')
        
        self._safe_tight_layout(fig)
        return self.fig_to_base64(fig)
    

    
    def create_comprehensive_visualization(self, comprehensive_result: Dict) -> Dict[str, str]:
        """포괄적인 CAM 분석 결과를 모두 시각화 (독립적인 함수들 사용)"""
        visualizations = {}
        
        # comprehensive_result를 인스턴스 변수로 저장하여 다른 함수에서 접근 가능하도록 함
        self._current_comprehensive_result = comprehensive_result
        
        try:
            # 디버깅: comprehensive_result 키 확인
            # print(f"    🔍 Comprehensive result keys: {list(comprehensive_result.keys()) if comprehensive_result else 'None'}")
            
            # CAM 결과 구성 (파일 기반 로딩)
            cam_file_path = comprehensive_result.get('cam_file_path')
            grayscale_cam = None
            
            # 원본 CAM 파일에서 로드 시도 (한 번만 로드하고 저장)
            if cam_file_path and os.path.exists(cam_file_path):
                try:
                    from data_utils.xai_analyzer import XAIAnalyzer
                    analyzer = XAIAnalyzer()
                    grayscale_cam = analyzer.load_cam_data_original(cam_file_path)
                    # CAM 데이터를 인스턴스 변수로 저장하여 재사용
                    self._current_cam_data = grayscale_cam
                    print(f"    ✅ Loaded original CAM data from: {cam_file_path}")
                    print(f"    CAM data info: shape={grayscale_cam.shape}, dtype={grayscale_cam.dtype}")
                    print(f"    CAM data stats: min={grayscale_cam.min():.6f}, max={grayscale_cam.max():.6f}, mean={grayscale_cam.mean():.6f}")
                except Exception as e:
                    print(f"    ⚠️  Failed to load original CAM data from file: {e}")
                    grayscale_cam = None
                    self._current_cam_data = None
            
            # 파일 로드 실패 시 메타데이터 기반 시뮬레이션
            if grayscale_cam is None:
                cam_metadata = comprehensive_result.get('cam_metadata', {})
                if cam_metadata:
                    np.random.seed(42)  # 재현성을 위한 시드
                    shape = cam_metadata.get('shape', (224, 224))
                    mean_val = cam_metadata.get('mean', 0)
                    std_val = cam_metadata.get('std', 0.1)
                    min_val = cam_metadata.get('min', 0)
                    max_val = cam_metadata.get('max', 1)
                    
                    # CAM 데이터 시뮬레이션
                    grayscale_cam = np.random.normal(mean_val, std_val, shape)
                    grayscale_cam = np.clip(grayscale_cam, min_val, max_val)
                    print(f"    ⚠️  Using simulated CAM data from metadata")
                    print(f"    Simulated CAM data info: shape={grayscale_cam.shape}, dtype={grayscale_cam.dtype}")
                    print(f"    Simulated CAM data stats: min={grayscale_cam.min():.6f}, max={grayscale_cam.max():.6f}, mean={grayscale_cam.mean():.6f}")
            
            cam_result = {
                'image_path': comprehensive_result.get('image_path'),
                'grayscale_cam': grayscale_cam,
                'target_layers': comprehensive_result.get('target_layers', []),
                'target_layer_index': comprehensive_result.get('target_layer_index'),
                'model_name': comprehensive_result.get('model_name', 'Unknown')
            }
            
            # 1. CAM 히트맵 시각화 (원본, CAM, 오버레이)
            if cam_result['image_path'] and cam_result['grayscale_cam'] is not None:
                try:
                    visualizations['cam_heatmap'] = self.visualize_cam_heatmap(cam_result)
                except Exception as e:
                    print(f"    ⚠️  Failed to generate CAM heatmap: {e}")
            
            # 2. 임계값 기반 활성 영역 시각화
            if cam_result['image_path'] and cam_result['grayscale_cam'] is not None:
                try:
                    visualizations['cam_threshold_analysis'] = self.visualize_cam_threshold_analysis(cam_result)
                except Exception as e:
                    print(f"    ⚠️  Failed to generate CAM threshold analysis: {e}")
            
            # 3. CAM 통계 시각화 (실제 CAM 데이터 사용)
            if comprehensive_result.get('cam_stats'):
                visualizations['cam_statistics'] = self.visualize_cam_statistics(
                    comprehensive_result['cam_stats'],
                    cam_data=grayscale_cam  # 실제 CAM 데이터 전달
                )
            
            # 5. Connected Components 시각화
            if comprehensive_result.get('components_analysis'):
                visualizations['connected_components'] = self.visualize_connected_components(
                    comprehensive_result['components_analysis']
                )
            
            # 6. 엔트로피 분석 시각화
            if comprehensive_result.get('entropy_results'):
                visualizations['entropy_analysis'] = self.visualize_entropy_analysis(
                    comprehensive_result['entropy_results'], 
                    cam_data=grayscale_cam  # 실제 CAM 데이터 전달
                )
            
            # 7. Centroid 분석 시각화
            if comprehensive_result.get('centroids'):
                visualizations['centroid_analysis'] = self.visualize_centroid_analysis(
                    comprehensive_result['centroids'], cam_result
                )
            
            # 8. Overlap 분석 시각화
            if comprehensive_result.get('overlap_results'):
                visualizations['overlap_analysis'] = self.visualize_overlap_analysis(
                    comprehensive_result['overlap_results'], cam_result
                )
                # Overlap 통계 시각화도 추가
                visualizations['overlap_statistics'] = self.visualize_overlap_statistics(
                    comprehensive_result['overlap_results']
                )
            
        except Exception as e:
            print(f"Error in comprehensive visualization: {e}")
        
        return visualizations

    def visualize_cam_heatmap(self, cam_result: Dict) -> str:
        """
        CAM 히트맵 시각화 (1열 3행: 원본, CAM 오버레이, 분포 그래프)
        
        Args:
            cam_result: CAM 분석 결과 (image_path, grayscale_cam 포함)
            
        Returns:
            str: base64 인코딩된 이미지
        """
        # 1열 3행 레이아웃: 원본 이미지, CAM 오버레이, 분포 그래프
        fig = plt.figure(figsize=(12, 16))  # 세로로 긴 레이아웃
        
        # GridSpec을 사용하여 행별로 다른 높이 설정 - 모든 이미지 동일한 크기
        gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.3)
        
        # 원본 이미지 로드
        original_img = cv2.imread(cam_result['image_path'])
        if original_img is None:
            raise ValueError(f"Failed to load image: {cam_result['image_path']}")
        
        # RGB 변환
        rgb_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        img = np.float32(rgb_img) / 255
        
        grayscale_cam = cam_result['grayscale_cam']
        
        # 1행: 원본 이미지
        ax1 = fig.add_subplot(gs[0])
        ax1.imshow(rgb_img)
        ax1.set_title('Original Image', fontsize=16, fontweight='bold')
        ax1.axis('off')
        
        # 2행: CAM 오버레이 이미지 (가장 큰 공간)
        ax2 = fig.add_subplot(gs[1])
        cam_overlay = show_yolocam_on_image(img, grayscale_cam, use_rgb=True)
        ax2.imshow(cam_overlay)
        
        # 타겟 레이어 정보 추가
        target_layer_index = cam_result.get('target_layer_index')
        model_name = cam_result.get('model_name', 'Unknown')
        
        if target_layer_index is not None:
            title_text = f'CAM Overlay\nModel: {model_name} | Target: Layer {target_layer_index}'
        else:
            title_text = f'CAM Overlay\nModel: {model_name}'
        
        ax2.set_title(title_text, fontsize=16, fontweight='bold')
        ax2.axis('off')
        
        # 3행: CAM 분포 히스토그램
        ax3 = fig.add_subplot(gs[2])
        cam_values = grayscale_cam.flatten()
        ax3.hist(cam_values, bins=50, alpha=0.7, color='blue', label='CAM Values', edgecolor='black')
        ax3.set_title('CAM Activation Distribution', fontsize=16, fontweight='bold')
        ax3.set_xlabel('Activation Value')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 통계 정보 추가
        mean_val = np.mean(cam_values)
        std_val = np.std(cam_values)
        max_val = np.max(cam_values)
        active_ratio = np.sum(cam_values > 0.1) / len(cam_values) * 100
        
        stats_text = f'Mean: {mean_val:.4f}\nStd: {std_val:.4f}\nMax: {max_val:.4f}\nActive (>0.1): {active_ratio:.1f}%'
        ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes, 
                verticalalignment='top', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
        
        self._safe_tight_layout(fig)
        return self.fig_to_base64(fig)
    
    def visualize_cam_threshold_analysis(self, cam_result: Dict, 
                                       percentiles: List[int] = [80, 85, 90, 95]) -> str:
        """
        임계값 기반 CAM 활성 영역 시각화 (히트맵 + 분포 + 오버레이)
        
        Args:
            cam_result: CAM 분석 결과
            percentiles: 비교할 임계값 백분위수 리스트
            
        Returns:
            str: base64 인코딩된 이미지
        """
        fig, axes = plt.subplots(3, len(percentiles), figsize=(4*len(percentiles), 12))
        
        # 원본 이미지 로드
        original_img = cv2.imread(cam_result['image_path'])
        if original_img is None:
            raise ValueError(f"Failed to load image: {cam_result['image_path']}")
        
        # RGB 변환
        rgb_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        img = np.float32(rgb_img) / 255
        
        grayscale_cam = cam_result['grayscale_cam']
        
        for i, percentile in enumerate(percentiles):
            # Adaptive thresholding 적용
            adaptive_result = self._apply_threshold_to_cam(grayscale_cam, percentile)
            
            # 첫 번째 행: Adaptive Thresholding 히트맵
            axes[0, i].imshow(adaptive_result, cmap='hot')
            axes[0, i].set_title(f'Adaptive {percentile}%', fontsize=14, fontweight='bold')
            axes[0, i].axis('off')
            
            # 두 번째 행: Adaptive 분포 히스토그램 (0 값 제외)
            adaptive_values = adaptive_result.flatten()
            # 0이 아닌 값만 필터링
            non_zero_values = adaptive_values[adaptive_values > 0]
            
            if len(non_zero_values) > 0:
                axes[1, i].hist(non_zero_values, bins=50, alpha=0.7, color='red', label='Non-zero Adaptive')
                axes[1, i].set_title(f'Adaptive Distribution ({percentile}%)\n(0 values excluded)', fontsize=14, fontweight='bold')
                axes[1, i].set_xlabel('Adaptive Value')
                axes[1, i].set_ylabel('Frequency')
                axes[1, i].legend()
                
                # 통계 정보 추가
                non_zero_count = len(non_zero_values)
                total_count = len(adaptive_values)
                zero_ratio = (total_count - non_zero_count) / total_count * 100
                stats_text = f'Non-zero: {non_zero_count:,}\nZero ratio: {zero_ratio:.1f}%'
                axes[1, i].text(0.02, 0.98, stats_text, transform=axes[1, i].transAxes, 
                               verticalalignment='top', fontsize=10, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))
            else:
                # 모든 값이 0인 경우
                axes[1, i].text(0.5, 0.5, 'All values are 0\n(No activation)', 
                               transform=axes[1, i].transAxes, ha='center', va='center',
                               fontsize=14, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
                axes[1, i].set_title(f'Adaptive Distribution ({percentile}%)', fontsize=14, fontweight='bold')
            
            # 세 번째 행: CAM 오버레이
            cam_overlay = show_yolocam_on_image(img, adaptive_result, use_rgb=True)
            axes[2, i].imshow(cam_overlay)
            axes[2, i].set_title(f'CAM Overlay ({percentile}%)', fontsize=14, fontweight='bold')
            axes[2, i].axis('off')
        
        self._safe_tight_layout(fig)
        return self.fig_to_base64(fig)
    

    
    def _apply_threshold_to_cam(self, cam: np.ndarray, percentile: int) -> np.ndarray:
        """
        CAM에 임계값을 적용하여 활성 영역만 남김
        
        Args:
            cam: CAM 데이터
            percentile: 임계값 백분위수
            
        Returns:
            np.ndarray: 임계값 적용된 CAM
        """
        threshold = np.percentile(cam, percentile)
        cam_filtered = np.where(cam > threshold, cam, 0)
        
        if cam_filtered.max() > 0:
            return cam_filtered / cam_filtered.max()
        else:
            return cam_filtered

    def visualize_cam_statistics(self, cam_stats: Dict, cam_data: np.ndarray = None) -> str:
        """CAM 통계 정보 시각화 - Percentile과 Skewness 분석 포함 (실제 CAM 데이터 사용)"""
        fig, axes = plt.subplots(3, 2, figsize=(12, 18))
        
        try:
            # CAM stats에서 직접 값 추출 (튜플 구조 처리)
            def extract_stat_value(stat_name, default_value=0):
                """CAM stats에서 값을 추출하는 헬퍼 함수"""
                if stat_name in cam_stats:
                    stat_data = cam_stats[stat_name]
                    if isinstance(stat_data, (list, tuple)) and len(stat_data) >= 2:
                        return stat_data[1]  # 값 부분
                    elif isinstance(stat_data, (int, float)):
                        return stat_data
                    else:
                        print(f"    ⚠️  Unknown stat format for {stat_name}: {type(stat_data)}")
                        return default_value
                else:
                    print(f"    ⚠️  Key {stat_name} not found in cam_stats")
                    return default_value
            
            # CAM stats에서 값 추출
            mean_val = extract_stat_value('mean', 0)
            max_val = extract_stat_value('max', 0)
            min_val = extract_stat_value('min', 0)
            std_val = extract_stat_value('std', 0)
            high_activation_ratio = extract_stat_value('high_activation_ratio', 15.0)
            total_pixels = extract_stat_value('total_pixels', 50176)  # 224x224 기본값
            
            # 1. CAM Value Distribution (Boxplot)
            q25_val = extract_stat_value('q25', 0)
            q50_val = extract_stat_value('q50', 0)
            q75_val = extract_stat_value('q75', 0)
            
            # 박스플롯 데이터 준비
            box_data = [min_val, q25_val, q50_val, q75_val, max_val]
            axes[0, 0].boxplot([box_data], labels=['CAM Values'], patch_artist=True, 
                              boxprops=dict(facecolor='lightblue', alpha=0.7))
            axes[0, 0].set_title('CAM Value Distribution', fontsize=16, fontweight='bold')
            axes[0, 0].set_ylabel('CAM Value')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 통계값 텍스트 추가
            stats_text = f'Mean: {mean_val:.4f}\nStd: {std_val:.4f}\nRange: {max_val-min_val:.4f}'
            axes[0, 0].text(0.02, 0.98, stats_text, transform=axes[0, 0].transAxes, 
                           verticalalignment='top', fontsize=12, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
            
            # 2. Percentile Analysis (기존 Quartile Analysis 대체)
            percentiles = cam_stats.get('percentiles', {})
            if percentiles:
                percentile_names = list(percentiles.keys())
                percentile_values = list(percentiles.values())
                
                bars = axes[0, 1].bar(percentile_names, percentile_values, 
                                     color=['lightblue', 'skyblue', 'blue', 'navy', 'purple', 
                                            'darkred', 'red', 'orange', 'yellow'], alpha=0.7)
                axes[0, 1].set_title('CAM Activation Percentiles', fontsize=16, fontweight='bold')
                axes[0, 1].set_ylabel('Activation Value')
                axes[0, 1].tick_params(axis='x', rotation=45)
                axes[0, 1].grid(True, alpha=0.3)
                
                # 값 표시 - 바 높이에 따라 위치 조정
                for bar, value in zip(bars, percentile_values):
                    height = bar.get_height()
                    # 바가 너무 작으면 위에, 크면 안쪽에 표시
                    if height < 0.01:
                        text_y = height + 0.001
                        va = 'bottom'
                        fontsize = 8
                    else:
                        text_y = height * 0.8
                        va = 'center'
                        fontsize = 10
                    
                    axes[0, 1].text(bar.get_x() + bar.get_width()/2., text_y,
                                   f'{value:.3f}', ha='center', va=va, fontsize=fontsize, fontweight='bold',
                                   bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.8))
                
                # 해석 정보 추가
                interpretation = cam_stats.get('percentile_interpretation', {})
                if interpretation.get('high_concentration'):
                    concentration_text = 'High Concentration'
                    color = 'red'
                elif interpretation.get('moderate_concentration'):
                    concentration_text = 'Moderate Concentration'
                    color = 'orange'
                else:
                    concentration_text = 'Low Concentration'
                    color = 'green'
                
                axes[0, 1].text(0.02, 0.98, concentration_text, transform=axes[0, 1].transAxes, 
                               verticalalignment='top', fontsize=12, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))
            else:
                axes[0, 1].text(0.5, 0.5, 'Percentile Analysis\nNot Available', 
                               transform=axes[0, 1].transAxes,
                               horizontalalignment='center', verticalalignment='center',
                               fontsize=16, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
                axes[0, 1].set_title('CAM Activation Percentiles', fontsize=16, fontweight='bold')
            
            # 3. 임계값별 활성화 분석
            thresholds = np.linspace(0, max_val, 20)
            activation_ratios = []
            
            # 실제 CAM 데이터 사용
            if cam_data is not None:
                cam_values = cam_data.flatten()
                for threshold in thresholds:
                    if threshold == 0:
                        activation_ratios.append(100.0)
                    else:
                        # 실제 데이터로 활성화 비율 계산
                        ratio = np.sum(cam_values >= threshold) / len(cam_values) * 100
                        activation_ratios.append(ratio)
            else:
                # CAM 데이터가 없는 경우 시뮬레이션 사용 (경고와 함께)
                for threshold in thresholds:
                    if threshold == 0:
                        activation_ratios.append(100.0)
                    else:
                        # 임계값 이상의 활성화 비율 계산 (시뮬레이션)
                        ratio = np.sum(np.random.normal(mean_val, std_val, total_pixels) >= threshold) / total_pixels * 100
                        activation_ratios.append(ratio)
            
            axes[1, 0].plot(thresholds, activation_ratios, 'o-', linewidth=2, markersize=4, 
                           color='red', markerfacecolor='orange')
            
            # 제목에 실제 데이터 사용 여부 표시
            if cam_data is not None:
                title = 'Activation Ratio vs Threshold (Real Data)'
            else:
                title = 'Activation Ratio vs Threshold (Simulated)'
            
            axes[1, 0].set_title(title, fontsize=16, fontweight='bold')
            axes[1, 0].set_xlabel('Threshold')
            axes[1, 0].set_ylabel('Activation Ratio (%)')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_ylim(0, 105)
            
            # 시뮬레이션 사용 시 경고 추가
            if cam_data is None:
                axes[1, 0].text(0.02, 0.98, '⚠️ SIMULATED DATA', 
                               transform=axes[1, 0].transAxes, verticalalignment='top',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='orange', alpha=0.7),
                               fontsize=12, fontweight='bold')
            
            # 4. Skewness Analysis (기존 CAM Value Histogram 대체)
            skewness = cam_stats.get('skewness', 0)
            percentile_skewness = cam_stats.get('percentile_skewness', 0)
            skewness_type = cam_stats.get('skewness_type', 'unknown')
            distribution_type = cam_stats.get('distribution_type', 'unknown')
            
            if skewness_type != 'unknown':
                # Skewness 값 비교
                skewness_types = ['Standard Skewness', 'Percentile Skewness']
                skewness_values = [skewness, percentile_skewness]
                colors = ['lightcoral', 'lightblue']
                
                bars = axes[1, 1].bar(skewness_types, skewness_values, color=colors, alpha=0.7)
                axes[1, 1].set_title('Skewness Analysis', fontsize=16, fontweight='bold')
                axes[1, 1].set_ylabel('Skewness Value')
                axes[1, 1].grid(True, alpha=0.3)
                
                # 값 표시 - 바 높이에 따라 위치 조정
                for bar, value in zip(bars, skewness_values):
                    height = bar.get_height()
                    # 바가 너무 작으면 위에, 크면 안쪽에 표시
                    if height < 0.1:
                        text_y = height + 0.01
                        va = 'bottom'
                    else:
                        text_y = height * 0.8
                        va = 'center'
                    
                    axes[1, 1].text(bar.get_x() + bar.get_width()/2., text_y,
                                   f'{value:.3f}', ha='center', va=va, fontsize=12, fontweight='bold',
                                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
                
                # 분포 타입 정보 추가 - 위치 조정
                type_text = f'Type: {skewness_type.replace("_", " ").title()}'
                axes[1, 1].text(0.02, 0.95, type_text, transform=axes[1, 1].transAxes, 
                               verticalalignment='top', fontsize=12, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
            else:
                axes[1, 1].text(0.5, 0.5, 'Skewness Analysis\nNot Available', 
                               transform=axes[1, 1].transAxes,
                               horizontalalignment='center', verticalalignment='center',
                               fontsize=16, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
                axes[1, 1].set_title('Skewness Analysis', fontsize=16, fontweight='bold')
            
            # 5. 품질 지표 (Quality Metrics)
            # 집중도 (Concentration) - 높은 값일수록 활성화가 집중됨
            concentration = (max_val - mean_val) / (max_val - min_val) if max_val != min_val else 0
            
            # 균등성 (Uniformity) - 표준편차가 작을수록 균등
            uniformity = 1 - (std_val / max_val) if max_val > 0 else 0
            
            # 신뢰도 (Confidence) - 활성화 비율과 평균값의 조합
            confidence = (high_activation_ratio / 100) * (mean_val / max_val) if max_val > 0 else 0
            
            quality_metrics = ['Concentration', 'Uniformity', 'Confidence']
            quality_values = [concentration, uniformity, confidence]
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            
            bars = axes[2, 0].bar(quality_metrics, quality_values, color=colors, alpha=0.8)
            axes[2, 0].set_title('Quality Metrics', fontsize=16, fontweight='bold')
            axes[2, 0].set_ylabel('Score')
            axes[2, 0].set_ylim(0, 1)
            axes[2, 0].grid(True, alpha=0.3)
            
            # 값 표시 - 바 높이에 따라 위치 조정
            for bar, value in zip(bars, quality_values):
                height = bar.get_height()
                # 바가 너무 작으면 위에, 크면 안쪽에 표시
                if height < 0.1:
                    text_y = height + 0.02
                    va = 'bottom'
                else:
                    text_y = height * 0.8
                    va = 'center'
                
                axes[2, 0].text(bar.get_x() + bar.get_width()/2., text_y,
                               f'{value:.3f}', ha='center', va=va, fontsize=12, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
            
            # 6. 종합 요약 (Comprehensive Summary)
            iqr = q75_val - q25_val
            
            summary_text = f"""CAM Analysis Summary:

Basic Stats:
• Mean: {mean_val:.4f}
• Std Dev: {std_val:.4f}
• Range: {max_val-min_val:.4f}

Activation:
• High Act. Ratio: {high_activation_ratio:.1f}%
• Total Pixels: {total_pixels:,}

Quality Scores:
• Concentration: {concentration:.3f}
• Uniformity: {uniformity:.3f}
• Confidence: {confidence:.3f}

Distribution:
• IQR: {iqr:.4f}
• Q50/Q25: {q50_val/q25_val:.2f if q25_val != 0 else 'N/A'}"""
            
            axes[2, 1].text(0.05, 0.95, summary_text, transform=axes[2, 1].transAxes, 
                           fontsize=14, verticalalignment='top',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
            axes[2, 1].set_title('Comprehensive Summary', fontsize=16, fontweight='bold')
            axes[2, 1].axis('off')
            
        except Exception as e:
            # 오류 발생 시 간단한 메시지 표시
            for ax in axes.flat:
                ax.text(0.5, 0.5, 'CAM Statistics\nNot Available', 
                       transform=ax.transAxes,
                       horizontalalignment='center', verticalalignment='center',
                       fontsize=16, fontweight='bold')
                ax.set_title('CAM Statistics', fontsize=16, fontweight='bold')
                ax.axis('off')
        
        self._safe_tight_layout(fig)
        return self.fig_to_base64(fig)
    

    def visualize_overlap_statistics(self, overlap_results: Dict) -> str:
        """
        Overlap 분석의 통계 정보를 시각화
        
        Args:
            overlap_results: overlap 분석 결과 딕셔너리
            
        Returns:
            str: base64 인코딩된 이미지
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        try:
            # 1. IoU 점수
            if 'iou' in overlap_results:
                iou_score = overlap_results['iou']
                axes[0, 0].bar(['IoU'], [iou_score], color='skyblue', alpha=0.7)
                axes[0, 0].set_title('Intersection over Union (IoU)', fontsize=14, fontweight='bold')
                axes[0, 0].set_ylabel('IoU Score')
                axes[0, 0].set_ylim(0, 1)
                axes[0, 0].grid(True, alpha=0.3)
                
                # IoU 품질 평가
                if iou_score > 0.7:
                    quality = 'Excellent'
                    color = 'green'
                elif iou_score > 0.5:
                    quality = 'Good'
                    color = 'orange'
                elif iou_score > 0.3:
                    quality = 'Fair'
                    color = 'yellow'
                else:
                    quality = 'Poor'
                    color = 'red'
                
                axes[0, 0].text(0.5, iou_score + 0.05, f'{quality}\n({iou_score:.3f})', 
                               ha='center', va='bottom', fontsize=12, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))
            else:
                axes[0, 0].text(0.5, 0.5, 'IoU Score\nNot Available', 
                               transform=axes[0, 0].transAxes,
                               horizontalalignment='center', verticalalignment='center',
                               fontsize=16, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
                axes[0, 0].set_title('IoU Score', fontsize=14, fontweight='bold')
            
            # 2. Coverage 메트릭
            coverage_metrics = []
            coverage_names = []
            coverage_colors = []
            
            if 'cam_coverage' in overlap_results:
                coverage_metrics.append(overlap_results['cam_coverage'])
                coverage_names.append('CAM Coverage')
                coverage_colors.append('lightgreen')
            
            if 'bbox_coverage' in overlap_results:
                coverage_metrics.append(overlap_results['bbox_coverage'])
                coverage_names.append('BBox Coverage')
                coverage_colors.append('lightcoral')
            
            if coverage_metrics:
                bars = axes[0, 1].bar(coverage_names, coverage_metrics, color=coverage_colors, alpha=0.7)
                axes[0, 1].set_title('Coverage Metrics', fontsize=14, fontweight='bold')
                axes[0, 1].set_ylabel('Coverage Ratio')
                axes[0, 1].set_ylim(0, 1)
                axes[0, 1].grid(True, alpha=0.3)
                
                # 값 표시 - 바 높이에 따라 위치 조정
                for bar, value in zip(bars, coverage_metrics):
                    height = bar.get_height()
                    # 바가 너무 작으면 위에, 크면 안쪽에 표시
                    if height < 0.1:
                        text_y = height + 0.02
                        va = 'bottom'
                    else:
                        text_y = height * 0.8
                        va = 'center'
                    
                    axes[0, 1].text(bar.get_x() + bar.get_width()/2., text_y,
                                   f'{value:.3f}', ha='center', va=va, fontsize=12, fontweight='bold',
                                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
            else:
                axes[0, 1].text(0.5, 0.5, 'Coverage Metrics\nNot Available', 
                               transform=axes[0, 1].transAxes,
                               horizontalalignment='center', verticalalignment='center',
                               fontsize=16, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
                axes[0, 1].set_title('Coverage Metrics', fontsize=14, fontweight='bold')
            
            # 3. 검출된 객체 정보
            if 'largest_class_name' in overlap_results:
                class_name = overlap_results['largest_class_name']
                confidence = overlap_results.get('largest_confidence', 0)
                
                info_text = f"""Detected Object:
                
Class: {class_name}
Confidence: {confidence:.3f}
BBox Index: {overlap_results.get('largest_bbox_idx', 'N/A')}"""
                
                axes[1, 0].text(0.5, 0.5, info_text, 
                               transform=axes[1, 0].transAxes, 
                               fontsize=14, fontweight='bold',
                               horizontalalignment='center', verticalalignment='center',
                               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
                axes[1, 0].set_title('Detected Object Info', fontsize=14, fontweight='bold')
                axes[1, 0].axis('off')
            else:
                axes[1, 0].text(0.5, 0.5, 'Object Info\nNot Available', 
                               transform=axes[1, 0].transAxes,
                               horizontalalignment='center', verticalalignment='center',
                               fontsize=16, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
                axes[1, 0].set_title('Detected Object Info', fontsize=14, fontweight='bold')
                axes[1, 0].axis('off')
            
            # 4. Overlap 요약 및 품질 평가
            iou_score = overlap_results.get('iou', 0)
            cam_coverage = overlap_results.get('cam_coverage', 0)
            bbox_coverage = overlap_results.get('bbox_coverage', 0)
            largest_class = overlap_results.get('largest_class_name', 'N/A')
            
            # 품질 평가
            if iou_score > 0.7:
                overlap_quality = 'Excellent'
                quality_color = 'green'
            elif iou_score > 0.5:
                overlap_quality = 'Good'
                quality_color = 'orange'
            elif iou_score > 0.3:
                overlap_quality = 'Fair'
                quality_color = 'yellow'
            else:
                overlap_quality = 'Poor'
                quality_color = 'red'
            
            summary_text = f"""Overlap Analysis Summary:

Metrics:
• IoU Score: {iou_score:.3f}
• CAM Coverage: {cam_coverage:.3f}
• BBox Coverage: {bbox_coverage:.3f}

Object:
• Largest Class: {largest_class}
• Quality: {overlap_quality}

Assessment:
• Overlap Quality: {overlap_quality}
• Model Focus: {'Good' if cam_coverage > 0.5 else 'Needs Improvement'}"""
            
            axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes, 
                           fontsize=12, verticalalignment='top',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor=quality_color, alpha=0.3))
            axes[1, 1].set_title('Comprehensive Summary', fontsize=14, fontweight='bold')
            axes[1, 1].axis('off')
            
        except Exception as e:
            print(f"    ❌ Error in overlap statistics visualization: {e}")
            # 오류 발생 시 모든 subplot에 오류 메시지 표시
            for ax in axes.flat:
                ax.text(0.5, 0.5, 'Statistics Error', 
                       transform=ax.transAxes,
                       horizontalalignment='center', verticalalignment='center',
                       fontsize=16, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.7))
                ax.set_title('Error', fontsize=14, fontweight='bold')
                ax.axis('off')
        
        self._safe_tight_layout(fig)
        return self.fig_to_base64(fig)

    def visualize_overlap_analysis(self, overlap_results: Dict, cam_result: Dict = None) -> str:
        """
        Overlap 분석 결과를 시각화
        
        Args:
            overlap_results: overlap 분석 결과 딕셔너리
            cam_result: CAM 분석 결과 (image_path, grayscale_cam 포함)
            
        Returns:
            str: base64 인코딩된 이미지
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        try:
            # 원본 이미지 로드
            original_img = None
            if cam_result and cam_result.get('image_path'):
                original_img = cv2.imread(cam_result['image_path'])
                if original_img is not None:
                    rgb_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                else:
                    print(f"    ⚠️  Failed to load image: {cam_result['image_path']}")
                    rgb_img = None
            else:
                print(f"    ⚠️  No image path provided in cam_result")
                rgb_img = None
            
            # CAM 데이터 가져오기
            grayscale_cam = None
            if cam_result and cam_result.get('grayscale_cam') is not None:
                grayscale_cam = cam_result['grayscale_cam']
            elif hasattr(self, '_current_cam_data') and self._current_cam_data is not None:
                grayscale_cam = self._current_cam_data
            else:
                print(f"    ⚠️  No CAM data available")
            
            # 1. 원본 이미지
            if rgb_img is not None:
                axes[0, 0].imshow(rgb_img)
                axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
            else:
                axes[0, 0].text(0.5, 0.5, 'Original Image\nNot Available', 
                               transform=axes[0, 0].transAxes,
                               horizontalalignment='center', verticalalignment='center',
                               fontsize=16, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
                axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
            axes[0, 0].axis('off')
            
            # 2. CAM 히트맵
            if grayscale_cam is not None:
                axes[0, 1].imshow(grayscale_cam, cmap='hot')
                axes[0, 1].set_title('CAM Heatmap', fontsize=14, fontweight='bold')
            else:
                axes[0, 1].text(0.5, 0.5, 'CAM Heatmap\nNot Available', 
                               transform=axes[0, 1].transAxes,
                               horizontalalignment='center', verticalalignment='center',
                               fontsize=16, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
                axes[0, 1].set_title('CAM Heatmap', fontsize=14, fontweight='bold')
            axes[0, 1].axis('off')
            
            # 3. 모든 bbox와 가장 큰 bbox 표시
            if rgb_img is not None:
                axes[0, 2].imshow(rgb_img)
                
                # bbox 정보가 있는 경우 표시
                if 'all_bboxes' in overlap_results and 'largest_bbox_idx' in overlap_results:
                    all_bboxes = overlap_results['all_bboxes']
                    largest_bbox_idx = overlap_results['largest_bbox_idx']
                    bbox_names = overlap_results.get('bbox_names', [f'Box_{i}' for i in range(len(all_bboxes))])
                    
                    # 색상 팔레트 생성
                    colors = plt.cm.Set3(np.linspace(0, 1, len(all_bboxes)))
                    
                    for i, (box, color, name) in enumerate(zip(all_bboxes, colors, bbox_names)):
                        x1, y1, x2, y2 = box
                        
                        if i == largest_bbox_idx:
                            # 가장 큰 bbox는 굵은 빨간색 선으로 표시
                            axes[0, 2].add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                                             fill=False, edgecolor='red', linewidth=4))
                            axes[0, 2].text(x1, y1-5, f'LARGEST: {name}', 
                                           color='red', fontsize=10, fontweight='bold',
                                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                        else:
                            # 다른 bbox는 얇은 선으로 표시
                            axes[0, 2].add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                                             fill=False, edgecolor=color, linewidth=2, alpha=0.7))
                            axes[0, 2].text(x1, y1-5, name, color=color, fontsize=8,
                                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
                    
                    axes[0, 2].set_title('All Detections (Red = Largest)', fontsize=14, fontweight='bold')
                else:
                    # bbox 정보가 없는 경우 기본 메시지
                    axes[0, 2].text(0.5, 0.5, 'BBox Information\nNot Available', 
                                   transform=axes[0, 2].transAxes,
                                   horizontalalignment='center', verticalalignment='center',
                                   fontsize=16, fontweight='bold',
                                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
                    axes[0, 2].set_title('All Detections', fontsize=14, fontweight='bold')
            else:
                axes[0, 2].text(0.5, 0.5, 'Image Not Available\nfor BBox Display', 
                               transform=axes[0, 2].transAxes,
                               horizontalalignment='center', verticalalignment='center',
                               fontsize=16, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
                axes[0, 2].set_title('All Detections', fontsize=14, fontweight='bold')
            axes[0, 2].axis('off')
            
            # 4. CAM 활성 영역
            if 'cam_active_mask' in overlap_results:
                cam_active_mask = overlap_results['cam_active_mask']
                axes[1, 0].imshow(cam_active_mask, cmap='gray')
                cam_active_area = overlap_results.get('cam_active_area', 0)
                
                # 임계값 정보 추가 (임계값 없이 모든 활성화 값 사용)
                threshold_info = overlap_results.get('threshold_info', {})
                threshold = threshold_info.get('threshold', 0.0)
                method = threshold_info.get('method', 'no_threshold')
                
                title_text = f'CAM Active Region\n({cam_active_area:,} pixels)\nAll Activations (>0)'
                axes[1, 0].set_title(title_text, fontsize=14, fontweight='bold')
                
                # 임계값 설명 추가
                explanation_text = f'White: > 0.0\nBlack: = 0.0\nMethod: {method}'
                axes[1, 0].text(0.02, 0.98, explanation_text, transform=axes[1, 0].transAxes, 
                               verticalalignment='top', fontsize=10, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
            else:
                axes[1, 0].text(0.5, 0.5, 'CAM Active Mask\nNot Available', 
                               transform=axes[1, 0].transAxes,
                               horizontalalignment='center', verticalalignment='center',
                               fontsize=16, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
                axes[1, 0].set_title('CAM Active Region', fontsize=14, fontweight='bold')
            axes[1, 0].axis('off')
            
            # 5. 가장 큰 bbox 마스크
            if 'bbox_mask' in overlap_results:
                bbox_mask = overlap_results['bbox_mask']
                axes[1, 1].imshow(bbox_mask, cmap='gray')
                bbox_area = overlap_results.get('bbox_area', 0)
                axes[1, 1].set_title(f'Largest Bbox Region\n({bbox_area:,} pixels)', 
                                   fontsize=14, fontweight='bold')
            else:
                axes[1, 1].text(0.5, 0.5, 'BBox Mask\nNot Available', 
                               transform=axes[1, 1].transAxes,
                               horizontalalignment='center', verticalalignment='center',
                               fontsize=16, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
                axes[1, 1].set_title('Largest Bbox Region', fontsize=14, fontweight='bold')
            axes[1, 1].axis('off')
            
            # 6. Overlap 시각화
            if all(key in overlap_results for key in ['bbox_mask', 'cam_active_mask', 'intersection_mask']):
                # Overlap 시각화 생성
                overlap_vis = np.zeros((*overlap_results['bbox_mask'].shape, 3))
                overlap_vis[overlap_results['bbox_mask']] = [1, 0, 0]  # 빨간색: bbox
                overlap_vis[overlap_results['cam_active_mask']] = [0, 1, 0]  # 초록색: CAM
                overlap_vis[overlap_results['intersection_mask']] = [1, 1, 0]  # 노란색: 교집합
                
                axes[1, 2].imshow(overlap_vis)
                iou_score = overlap_results.get('iou', 0)
                axes[1, 2].set_title(f'Overlap Visualization\nIoU: {iou_score:.4f}', 
                                   fontsize=14, fontweight='bold')
                
                # 범례 추가
                legend_elements = [
                    plt.Rectangle((0, 0), 1, 1, facecolor='red', alpha=0.7, label='Largest Bbox'),
                    plt.Rectangle((0, 0), 1, 1, facecolor='green', alpha=0.7, label='CAM Active'),
                    plt.Rectangle((0, 0), 1, 1, facecolor='yellow', alpha=0.7, label='Intersection')
                ]
                axes[1, 2].legend(handles=legend_elements, loc='upper right', fontsize=10)
            else:
                axes[1, 2].text(0.5, 0.5, 'Overlap Visualization\nNot Available', 
                               transform=axes[1, 2].transAxes,
                               horizontalalignment='center', verticalalignment='center',
                               fontsize=16, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
                axes[1, 2].set_title('Overlap Visualization', fontsize=14, fontweight='bold')
            axes[1, 2].axis('off')
            
        except Exception as e:
            print(f"    ❌ Error in overlap visualization: {e}")
            # 오류 발생 시 모든 subplot에 오류 메시지 표시
            for ax in axes.flat:
                ax.text(0.5, 0.5, 'Visualization Error', 
                       transform=ax.transAxes,
                       horizontalalignment='center', verticalalignment='center',
                       fontsize=16, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.7))
                ax.set_title('Error', fontsize=14, fontweight='bold')
                ax.axis('off')
        
        self._safe_tight_layout(fig)
        return self.fig_to_base64(fig)