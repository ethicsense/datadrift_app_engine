import cv2
import sys
import os

import numpy as np
from PIL import Image
import hashlib
from typing import List, Tuple, Dict, Optional, Union

import matplotlib.pyplot as plt
import seaborn as sns

import torch
from yolo_cam.eigen_cam import EigenCAM as YOLO_EigenCAM
from yolo_cam.utils.image import show_cam_on_image as show_yolocam_on_image
from yolo_cam.utils.image import scale_cam_image as scale_yolocam_image

from scipy import ndimage
from scipy.stats import entropy
from skimage.measure import shannon_entropy

from datetime import datetime


class XAIAnalyzer:
    """XAI (Explainable AI) 분석을 위한 클래스 - CAM, GradCAM 등 지원"""
    
    def __init__(self, device: Optional[str] = None, model_path: Optional[str] = None):
        """
        XAI 분석기 초기화
        
        Args:
            device: 사용할 디바이스 (None이면 자동 선택)
            model_path: 모델 파일 경로 (None이면 기본 모델 사용)
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model = None
        self.model_path = model_path
        self.model_name = None
        self.class_names = {}
        self.feature_maps = None
        self.gradients = None
        self.target_layers = None  # 타겟 레이어를 저장할 변수 추가
        
        print(f"Using device: {self.device}")
    
    def load_model(self, model_path: str):
        """
        YOLO 모델을 로드합니다.
        
        Args:
            model_path: 모델 파일 경로
        """
        try:
            from ultralytics import YOLO
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            self.model = YOLO(model_path)
            self.model_name = os.path.basename(model_path)
            
            # 모델을 지정된 디바이스로 이동
            self.model.to(self.device)
            
            # 클래스명 가져오기
            if hasattr(self.model, 'names'):
                self.class_names = self.model.names
            
            # 타겟 레이어를 한 번만 찾아서 저장
            self.target_layers = self.find_target_layer()
            
            print(f"Model loaded successfully: {self.model_name}")
            print(f"Number of classes: {len(self.class_names)}")
            if self.target_layers:
                print(f"Target layer found: {self.target_layers}")
            
        except ImportError:
            print("Error: ultralytics package not found. Please install it with: pip install ultralytics")
            raise
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def parse_detections(self, results) -> Tuple[np.ndarray, List, List]:
        """
        YOLO 모델의 결과를 박스, 색상, 클래스명으로 파싱
        
        Args:
            results: YOLO 모델의 추론 결과
            
        Returns:
            Tuple[np.ndarray, List, List]: (boxes, colors, names)
        """
        boxes = []
        colors = []
        names = []
        
        if not results or len(results) == 0:
            return np.array([]), [], []
        
        for result in results[0].boxes:
            box = result.xyxy[0].cpu().numpy()  # x1, y1, x2, y2 형식
            conf = float(result.conf)
            cls = int(result.cls)
            name = results[0].names[cls]
            
            boxes.append(box)
            colors.append(self._generate_color(cls))  # 클래스별 고유 색상
            names.append(f"{name} {conf:.2f}")
        
        return np.array(boxes), colors, names
    
    def _generate_color(self, class_id: int) -> List[int]:
        """
        클래스 ID에 따른 고유한 색상 생성
        
        Args:
            class_id: 클래스 ID
            
        Returns:
            List[int]: RGB 색상 값 [R, G, B]
        """
        np.random.seed(class_id)
        color = np.random.randint(0, 255, size=3).tolist()
        return color
    
    def draw_detections(self, boxes: np.ndarray, colors: List, names: List, image: np.ndarray) -> np.ndarray:
        """
        검출 결과를 이미지에 시각화
        
        Args:
            boxes: 검출된 박스 좌표
            colors: 각 박스의 색상
            names: 각 박스의 클래스명과 신뢰도
            image: 원본 이미지
            
        Returns:
            np.ndarray: 시각화된 이미지
        """
        image_copy = image.copy()
        
        for box, color, name in zip(boxes, colors, names):
            x1, y1, x2, y2 = map(int, box)
            
            # 박스 그리기
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 2)
            
            # 텍스트 배경 그리기
            (text_w, text_h), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(image_copy, (x1, y1-text_h-4), (x1+text_w, y1), color, -1)
            
            # 텍스트 그리기
            cv2.putText(image_copy, name, (x1, y1-4), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        
        return image_copy
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        이미지를 전처리합니다.
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            np.ndarray: 전처리된 이미지
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        return image
    
    def calculate_file_hash(self, file_path: str) -> str:
        """
        파일의 MD5 해시를 계산합니다.
        
        Args:
            file_path: 파일 경로
            
        Returns:
            str: MD5 해시값
        """
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()
    
    def get_model_info(self) -> Dict:
        """
        현재 로드된 모델의 정보를 반환합니다.
        
        Returns:
            Dict: 모델 정보
        """
        if self.model is None:
            return {}
        
        info = {
            'model_name': self.model_name,
            'device': str(self.device),
            'num_classes': len(self.class_names),
            'class_names': self.class_names.copy(),
            'target_layer': str(self.target_layers) if self.target_layers else None
        }
        
        return info
    
    def find_target_layer(self) -> Optional[torch.nn.Module]:
        """
        YOLO 모델에서 CAM 분석에 적합한 타겟 레이어를 찾습니다.
        
        Returns:
            Optional[torch.nn.Module]: 타겟 레이어 또는 None
        """
        if self.model is None:
            print("Model not loaded. Please load model first.")
            return None
        
        target_layer = None
        
        # 모델의 레이어들을 역순으로 탐색하여 concat() 레이어 찾기
        for idx, layers in reversed(list(enumerate(self.model.model.model))):
            if str(layers).lower() == "concat()":
                print(f"마지막 concat()의 인덱스: {idx}")
                print(f"마지막 concat() 모델: {layers}")
                target_layer = layers
                break
        else:
            print("concat()을 찾을 수 없습니다.")
            # 기본값으로 마지막 레이어 사용
            target_layer = self.model.model.model[-1]
            print(f"기본 타겟 레이어 사용: {target_layer}")
        
        return target_layer
    
    def get_target_layers(self) -> Optional[List]:
        """
        저장된 타겟 레이어를 반환합니다.
        
        Returns:
            Optional[List]: 타겟 레이어 리스트 또는 None
        """
        if self.target_layers is None:
            print("Target layers not found. Please load model first.")
            return None
        
        return [self.target_layers]
    
    def generate_cam(self, image_path: str, target_layers: Optional[List] = None, 
                    use_rgb: bool = True) -> Dict:
        """
        이미지에 대해 CAM (Class Activation Mapping)을 생성합니다.
        
        Args:
            image_path: 이미지 파일 경로
            target_layers: CAM 분석에 사용할 타겟 레이어 리스트 (None이면 자동 선택)
            use_rgb: RGB 이미지 사용 여부
            
        Returns:
            Dict: CAM 분석 결과
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please load model first.")
        
        try:
            # 이미지 전처리
            rgb_img = self.preprocess_image(image_path)
            img = np.float32(rgb_img) / 255
            
            # 타겟 레이어 설정
            if target_layers is None:
                target_layers = self.get_target_layers()
                if target_layers is None:
                    raise ValueError("Could not find suitable target layer")
            
            # YOLO_EigenCAM 생성
            cam = YOLO_EigenCAM(self.model, target_layers, task='od')
            
            # CAM 생성
            grayscale_cam = cam(rgb_img)[0, :, :]
            
            # CAM을 원본 이미지에 오버레이
            cam_image = show_yolocam_on_image(img, grayscale_cam, use_rgb=use_rgb)
            
            # 결과 저장
            result = {
                'original_image': rgb_img,
                'grayscale_cam': grayscale_cam,
                'cam_image': cam_image,
                'target_layers': [str(layer) for layer in target_layers],
                'image_path': image_path
            }
            
            return result
            
        except Exception as e:
            print(f"Error generating CAM for {image_path}: {e}")
            return None
    
    def save_cam_result(self, cam_result: Dict, output_path: str, save_original: bool = True):
        """
        CAM 결과를 파일로 저장합니다.
        
        Args:
            cam_result: CAM 분석 결과
            output_path: 저장할 파일 경로
            save_original: 원본 이미지도 함께 저장할지 여부
        """
        if cam_result is None:
            print("No CAM result to save")
            return
        
        try:
            # CAM 이미지 저장
            cam_image = cam_result['cam_image']
            cv2.imwrite(output_path, cam_image)
            print(f"CAM image saved to: {output_path}")
            
            # 원본 이미지도 저장 (선택사항)
            if save_original:
                original_path = output_path.replace('.jpg', '_original.jpg').replace('.png', '_original.png')
                cv2.imwrite(original_path, cam_result['original_image'])
                print(f"Original image saved to: {original_path}")
                
        except Exception as e:
            print(f"Error saving CAM result: {e}")
    
    def visualize_cam(self, cam_result: Dict, figsize: Tuple[int, int] = (15, 5)):
        """
        CAM 결과를 시각화합니다.
        
        Args:
            cam_result: CAM 분석 결과
            figsize: 그래프 크기
        """
        if cam_result is None:
            print("No CAM result to visualize")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # 원본 이미지
        axes[0].imshow(cv2.cvtColor(cam_result['original_image'], cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Grayscale CAM
        axes[1].imshow(cam_result['grayscale_cam'], cmap='jet')
        axes[1].set_title('Grayscale CAM')
        axes[1].axis('off')
        
        # CAM 오버레이 이미지
        axes[2].imshow(cv2.cvtColor(cam_result['cam_image'], cv2.COLOR_BGR2RGB))
        axes[2].set_title('CAM Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_image_with_cam(self, image_path: str, output_dir: Optional[str] = None,
                             save_results: bool = True, visualize: bool = True) -> Dict:
        """
        이미지를 CAM과 함께 분석합니다.
        
        Args:
            image_path: 이미지 파일 경로
            output_dir: 결과 저장 디렉토리 (None이면 저장하지 않음)
            save_results: 결과를 파일로 저장할지 여부
            visualize: 결과를 시각화할지 여부
            
        Returns:
            Dict: 분석 결과
        """
        print(f"Analyzing image with CAM: {image_path}")
        
        # CAM 생성
        cam_result = self.generate_cam(image_path)
        
        if cam_result is None:
            print(f"Failed to generate CAM for {image_path}")
            return None
        
        # 결과 저장
        if save_results and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(output_dir, f"{name}_cam{ext}")
            self.save_cam_result(cam_result, output_path)
        
        # 시각화
        if visualize:
            self.visualize_cam(cam_result)
        
        return cam_result
    
    def calculate_cam_statistics(self, cam: np.ndarray) -> Dict:
        """
        CAM 활성도 통계를 계산합니다.
        
        Args:
            cam: CAM 데이터 (2D numpy array)
            
        Returns:
            Dict: CAM 통계 정보
        """
        cam_stats = {
            'mean': ('평균', np.mean(cam)),
            'max': ('최대값', np.max(cam)),
            'min': ('최소값', np.min(cam)),
            'sum': ('합계', np.sum(cam)),
            'std': ('표준편차', np.std(cam)),
            'median': ('중앙값', np.median(cam)),
            'variance': ('분산', np.var(cam)),
            'range': ('범위', np.max(cam) - np.min(cam)),
            'shape': ('크기', cam.shape),
            'total_pixels': ('총 픽셀 수', cam.size)
        }
        
        # 사분위수 정보
        q25, q50, q75 = np.percentile(cam, [25, 50, 75])
        cam_stats.update({
            'q25': ('1사분위수', q25),
            'q50': ('중앙값', q50),
            'q75': ('3사분위수', q75),
            'iqr': ('사분위 범위', q75 - q25)
        })
        
        # 상위 10% 활성도 정보
        threshold_90 = np.percentile(cam, 90)
        high_activation_mask = cam > threshold_90
        cam_stats.update({
            'threshold_90': ('상위 10% 임계값', threshold_90),
            'high_activation_pixels': ('상위 10% 픽셀 수', np.sum(high_activation_mask)),
            'high_activation_ratio': ('상위 10% 비율', np.sum(high_activation_mask) / cam.size * 100)
        })
        
        return cam_stats
    
    def adaptive_thresholding(self, cam: np.ndarray, percentile: int = 85) -> np.ndarray:
        """
        Adaptive 쓰레스홀딩을 적용한 CAM을 생성합니다.
        
        Args:
            cam: CAM 데이터
            percentile: 임계값 백분위수
            
        Returns:
            np.ndarray: Adaptive 쓰레스홀딩 적용된 CAM
        """
        threshold = np.percentile(cam, percentile)
        cam_filtered = np.where(cam > threshold, cam, 0)
        
        if cam_filtered.max() > 0:
            adaptive_cam = cam_filtered / cam_filtered.max()
        else:
            adaptive_cam = cam_filtered
        
        return adaptive_cam
    
    def analyze_connected_components(self, cam: np.ndarray, threshold_percentile: int = 85) -> Dict:
        """
        Connected Components Analysis를 통한 활성화 영역 구조 분석
        
        Args:
            cam: CAM 데이터
            threshold_percentile: 임계값 백분위수
            
        Returns:
            Dict: 연결된 컴포넌트 분석 결과
        """
        threshold = np.percentile(cam, threshold_percentile)
        binary_mask = cam > threshold
        
        # 기본 활성화 영역 통계
        active_pixels = np.sum(binary_mask)
        active_ratio = active_pixels / cam.size * 100
        
        # Connected Components 분석
        labeled_mask, num_components = ndimage.label(binary_mask)
        
        result = {
            'threshold': threshold,
            'active_pixels': active_pixels,
            'active_ratio': active_ratio,
            'num_components': num_components,
            'binary_mask': binary_mask,
            'labeled_mask': labeled_mask
        }
        
        if num_components > 0:
            # 각 컴포넌트 분석
            component_sizes = []
            component_centroids = []
            component_bboxes = []
            component_densities = []
            circularities = []
            
            for i in range(1, num_components + 1):
                component_mask = labeled_mask == i
                size = np.sum(component_mask)
                component_sizes.append(size)
                
                # 중심점 계산
                y_coords, x_coords = np.where(component_mask)
                centroid_y = np.mean(y_coords)
                centroid_x = np.mean(x_coords)
                component_centroids.append((centroid_x, centroid_y))
                
                # 바운딩 박스 계산
                min_y, max_y = np.min(y_coords), np.max(y_coords)
                min_x, max_x = np.min(x_coords), np.max(x_coords)
                bbox_area = (max_y - min_y + 1) * (max_x - min_x + 1)
                component_bboxes.append((min_x, min_y, max_x, max_y))
                component_densities.append(size / bbox_area)
                
                # 원형도 계산
                contours, _ = cv2.findContours(component_mask.astype(np.uint8), 
                                             cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    contour = contours[0]
                    area = cv2.contourArea(contour)
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter ** 2)
                        circularities.append(circularity)
                    else:
                        circularities.append(0)
                else:
                    circularities.append(0)
            
            result.update({
                'component_sizes': np.array(component_sizes),
                'component_centroids': component_centroids,
                'component_bboxes': component_bboxes,
                'component_densities': np.array(component_densities),
                'circularities': np.array(circularities),
                'size_stats': {
                    'max': np.max(component_sizes),
                    'min': np.min(component_sizes),
                    'mean': np.mean(component_sizes),
                    'median': np.median(component_sizes),
                    'std': np.std(component_sizes)
                }
            })
        
        return result
    
    def calculate_cam_centroids(self, cam: np.ndarray, methods: List[str] = None) -> Dict:
        """
        CAM의 centroid 좌표를 다양한 방법으로 계산
        
        Args:
            cam: CAM 데이터
            methods: 사용할 방법 리스트 ['weighted', 'threshold', 'max', 'components']
            
        Returns:
            Dict: 각 방법별 centroid 좌표
        """
        if methods is None:
            methods = ['weighted', 'threshold', 'max', 'components']
        
        centroids = {}
        
        # 1. 활성도 가중 평균 centroid
        if 'weighted' in methods:
            y_coords, x_coords = np.meshgrid(np.arange(cam.shape[0]), np.arange(cam.shape[1]), indexing='ij')
            total_weight = np.sum(cam)
            if total_weight > 0:
                weighted_x = np.sum(x_coords * cam) / total_weight
                weighted_y = np.sum(y_coords * cam) / total_weight
            else:
                weighted_x, weighted_y = cam.shape[1] / 2, cam.shape[0] / 2
            centroids['weighted'] = (weighted_x, weighted_y)
        
        # 2. 임계값 기반 centroid
        if 'threshold' in methods:
            threshold = np.percentile(cam, 85)
            active_mask = cam > threshold
            if np.sum(active_mask) > 0:
                y_coords, x_coords = np.where(active_mask)
                threshold_x = np.mean(x_coords)
                threshold_y = np.mean(y_coords)
            else:
                threshold_x, threshold_y = cam.shape[1] / 2, cam.shape[0] / 2
            centroids['threshold'] = (threshold_x, threshold_y)
        
        # 3. 최대 활성도 위치
        if 'max' in methods:
            max_idx = np.unravel_index(np.argmax(cam), cam.shape)
            max_y, max_x = max_idx
            centroids['max'] = (max_x, max_y)
        
        # 4. Connected Components 기반 centroid
        if 'components' in methods:
            threshold = np.percentile(cam, 85)
            binary_mask = cam > threshold
            labeled_mask, num_components = ndimage.label(binary_mask)
            
            if num_components > 0:
                component_sizes = []
                component_centroids = []
                
                for i in range(1, num_components + 1):
                    component_mask = labeled_mask == i
                    size = np.sum(component_mask)
                    component_sizes.append(size)
                    
                    y_coords, x_coords = np.where(component_mask)
                    centroid_y = np.mean(y_coords)
                    centroid_x = np.mean(x_coords)
                    component_centroids.append((centroid_x, centroid_y))
                
                largest_idx = np.argmax(component_sizes)
                largest_x, largest_y = component_centroids[largest_idx]
                centroids['components'] = (largest_x, largest_y)
            else:
                centroids['components'] = (cam.shape[1] / 2, cam.shape[0] / 2)
        
        return centroids
    
    def calculate_cam_entropy(self, cam: np.ndarray, methods: List[str] = None) -> Dict:
        """
        CAM 데이터의 다양한 엔트로피 계산
        
        Args:
            cam: CAM 데이터
            methods: 사용할 엔트로피 방법 리스트
            
        Returns:
            Dict: 엔트로피 분석 결과
        """
        if methods is None:
            methods = ['shannon', 'spatial', 'histogram', 'conditional']
        
        entropy_results = {}
        
        # 1. Shannon 엔트로피
        if 'shannon' in methods:
            cam_normalized = (cam - cam.min()) / (cam.max() - cam.min())
            cam_discrete = (cam_normalized * 255).astype(np.uint8)
            shannon_ent = shannon_entropy(cam_discrete)
            entropy_results['shannon'] = shannon_ent
        
        # 2. 공간적 엔트로피
        if 'spatial' in methods:
            h_diff = np.diff(cam, axis=1)
            v_diff = np.diff(cam, axis=0)
            d_diff = np.diff(np.diff(cam, axis=0), axis=1)
            
            h_ent = shannon_entropy((h_diff * 255).astype(np.uint8))
            v_ent = shannon_entropy((v_diff * 255).astype(np.uint8))
            d_ent = shannon_entropy((d_diff * 255).astype(np.uint8))
            
            spatial_ent = (h_ent + v_ent + d_ent) / 3
            entropy_results['spatial'] = spatial_ent
            entropy_results['spatial_directions'] = {'horizontal': h_ent, 'vertical': v_ent, 'diagonal': d_ent}
        
        # 3. 히스토그램 엔트로피
        if 'histogram' in methods:
            non_zero_cam = cam.flatten()[cam.flatten() > 0]
            if len(non_zero_cam) > 0:
                activation_ratio = len(non_zero_cam) / len(cam.flatten())
                hist, bins = np.histogram(non_zero_cam, bins=50, density=True)
                hist = hist[hist > 0]
                hist_ent = entropy(hist)
                
                entropy_results['histogram'] = hist_ent
                entropy_results['activation_ratio'] = activation_ratio
                entropy_results['non_zero_count'] = len(non_zero_cam)
            else:
                entropy_results['histogram'] = 0.0
                entropy_results['activation_ratio'] = 0.0
                entropy_results['non_zero_count'] = 0
        
        # 4. 조건부 엔트로피
        if 'conditional' in methods:
            thresholds = [50, 75, 85, 90, 95]
            conditional_ents = {}
            
            for thresh in thresholds:
                threshold_val = np.percentile(cam, thresh)
                active_mask = cam > threshold_val
                inactive_mask = cam <= threshold_val
                
                if np.sum(active_mask) > 0:
                    active_ent = shannon_entropy((cam[active_mask] * 255).astype(np.uint8))
                else:
                    active_ent = 0
                    
                if np.sum(inactive_mask) > 0:
                    inactive_ent = shannon_entropy((cam[inactive_mask] * 255).astype(np.uint8))
                else:
                    inactive_ent = 0
                
                active_ratio = np.sum(active_mask) / cam.size
                conditional_ent = active_ratio * active_ent + (1 - active_ratio) * inactive_ent
                conditional_ents[thresh] = conditional_ent
            
            entropy_results['conditional'] = conditional_ents
        
        return entropy_results
    
    def calculate_cam_bbox_overlap(self, cam: np.ndarray, boxes: np.ndarray, 
                                  names: List[str], threshold_percentile: int = 85) -> Dict:
        """
        CAM 활성 영역과 가장 큰 bbox 간의 overlap 계산
        
        Args:
            cam: CAM 데이터
            boxes: 검출된 박스 좌표
            names: 클래스명 리스트
            threshold_percentile: 임계값 백분위수
            
        Returns:
            Dict: Overlap 분석 결과
        """
        if len(boxes) == 0:
            return None
        
        # 가장 큰 bbox 찾기
        areas = []
        for box in boxes:
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            areas.append(area)
        
        largest_idx = np.argmax(areas)
        largest_bbox = boxes[largest_idx]
        largest_area = areas[largest_idx]
        largest_name = names[largest_idx]
        
        # bbox 정보 추출
        x1, y1, x2, y2 = largest_bbox
        
        # bbox를 이미지 경계 내로 제한
        img_height, img_width = cam.shape
        x1 = max(0, min(int(x1), img_width-1))
        y1 = max(0, min(int(y1), img_height-1))
        x2 = max(0, min(int(x2), img_width-1))
        y2 = max(0, min(int(y2), img_height-1))
        
        # bbox 마스크 생성
        bbox_mask = np.zeros_like(cam, dtype=bool)
        bbox_mask[y1:y2+1, x1:x2+1] = True
        
        # CAM 활성 영역 마스크 생성
        threshold = np.percentile(cam, threshold_percentile)
        cam_active_mask = cam > threshold
        
        # Overlap 계산
        intersection = np.logical_and(bbox_mask, cam_active_mask)
        union = np.logical_or(bbox_mask, cam_active_mask)
        
        intersection_area = np.sum(intersection)
        union_area = np.sum(union)
        iou = intersection_area / union_area if union_area > 0 else 0
        
        bbox_area = np.sum(bbox_mask)
        cam_active_area = np.sum(cam_active_mask)
        
        cam_coverage = intersection_area / bbox_area if bbox_area > 0 else 0
        bbox_coverage = intersection_area / cam_active_area if cam_active_area > 0 else 0
        
        return {
            'iou': iou,
            'cam_coverage': cam_coverage,
            'bbox_coverage': bbox_coverage,
            'intersection_area': intersection_area,
            'bbox_area': bbox_area,
            'cam_active_area': cam_active_area,
            'union_area': union_area,
            'bbox_coords': (x1, y1, x2, y2),
            'bbox_mask': bbox_mask,
            'cam_active_mask': cam_active_mask,
            'intersection_mask': intersection,
            'largest_bbox_idx': largest_idx,
            'all_areas': areas,
            'largest_class_name': largest_name,
            'all_class_names': names,
        }
    
    def comprehensive_cam_analysis(self, image_path: str, target_layers: Optional[List] = None,
                                 save_visualizations: bool = False, output_dir: Optional[str] = None) -> Dict:
        """
        CAM에 대한 포괄적인 분석을 수행합니다.
        
        Args:
            image_path: 이미지 파일 경로
            target_layers: 타겟 레이어 리스트
            save_visualizations: 시각화 결과 저장 여부 (CAM 이미지 등)
            output_dir: 시각화 결과 저장 디렉토리 (save_visualizations=True일 때만 사용)
            
        Returns:
            Dict: 포괄적인 CAM 분석 결과
        """
        print(f"Comprehensive CAM analysis for: {image_path}")
        
        # CAM 생성 (타겟 레이어가 제공되지 않으면 저장된 타겟 레이어 사용)
        if target_layers is None:
            target_layers = self.get_target_layers()
        
        cam_result = self.generate_cam(image_path, target_layers)
        if cam_result is None:
            return None
        
        grayscale_cam = cam_result['grayscale_cam']
        
        # 1. 기본 통계
        cam_stats = self.calculate_cam_statistics(grayscale_cam)
        
        # 2. Adaptive 쓰레스홀딩
        adaptive_cam = self.adaptive_thresholding(grayscale_cam)
        
        # 3. Connected Components 분석
        components_analysis = self.analyze_connected_components(grayscale_cam)
        
        # 4. Centroid 계산
        centroids = self.calculate_cam_centroids(grayscale_cam)
        
        # 5. 엔트로피 분석
        entropy_results = self.calculate_cam_entropy(grayscale_cam)
        
        # 6. 객체 검출 및 Overlap 분석
        rgb_img = cam_result['original_image']
        results = self.model(rgb_img)
        boxes, colors, names = self.parse_detections(results)
        
        overlap_results = None
        if len(boxes) > 0:
            overlap_results = self.calculate_cam_bbox_overlap(grayscale_cam, boxes, names)
        
        # 시각화 결과 저장 (선택사항)
        visualization_paths = {}
        if save_visualizations and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)
            
            # CAM 이미지 저장
            cam_image_path = os.path.join(output_dir, f"{name}_cam_overlay{ext}")
            cv2.imwrite(cam_image_path, cam_result['cam_image'])
            visualization_paths['cam_overlay'] = cam_image_path
            
            # 원본 이미지 저장
            original_image_path = os.path.join(output_dir, f"{name}_original{ext}")
            cv2.imwrite(original_image_path, cam_result['original_image'])
            visualization_paths['original_image'] = original_image_path
            
            print(f"Visualization files saved to: {output_dir}")
        
        # 결과 통합 (캐시에 저장될 데이터) - 크기 최적화
        comprehensive_result = {
            'image_path': image_path,
            'cam_stats': cam_stats,
            'components_analysis': {
                'threshold': components_analysis['threshold'],
                'active_pixels': components_analysis['active_pixels'],
                'active_ratio': components_analysis['active_ratio'],
                'num_components': components_analysis['num_components'],
                # 큰 배열 데이터는 제거
                'size_stats': components_analysis.get('size_stats', {})
            },
            'centroids': centroids,
            'entropy_results': entropy_results,
            'detection_results': {
                'boxes': boxes.tolist() if len(boxes) > 0 else [],  # numpy array를 list로 변환
                'names': names
                # colors는 제거 (재생성 가능)
            },
            'overlap_results': overlap_results,
            'visualization_paths': visualization_paths if save_visualizations else {},
            'analysis_timestamp': datetime.now().isoformat(),
            'model_name': self.model_name
        }
        
        return comprehensive_result