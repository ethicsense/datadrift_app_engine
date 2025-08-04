import torch
import cv2
import numpy as np
from PIL import Image
import os
import hashlib
from typing import List, Tuple, Dict, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns


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
        
        print(f"Using device: {self.device}")
    
    def load_model(self, model_name: str = "yolov8n.pt", model_path: Optional[str] = None):
        """
        YOLO 모델을 로드합니다.
        
        Args:
            model_name: 사용할 YOLO 모델명 (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)
            model_path: 모델 파일 경로 (None이면 기본 경로 사용)
        """
        try:
            from ultralytics import YOLO
            
            if model_path and os.path.exists(model_path):
                self.model = YOLO(model_path)
                self.model_name = os.path.basename(model_path)
            else:
                self.model = YOLO(model_name)
                self.model_name = model_name
            
            # 모델을 지정된 디바이스로 이동
            self.model.to(self.device)
            
            # 클래스명 가져오기
            if hasattr(self.model, 'names'):
                self.class_names = self.model.names
            
            print(f"Model loaded successfully: {self.model_name}")
            print(f"Number of classes: {len(self.class_names)}")
            
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
            'class_names': self.class_names.copy()
        }
        
        return info

def run_xai_analysis(directories: List[str], formats: List[str] = ['jpg', 'jpeg', 'png'], 
                    model_name: str = "yolov8n.pt", device: Optional[str] = None):
    """
    디렉토리의 이미지들에 대해 XAI 분석을 수행합니다.
    
    Args:
        directories: 분석할 디렉토리 리스트
        formats: 분석할 이미지 포맷 리스트
        model_name: 사용할 YOLO 모델명
        device: 사용할 디바이스
    """
    print("Starting XAI analysis...")
    
    analyzer = XAIAnalyzer(device=device)
    analyzer.load_model(model_name)
    
    for directory in directories:
        if not os.path.exists(directory):
            print(f"Error: Directory {directory} does not exist.")
            continue
        
        print(f"\nAnalyzing directory: {directory}")
        
        # 디렉토리 탐색
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(tuple(formats)):
                    file_path = os.path.join(root, file)
                    print(f"Processing {file}...")
                    
                    # TODO: 여기에 CAM/GradCAM 분석 로직 추가 예정
                    # analyzer.generate_cam(file_path)
                    # analyzer.generate_gradcam(file_path)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python xai_analyzer.py <directory> [model_name]")
        sys.exit(1)
    
    directory = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "yolov8n.pt"
    
    run_xai_analysis([directory], model_name=model_name) 