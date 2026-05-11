from ultralytics import YOLO
import torch
import sys
import os

def train_yolo(data_path, model_path, project="runs", name="exp", epochs=100, batch_size=16, img_size=640, learning_rate=0.001):
    print(f"""
    Starting YOLO training with parameters:
    - Project: {project}
    - Name: {name}
    - Epochs: {epochs}
    - Batch size: {batch_size}
    - Image size: {img_size}
    - Learning rate: {learning_rate}
    """)

    # Mac MPS 지원 추가
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using Mac MPS (Metal Performance Shaders)")
    elif torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA")
    else:
        device = "cpu"
        print("Using CPU")

    # ddoc 패키지 내에서 flask_webapp 모듈의 경로를 찾아서 logs 디렉토리 설정
    try:
        # ddoc.flask_webapp 모듈을 import해서 경로 찾기
        import ddoc.flask_webapp as flask_webapp_module
        flask_webapp_dir = os.path.dirname(flask_webapp_module.__file__)
    except ImportError:
        try:
            # 직접 flask_webapp import 시도
            import flask_webapp
            flask_webapp_dir = os.path.dirname(flask_webapp.__file__)
        except ImportError:
            # 개발 환경에서 현재 파일 기준으로 설정
            print("🔧 Error: flask_webapp module not found")
            print("🔧 Development mode: Using local file path for flask_webapp directory")
            flask_webapp_dir = os.path.dirname(os.path.abspath(__file__))
    
    project_path = os.path.join(flask_webapp_dir, "logs", project)
    
    # logs 디렉토리가 존재하지 않으면 생성
    os.makedirs(project_path, exist_ok=True)
    
    print(f"Model and logs will be saved to: {project_path}")
    
    model = YOLO(model_path)
    model.train(
        data=data_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        lr0=learning_rate,
        device=device,
        project=project_path,  # 수정된 project 경로 사용
        name=name,
        exist_ok=True,
    )
    print("Training completed.")

def train_ocr(dataset_path, model_path, epochs=10, imgsz=640, batch=4, device="cpu"):

    pass

def train_custom_model(dataset_path, model_path, epochs=10, imgsz=640, batch=4, device="cpu"):

    pass