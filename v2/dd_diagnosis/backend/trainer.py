from ultralytics import YOLO
import torch
import os


def train_yolo(
    data_path,
    model_path,
    project="runs",
    name="exp",
    epochs=100,
    batch_size=16,
    img_size=640,
    learning_rate=0.001,
):
    print(f"""
    Starting YOLO training with parameters:
    - Project: {project}
    - Name: {name}
    - Epochs: {epochs}
    - Batch size: {batch_size}
    - Image size: {img_size}
    - Learning rate: {learning_rate}
    """)

    if torch.backends.mps.is_available():
        device = "mps"
        print("Using Mac MPS (Metal Performance Shaders)")
    elif torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA")
    else:
        device = "cpu"
        print("Using CPU")

    package_dir = os.path.dirname(os.path.abspath(__file__))
    project_path = os.path.join(package_dir, "logs", project)

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
        project=project_path,
        name=name,
        exist_ok=True,
    )
    print("Training completed.")


def train_ocr(dataset_path, model_path, epochs=10, imgsz=640, batch=4, device="cpu"):
    pass


def train_custom_model(dataset_path, model_path, epochs=10, imgsz=640, batch=4, device="cpu"):
    pass
