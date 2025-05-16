import fiftyone as fo
import fiftyone.brain as fob
import clip
import torch

import cv2
import numpy as np
from PIL import Image

from flask import Flask, render_template, request, redirect, url_for, Response, send_file, jsonify, g, current_app
from flask import session as flask_session
from flask_socketio import SocketIO, emit
from flask_cors import CORS

import os
import zipfile
import sys
import argparse
from tqdm import tqdm
import threading
import io
import time
import subprocess
import atexit
import json
from tqdm import tqdm
import shutil
import psutil
import platform

from trainer import train_yolo
from utils import TensorboardManager, FiftyoneManager, CaptureOutput, InputDataLoader, MilvusManager

def get_milvus_manager(db_path):
    if 'milvus_manager' not in g:
        g.milvus_manager = MilvusManager()
        g.milvus_manager.connect(db_path)

    return g.milvus_manager

def is_wsl():
    # WSL 확인을 위한 여러 방법 시도
    try:
        # 1. /proc/version 파일 확인
        with open('/proc/version', 'r') as f:
            if 'microsoft' in f.read().lower():
                return True
            
        # 2. WSL 환경 변수 확인
        if 'WSL_DISTRO_NAME' in os.environ:
            return True
            
        # 3. uname 확인
        if 'microsoft' in platform.uname().release.lower():
            return True
            
    except:
        pass
    return False

def kill_process_on_port(port):
    for proc in psutil.process_iter(['pid', 'name']):
        for conn in proc.connections(kind='inet'):
            if conn.laddr.port == port:
                proc.terminate()
                print(f"Terminated process {proc.info['name']} (PID: {proc.info['pid']}) using port {port}")
                      
parser = argparse.ArgumentParser()

# parser.add_argument("--dataset_dir", type=str, required=True, help="Importing dataset path")
# parser.add_argument("--dataset_name", type=str, default="imported_dataset", help="Name the dataset you are importing")
parser.add_argument("--port", type=int, default=8159, help="Port to run the FiftyOne app on")
parser.add_argument("--db_path", type=str, default="DAE_data.db", help="Path to the Milvus database")
# parser.add_argument("--dataset_type", type=str, default=None, help="dataset type (51, yolo)")

args = parser.parse_args()

# 데이터셋 로드 및 세션 생성은 애플리케이션 시작 시 한 번만 수행
tsb_runner = TensorboardManager(port=6006)
atexit.register(tsb_runner.stop)

fom_runner = FiftyoneManager(port=args.port)
fiftyone_thread = fom_runner.start()

app = Flask(__name__, static_folder='static')
CORS(app)
socketio = SocketIO(app)
app.secret_key = os.urandom(24)

## logger instance
capture_stream = CaptureOutput()
sys.stdout = capture_stream

def init_app(app):
    app.model_storage = None

@app.route('/')
def index():
    return redirect(url_for('datadoctor'))

@app.route('/datadoctor')
def datadoctor():
    return render_template('hellodd.html')

@app.route('/dsampler/init_data')
def init_data():
    return render_template('init_data.html')

@app.route('/dsampler/get_existing_datasets', methods=['GET'])
def get_existing_datasets():
    existing_datasets = fo.list_datasets()

    return jsonify(existing_datasets)

@app.route('/dsampler/load_existing_dataset', methods=['POST'])
def load_existing_dataset():
    dataset_name = request.form.get('saved-datasets')
    dataset = fo.load_dataset(dataset_name)
    milvus_manager = get_milvus_manager(args.db_path)

    embeddings_by_sample_id = fom_runner.collect_image_embeddings_by_sample_id(dataset, db_client=milvus_manager)
    results = fob.compute_visualization(
        dataset,
        embeddings=embeddings_by_sample_id,
        brain_key="clip_embeddings",
        plot_points=True,
        verbose=True,
    )
    fom_runner.set_dataset(dataset, results)

    return redirect(url_for('dataclinic', fiftyone_port_number=fom_runner.port))

@app.route('/dsampler/delete_dataset', methods=['POST'])
def delete_dataset():
    try:
        data = request.get_json()
        if not data:
            raise ValueError("No JSON data received")
            
        dataset_name = data.get('dataset_name')
        if not dataset_name:
            raise ValueError("No dataset_name provided in request")
            
        print(f"Attempting to delete dataset: {dataset_name}")
        status_log = []
        
        try:
            milvus_manager = get_milvus_manager(args.db_path)
        except Exception as e:
            raise Exception(f"Failed to connect to Milvus: {str(e)}")

        try:
            if fo.dataset_exists(dataset_name):
                print(f"Found FiftyOne dataset: {dataset_name}")
                fo.delete_dataset(dataset_name)
                status_log.append(f"Deleted FiftyOne dataset: {dataset_name}")
                print(f"Deleted FiftyOne dataset: {dataset_name}")
            else:
                print(f"FiftyOne dataset not found: {dataset_name}")

            if milvus_manager.has_collection(dataset_name):
                print(f"Found Milvus collection: {dataset_name}")
                milvus_manager.drop_collection(dataset_name)
                status_log.append(f"Deleted Milvus collection: {dataset_name}")
                print(f"Deleted Milvus collection: {dataset_name}")
            else:
                print(f"Milvus collection not found: {dataset_name}")

            if not status_log:
                return jsonify({'message': f"No datasets or collections found to delete: {dataset_name}"}), 404

            return jsonify({'message': "\n".join(status_log)})
            
        except Exception as e:
            raise Exception(f"Error during dataset/collection deletion: {str(e)}")
    
    except ValueError as e:
        error_msg = f"Invalid request: {str(e)}"
        print(error_msg)
        return jsonify({'message': error_msg}), 400
    except Exception as e:
        error_msg = f"Error deleting dataset: {str(e)}"
        print(error_msg)
        return jsonify({'message': error_msg}), 500

@app.route('/dsampler/upload', methods=['POST'])
def upload_file():
    UPLOAD_FOLDER = './datasets/uploads'
    ref_dataset = None
    cur_dataset = None
    test_dataset = None

    print("Starting file upload process...")
    for key in request.files:
        file = request.files[key]
        selected_format = request.form.get(f"{key.split('-')[0]}-format")
        print(f"Processing file: {file.filename}, Format: {selected_format}")

        if file and file.filename.endswith('.zip'):
            try:
                # uploads 폴더가 없으면 생성
                if not os.path.exists(UPLOAD_FOLDER):
                    os.makedirs(UPLOAD_FOLDER)

                # 압축 파일 저장 및 해제
                zip_path = os.path.join(UPLOAD_FOLDER, file.filename)
                temp_dir = os.path.join(UPLOAD_FOLDER, 'temp_extract')  # 임시 압축해제 디렉토리
                data_name = os.path.splitext(file.filename)[0]  # 압축파일명을 데이터셋 이름으로 사용
                data_dir = os.path.join(UPLOAD_FOLDER, data_name)  # 최종 데이터 디렉토리
                
                try:
                    # 임시 디렉토리 정리
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
                    os.makedirs(temp_dir)

                    # 최종 데이터 디렉토리 생성
                    if not os.path.exists(data_dir):
                        os.makedirs(data_dir)

                    # 기존 데이터셋 제거
                    if fo.dataset_exists(data_name):
                        print(f"Removing existing dataset: {data_name}")
                        fo.delete_dataset(data_name)

                    # 압축 파일 저장
                    file.save(zip_path)
                    print(f"File saved to: {zip_path}")

                    # 임시 디렉토리에 압축 해제
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        contents = [member for member in zip_ref.namelist() if not member.endswith('.DS_Store')]
                        total_size = sum(zip_ref.getinfo(member).file_size for member in contents)
                        
                        with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Extracting {file.filename}") as pbar:
                            zip_ref.extractall(temp_dir, members=contents)
                            pbar.update(total_size)

                    macosx_path = os.path.join(temp_dir, '__MACOSX')
                    if os.path.exists(macosx_path):
                        print("Removing __MACOSX directory...")
                        shutil.rmtree(macosx_path)

                    # 디렉토리 구조 분석 및 파일 이동
                    def find_dataset_root(start_path):
                        """데이터셋의 실제 루트 디렉토리를 찾는 함수"""
                        for root, dirs, files in os.walk(start_path):
                            # 'dataset.yaml' 파일과 'images', 'labels' 디렉토리가 모두 있는지 확인
                            if 'dataset.yaml' in files and 'images' in dirs and 'labels' in dirs:
                                return root
                            # 'dataset.yaml' 파일과 'train', 'valid', 'test' 디렉토리가 있는지 확인
                            if 'dataset.yaml' in files and 'train' in dirs and 'valid' in dirs and 'test' in dirs:
                                return root

                        # filename/filename/ 구조 처리
                        for root, dirs, files in os.walk(start_path):
                            for dir_name in dirs:
                                subdir_path = os.path.join(root, dir_name)
                                subdirs = os.listdir(subdir_path)
                                if 'dataset.yaml' in subdirs and ('images' in subdirs and 'labels' in subdirs or
                                                                'train' in subdirs and 'valid' in subdirs and 'test' in subdirs):
                                    return subdir_path

                        return None

                    dataset_root = find_dataset_root(temp_dir)
                    if dataset_root:
                        print(f"Found dataset root at: {dataset_root}")
                        
                        # 필요한 파일들을 최종 위치로 이동
                        for item in os.listdir(dataset_root):
                            src_path = os.path.join(dataset_root, item)
                            dst_path = os.path.join(data_dir, item)
                            
                            if os.path.exists(dst_path):
                                if os.path.isdir(dst_path):
                                    shutil.rmtree(dst_path)
                                else:
                                    os.remove(dst_path)
                            
                            shutil.move(src_path, dst_path)
                            print(f"Moved {item} to final location")
                    else:
                        raise Exception("Could not find valid dataset structure in the ZIP file")

                    # 임시 디렉토리 삭제
                    shutil.rmtree(temp_dir)
                    
                    # 최종 디렉토리 구조 확인
                    expected_paths_1 = [
                        os.path.join(data_dir, 'dataset.yaml'),
                        os.path.join(data_dir, 'images', 'train'),
                        os.path.join(data_dir, 'images', 'val'),
                        os.path.join(data_dir, 'labels', 'train'),
                        os.path.join(data_dir, 'labels', 'val')
                    ]

                    expected_paths_2 = [
                        os.path.join(data_dir, 'dataset.yaml'),
                        os.path.join(data_dir, 'train', 'images'),
                        os.path.join(data_dir, 'train', 'labels'),
                        os.path.join(data_dir, 'valid', 'images'),
                        os.path.join(data_dir, 'valid', 'labels')
                    ]

                    # test 디렉토리 경로 추가 (있는 경우에만)
                    test_paths_1 = [
                        os.path.join(data_dir, 'images', 'test'),
                        os.path.join(data_dir, 'labels', 'test')
                    ]
                    test_paths_2 = [
                        os.path.join(data_dir, 'test', 'images'),
                        os.path.join(data_dir, 'test', 'labels')
                    ]

                    # 기본 구조 확인
                    if not any(all(os.path.exists(path) for path in expected_paths) for expected_paths in [expected_paths_1, expected_paths_2]):
                        raise Exception(f"Expected dataset structure not found")

                    # test 디렉토리 확인 (있는 경우에만)
                    if any(os.path.exists(path) for path in test_paths_1):
                        missing_paths = [path for path in test_paths_1 if not os.path.exists(path)]
                        if missing_paths:
                            raise Exception(f"Test directory structure is incomplete. Missing: {', '.join(missing_paths)}")
                    elif any(os.path.exists(path) for path in test_paths_2):
                        missing_paths = [path for path in test_paths_2 if not os.path.exists(path)]
                        if missing_paths:
                            raise Exception(f"Test directory structure is incomplete. Missing: {', '.join(missing_paths)}")
                    
                    print(f"Directory structure verified successfully")

                    # 데이터셋 로드 - data_name을 데이터셋 이름으로 사용
                    loader = InputDataLoader(data_dir, selected_format, data_name)
                    if key == "ref-upload":
                        ref_dataset = loader.get_img_data()
                        loader.add_tags("ref")
                    elif key == "cur-upload":
                        cur_dataset = loader.get_img_data()
                        loader.add_tags("cur")
                    elif key == "test-upload":
                        test_dataset = loader.get_img_data()
                        loader.add_tags("test")

                    # 원본 압축 파일 삭제
                    os.remove(zip_path)
                    print(f"Successfully processed {file.filename} as dataset '{data_name}'")

                except Exception as e:
                    print(f"Error during processing {file.filename}: {e}")
                    # 에러 발생 시 정리
                    if os.path.exists(data_dir):
                        shutil.rmtree(data_dir)
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
                    if os.path.exists(zip_path):
                        os.remove(zip_path)
                    return jsonify({'error': str(e)}), 500

                print(f"Successfully processed {file.filename} and cleaned up uploads folder")

            except Exception as e:
                print(f"Error during processing {file.filename}: {e}")
                # 에러 발생 시 uploads 폴더 정리
                if os.path.exists(UPLOAD_FOLDER):
                    for item in os.listdir(UPLOAD_FOLDER):
                        item_path = os.path.join(UPLOAD_FOLDER, item)
                        try:
                            if os.path.isfile(item_path):
                                os.remove(item_path)
                            elif os.path.isdir(item_path):
                                shutil.rmtree(item_path)
                        except:
                            pass
                return jsonify({'error': str(e)}), 500

    merged_dataset_name = request.form.get('merged-dataset-name')

    # 기존 데이터셋 삭제
    if fo.dataset_exists(merged_dataset_name):
        print(f"Deleting existing dataset: {merged_dataset_name}")
        fo.delete_dataset(merged_dataset_name)

    print(f"Merged Dataset Name: {merged_dataset_name}")  # 디버그 출력
    merged_dataset = fo.Dataset(merged_dataset_name, persistent=True)
    
    # 분기 처리
    print("\n=== Dataset Merge Debug Info ===")
    print(f"ref_dataset exists: {ref_dataset is not None}")
    print(f"cur_dataset exists: {cur_dataset is not None}")
    print(f"test_dataset exists: {test_dataset is not None}")
    
    if ref_dataset and not cur_dataset and not test_dataset:
        print(f"Case 1: Adding only Ref Dataset... {ref_dataset.name}")
        print(f"Ref dataset sample count: {len(ref_dataset)}")
        merged_dataset.add_samples(ref_dataset)
    elif ref_dataset and cur_dataset and not test_dataset:
        print(f"Case 2: Adding Ref and Cur Datasets...")
        print(f"Ref dataset: {ref_dataset.name}, sample count: {len(ref_dataset)}")
        print(f"Cur dataset: {cur_dataset.name}, sample count: {len(cur_dataset)}")
        merged_dataset.add_samples(ref_dataset)
        merged_dataset.add_samples(cur_dataset)
    elif ref_dataset and cur_dataset and test_dataset:
        print(f"Case 3: Adding Ref, Cur, and Test Datasets...")
        print(f"Ref dataset: {ref_dataset.name}, sample count: {len(ref_dataset)}")
        print(f"Cur dataset: {cur_dataset.name}, sample count: {len(cur_dataset)}")
        print(f"Test dataset: {test_dataset.name}, sample count: {len(test_dataset)}")
        merged_dataset.add_samples(ref_dataset)
        merged_dataset.add_samples(cur_dataset)
        merged_dataset.add_samples(test_dataset)
    else:
        print("No valid dataset combination found")
        print("Current state:")
        print(f"- ref_dataset: {ref_dataset.name if ref_dataset else 'None'}")
        print(f"- cur_dataset: {cur_dataset.name if cur_dataset else 'None'}")
        print(f"- test_dataset: {test_dataset.name if test_dataset else 'None'}")
    
    print(f"\nFinal merged dataset sample count: {len(merged_dataset)}")
    merged_dataset.save()
    print("=== End Dataset Merge Debug Info ===\n")

    print("Deleting Temporary Datasets...")
    # 개별 데이터셋 삭제
    if ref_dataset:
        fo.delete_dataset(ref_dataset.name)
    if cur_dataset:
        fo.delete_dataset(cur_dataset.name)
    if test_dataset:
        fo.delete_dataset(test_dataset.name)

    if merged_dataset:
        print("\n=== Starting Embedding and Visualization Process ===")
        print(f"Dataset name: {merged_dataset.name}")
        print(f"Dataset size: {len(merged_dataset)}")
        
        print("\nCalculating Embeddings...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        model, preprocess = clip.load("ViT-B/16", device=device)
        data = fom_runner.get_embeddings(merged_dataset, device, model, preprocess)
        embeddings_by_sample_id = fom_runner.collect_image_embeddings_by_sample_id(data)
        print(f"Total embeddings to insert: {len(data)}")

        print("\nInserting Embeddings to Milvus...")
        milvus_manager = get_milvus_manager(args.db_path)
        milvus_manager.create_collection(merged_dataset.name)
        milvus_manager.insert(merged_dataset.name, data)
        print(f"Successfully inserted embeddings to Milvus collection: {merged_dataset.name}")

        print("\nComputing Visualization...")
        results = fob.compute_visualization(
            merged_dataset,
            embeddings=embeddings_by_sample_id,
            brain_key="clip_embeddings",
            plot_points=True,
            verbose=True,
        )
        fom_runner.set_dataset(merged_dataset, results)
        print("=== Completed Embedding and Visualization Process ===\n")
    else:
        print("\n=== Error: No Valid Dataset for Processing ===")
        print("merged_dataset is None or empty")
        print(f"Dataset name: {merged_dataset_name}")
        print(f"Dataset exists: {fo.dataset_exists(merged_dataset_name)}")
        print("=== End Error Report ===\n")

    if fom_runner.session.dataset:
        print("Successfully processed all datasets. Starting FiftyOne Visualization...")  # 디버그 출력
        # /dataclinic 페이지로 리다이렉트하며 데이터 전달
        return redirect(url_for(
            'dataclinic',
            fiftyone_port_number=fom_runner.port,
        ))
    
    else:
        print("No valid files processed.")  # 디버그 출력
        return jsonify({'message': 'Invalid file type'}), 400

@app.route('/dsampler/dataclinic')
def dataclinic():
    # list_views를 가져옵니다.
    list_views = fom_runner.session.dataset.list_saved_views()
    
    return render_template(
        'dataclinic.html',
        fiftyone_port_number=fom_runner.port,
        list_views=list_views,
    )

# # @app.route('/save', methods=['POST'])
# # def save_dataset():
# #     # 데이터셋 변경사항 저장 및 내보내기
# #     print()
# #     print("Save Changes of Dataset...")
# #     print()
# #     # export_dir = args.dataset_dir
# #     # dataset.export(
# #     #     export_dir=export_dir,
# #     #     dataset_type=fo.types.FiftyOneDataset,
# #     # )
# #     dataset.save()

# #     return redirect(url_for('home'))

@app.route('/dsampler/export', methods=['POST'])
def export_selected_view():
    # 선택된 뷰 가져오기
    selected_view = request.form.get('selected_view')
    selected_format = request.form.get('selected_format')
    print(f"Selected view: {selected_view}, Selected format: {selected_format}")  # 디버깅을 위한 출력

    if selected_view and selected_format:
        print(f"Exporting View: {selected_view}")
        view = fom_runner.session.dataset.load_saved_view(selected_view)

        if len(view) <= 4:
            print("Error: The selected view contains 4 or fewer samples.")
            return jsonify({'error': 'The selected view contains 4 or fewer samples. Please select a view with more samples.'}), 400

        view_export_dir = f"./datasets/exported_datasets/{fom_runner.session.dataset.name}_{selected_view}"
        label_field = "ground_truth"

        # 전체 샘플을 train, val, test로 스플릿
        splits = ['train', 'val', 'test']
        split_ratios = [0.7, 0.2, 0.1]  # 예시 비율
        view.shuffle(seed=42)  # 랜덤 시드로 셔플

        # 기존의 train, val, test 태그 삭제 및 새로 추가
        num_samples = len(view)
        split_indices = np.cumsum([int(r * num_samples) for r in split_ratios])

        for idx, sample in enumerate(view):
            # 기존 태그 중 train, val, test 제거
            sample.tags = [tag for tag in sample.tags if tag not in splits]

            # 새로운 스플릿 태그 추가
            if idx < split_indices[0]:
                sample.tags.append('train')
            elif idx < split_indices[1]:
                sample.tags.append('val')
            else:
                sample.tags.append('test')
            sample.save()

        # 내보내기 포맷 선택
        if selected_format == "YOLOv5Dataset":
            dataset_type = fo.types.YOLOv5Dataset
        elif selected_format == "FiftyOneDataset":
            dataset_type = fo.types.FiftyOneDataset

        # 내보내기
        for split in splits:
            split_view = view.match_tags(split)
            split_view.export(
                export_dir=view_export_dir,
                dataset_type=dataset_type,
                label_field=label_field,
                split=split,
            )

        print(f"Exported to {view_export_dir}")

    return redirect(url_for('dataclinic'))

@app.route('/dsampler/train_page', methods=['GET', 'POST'])
def train_page():
    # Exported datasets directory
    export_dir = './datasets/exported_datasets/'
    # Models directory
    models_dir = './models/'

    # exported_datasets 디렉토리가 없으면 생성
    if not os.path.exists(export_dir):
        print(f"Exported datasets directory not found. Creating directory: {export_dir}")
        os.makedirs(export_dir)

    # models 디렉토리가 없으면 생성
    if not os.path.exists(models_dir):
        print(f"Models directory not found. Creating directory: {models_dir}")
        os.makedirs(models_dir)

    # 데이터셋과 모델의 경로를 저장
    datasets = [d for d in os.listdir(export_dir) if os.path.isdir(os.path.join(export_dir, d))]
    models = [m for m in os.listdir(models_dir) if os.path.isfile(os.path.join(models_dir, m))]

    return render_template('train_page.html', datasets=datasets, models=models)

@app.route('/dsampler/train', methods=['GET', 'POST'])
def train():
    if request.method == 'POST':
        print(f"Train Dataset : {request.form.get('selected_dataset')}")
        print(f"Target Model : {request.form.get('selected_model')}")
        print()
        # 파라미터 값 가져오기
        project = request.form.get('project', 'runs')
        name = request.form.get('name', 'exp')
        epochs = int(request.form.get('epochs', 100))
        batch_size = int(request.form.get('batch_size', 16))
        img_size = int(request.form.get('img_size', 640))
        learning_rate = float(request.form.get('learning_rate', 0.001))

        flask_session['project'] = project
        flask_session['run'] = name
        
        selected_dataset = "datasets/exported_datasets/" + request.form.get('selected_dataset') + '/dataset.yaml'
        selected_model = "models/" + request.form.get('selected_model')
        log_dir = "logs/" + project + "/" + name

        # 별도의 스레드에서 훈련 시작 (파라미터 전달)
        training_thread = threading.Thread(
            target=train_yolo,
            args=(
                selected_dataset,
                selected_model,
                project,
                name,
                epochs,
                batch_size,
                img_size,
                learning_rate
            )
        )
        training_thread.start()

        tensorboard_thread = threading.Thread(
            target=lambda: tsb_runner.start(logdir=log_dir)
        )
        tensorboard_thread.start()
        tsb_runner.emit_event(socketio, 'tensorboard_ready', {'status': 'ready'})

        return render_template('train_page.html', project=project, name=name, epochs=epochs, batch_size=batch_size, img_size=img_size, learning_rate=learning_rate)

    return render_template('train_page.html', project='runs', name='exp', epochs=100, batch_size=16, img_size=640, learning_rate=0.001)

@app.route('/dsampler/download_model')
def download_model():
    project = flask_session.get('project', 'runs')
    run = flask_session.get('run', 'exp')
    model_path = f'logs/{project}/{run}/weights/best.pt'
    print()
    print(f"Downloading Model : {model_path}")
    print()

    return send_file(model_path, as_attachment=True)

@app.route('/dsampler/stream_logs')
def stream_logs():
    def generate():
        while True:
            # 여러 줄의 로그를 한 번에 가져옴
            lines = capture_stream.get_output().splitlines()
            if lines:
                for line in lines:
                    yield f"data: {line}\n\n"
                capture_stream.clear_output()  # 로그 전송 후 초기화
            time.sleep(1)  # Add a small delay to prevent high CPU usage

    return Response(generate(), mimetype='text/event-stream')







def parse_detections(results):
    """YOLO 모델의 결과를 박스, 색상, 클래스명으로 파싱"""
    boxes = []
    colors = []
    names = []
    
    for result in results[0].boxes:
        box = result.xyxy[0].cpu().numpy()  # x1, y1, x2, y2 형식
        conf = float(result.conf)
        cls = int(result.cls)
        name = results[0].names[cls]
        
        boxes.append(box)
        colors.append(generate_color(cls))  # 클래스별 고유 색상
        names.append(f"{name} {conf:.2f}")
    
    return np.array(boxes), colors, names

def generate_color(class_id):
    """클래스 ID에 따른 고유한 색상 생성"""
    np.random.seed(class_id)
    color = np.random.randint(0, 255, size=3).tolist()
    return color

def draw_detections(boxes, colors, names, image):
    """검출 결과를 이미지에 시각화"""
    for box, color, name in zip(boxes, colors, names):
        x1, y1, x2, y2 = map(int, box)
        
        # 박스 그리기
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # 텍스트 배경 그리기
        (text_w, text_h), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(image, (x1, y1-text_h-4), (x1+text_w, y1), color, -1)
        
        # 텍스트 그리기
        cv2.putText(image, name, (x1, y1-4), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    
    return image

def get_image_from_request(request, grayscale=False):
    print(request.files)
    if request.files and "file" in request.files:
        if request.files["file"].content_type.startswith("image"):
            data = request.files["file"].read()
            bgr = cv2.imdecode(
                np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if grayscale else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        raise TypeError("file content type should be image.")
    raise ValueError("file field not found from request.")

def get_model():
    if current_app.model_storage is None:
        raise ValueError("Model not loaded. Please load the model first.")
    return current_app.model_storage

def visualize_layer_sequence(model, rgb_img, img, cam_class, cam_kwargs, task='od', use_rgb=True):
    """
    모델의 레이어별 CAM 시각화를 수행하는 함수
    
    Args:
        model: YOLO 모델
        rgb_img: RGB 형식의 입력 이미지
        img: 정규화된 입력 이미지
        task: CAM 작업 유형 ('od' for object detection)
        use_rgb: RGB 시각화 사용 여부
    
    Returns:
        str: 저장된 시퀀스 이미지의 경로
    """
    from datadoctor.yolo_cam.utils.image import show_cam_on_image as show_yolocam_on_image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    total_layers = len(model.model.model) - 1
    n_seq = 7

    # 1, total_layers-1은 반드시 포함, 나머지는 균등하게 샘플링
    if total_layers <= 2:
        layer_indices = [1, total_layers-1]
    else:
        # 중간값 균등 샘플링 (2 ~ total_layers-2)
        if n_seq > 2:
            mid_indices = np.linspace(2, total_layers-2, n_seq-2, dtype=int)
            layer_indices = [1] + list(mid_indices) + [total_layers-1]
        else:
            layer_indices = [1, total_layers-1]

    # 메인 플롯 생성
    fig = plt.figure(figsize=(21, 7))
    gs = plt.GridSpec(2, 1, height_ratios=[0.2, 1], hspace=0)
    
    # 깊이 표시 컬러바 추가
    ax_depth = fig.add_subplot(gs[0])
    gradient = np.linspace(0, 1, 100).reshape(1, -1)
    ax_depth.imshow(gradient, cmap='Blues', aspect='auto')
    ax_depth.set_xticks([0, 99])
    ax_depth.set_xticklabels(['Shallow', 'Deep'], fontsize=18)
    ax_depth.set_yticks([])
    ax_depth.set_title('Layer Depth', pad=10, fontsize=22)
    
    # 각 레이어의 CAM 생성 및 시각화
    for idx, layer_idx in enumerate(layer_indices):
        try:
            target_layer = [model.model.model[layer_idx]]
            cam = cam_class(model, target_layer, **cam_kwargs)
            grayscale_cam = cam(rgb_img)[0, :, :]
            cam_image = show_yolocam_on_image(img, grayscale_cam, use_rgb=use_rgb)
            
            ax = plt.subplot(2, n_seq, n_seq + idx + 1)
            ax.imshow(cam_image)
            ax.set_title(f'Layer {layer_idx}', fontsize=18)
            ax.axis('off')
        except Exception as e:
            print(f"Error processing layer {layer_idx}: {str(e)}")
            continue
    
    plt.suptitle('Layer-wise Visualization (Shallow → Deep)', y=1.1, fontsize=18)
    plt.subplots_adjust(left=0, right=1, top=0.98, bottom=0.01, wspace=0.05, hspace=0.001)
    
    # 이미지 저장
    output_dir = 'static/cam_results'
    os.makedirs(output_dir, exist_ok=True)
    timestamp = int(time.time())
    sequence_path = f'{output_dir}/layer_sequence_{timestamp}.jpg'
    plt.savefig(sequence_path, bbox_inches='tight', dpi=300, pad_inches=0.08)
    plt.close()
    
    return sequence_path





@app.route('/camvis/upload_page')
def cam_upload():
    models = [m for m in os.listdir('./models')]
    return render_template('camupload.html', models=models)

@app.route('/api/model/load', methods=['POST'])
def load_model():
    from ultralytics import YOLO

    data = request.json
    model_path = os.path.join('models', data['model_name'])
    use_gpu = data['use_gpu']
    
    try:
        device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        print(model_path)
        current_app.model_storage = YOLO(model_path)
        current_app.model_storage.to(device)
        layer_count = len(current_app.model_storage.model.model) - 1  # 모델 구조에 따라 조정 필요
        
        return jsonify({
            'success': True,
            'layer_count': layer_count
        })
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/camvis/run_cam', methods=['POST'])
def run_cam():
    # try:
        use_grayscale = request.form.get('useGrayscale') == 'true'
        img = get_image_from_request(request, grayscale=use_grayscale)
        img = cv2.resize(img, (640, 640))  # YOLO 입력 크기로 리사이즈
        rgb_img = img.copy()  # 원본 복사본 저장
        img = np.float32(img) / 255  # 정규화

        task = request.form.get('task')
        use_rgb = request.form.get('useRgb') == 'false'

        from datadoctor.yolo_cam.eigen_cam import EigenCAM as YOLO_EigenCAM
        from datadoctor.yolo_cam.utils.image import show_cam_on_image as show_yolocam_on_image
        from datadoctor.yolo_cam.utils.image import scale_cam_image as scale_yolocam_image
        import matplotlib.pyplot as plt

        model = get_model()
        results = model(rgb_img)
        boxes, colors, names = parse_detections(results)
        target_layer = [model.model.model[int(request.form.get('targetLayer'))]]

        # CAM 계산
        cam = YOLO_EigenCAM(model, target_layer, task=task)
        grayscale_cam = cam(rgb_img)[0, :, :]

        # CAM 이미지 시각화
        cam_image = show_yolocam_on_image(img, grayscale_cam, use_rgb=use_rgb)
        # 흑백
        g_scale_normalized = (grayscale_cam * 255).astype(np.uint8)
        g_scale_image = cv2.resize(np.stack([g_scale_normalized] * 3, axis=2), (640, 640))

        # 이미지 저장 경로 설정
        output_dir = 'static/cam_results'
        os.makedirs(output_dir, exist_ok=True)
        timestamp = int(time.time())

        # 원본 이미지 저장
        original_path = f'{output_dir}/original_{timestamp}.jpg'
        cv2.imwrite(original_path, cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))

        # CAM 결과 이미지 저장
        cam_path = f'{output_dir}/cam_{timestamp}.jpg'
        cv2.imwrite(cam_path, cam_image)

        # 흑백 이미지 저장
        g_scale_path = f'{output_dir}/g_scale_{timestamp}.jpg'
        cv2.imwrite(g_scale_path, g_scale_image)

        # 레이어 시퀀스 시각화 수행
        sequence_path = visualize_layer_sequence(
            model, rgb_img, img,
            cam_class=YOLO_EigenCAM,
            cam_kwargs={'task': task},
            use_rgb=use_rgb
        )

        # 추론 결과 이미지 저장
        inference_img = draw_detections(boxes, colors, names, rgb_img)
        inference_path = f'{output_dir}/inference_{timestamp}.jpg'
        cv2.imwrite(inference_path, cv2.cvtColor(inference_img, cv2.COLOR_RGB2BGR))
        
        # 세션에 결과 저장
        flask_session['cam_results'] = {
            'original_image': original_path,
            'inference_image': inference_path,
            'cam_image': cam_path,
            'g_scale_image': g_scale_path,
            'sequence_image': sequence_path,
            'detection_results': [
                {
                    'class_name': result.names[int(result.boxes.cls[0])],
                    'confidence': float(result.boxes.conf[0]),
                    'bbox': result.boxes.xyxy[0].tolist()
                }
                for result in results
            ]
        }
        
        return jsonify({
            'success': True,
            'redirect_url': url_for('cam_result')
        })

    # except ValueError as e:
    #     return jsonify({'error': str(e)}), 400
    # except Exception as e:
    #     return jsonify({'error': str(e)}), 500

@app.route('/camvis/result')
def cam_result():
    results = flask_session.get('cam_results', {})
    return render_template('cam_result.html',
        original_image=results.get('original_image'),
        inference_image=results.get('inference_image'),
        cam_image=results.get('cam_image'),
        g_scale_image=results.get('g_scale_image'),
        sequence_image=results.get('sequence_image'),
        detection_results=results.get('detection_results', [])
    )

# @socketio.on('check_fiftyone_ready')
# def handle_check_fiftyone_ready():
#     # FiftyOne 세션이 준비되었는지 확인하는 로직을 추가하세요.
#     # 예를 들어, 세션이 이미 실행 중이라면 'ready' 상태를 emit합니다.
#     if fiftyone_thread and fiftyone_thread.is_alive():
#         emit('fiftyone_ready', {'status': 'ready'})
#     else:
#         emit('fiftyone_ready', {'status': 'not_ready'})

if __name__ == "__main__":
    init_app(app)
    socketio.run(app, port=5555, debug=False)