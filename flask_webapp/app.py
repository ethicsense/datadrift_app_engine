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
from datetime import datetime
import subprocess
import atexit
import json
from tqdm import tqdm
import shutil
import platform

from trainer import train_yolo
from utils import TensorboardManager, FiftyoneManager, CaptureOutput, InputDataLoader, MilvusManager

def get_milvus_manager(db_path=None):
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

def create_parser():
    """Flask 앱 전용 argument parser 생성"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, help="Port to run the FiftyOne app on")
    return parser

# 데이터셋 로드 및 세션 생성은 애플리케이션 시작 시 한 번만 수행
tsb_runner = TensorboardManager(port=6006)
atexit.register(tsb_runner.stop)

# FiftyOne Manager는 config 시스템을 통해 관리

app = Flask(__name__, static_folder='static')
CORS(app)
socketio = SocketIO(app)
app.secret_key = os.urandom(24)

## logger instance
capture_stream = CaptureOutput()
sys.stdout = capture_stream

def get_project_paths():
    """flask_webapp 디렉토리 기준으로 주요 디렉토리 경로들을 반환합니다."""
    try:
        from config import config
        return config.get_project_paths()
    except ImportError:
        # config가 없는 경우 기존 방식 사용
        flask_webapp_dir = os.path.dirname(os.path.abspath(__file__))
        
        return {
            'flask_webapp_dir': flask_webapp_dir,
            'models_dir': os.path.join(flask_webapp_dir, 'models'),
            'datasets_uploads': os.path.join(flask_webapp_dir, 'datasets', 'uploads'),
            'datasets_exported': os.path.join(flask_webapp_dir, 'datasets', 'exported_datasets'),
            'logs_dir': os.path.join(flask_webapp_dir, 'logs'),
            'static_cam_results': os.path.join(flask_webapp_dir, 'static', 'cam_results'),
            'static_perturbation_results': os.path.join(flask_webapp_dir, 'static', 'perturbation_results')
        }

def get_fiftyone_manager():
    """FiftyOne Manager를 config 시스템을 통해 가져오는 공통 함수"""
    try:
        from config import config
        fom_runner, fiftyone_thread = config.get_fiftyone_manager()
        print(f"✅ Using FiftyOne Manager from config (port: {fom_runner.port})")
        return fom_runner, fiftyone_thread
    except ImportError:
        from utils import FiftyoneManager
        fom_runner = FiftyoneManager(port=8159)
        fiftyone_thread = fom_runner.start()
        print(f"✅ Using FiftyOne Manager from fallback (port: {fom_runner.port})")
        return fom_runner, fiftyone_thread

def init_app(app):
    app.model_storage = None
    
    # 설정 초기화 및 디렉토리 생성
    try:
        from config import config
        config.ensure_directories()
        print("✅ Using centralized configuration")
    except ImportError:
        # config가 없는 경우 기존 방식 사용
        paths = get_project_paths()
        required_dirs = [
            paths['models_dir'],
            paths['datasets_uploads'],
            paths['datasets_exported'],
            paths['logs_dir'],
            paths['static_cam_results'],
            paths['static_perturbation_results']
        ]
        
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                print(f"Required directory not found. Creating directory: {dir_path}")
                os.makedirs(dir_path, exist_ok=True)
        print("⚠️  Using fallback configuration")

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
    milvus_manager = get_milvus_manager()

    # FiftyOne Manager 초기화
    fom_runner, fiftyone_thread = get_fiftyone_manager()

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
            milvus_manager = get_milvus_manager()
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
    paths = get_project_paths()
    UPLOAD_FOLDER = paths['datasets_uploads']
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
        
        # FiftyOne Manager 초기화
        fom_runner, fiftyone_thread = get_fiftyone_manager()
        
        print("\nCalculating Embeddings...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        model, preprocess = clip.load("ViT-B/16", device=device)
        data = fom_runner.get_embeddings(merged_dataset, device, model, preprocess)
        embeddings_by_sample_id = fom_runner.collect_image_embeddings_by_sample_id(data)
        print(f"Total embeddings to insert: {len(data)}")

        print("\nInserting Embeddings to Milvus...")
        milvus_manager = get_milvus_manager()
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
    # FiftyOne Manager 초기화
    fom_runner, fiftyone_thread = get_fiftyone_manager()
    
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
    # FiftyOne Manager 초기화
    fom_runner, fiftyone_thread = get_fiftyone_manager()
    
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

        paths = get_project_paths()
        view_export_dir = f"{paths['datasets_exported']}/{fom_runner.session.dataset.name}_{selected_view}"
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
    paths = get_project_paths()
    # Exported datasets directory
    export_dir = paths['datasets_exported'] + '/'
    # Models directory
    models_dir = paths['models_dir'] + '/'

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
        
        paths = get_project_paths()
        selected_dataset = f"{paths['datasets_exported']}/" + request.form.get('selected_dataset') + '/dataset.yaml'
        selected_model = f"{paths['models_dir']}/" + request.form.get('selected_model')
        log_dir = f"{paths['logs_dir']}/" + project + "/" + name

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
    paths = get_project_paths()
    model_path = f"{paths['logs_dir']}/{project}/{run}/weights/best.pt"
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

def visualize_layer_sequence(model, rgb_img, img, cam_kwargs, use_rgb, output_dir, file_name):
    """
    모델의 레이어별 CAM 시각화를 수행하는 함수
    
    Args:
        model: YOLO 모델
        rgb_img: RGB 형식의 입력 이미지
        img: 정규화된 입력 이미지
        cam_class: CAM 클래스
        cam_kwargs: CAM 생성에 필요한 키워드 인자
        use_rgb: RGB 시각화 사용 여부
        output_dir: 저장 경로
        file_name: 파일 이름
    
    Returns:
        str: 저장된 시퀀스 이미지의 경로
    """
    from yolo_cam.eigen_cam import EigenCAM as YOLO_EigenCAM
    from yolo_cam.utils.image import show_cam_on_image as show_yolocam_on_image
    from yolo_cam.utils.image import scale_cam_image as scale_yolocam_image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    total_layers = len(model.model.model) - 1
    n_cols = 5  # 한 줄에 5개의 서브플롯
    n_rows = (total_layers + n_cols - 1) // n_cols  # 올림 나눗셈

    # 메인 플롯 생성 - 왼쪽에 colorbar를 위한 공간 확보
    fig = plt.figure(figsize=(27, 5 * n_rows))
    
    # colorbar를 위한 axes 생성
    cbar_ax = fig.add_axes([0.02, 0.15, 0.02, 0.7])  # [left, bottom, width, height]

    # im 변수 초기화
    im = None
    
    # 각 레이어의 CAM 생성 및 시각화
    for layer_idx in range(1, total_layers):
        try:
            target_layer = [model.model.model[layer_idx]]
            cam = YOLO_EigenCAM(model, target_layer, **cam_kwargs)
            grayscale_cam = cam(rgb_img)[0, :, :]
            # use_rgb 값을 반전시켜서 전달
            cam_image = show_yolocam_on_image(img, grayscale_cam, use_rgb=not use_rgb)

            layer_path = f'{output_dir}/{file_name}_layer_{layer_idx}.jpg'
            plt.figure(figsize=(10, 10))
            plt.imshow(cam_image)
            plt.axis('off')
            plt.savefig(layer_path, bbox_inches='tight', dpi=300)
            plt.close()
            
            # 첫 번째 서브플롯에 Shallow 표시
            if layer_idx == 1:
                ax = plt.subplot(n_rows, n_cols, 1)
                ax.text(0.5, 0.5, 'Shallow', fontsize=24, ha='center', va='center', 
                       color='blue', weight='bold', transform=ax.transAxes)
                ax.axis('off')
                
                # 실제 레이어 1 이미지
                ax = plt.subplot(n_rows, n_cols, 2)
                im = ax.imshow(cam_image, vmin=0, vmax=1, cmap='jet')
                ax.set_title(f'Layer {layer_idx}', fontsize=18)
                ax.axis('off')
            # 마지막 서브플롯에 Deep 표시
            elif layer_idx == total_layers - 1:
                # 마지막 레이어 이미지
                ax = plt.subplot(n_rows, n_cols, n_cols * n_rows - 2)
                im = ax.imshow(cam_image, vmin=0, vmax=1, cmap='jet')
                ax.set_title(f'Layer {layer_idx}', fontsize=18)
                ax.axis('off')
                
                # Deep 텍스트
                ax = plt.subplot(n_rows, n_cols, n_cols * n_rows - 1)
                ax.text(0.5, 0.5, 'Deep', fontsize=24, ha='center', va='center',
                       color='blue', weight='bold', transform=ax.transAxes)
                ax.axis('off')
            else:
                # 중간 레이어들
                ax = plt.subplot(n_rows, n_cols, layer_idx + 1)  # +1 because first position is for Shallow
                im = ax.imshow(cam_image, vmin=0, vmax=1, cmap='jet')
                ax.set_title(f'Layer {layer_idx}', fontsize=18)
                ax.axis('off')
                
        except Exception as e:
            print(f"Error processing layer {layer_idx}: {str(e)}")
            continue
    
    # colorbar 추가 (im이 None이 아닌 경우에만)
    if im is not None:
        cbar = plt.colorbar(im, cax=cbar_ax, orientation='vertical')
        cbar.set_label('Gradient Score', fontsize=16, labelpad=20)
        cbar.ax.yaxis.set_label_position('left')
        cbar.ax.tick_params(labelsize=14)
    else:
        # im이 None인 경우 (레이어 처리 실패) colorbar axes 제거
        cbar_ax.remove()
    
    plt.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.05, wspace=0.1, hspace=0.2)
    
    # 이미지 저장
    sequence_path = f'{output_dir}/{file_name}_layer_sequence.jpg'
    plt.savefig(sequence_path, bbox_inches='tight', dpi=300, pad_inches=0.08)
    plt.close()
    
    return sequence_path






@app.route('/camvis/upload_page')
def cam_upload():
    paths = get_project_paths()
    models = [m for m in os.listdir(paths['models_dir'])]
    return render_template('camupload.html', models=models)

@app.route('/perturbation/upload_page')
def perturbation_vis():
    paths = get_project_paths()
    models = [m for m in os.listdir(paths['models_dir'])]
    return render_template('perturbation_upload.html', models=models)



@app.route('/api/model/load', methods=['POST'])
def load_model():
    from ultralytics import YOLO

    data = request.json
    paths = get_project_paths()
    model_path = os.path.join(paths['models_dir'], data['model_name'])
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

@app.route('/api/model/visualize', methods=['POST'])
def visualize_model():
    import netron
    import psutil
    import subprocess
    
    def kill_netron_process(port):
        try:
            # netstat을 사용하여 포트를 사용하는 프로세스의 PID 찾기
            cmd = f"lsof -i :{port} -t"
            pid = subprocess.check_output(cmd, shell=True).decode().strip()
            
            if pid:
                # netron 프로세스만 종료
                if 'netron' in psutil.Process(int(pid)).name().lower():
                    print(f"Found netron process (PID: {pid})")
                    subprocess.run(f"kill {pid}", shell=True)
                    print(f"Terminated netron process (PID: {pid})")
                    return True
        except subprocess.CalledProcessError:
            # 포트를 사용하는 프로세스가 없는 경우
            return False
        except Exception as e:
            print(f"Error killing process: {str(e)}")
            return False
        return False
    
    data = request.json
    paths = get_project_paths()
    model_path = os.path.join(paths['models_dir'], data['model_name'])
    port = 8888
    
    try:
        # 기존 netron 프로세스 종료
        if kill_netron_process(port):
            print(f"Successfully terminated existing netron process on port {port}")
            time.sleep(1)  # 프로세스가 완전히 종료될 때까지 잠시 대기
        
        # netron 서버 시작 (비동기로 실행)
        def start_netron():
            netron.start(model_path, address=port, browse=False)
        
        # 별도의 스레드에서 netron 서버 시작
        import threading
        thread = threading.Thread(target=start_netron)
        thread.daemon = True  # 메인 스레드가 종료되면 함께 종료
        thread.start()
        
        # 서버가 시작될 때까지 잠시 대기
        time.sleep(0.5)
        
        return jsonify({
            'success': True,
            'url': f'http://localhost:{port}'
        })
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/camvis/run_cam', methods=['POST'])
def run_cam():
    files = []
    for key in request.files:
        if key.startswith('file_'):
            files.append(request.files[key])
    
    if not files:
        return jsonify({'error': 'No images uploaded'}), 400
    
    # 프로젝트 이름 가져오기
    project_name = request.form.get('projectName', 'default_project')
    
    # 이미지 저장 경로 설정 - 프로젝트별 디렉토리 생성
    paths = get_project_paths()
    base_output_dir = paths['static_cam_results']
    project_output_dir = os.path.join(base_output_dir, project_name)
    os.makedirs(project_output_dir, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')

    use_grayscale = request.form.get('useGrayscale') == 'true'
    task = request.form.get('task')
    use_rgb = request.form.get('useRgb') == 'true'
    model = get_model()

    all_results = []

    for idx, file in enumerate(files):
        file_name = f'{timestamp}_{idx}'

        data = file.read()
        bgr = cv2.imdecode(
            np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))  # YOLO 입력 크기로 리사이즈
        rgb_img = img.copy()
        img = np.float32(img) / 255  # 정규화

        results = model(rgb_img)
        boxes, colors, names = parse_detections(results)

        # 원본 이미지 저장
        original_path = f'{project_output_dir}/{file_name}_original.jpg'
        cv2.imwrite(original_path, cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))

        # 레이어 시퀀스 시각화 수행
        sequence_path = visualize_layer_sequence(
            model, rgb_img, img,
            cam_kwargs={'task': task},
            use_rgb=use_rgb,
            output_dir=project_output_dir,
            file_name=file_name
        )

        # 추론 결과 이미지 저장
        inference_img = draw_detections(boxes, colors, names, rgb_img)
        inference_path = f'{project_output_dir}/{file_name}_inference.jpg'
        cv2.imwrite(inference_path, cv2.cvtColor(inference_img, cv2.COLOR_RGB2BGR))

        # 각 이미지의 결과 저장
        image_result = {
            'original_image': f'cam_results/{project_name}/{file_name}_original.jpg',
            'inference_image': f'cam_results/{project_name}/{file_name}_inference.jpg',
            'sequence_image': f'cam_results/{project_name}/{file_name}_layer_sequence.jpg',
            'detection_results': [
                {
                    'class_name': result.names[int(result.boxes.cls[0])],
                    'confidence': float(result.boxes.conf[0]),
                    'bbox': result.boxes.xyxy[0].tolist()
                }
                for result in results
            ],
            'file_name': file_name,
            'total_layers': len(model.model.model) - 1
        }
        all_results.append(image_result)
    
    # 세션에 결과 저장
    flask_session['cam_results'] = {
        'images': all_results,
        'total_images': len(all_results),
        'project_name': project_name
    }
    
    return jsonify({
        'success': True,
        'redirect_url': url_for('cam_result')
    })

@app.route('/camvis/result')
def cam_result():
    results = flask_session.get('cam_results', {})
    if not results:
        return redirect(url_for('cam_upload'))  # 결과가 없으면 업로드 페이지로 리다이렉트
        
    return render_template('cam_result.html',
        images=results.get('images', []),  # 이미지 리스트
        image_count=results.get('total_images', 0),  # 세션에서 이미지 개수 가져오기
        project_name=results.get('project_name', 'default_project')
    )

@app.route('/api/project/check', methods=['POST'])
def check_project_exists():
    data = request.json
    project_name = data.get('project_name', '')
    
    if not project_name:
        return jsonify({'error': 'Project name is required'}), 400
    
    # 프로젝트 디렉토리 경로 확인
    paths = get_project_paths()
    project_dir = os.path.join(paths['static_cam_results'], project_name)
    exists = os.path.exists(project_dir)
    
    return jsonify({
        'exists': exists,
        'project_name': project_name
    })

@app.route('/api/project/list', methods=['GET'])
def get_project_list():
    paths = get_project_paths()
    base_dir = paths['static_cam_results']
    projects = []
    
    if os.path.exists(base_dir):
        for project_name in os.listdir(base_dir):
            project_dir = os.path.join(base_dir, project_name)
            if os.path.isdir(project_dir):
                # 프로젝트 디렉토리에서 original 이미지들 찾기
                original_images = []
                for file in os.listdir(project_dir):
                    if file.endswith('_original.jpg'):
                        original_images.append({
                            'filename': file,
                            'path': f'cam_results/{project_name}/{file}'
                        })
                
                if original_images:  # original 이미지가 있는 프로젝트만 포함
                    projects.append({
                        'name': project_name,
                        'original_images': original_images,
                        'image_count': len(original_images)
                    })
    
    return jsonify({'projects': projects})

@app.route('/api/project/load', methods=['POST'])
def load_existing_project():
    data = request.json
    project_name = data.get('project_name', '')
    
    if not project_name:
        return jsonify({'error': 'Project name is required'}), 400
    
    # 프로젝트 경로 가져오기
    paths = get_project_paths()
    project_dir = os.path.join(paths['static_cam_results'], project_name)
    if not os.path.exists(project_dir):
        return jsonify({'error': 'Project not found'}), 404
    
    # 프로젝트 디렉토리에서 결과 파일들 찾기
    all_results = []
    
    # original 이미지들을 기준으로 결과 구성
    for file in os.listdir(project_dir):
        if file.endswith('_original.jpg'):
            base_name = file.replace('_original.jpg', '')
            
            # 관련 파일들 확인
            inference_file = f'{base_name}_inference.jpg'
            sequence_file = f'{base_name}_layer_sequence.jpg'
            
            # 레이어 파일들 찾기
            layer_files = []
            for layer_file in os.listdir(project_dir):
                if layer_file.startswith(base_name) and layer_file.endswith('.jpg') and 'layer_' in layer_file:
                    layer_files.append(layer_file)
            
            # 결과 구성
            image_result = {
                'original_image': f'cam_results/{project_name}/{file}',
                'inference_image': f'cam_results/{project_name}/{inference_file}' if os.path.exists(os.path.join(project_dir, inference_file)) else None,
                'sequence_image': f'cam_results/{project_name}/{sequence_file}' if os.path.exists(os.path.join(project_dir, sequence_file)) else None,
                'file_name': base_name,
                'total_layers': len(layer_files),
                'detection_results': []  # 기존 결과에서는 detection 결과를 저장하지 않았으므로 빈 배열
            }
            all_results.append(image_result)
    
    # 세션에 결과 저장
    flask_session['cam_results'] = {
        'images': all_results,
        'total_images': len(all_results),
        'project_name': project_name
    }
    
    return jsonify({
        'success': True,
        'redirect_url': url_for('cam_result')
    })

@app.route('/perturbation/run_perturbation', methods=['POST'])
def run_perturbation():
    try:
        # 모델 로드 확인
        if current_app.model_storage is None:
            return jsonify({'error': 'Model not loaded. Please load the model first.'}), 400
        
        # 폼 데이터에서 값들 가져오기
        model = current_app.model_storage
        project_name = request.form.get('projectName')
        task = request.form.get('task')
        
        # 마스킹 데이터 파싱
        mask_data = json.loads(request.form.get('mask_data', '[]'))
        masking_type = request.form.get('masking_type')
        canvas_width = int(request.form.get('canvas_width', 800))
        canvas_height = int(request.form.get('canvas_height', 600))
        
        print(f"Received mask data: {len(mask_data)} points")
        print(f"Mask data sample: {mask_data[:3] if mask_data else 'No data'}")
        print(f"Masking type: {masking_type}")
        print(f"Canvas dimensions: {canvas_width} x {canvas_height}")
        
        # Extra Perturbations 설정
        extra_perturbations_enabled = request.form.get('extra_perturbations_enabled') == 'true'
        extra_perturbations = {'enabled': extra_perturbations_enabled}
        
        if extra_perturbations_enabled:
            extra_perturbations.update({
                'brightness': int(request.form.get('brightness', 0)),
                'rotation': int(request.form.get('rotation', 0)),
                'scale': float(request.form.get('scale', 1.0))
            })
        
        # 이미지 파일 처리
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # 이미지 로드 및 전처리
        data = image_file.read()
        bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        original_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        
        # 원본 이미지 크기 저장
        original_height, original_width = original_img.shape[:2]
        print(f"Original image size: {original_width} x {original_height}")
        
        # 마스크 좌표를 원본 이미지 크기에 맞게 조정
        if mask_data:
            print(f"Received mask data: {len(mask_data)} points")
            print(f"Mask data sample: {mask_data[:3]}")
            print(f"Canvas dimensions: {canvas_width} x {canvas_height}")
            print(f"Original image size: {original_width} x {original_height}")
            
            # 캔버스 좌표를 원본 이미지 좌표로 변환
            transformed_mask_data = []
            for point in mask_data:
                # 캔버스 좌표 → 원본 이미지 좌표 변환
                scale_x = original_width / canvas_width
                scale_y = original_height / canvas_height
                
                transformed_x = int(point['x'] * scale_x)
                transformed_y = int(point['y'] * scale_y)
                
                # 좌표가 이미지 범위 내에 있는지 확인
                if 0 <= transformed_x < original_width and 0 <= transformed_y < original_height:
                    transformed_mask_data.append({'x': transformed_x, 'y': transformed_y})
            
            print(f"Transformed mask data: {len(transformed_mask_data)} points")
            print(f"Transformed sample: {transformed_mask_data[:3] if transformed_mask_data else 'No valid points'}")
            
            # 변환된 마스크 데이터 사용
            mask_data = transformed_mask_data
        
        # 마스킹 설정 정보 구성
        masking_settings = {
            'masking_type': masking_type,
            'mask_data': mask_data,  # 변환된 마스킹 좌표 (원본 이미지 기준)
            'mask_points_count': len(mask_data),
            'original_canvas_points': request.form.get('original_canvas_points'),  # 원본 캔버스 좌표 (1000x800 기준)
            'canvas_dimensions': {
                'width': canvas_width,
                'height': canvas_height
            },
            'image_dimensions': {
                'width': original_width,
                'height': original_height
            },
            'extra_perturbations': extra_perturbations,
            'model_info': {
                'model_name': getattr(model, 'ckpt_path', 'Unknown').split('/')[-1] if hasattr(model, 'ckpt_path') else 'Unknown',
                'use_gpu': str(model.device).startswith('cuda'),
                'device': str(model.device)
            }
        }
        
        # 프론트엔드에서 전송된 완전한 마스킹 데이터가 있는지 확인
        complete_mask_data = request.form.get('complete_mask_data')
        if complete_mask_data:
            try:
                complete_mask = json.loads(complete_mask_data)
                masking_settings['complete_mask_data'] = complete_mask
                print(f"Complete mask data received: {len(complete_mask.get('canvasPoints', []))} canvas points")
            except json.JSONDecodeError:
                print("Failed to parse complete mask data")
        
        # 원본 이미지에 마스크 적용
        print("Applying perturbation to original image...")
        perturbed_original = apply_perturbation(
            original_img, 
            mask_data, 
            masking_type, 
            extra_perturbations
        )
        
        # 교란된 원본 이미지를 모델 입력 크기로 리사이즈 (640x640)
        perturbed_img = cv2.resize(perturbed_original, (640, 640))
        print(f"Resized perturbed image for model input: {perturbed_img.shape[1]} x {perturbed_img.shape[0]}")
        
        # 원본 이미지도 모델 입력 크기로 리사이즈 (640x640)
        original_img = cv2.resize(original_img, (640, 640))
        print(f"Resized original image for model input: {original_img.shape[1]} x {original_img.shape[0]}")
        
        # 1. 원본 이미지에 대한 추론 수행
        print("Performing inference on original image...")
        original_results = model(original_img)
        original_boxes, original_colors, original_names = parse_detections(original_results)
        
        # 원본 추론 결과 저장
        original_detections = []
        for i, result in enumerate(original_results):
            if len(result.boxes) > 0:
                for j in range(len(result.boxes)):
                    detection = {
                        'class_name': result.names[int(result.boxes.cls[j])],
                        'confidence': float(result.boxes.conf[j]),
                        'bbox': result.boxes.xyxy[j].cpu().numpy().tolist(),  # [x1, y1, x2, y2]
                        'class_id': int(result.boxes.cls[j]),
                        'detection_id': f"orig_{i}_{j}"
                    }
                    original_detections.append(detection)
        
        # 2. 교란된 이미지에 대한 추론 수행
        print("Performing inference on perturbed image...")
        
        # 3. 교란된 이미지에 대한 추론 수행
        print("Performing inference on perturbed image...")
        perturbed_results = model(perturbed_img)
        perturbed_boxes, perturbed_colors, perturbed_names = parse_detections(perturbed_results)
        
        # 교란된 이미지 추론 결과 저장
        perturbed_detections = []
        for i, result in enumerate(perturbed_results):
            if len(result.boxes) > 0:
                for j in range(len(result.boxes)):
                    detection = {
                        'class_name': result.names[int(result.boxes.cls[j])],
                        'confidence': float(result.boxes.conf[j]),
                        'bbox': result.boxes.xyxy[j].cpu().numpy().tolist(),  # [x1, y1, x2, y2]
                        'class_id': int(result.boxes.cls[j]),
                        'detection_id': f"pert_{i}_{j}"
                    }
                    perturbed_detections.append(detection)
        
        # 4. 같은 객체 매칭 수행
        print("Matching objects between original and perturbed images...")
        matched_pairs = match_objects(original_detections, perturbed_detections)
        
        # 결과 저장 경로 설정
        paths = get_project_paths()
        output_dir = os.path.join(paths['static_perturbation_results'], project_name)
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        # 5. 결과 이미지들 저장 (모델 입력용 640x640만 저장)
        original_path = os.path.join(output_dir, f'{timestamp}_original.jpg')
        cv2.imwrite(original_path, cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR))
        
        perturbed_path = os.path.join(output_dir, f'{timestamp}_perturbed.jpg')
        cv2.imwrite(perturbed_path, cv2.cvtColor(perturbed_img, cv2.COLOR_RGB2BGR))
        
        # 원본 추론 결과 이미지 저장
        original_inference_img = draw_detections(original_boxes, original_colors, original_names, original_img.copy())
        original_inference_path = os.path.join(output_dir, f'{timestamp}_original_inference.jpg')
        cv2.imwrite(original_inference_path, cv2.cvtColor(original_inference_img, cv2.COLOR_RGB2BGR))
        
        # 교란된 이미지 추론 결과 저장
        perturbed_inference_img = draw_detections(perturbed_boxes, perturbed_colors, perturbed_names, perturbed_img.copy())
        perturbed_inference_path = os.path.join(output_dir, f'{timestamp}_perturbed_inference.jpg')
        cv2.imwrite(perturbed_inference_path, cv2.cvtColor(perturbed_inference_img, cv2.COLOR_RGB2BGR))
        
        # 6. 결과 데이터 구성
        result_data = {
            'project_name': project_name,  # 프로젝트 이름 추가
            'original_image': f'perturbation_results/{project_name}/{timestamp}_original.jpg',
            'perturbed_image': f'perturbation_results/{project_name}/{timestamp}_perturbed.jpg',
            'original_inference_image': f'perturbation_results/{project_name}/{timestamp}_original_inference.jpg',
            'perturbed_inference_image': f'perturbation_results/{project_name}/{timestamp}_perturbed_inference.jpg',
            'original_detections': original_detections,
            'perturbed_detections': perturbed_detections,
            'matched_pairs': matched_pairs,
            'perturbation_settings': masking_settings,
            'file_name': timestamp
        }
        
        # 결과 데이터를 JSON 파일로 저장
        result_json_path = os.path.join(output_dir, f'{timestamp}_results.json')
        with open(result_json_path, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        # 세션에 결과 저장 (최소한의 데이터만)
        flask_session['perturbation_results'] = {
            'project_name': project_name,
            'timestamp': timestamp,
            'has_results': True
        }
        
        # 디버깅 로그 추가
        print(f"Session data saved - project_name: {project_name}")
        print(f"Result data keys: {list(result_data.keys())}")
        print(f"Session perturbation_results keys: {list(flask_session['perturbation_results'].keys())}")
        
        return jsonify({
            'success': True,
            'redirect_url': url_for('perturbation_result'),
            'timestamp': timestamp
        })
        
    except Exception as e:
        print(f"Error in run_perturbation: {str(e)}")
        return jsonify({'error': str(e)}), 500

def calculate_iou(box1, box2):
    """
    두 바운딩 박스 간의 IoU를 계산하는 함수
    
    Args:
        box1, box2: [x1, y1, x2, y2] 형식의 바운딩 박스
    
    Returns:
        float: IoU 값 (0.0 ~ 1.0)
    """
    # 교집합 영역 계산
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # 합집합 영역 계산
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def calculate_center_distance(box1, box2):
    """
    두 바운딩 박스의 중심점 간 거리를 계산하는 함수
    
    Args:
        box1, box2: [x1, y1, x2, y2] 형식의 바운딩 박스
    
    Returns:
        float: 중심점 간 유클리드 거리
    """
    center1_x = (box1[0] + box1[2]) / 2
    center1_y = (box1[1] + box1[3]) / 2
    center2_x = (box2[0] + box2[2]) / 2
    center2_y = (box2[1] + box2[3]) / 2
    
    return np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)

def match_objects(original_detections, perturbed_detections, iou_threshold=0.3, distance_threshold=100):
    """
    원본 이미지와 교란된 이미지의 객체들을 매칭하는 함수
    
    Args:
        original_detections: 원본 이미지의 검출 결과 리스트
        perturbed_detections: 교란된 이미지의 검출 결과 리스트
        iou_threshold: IoU 매칭 임계값 (기본값: 0.3)
        distance_threshold: 중심점 거리 임계값 (기본값: 100픽셀)
    
    Returns:
        list: 매칭된 객체 쌍들의 리스트
    """
    matched_pairs = []
    used_original = set()
    used_perturbed = set()
    
    # 1단계: IoU 기반 매칭 (높은 우선순위)
    for i, orig_det in enumerate(original_detections):
        if i in used_original:
            continue
            
        best_iou = 0
        best_match_idx = -1
        
        for j, pert_det in enumerate(perturbed_detections):
            if j in used_perturbed:
                continue
                
            # 같은 클래스인 경우에만 매칭 시도
            if orig_det['class_id'] == pert_det['class_id']:
                iou = calculate_iou(orig_det['bbox'], pert_det['bbox'])
                
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_match_idx = j
        
        if best_match_idx != -1:
            matched_pairs.append({
                'original_detection': orig_det,
                'perturbed_detection': perturbed_detections[best_match_idx],
                'iou': best_iou,
                'match_type': 'iou',
                'confidence_change': perturbed_detections[best_match_idx]['confidence'] - orig_det['confidence']
            })
            used_original.add(i)
            used_perturbed.add(best_match_idx)
    
    # 2단계: 중심점 거리 기반 매칭 (낮은 우선순위)
    for i, orig_det in enumerate(original_detections):
        if i in used_original:
            continue
            
        best_distance = float('inf')
        best_match_idx = -1
        
        for j, pert_det in enumerate(perturbed_detections):
            if j in used_perturbed:
                continue
                
            # 같은 클래스이고 신뢰도가 높은 경우에만 매칭 시도
            if (orig_det['class_id'] == pert_det['class_id'] and 
                pert_det['confidence'] > 0.5):  # 신뢰도 임계값
                
                distance = calculate_center_distance(orig_det['bbox'], pert_det['bbox'])
                
                if distance < best_distance and distance <= distance_threshold:
                    best_distance = distance
                    best_match_idx = j
        
        if best_match_idx != -1:
            matched_pairs.append({
                'original_detection': orig_det,
                'perturbed_detection': perturbed_detections[best_match_idx],
                'center_distance': best_distance,
                'match_type': 'distance',
                'confidence_change': perturbed_detections[best_match_idx]['confidence'] - orig_det['confidence']
            })
            used_original.add(i)
            used_perturbed.add(best_match_idx)
    
    # 3단계: 매칭되지 않은 객체들 기록
    unmatched_original = [i for i in range(len(original_detections)) if i not in used_original]
    unmatched_perturbed = [i for i in range(len(perturbed_detections)) if i not in used_perturbed]
    
    print(f"Matched pairs: {len(matched_pairs)}")
    print(f"Unmatched original: {len(unmatched_original)}")
    print(f"Unmatched perturbed: {len(unmatched_perturbed)}")
    
    return {
        'matched_pairs': matched_pairs,
        'unmatched_original_indices': unmatched_original,
        'unmatched_perturbed_indices': unmatched_perturbed,
        'total_original': len(original_detections),
        'total_perturbed': len(perturbed_detections)
    }

def apply_perturbation(image, mask_data, masking_type, extra_perturbations=None):
    """
    원본 이미지에서 선택된 영역에 마스킹 타입을 적용하고, 
    Extra Perturbations가 활성화된 경우 추가 교란을 적용하는 함수
    
    Args:
        image: 원본 이미지 (numpy array)
        mask_data: 마스킹 영역 데이터 (list of dict with x, y coordinates)
        masking_type: 마스킹 타입 ('black', 'white', 'gaussian', 'blur', 'inductive_bias')
        extra_perturbations: 추가 교란 설정 (dict with brightness, rotation, scale values)
    
    Returns:
        numpy array: 교란된 이미지
    """
    img = image.copy()
    height, width = img.shape[:2]
    
    # 마스킹 영역을 위한 마스크 생성
    mask = np.zeros((height, width), dtype=np.uint8)
    
    if len(mask_data) > 0:
        # 마스킹 데이터를 numpy 배열로 변환
        mask_points = np.array([[point['x'], point['y']] for point in mask_data], dtype=np.int32)
        
        print(f"Mask points shape: {mask_points.shape}")
        print(f"Mask points sample: {mask_points[:3]}")
        print(f"Image shape: {img.shape}")
        
        # 마스킹 영역을 채우기
        cv2.fillPoly(mask, [mask_points], 255)
        
        # 마스크가 제대로 생성되었는지 확인
        mask_area = np.sum(mask > 0)
        print(f"Mask area: {mask_area} pixels")
        
        # 마스킹 타입에 따른 처리
        if masking_type == 'black':
            # 검은색으로 마스킹
            img[mask > 0] = [0, 0, 0]
            
        elif masking_type == 'white':
            # 흰색으로 마스킹
            img[mask > 0] = [255, 255, 255]
            
        elif masking_type == 'gaussian':
            # 가우시안 노이즈 적용
            noise = np.random.normal(0, 50, img.shape).astype(np.uint8)
            img[mask > 0] = np.clip(img[mask > 0] + noise[mask > 0], 0, 255)
            
        elif masking_type == 'blur':
            # 블러 처리
            blurred = cv2.GaussianBlur(img, (15, 15), 0)
            img[mask > 0] = blurred[mask > 0]
            
        elif masking_type == 'inductive_bias':
            # Inductive Bias: 마스킹된 영역을 이미지 전체에서 무작위 위치로 이동
            mask_coords = np.where(mask > 0)
            if len(mask_coords[0]) > 0:
                # 마스킹된 영역의 바운딩 박스 계산
                min_y, max_y = np.min(mask_coords[0]), np.max(mask_coords[0])
                min_x, max_x = np.min(mask_coords[1]), np.max(mask_coords[1])
                mask_height = max_y - min_y
                mask_width = max_x - min_x
                
                # 무작위 위치 생성 (이미지 경계 내에서)
                max_offset_x = width - mask_width
                max_offset_y = height - mask_height
                
                if max_offset_x > 0 and max_offset_y > 0:
                    # 무작위 오프셋 생성
                    random_offset_x = np.random.randint(0, max_offset_x)
                    random_offset_y = np.random.randint(0, max_offset_y)
                    
                    # 이동된 마스킹 영역 생성
                    shifted_mask = np.zeros_like(mask)
                    shifted_points = mask_points - np.array([min_x, min_y]) + np.array([random_offset_x, random_offset_y])
                    shifted_points = np.clip(shifted_points, 0, [width-1, height-1])
                    cv2.fillPoly(shifted_mask, [shifted_points.astype(np.int32)], 255)
                    
                    # 원본 영역을 검은색으로, 이동된 영역을 원본으로 복사
                    img[mask > 0] = [0, 0, 0]
                    img[shifted_mask > 0] = image[shifted_mask > 0]
    
    # Extra Perturbations 적용 (활성화된 경우에만)
    if extra_perturbations and extra_perturbations.get('enabled', False):
        # 밝기 조정
        if 'brightness' in extra_perturbations:
            brightness_value = extra_perturbations['brightness']
            if brightness_value != 0:
                factor = 1.0 + (brightness_value / 100.0)  # -1.0 ~ 2.0
                img = cv2.convertScaleAbs(img, alpha=factor, beta=0)
        
        # 회전
        if 'rotation' in extra_perturbations:
            rotation_value = extra_perturbations['rotation']
            if rotation_value != 0:
                center = (width // 2, height // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, rotation_value, 1.0)
                img = cv2.warpAffine(img, rotation_matrix, (width, height))
        
        # 크기 조정
        if 'scale' in extra_perturbations:
            scale_value = extra_perturbations['scale']
            if scale_value != 1.0:
                new_height = int(height * scale_value)
                new_width = int(width * scale_value)
                img = cv2.resize(img, (new_width, new_height))
                # 원래 크기로 다시 리사이즈
                img = cv2.resize(img, (width, height))
    
    return img

@app.route('/perturbation/result')
def perturbation_result():
    results = flask_session.get('perturbation_results', {})
    if not results or not results.get('has_results'):
        return redirect(url_for('perturbation_vis'))  # 결과가 없으면 업로드 페이지로 리다이렉트
    
    # 프로젝트 이름과 타임스탬프로 결과 파일에서 데이터 로드
    project_name = results.get('project_name')
    timestamp = results.get('timestamp')
    
    if not project_name or not timestamp:
        return redirect(url_for('perturbation_vis'))
    
    # 결과 데이터 파일에서 정보 로드
    paths = get_project_paths()
    result_data_file = os.path.join(paths['static_perturbation_results'], project_name, f'{timestamp}_results.json')
    
    if not os.path.exists(result_data_file):
        return redirect(url_for('perturbation_vis'))
    
    try:
        with open(result_data_file, 'r') as f:
            result_data = json.load(f)
        
        # 템플릿에 필요한 데이터만 전달
        return render_template('perturbation_result.html',
            images=[result_data],  # 단일 결과를 배열로 감싸기
            image_count=1,
            project_name=project_name
        )
    except Exception as e:
        print(f"Error loading result data: {str(e)}")
        return redirect(url_for('perturbation_vis'))

@app.route('/perturbation/re-edit-page')
def perturbation_re_edit_page():
    """재편집 전용 페이지"""
    paths = get_project_paths()
    models = [m for m in os.listdir(paths['models_dir'])]
    return render_template('perturbation_re_edit.html', models=models)

@app.route('/perturbation/re-edit')
def perturbation_re_edit():
    """재편집 전용 페이지"""
    return render_template('perturbation_re_edit.html')

@app.route('/api/perturbation/project/list', methods=['GET'])
def get_perturbation_project_list():
    paths = get_project_paths()
    base_dir = paths['static_perturbation_results']
    projects = []
    
    if os.path.exists(base_dir):
        for project_name in os.listdir(base_dir):
            project_dir = os.path.join(base_dir, project_name)
            if os.path.isdir(project_dir):
                # 프로젝트 디렉토리에서 original 이미지들 찾기
                original_images = []
                for file in os.listdir(project_dir):
                    if file.endswith('_original.jpg'):
                        original_images.append({
                            'filename': file,
                            'path': f'perturbation_results/{project_name}/{file}'
                        })
                
                if original_images:  # original 이미지가 있는 프로젝트만 포함
                    # 가장 최근 이미지 사용
                    latest_image = original_images[0]
                    
                    # 실제 파일 존재 여부 확인
                    actual_file_path = os.path.join(project_dir, latest_image['filename'])
                    if not os.path.exists(actual_file_path):
                        print(f"Warning: File does not exist: {actual_file_path}")
                        continue
                    
                    # 프로젝트 정보 구성 (기본값)
                    project_info = {
                        'name': project_name,
                        'original_image': latest_image['path'],
                        'original_detections': 0,
                        'perturbed_detections': 0,
                        'matched_objects': 0
                    }
                    
                    # 결과 파일이 있다면 더 자세한 정보 로드
                    result_files = [f for f in os.listdir(project_dir) if f.endswith('_original_inference.jpg')]
                    if result_files:
                        # 가장 최근 결과 파일 사용
                        latest_result = result_files[0]
                        base_name = latest_result.replace('_original_inference.jpg', '')
                        
                        # 결과 데이터 파일 확인 (JSON 형태로 저장된 경우)
                        result_data_file = os.path.join(project_dir, f'{base_name}_results.json')
                        if os.path.exists(result_data_file):
                            try:
                                with open(result_data_file, 'r') as f:
                                    result_data = json.load(f)
                                    if 'original_detections' in result_data:
                                        project_info['original_detections'] = len(result_data['original_detections'])
                                    if 'perturbed_detections' in result_data:
                                        project_info['perturbed_detections'] = len(result_data['perturbed_detections'])
                                    if 'matched_pairs' in result_data:
                                        project_info['matched_objects'] = len(result_data['matched_pairs']['matched_pairs'])
                            except:
                                pass
                    
                    projects.append(project_info)
    
    # print(f"Returning {len(projects)} projects")
    # for project in projects:
    #     print(f"Project: {project['name']}, Image: {project['original_image']}")
    
    return jsonify({'projects': projects})

@app.route('/api/perturbation/project/load', methods=['POST'])
def load_existing_perturbation_project():
    data = request.json
    project_name = data.get('project_name', '')
    
    if not project_name:
        return jsonify({'error': 'Project name is required'}), 400
    
    # 프로젝트 경로 가져오기
    paths = get_project_paths()
    project_dir = os.path.join(paths['static_perturbation_results'], project_name)
    if not os.path.exists(project_dir):
        return jsonify({'error': 'Project not found'}), 404
    
    # 프로젝트 디렉토리에서 결과 파일들 찾기
    result_files = [f for f in os.listdir(project_dir) if f.endswith('_original_inference.jpg')]
    if not result_files:
        return jsonify({'error': 'No results found for this project'}), 404
    
    # 가장 최근 결과 파일 사용
    latest_result = result_files[0]
    base_name = latest_result.replace('_original_inference.jpg', '')
    
    # 관련 파일들 확인
    original_file = f'{base_name}_original.jpg'
    perturbed_file = f'{base_name}_perturbed.jpg'
    original_inference_file = f'{base_name}_original_inference.jpg'
    perturbed_inference_file = f'{base_name}_perturbed_inference.jpg'
    
    # 결과 구성
    result_data = {
        'project_name': project_name,  # 프로젝트 이름 추가
        'original_image': f'perturbation_results/{project_name}/{original_file}',
        'perturbed_image': f'perturbation_results/{project_name}/{perturbed_file}',
        'original_inference_image': f'perturbation_results/{project_name}/{original_inference_file}',
        'perturbed_inference_image': f'perturbation_results/{project_name}/{perturbed_inference_file}',
        'original_detections': [],
        'perturbed_detections': [],
        'matched_pairs': {
            'matched_pairs': [],
            'unmatched_original_indices': [],
            'unmatched_perturbed_indices': [],
            'total_original': 0,
            'total_perturbed': 0
        },
        'perturbation_settings': {
            'masking_type': 'unknown',
            'mask_points_count': 0,
            'extra_perturbations': {'enabled': False}
        },
        'file_name': base_name
    }
    
    # 결과 데이터 파일 확인
    result_data_file = os.path.join(project_dir, f'{base_name}_results.json')
    if os.path.exists(result_data_file):
        try:
            with open(result_data_file, 'r') as f:
                saved_data = json.load(f)
                # 저장된 데이터로 업데이트
                result_data.update(saved_data)
        except:
            pass
    
            # 세션에 결과 저장 (최소한의 데이터만)
        flask_session['perturbation_results'] = {
            'project_name': project_name,
            'timestamp': base_name,
            'has_results': True
        }
    
    return jsonify({
        'success': True,
        'redirect_url': url_for('perturbation_result')
    })

@app.route('/api/project/delete', methods=['POST'])
def delete_cam_project():
    try:
        data = request.json
        if not data:
            raise ValueError("No JSON data received")
            
        project_name = data.get('project_name')
        if not project_name:
            raise ValueError("No project_name provided in request")
            
        print(f"Attempting to delete CAM project: {project_name}")
        
        # 프로젝트 디렉토리 경로 확인
        paths = get_project_paths()
        project_dir = os.path.join(paths['static_cam_results'], project_name)
        
        if os.path.exists(project_dir):
            shutil.rmtree(project_dir)
            print(f"Deleted CAM project directory: {project_dir}")
            return jsonify({'message': f'Project "{project_name}" deleted successfully'})
        else:
            return jsonify({'message': f'Project "{project_name}" not found'}), 404
            
    except ValueError as e:
        error_msg = f"Invalid request: {str(e)}"
        print(error_msg)
        return jsonify({'error': error_msg}), 400
    except Exception as e:
        error_msg = f"Error deleting CAM project: {str(e)}"
        print(error_msg)
        return jsonify({'error': error_msg}), 500

@app.route('/api/perturbation/project/delete', methods=['POST'])
def delete_perturbation_project():
    try:
        data = request.json
        if not data:
            raise ValueError("No JSON data received")
            
        project_name = data.get('project_name')
        if not project_name:
            raise ValueError("No project_name provided in request")
            
        print(f"Attempting to delete perturbation project: {project_name}")
        
        # 프로젝트 디렉토리 경로 확인
        paths = get_project_paths()
        project_dir = os.path.join(paths['static_perturbation_results'], project_name)
        
        if os.path.exists(project_dir):
            shutil.rmtree(project_dir)
            print(f"Deleted perturbation project directory: {project_dir}")
            return jsonify({'message': f'Project "{project_name}" deleted successfully'})
        else:
            return jsonify({'message': f'Project "{project_name}" not found'}), 404
            
    except ValueError as e:
        error_msg = f"Invalid request: {str(e)}"
        print(error_msg)
        return jsonify({'error': error_msg}), 400
    except Exception as e:
        error_msg = f"Error deleting perturbation project: {str(e)}"
        print(error_msg)
        return jsonify({'error': error_msg}), 500

@app.route('/api/perturbation/check-re-edit-data')
def check_re_edit_data():
    """재편집 데이터가 있는지 확인하는 API"""
    try:
        re_edit_data = flask_session.get('re_edit_data')
        
        if re_edit_data:
            return jsonify({
                'has_re_edit_data': True,
                're_edit_data': re_edit_data
            })
        else:
            return jsonify({
                'has_re_edit_data': False
            })
            
    except Exception as e:
        print(f"Error in check_re_edit_data: {str(e)}")
        return jsonify({
            'has_re_edit_data': False,
            'error': str(e)
        }), 500

@app.route('/api/perturbation/re-edit', methods=['POST'])
def re_edit_perturbation():
    """현재 보고 있는 perturbation 프로젝트를 재편집하기 위한 API"""
    try:
        data = request.json
        project_name = data.get('project_name')
        
        if not project_name:
            return jsonify({'error': 'Project name is required'}), 400
        
        # 현재 세션에서 perturbation_results 가져오기
        perturbation_results = flask_session.get('perturbation_results')
        print(f"Current session perturbation_results: {perturbation_results}")
        
        if not perturbation_results or not perturbation_results.get('has_results'):
            print(f"No perturbation results found in session")
            return jsonify({'error': 'No current perturbation results found'}), 400
        
        # 현재 세션에서 프로젝트 정보 가져오기
        session_project_name = perturbation_results.get('project_name')
        session_timestamp = perturbation_results.get('timestamp')
        print(f"Session project: {session_project_name}, timestamp: {session_timestamp}")
        
        # 프로젝트 이름이 일치하는지 확인
        if session_project_name != project_name:
            print(f"Project name mismatch: session={session_project_name}, requested={project_name}")
            return jsonify({'error': f'Project name mismatch. Current: {session_project_name}, Requested: {project_name}'}), 400
        
        # 재편집을 위한 데이터 구성 (최소한의 데이터만)
        re_edit_data = {
            'project_name': project_name,
            'timestamp': session_timestamp,
            'has_settings': True
        }
        
        # 세션에 재편집 데이터 저장 (최소화)
        flask_session['re_edit_data'] = re_edit_data
        
        print(f"Re-edit data prepared for project: {project_name}")
        print(f"Re-edit data: {re_edit_data}")
        
        return jsonify({
            'success': True,
            'redirect_url': url_for('perturbation_vis')
        })
        
    except Exception as e:
        print(f"Error in re_edit_perturbation: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/perturbation/get-re-edit-data', methods=['POST'])
def get_re_edit_data():
    """재편집에 필요한 데이터를 파일에서 직접 로드하는 API"""
    try:
        data = request.json
        project_name = data.get('project_name')
        timestamp = data.get('timestamp')
        
        if not project_name or not timestamp:
            return jsonify({'error': 'Project name and timestamp are required'}), 400
        
        # 프로젝트 경로 가져오기
        paths = get_project_paths()
        project_dir = os.path.join(paths['static_perturbation_results'], project_name)
        
        if not os.path.exists(project_dir):
            return jsonify({'error': 'Project not found'}), 404
        
        # 결과 데이터 파일에서 정보 로드
        result_data_file = os.path.join(project_dir, f'{timestamp}_results.json')
        if not os.path.exists(result_data_file):
            return jsonify({'error': 'Results file not found'}), 404
        
        with open(result_data_file, 'r') as f:
            result_data = json.load(f)
        
        # 재편집에 필요한 데이터만 반환
        re_edit_data = {
            'project_name': project_name,
            'original_image': result_data.get('original_image'),
            'perturbation_settings': result_data.get('perturbation_settings'),
            'file_name': timestamp  # 타임스탬프를 file_name으로 추가
        }
        
        return jsonify({
            'success': True,
            're_edit_data': re_edit_data
        })
        
    except Exception as e:
        print(f"Error in get_re_edit_data: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/perturbation_compare')
def perturbation_compare_page():
    """교란 분석 비교 페이지"""
    return render_template('perturbation_compare.html')


@app.route('/api/perturbation/comparisons/<project_name>')
def get_perturbation_comparisons(project_name):
    """특정 프로젝트의 re-edit 비교 목록을 반환 (간단한 폴더 스캔)"""
    try:
        # 프로젝트 경로 가져오기
        paths = get_project_paths()
        project_dir = os.path.join(paths['static_perturbation_results'], project_name)
        
        if not os.path.exists(project_dir):
            return jsonify({'error': 'Project not found'}), 404
        
        comparisons = []
        
        # ca 폴더들 찾기 (간단한 폴더 스캔)
        for item in os.listdir(project_dir):
            if item.startswith('ca') and os.path.isdir(os.path.join(project_dir, item)):
                ca_dir = os.path.join(project_dir, item)
                
                # ca 폴더 내의 이미지 파일들 찾기
                image_files = []
                for file in os.listdir(ca_dir):
                    if file.endswith('_re_edit_perturbed.jpg'):
                        image_files.append(file)
                
                if image_files:
                    # 가장 최근 이미지 파일 사용
                    latest_image = max(image_files)
                    # 파일명에서 timestamp 추출
                    timestamp = latest_image.replace('_re_edit_perturbed.jpg', '')
                    
                    comparisons.append({
                        'ca_directory': item,
                        'comparison_timestamp': timestamp,
                        'image_file': latest_image,
                        'created_time': os.path.getctime(os.path.join(ca_dir, latest_image))
                    })
        
        # 생성 시간 순으로 정렬 (최신순)
        comparisons.sort(key=lambda x: x['created_time'], reverse=True)
        
        return jsonify({
            'success': True,
            'comparisons': comparisons
        })
        
    except Exception as e:
        print(f"Error in get_perturbation_comparisons: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/perturbation/result/<project_name>/<timestamp>')
def get_perturbation_result_data(project_name, timestamp):
    """특정 프로젝트의 교란 분석 결과 데이터를 반환"""
    try:
        # ca 파라미터 확인 (비교 데이터인지 확인)
        ca_directory = request.args.get('ca')
        
        # 프로젝트 경로 가져오기
        paths = get_project_paths()
        print(f"Static perturbation results path: {paths['static_perturbation_results']}")
        
        if ca_directory:
            # 비교 데이터 로드 (ca 디렉터리에서)
            result_json_path = os.path.join(paths['static_perturbation_results'], project_name, ca_directory, f'{timestamp}_comparison_results.json')
            print(f"Loading comparison data from: {result_json_path}")
        else:
            # 일반 결과 데이터 로드
            result_json_path = os.path.join(paths['static_perturbation_results'], project_name, f'{timestamp}_results.json')
            print(f"Loading original data from: {result_json_path}")
        
        # 디렉터리 존재 여부 확인
        dir_path = os.path.dirname(result_json_path)
        print(f"Directory path: {dir_path}")
        print(f"Directory exists: {os.path.exists(dir_path)}")
        
        if os.path.exists(dir_path):
            files_in_dir = os.listdir(dir_path)
            print(f"Files in directory: {files_in_dir}")
        
        if not os.path.exists(result_json_path):
            return jsonify({'error': f'Result file not found: {result_json_path}'}), 404
        
        # JSON 파일 읽기
        with open(result_json_path, 'r') as f:
            result_data = json.load(f)
        
        return jsonify({
            'success': True,
            'data': result_data
        })
        
    except Exception as e:
        print(f"Error in get_perturbation_result_data: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/perturbation/compare', methods=['POST'])
def run_perturbation_comparison():
    """re-edit된 교란을 적용하고 기존 프로젝트와 비교 분석 수행"""
    try:
        # 폼 데이터에서 값들 가져오기
        project_name = request.form.get('projectName')
        original_timestamp = request.form.get('originalTimestamp')
        
        if not project_name or not original_timestamp:
            return jsonify({'error': 'Missing required parameters'}), 400
        
        # 모델 로드 확인
        if current_app.model_storage is None:
            return jsonify({'error': 'Model not loaded. Please load the model first.'}), 400
        
        model = current_app.model_storage
        
        # 프로젝트 경로 설정
        paths = get_project_paths()
        project_dir = os.path.join(paths['static_perturbation_results'], project_name)
        
        # ca 디렉터리 생성 (자동 증가)
        ca_dir = create_comparison_directory(project_dir)
        
        # 원본 프로젝트 데이터 로드
        original_result_path = os.path.join(project_dir, f'{original_timestamp}_results.json')
        print(f"Looking for original result file: {original_result_path}")
        print(f"Project directory: {project_dir}")
        print(f"Original timestamp: {original_timestamp}")
        
        if not os.path.exists(original_result_path):
            # 디렉터리 내용 확인
            if os.path.exists(project_dir):
                files = os.listdir(project_dir)
                print(f"Files in project directory: {files}")
            else:
                print(f"Project directory does not exist: {project_dir}")
            return jsonify({'error': f'Original project data not found: {original_result_path}'}), 404
        
        with open(original_result_path, 'r') as f:
            original_data = json.load(f)
        
        # re-edit 설정을 폼 데이터에서 직접 가져오기
        masking_type = request.form.get('masking_type')
        mask_data_str = request.form.get('mask_data')
        canvas_width = int(request.form.get('canvas_width', 1000))
        canvas_height = int(request.form.get('canvas_height', 800))
        
        # 마스크 데이터 파싱
        try:
            mask_data = json.loads(mask_data_str) if mask_data_str else []
        except:
            mask_data = []
        
        # Extra Perturbations 설정
        extra_perturbations_enabled = request.form.get('extra_perturbations_enabled') == 'true'
        extra_perturbations = {'enabled': extra_perturbations_enabled}
        
        if extra_perturbations_enabled:
            extra_perturbations.update({
                'brightness': int(request.form.get('brightness', 0)),
                'rotation': int(request.form.get('rotation', 0)),
                'scale': float(request.form.get('scale', 1.0))
            })
        
        # 원본 이미지 로드 (Flask static 폴더 경로 사용)
        original_image_path = os.path.join(current_app.static_folder, original_data['original_image'])
        print(f"Looking for original image: {original_image_path}")
        print(f"Original data keys: {list(original_data.keys())}")
        print(f"Original image path from data: {original_data.get('original_image')}")
        
        if not os.path.exists(original_image_path):
            # static 디렉터리 확인
            static_dir = current_app.static_folder
            print(f"Flask static folder: {static_dir}")
            if os.path.exists(static_dir):
                print(f"Static directory exists: {static_dir}")
                # perturbation_results 디렉터리 확인
                perturbation_results_dir = os.path.join(static_dir, 'perturbation_results')
                if os.path.exists(perturbation_results_dir):
                    print(f"Perturbation results directory exists: {perturbation_results_dir}")
                    # 프로젝트 디렉터리 확인
                    if os.path.exists(project_dir):
                        project_files = os.listdir(project_dir)
                        print(f"Files in project directory: {project_files}")
                    else:
                        print(f"Project directory does not exist: {project_dir}")
                else:
                    print(f"Perturbation results directory does not exist: {perturbation_results_dir}")
            else:
                print(f"Static directory does not exist: {static_dir}")
            return jsonify({'error': f'Original image not found: {original_image_path}'}), 404
        
        original_img = cv2.imread(original_image_path)
        print(f"Image loaded successfully: {original_img is not None}")
        if original_img is not None:
            print(f"Image shape: {original_img.shape}")
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            original_height, original_width = original_img.shape[:2]
            print(f"Image dimensions: {original_width}x{original_height}")
        else:
            print(f"Failed to load image from: {original_image_path}")
            return jsonify({'error': f'Failed to load original image: {original_image_path}'}), 404
        
        # re-edit 설정 구성 (이미지 로드 후)
        updated_settings = {
            'masking_type': masking_type,
            'mask_data': mask_data,
            'mask_points_count': len(mask_data),
            'canvas_dimensions': {
                'width': canvas_width,
                'height': canvas_height
            },
            'image_dimensions': {
                'width': original_width,
                'height': original_height
            },
            'extra_perturbations': extra_perturbations,
            'model_info': {
                'model_name': getattr(model, 'ckpt_path', 'Unknown').split('/')[-1] if hasattr(model, 'ckpt_path') else 'Unknown',
                'use_gpu': str(model.device).startswith('cuda'),
                'device': str(model.device)
            }
        }
        
        print(f"Applying re-edit perturbation to original image...")
        print(f"Masking type: {masking_type}")
        print(f"Extra perturbations: {extra_perturbations}")
        
        # 마스크 데이터를 원본 이미지 크기로 스케일링
        print(f"Scaling mask data from canvas ({canvas_width}x{canvas_height}) to original image ({original_width}x{original_height})")
        scaled_mask_data = []
        for point in mask_data:
            # 캔버스 좌표를 원본 이미지 좌표로 변환
            scaled_x = int((point['x'] / canvas_width) * original_width)
            scaled_y = int((point['y'] / canvas_height) * original_height)
            scaled_mask_data.append({'x': scaled_x, 'y': scaled_y})
        
        print(f"Scaled mask data points: {len(scaled_mask_data)}")
        if len(scaled_mask_data) > 0:
            print(f"Sample scaled point: {scaled_mask_data[0]}")
        
        # 교란 적용 (스케일링된 마스크 데이터 사용)
        perturbed_img = apply_perturbation(original_img, scaled_mask_data, masking_type, extra_perturbations)
        
        # 모델 입력 크기로 리사이즈 (640x640)
        model_input_img = cv2.resize(perturbed_img, (640, 640))
        
        # 모델 추론 수행 (re-edit perturbed image만)
        print("Performing inference on re-edit perturbed image...")
        
        # re-edit 교란된 이미지 추론
        re_edit_results = model(model_input_img)
        re_edit_boxes, re_edit_colors, re_edit_names = parse_detections(re_edit_results)
        
        # 기존 perturbed image 데이터 재사용 (추론 불필요)
        original_detections = original_data['perturbed_detections']  # 기존 교란된 이미지의 검출 결과
        
        # re-edit 교란된 이미지 추론 결과 저장
        re_edit_detections = []
        for i, result in enumerate(re_edit_results):
            if len(result.boxes) > 0:
                for j in range(len(result.boxes)):
                    detection = {
                        'class_name': result.names[int(result.boxes.cls[j])],
                        'confidence': float(result.boxes.conf[j]),
                        'bbox': result.boxes.xyxy[j].cpu().numpy().tolist(),
                        'class_id': int(result.boxes.cls[j]),
                        'detection_id': f"re_edit_{i}_{j}"
                    }
                    re_edit_detections.append(detection)
        
        # 객체 매칭 (기존 perturbed vs re-edit perturbed)
        matched_pairs = match_objects(original_detections, re_edit_detections)
        
        # 추론 결과 이미지 생성
        # 기존 perturbed inference image는 기존 데이터에서 가져오기
        original_inference_path = os.path.join(current_app.static_folder, original_data['perturbed_inference_image'])
        print(f"Loading original inference image: {original_inference_path}")
        original_inference_img = cv2.imread(original_inference_path)
        
        if original_inference_img is not None:
            print(f"Original inference image loaded successfully, shape: {original_inference_img.shape}")
            original_inference_img = cv2.cvtColor(original_inference_img, cv2.COLOR_BGR2RGB)
        else:
            print(f"Failed to load original inference image from: {original_inference_path}")
            return jsonify({'error': f'Failed to load original inference image: {original_inference_path}'}), 404
        
        # re-edit perturbed inference image 생성
        re_edit_inference_img = draw_detections(re_edit_boxes, re_edit_colors, re_edit_names, model_input_img.copy())
        
        # 이미지 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # re-edit 교란된 이미지만 저장 (기존 이미지는 경로 참조)
        re_edit_perturbed_path = os.path.join(ca_dir, f'{timestamp}_re_edit_perturbed.jpg')
        cv2.imwrite(re_edit_perturbed_path, cv2.cvtColor(perturbed_img, cv2.COLOR_RGB2BGR))
        
        # re-edit 추론 결과 이미지만 저장 (기존 추론 결과는 경로 참조)
        re_edit_inference_save_path = os.path.join(ca_dir, f'{timestamp}_re_edit_perturbed_inference.jpg')
        cv2.imwrite(re_edit_inference_save_path, cv2.cvtColor(re_edit_inference_img, cv2.COLOR_RGB2BGR))
        
        # 비교 분석 수행 (기존 perturbed vs re-edit perturbed)
        comparison_analysis = perform_comparison_analysis(original_data, {
            'original_detections': original_detections,  # 기존 perturbed detections
            'perturbed_detections': re_edit_detections,  # re-edit perturbed detections
            'matched_pairs': matched_pairs
        }, updated_settings)
        
        # 비교 결과 데이터 구성
        comparison_data = {
            'project_name': project_name,
            'original_timestamp': original_timestamp,
            'comparison_timestamp': timestamp,
            'ca_directory': os.path.basename(ca_dir),
            # 기존 이미지들은 상위 디렉터리 경로 참조
            'original_perturbed_image': original_data['perturbed_image'],  # 기존 교란된 이미지 경로
            'original_perturbed_inference_image': original_data['perturbed_inference_image'],  # 기존 추론 결과 경로
            # 새로 생성된 이미지들은 ca 디렉터리 경로
            're_edit_perturbed_image': f'perturbation_results/{project_name}/{os.path.basename(ca_dir)}/{timestamp}_re_edit_perturbed.jpg',
            're_edit_perturbed_inference_image': f'perturbation_results/{project_name}/{os.path.basename(ca_dir)}/{timestamp}_re_edit_perturbed_inference.jpg',
            'original_perturbed_detections': original_detections,  # 기존 perturbed detections
            're_edit_perturbed_detections': re_edit_detections,    # re-edit perturbed detections
            'matched_pairs': matched_pairs,
            'comparison_analysis': comparison_analysis,
            'perturbation_settings': updated_settings,
            'file_name': timestamp
        }
        
        # 비교 결과 JSON 파일 저장
        comparison_json_path = os.path.join(ca_dir, f'{timestamp}_comparison_results.json')
        with open(comparison_json_path, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        print(f"Comparison analysis completed and saved to: {ca_dir}")
        
        return jsonify({
            'success': True,
            'ca_directory': os.path.basename(ca_dir),
            'comparison_timestamp': timestamp,
            'redirect_url': f'/perturbation_compare?project={project_name}&original={original_timestamp}&comparison={timestamp}&ca={os.path.basename(ca_dir)}'
        })
        
    except Exception as e:
        print(f"Error in run_perturbation_comparison: {str(e)}")
        return jsonify({'error': str(e)}), 500


def create_comparison_directory(project_dir):
    """ca{숫자} 형태의 비교 디렉터리 생성"""
    ca_counter = 1
    while True:
        ca_dir_name = f"ca{ca_counter:02d}"  # ca01, ca02, ca03, ...
        ca_dir_path = os.path.join(project_dir, ca_dir_name)
        
        if not os.path.exists(ca_dir_path):
            os.makedirs(ca_dir_path)
            print(f"Created comparison directory: {ca_dir_path}")
            return ca_dir_path
        
        ca_counter += 1


def perform_comparison_analysis(original_data, comparison_results, updated_settings):
    """기존 프로젝트와 업데이트된 프로젝트, 그리고 새로운 비교 결과를 분석"""
    
    # 기본 통계
    original_objects = len(original_data['perturbed_detections'])  # 기존 perturbed detections
    updated_objects = len(comparison_results['perturbed_detections'])  # re-edit perturbed detections
    comparison_objects = len(comparison_results['perturbed_detections'])  # re-edit perturbed detections
    
    # 클래스별 분석
    def get_class_distribution(detections):
        class_dist = {}
        for detection in detections:
            class_name = detection['class_name']
            if class_name not in class_dist:
                class_dist[class_name] = 0
            class_dist[class_name] += 1
        return class_dist
    
    original_classes = get_class_distribution(original_data['perturbed_detections'])  # 기존 perturbed detections
    updated_classes = get_class_distribution(comparison_results['perturbed_detections'])  # re-edit perturbed detections
    comparison_classes = get_class_distribution(comparison_results['perturbed_detections'])  # re-edit perturbed detections
    
    # 컨피던스 분석
    def get_confidence_stats(detections):
        if not detections:
            return {'mean': 0, 'min': 0, 'max': 0}
        confidences = [d['confidence'] for d in detections]
        return {
            'mean': sum(confidences) / len(confidences),
            'min': min(confidences),
            'max': max(confidences)
        }
    
    original_conf = get_confidence_stats(original_data['perturbed_detections'])  # 기존 perturbed detections
    updated_conf = get_confidence_stats(comparison_results['perturbed_detections'])  # re-edit perturbed detections
    comparison_conf = get_confidence_stats(comparison_results['perturbed_detections'])  # re-edit perturbed detections
    
    # 교란 설정 변화
    original_settings = original_data['perturbation_settings']
    # updated_settings는 이미 매개변수로 받음
    
    perturbation_changes = {
        'masking_type_changed': original_settings['masking_type'] != updated_settings['masking_type'],
        'masking_type_original': original_settings['masking_type'],
        'masking_type_updated': updated_settings['masking_type'],
        'extra_perturbations_changed': original_settings.get('extra_perturbations', {}) != updated_settings.get('extra_perturbations', {}),
        'extra_perturbations_original': original_settings.get('extra_perturbations', {}),
        'extra_perturbations_updated': updated_settings.get('extra_perturbations', {})
    }
    
    # 분석 결과 구성
    analysis = {
        'object_count_analysis': {
            'original': original_objects,
            'updated': updated_objects,
            'comparison': comparison_objects,
            'original_to_updated_change': updated_objects - original_objects,
            'original_to_comparison_change': comparison_objects - original_objects
        },
        'class_distribution_analysis': {
            'original': original_classes,
            'updated': updated_classes,
            'comparison': comparison_classes
        },
        'confidence_analysis': {
            'original': original_conf,
            'updated': updated_conf,
            'comparison': comparison_conf,
            'original_to_updated_change': updated_conf['mean'] - original_conf['mean'],
            'original_to_comparison_change': comparison_conf['mean'] - original_conf['mean']
        },
        'perturbation_changes': perturbation_changes,
        'detection_rate_analysis': {
            'original_rate': 100.0 if original_objects > 0 else 0,
            'updated_rate': 100.0 if updated_objects > 0 else 0,
            'comparison_rate': 100.0 if comparison_objects > 0 else 0
        }
    }
    
    return analysis


# @socketio.on('check_fiftyone_ready')
# def handle_check_fiftyone_ready():
#     # FiftyOne 세션이 준비되었는지 확인하는 로직을 추가하세요.
#     # 예를 들어, 세션이 이미 실행 중이라면 'ready' 상태를 emit합니다.
#     if fiftyone_thread and fiftyone_thread.is_alive():
#         emit('fiftyone_ready', {'status': 'ready'})
#     else:
#         emit('fiftyone_ready', {'status': 'not_ready'})

if __name__ == "__main__":
    # argparse 처리 (직접 실행할 때만)
    parser = create_parser()
    args = parser.parse_args()
    
    # config에서 기본값 가져오기
    try:
        from config import config
        flask_port = config.flask_port
        fiftyone_port = args.port if args.port is not None else config.fiftyone_port
        
        # FiftyOne 포트 설정
        config.set_fiftyone_port(fiftyone_port)
        print(f"✅ Using config values - Flask: {flask_port}, FiftyOne: {fiftyone_port}")
    except ImportError:
        # config가 없는 경우 기본값 사용
        flask_port = 5555
        fiftyone_port = args.port if args.port is not None else 8159
        print(f"⚠️  Config not available, using fallback values - Flask: {flask_port}, FiftyOne: {fiftyone_port}")
    
    init_app(app)
    socketio.run(app, port=flask_port, debug=False)