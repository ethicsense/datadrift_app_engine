import torch
import clip
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
from sklearn.metrics.pairwise import cosine_similarity
from yellowbrick.cluster import KElbowVisualizer

from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import io, filters, img_as_float
import json

import numpy as np
import hashlib
import random

import argparse
import os
import sys
import subprocess
from datetime import datetime
import time

# 캐시 매니저 import
from cache_utils.cache_manager import get_cache_manager, save_analysis_data, get_cached_analysis_data

# 데이터 유틸리티 import
from data_utils import run_attribute_analysis


def format_time(seconds):
    """초를 HH:MM:SS 형식으로 변환합니다."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remaining_seconds = seconds % 60
    
    return f"{hours:02d}:{minutes:02d}:{remaining_seconds:05.2f}"

def calculate_file_hash(file_path):
    """파일의 MD5 해시를 계산합니다."""
    import hashlib
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()


def calculate_cache_hash(cache_data):
    """캐시 데이터의 해시를 계산합니다."""
    import hashlib
    import json
    hasher = hashlib.md5()
    
    # 파일명과 해시만 사용하여 캐시 해시 계산
    file_hashes = {}
    for file_name, data in cache_data.items():
        if 'hash' in data:
            file_hashes[file_name] = data['hash']
    
    # 정렬된 딕셔너리를 JSON으로 변환하여 해시 계산
    sorted_data = json.dumps(file_hashes, sort_keys=True)
    hasher.update(sorted_data.encode())
    return hasher.hexdigest()


def calculate_mmd_batch(X, Y, batch_size=1000, gamma=None):
    batch_size = min(batch_size, min(len(X), len(Y)))
    
    if len(X) > batch_size:
        idx = np.random.choice(len(X), batch_size, replace=False)
        X = X[idx]
    if len(Y) > batch_size:
        idx = np.random.choice(len(Y), batch_size, replace=False)
        Y = Y[idx]
    
    if gamma is None:
        X_combined = np.vstack([X[:batch_size], Y[:batch_size]])
        pairwise_dists = np.sum((X_combined[:, np.newaxis] - X_combined) ** 2, axis=2)
        gamma = 1.0 / np.median(pairwise_dists[pairwise_dists > 0])

    def kernel(X, Y):
        X_norm = np.sum(X ** 2, axis=1)
        Y_norm = np.sum(Y ** 2, axis=1)
        distances = X_norm[:, np.newaxis] + Y_norm - 2 * np.dot(X, Y.T)
        return np.exp(-gamma * distances)

    K_XX = kernel(X, X)
    K_YY = kernel(Y, Y)
    K_XY = kernel(X, Y)

    n_X = X.shape[0]
    n_Y = Y.shape[0]

    mmd = (np.sum(K_XX) - np.trace(K_XX)) / (n_X * (n_X - 1))
    mmd += (np.sum(K_YY) - np.trace(K_YY)) / (n_Y * (n_Y - 1))
    mmd -= 2 * np.mean(K_XY)

    return np.sqrt(max(mmd, 0))


def run_attribute_analysis_wrapper(directories, formats):
    """
    속성 분석을 수행하고 캐시에 저장합니다.
    디렉토리 탐색하면서 파일을 바로 처리합니다.
    
    Returns:
        dict: 각 디렉토리별 처리된 파일 수 정보
    """
    print("Starting attribute analysis...")
    
    # 분석 모듈 한 번만 로드
    from data_utils import AttributeAnalyzer
    analyzer = AttributeAnalyzer()
    
    processing_stats = {}
    
    for directory in directories:
        if not os.path.exists(directory):
            print(f"Error: Directory {directory} does not exist.")
            continue
        
        # 캐시 매니저 사용
        cache_manager = get_cache_manager(directory)
        
        # 기존 캐시 로드 (새로운 키 값으로 시도, 없으면 기존 키 값 사용)
        existing_cache = get_cached_analysis_data(directory, "attribute_analysis") or {}
        print(f"\nAnalyzing directory: {directory}")
        
        new_cache = {}
        total_files = 0
        processed_files = 0
        skipped_files = 0
        
        # 디렉토리 탐색하면서 바로 처리
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(tuple(formats)):
                    file_path = os.path.join(root, file)
                    total_files += 1
                    
                    # 캐시 확인
                    need_analysis = False
                    if file not in existing_cache:
                        need_analysis = True
                    else:
                        # 해시 확인
                        current_hash = calculate_file_hash(file_path)
                        if existing_cache[file]['hash'] != current_hash:
                            need_analysis = True
                    
                    if need_analysis:
                        print(f"Analyzing {file}...")
                        result = analyzer.analyze_image_attributes(file_path)
                        
                        if result:
                            new_cache[file] = result
                            processed_files += 1
                            print(f"Processed {file}")
                        else:
                            print(f"Failed to process {file}")
                    else:
                        skipped_files += 1
                        print(f"Skipping unchanged file: {file}")
        
        print(f"Total files found: {total_files}")
        print(f"Files processed: {processed_files}")
        print(f"Files skipped (cached): {skipped_files}")
        print(f"Files in cache: {len(existing_cache)}")
        
        if not new_cache:
            print("All files are up to date in cache.")
        
        # 기존 캐시와 새 캐시 병합
        existing_cache.update(new_cache)
        
        # 캐시에 저장
        save_analysis_data(directory, existing_cache, "attribute_analysis")
        
        print(f"\nSaved attribute analysis cache to {directory}")
        print(f"Cache location: {cache_manager.cache_dir}")
        print(f"Total files in cache: {len(existing_cache)}")
        print(f"New files processed: {len(new_cache)}")
        
        # 처리 통계 저장
        processing_stats[directory] = {
            'total_files': total_files,
            'processed_files': processed_files,
            'skipped_files': skipped_files
        }
    
    return processing_stats


def run_drift_analysis(directories, formats, model=None, device=None, n_clusters=None, method='kmeans'):
    """
    임베딩 추출과 클러스터링 분석을 수행하고 캐시에 저장합니다.
    디렉토리 탐색하면서 파일을 바로 처리합니다.

    methods = ['kmeans', 'dbscan', 'hierarchical']
    
    Returns:
        dict: 각 디렉토리별 처리된 파일 수 정보
    """
    print("Starting drift analysis (embedding extraction and clustering)...")
    
    # 분석 모듈 한 번만 로드
    from data_utils import EmbeddingManager
    manager = EmbeddingManager(device)
    manager.load_model("ViT-B/16")
    
    processing_stats = {}
    
    for directory in directories:
        if not os.path.exists(directory):
            print(f"Error: Directory {directory} does not exist.")
            continue
        
        # 캐시 매니저 사용
        cache_manager = get_cache_manager(directory)
        
        # 기존 캐시 로드 (새로운 키 값으로 시도, 없으면 기존 키 값 사용)
        existing_cache = get_cached_analysis_data(directory, "embedding_analysis") or {}
        print(f"\nExtracting embeddings for directory: {directory}")
        
        new_cache = {}
        total_files = 0
        processed_files = 0
        skipped_files = 0
        
        # 디렉토리 탐색하면서 바로 처리
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(tuple(formats)):
                    file_path = os.path.join(root, file)
                    total_files += 1
                    
                    # 캐시 확인
                    need_analysis = False
                    if file not in existing_cache:
                        need_analysis = True
                    else:
                        # 해시 확인
                        current_hash = calculate_file_hash(file_path)
                        if existing_cache[file]['hash'] != current_hash:
                            need_analysis = True
                    
                    if need_analysis:
                        print(f"Extracting embedding for {file}...")
                        result = manager.extract_embedding(file_path)
                        
                        if result:
                            new_cache[file] = result
                            processed_files += 1
                            print(f"Processed {file}")
                        else:
                            print(f"Failed to process {file}")
                    else:
                        skipped_files += 1
                        print(f"Skipping unchanged file: {file}")
        
        print(f"Total files found: {total_files}")
        print(f"Files processed: {processed_files}")
        print(f"Files skipped (cached): {skipped_files}")
        print(f"Files in cache: {len(existing_cache)}")
        
        if not new_cache:
            print("All files are up to date in cache.")
        else:
            # 기존 캐시와 새 캐시 병합
            existing_cache.update(new_cache)
            
            # 캐시에 저장
            save_analysis_data(directory, existing_cache, "embedding_analysis")
            
            print(f"\nSaved embedding extraction cache to {directory}")
            print(f"Cache location: {cache_manager.cache_dir}")
            print(f"Total files in cache: {len(existing_cache)}")
            print(f"New files processed: {len(new_cache)}")
        
        # 클러스터링 분석 수행
        print(f"\nStarting clustering analysis for {directory}")
        
        # 기존 클러스터링 캐시 로드
        existing_clustering_cache = get_cached_analysis_data(directory, "clustering_analysis") or {}
        
        # 클러스터링 재수행 여부 확인
        # 임베딩 데이터의 해시를 기반으로 클러스터링 캐시의 유효성 검사
        embedding_hash = calculate_cache_hash(existing_cache)
        clustering_hash = existing_clustering_cache.get('_embedding_hash', '')
        
        if embedding_hash == clustering_hash and existing_clustering_cache:
            print(f"Clustering cache is up to date for {directory}")
        else:
            print(f"Recomputing clustering for {directory}")
            
            # 임베딩 데이터 수집
            embeddings = []
            file_paths = []
            file_names = []
            
            for file_name, data in existing_cache.items():
                if 'embedding' in data:
                    embeddings.append(data['embedding'])
                    file_paths.append(data['path'])
                    file_names.append(file_name)
            
            if len(embeddings) < 2:
                print(f"Not enough data for clustering in {directory}")
                continue
            
            # 클러스터링 분석 수행
            clustering_result = manager.perform_clustering(embeddings, file_names, file_paths, n_clusters, method)
            
            if clustering_result:
                # 임베딩 해시를 클러스터링 결과에 추가
                clustering_result['clustering_results']['_embedding_hash'] = embedding_hash
                
                # 클러스터링 결과를 캐시에 저장
                save_analysis_data(directory, clustering_result['clustering_results'], "clustering_analysis")
                
                print(f"\nSaved clustering analysis cache to {directory}")
                print(f"Cache location: {cache_manager.cache_dir}")
                print(f"Method: {clustering_result['method']}")
                print(f"Number of clusters: {clustering_result['n_clusters']}")
                print(f"Total samples: {clustering_result['total_samples']}")
            else:
                print(f"Failed to perform clustering analysis for {directory}")
        
        # 처리 통계 저장
        processing_stats[directory] = {
            'total_files': total_files,
            'processed_files': processed_files,
            'skipped_files': skipped_files
        }
    
    return processing_stats


def run_xai_analysis(directories, formats, model_path=None, device=None):
    """
    XAI 분석을 수행하고 캐시에 저장합니다.
    디렉토리 탐색하면서 파일을 바로 처리합니다.
    
    Args:
        directories: 분석할 디렉토리 리스트
        formats: 분석할 이미지 포맷 리스트
        model_path: YOLO 모델 파일 경로
        device: 사용할 디바이스
        
    Returns:
        dict: 각 디렉토리별 처리된 파일 수 정보
    """
    if not model_path:
        print("❌ Error: Model path is required for XAI analysis")
        return {}
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found: {model_path}")
        return {}
    
    print("Starting XAI analysis...")
    
    # 분석 모듈 한 번만 로드
    from data_utils.xai_analyzer import XAIAnalyzer
    
    processing_stats = {}
    
    for directory in directories:
        if not os.path.exists(directory):
            print(f"Error: Directory {directory} does not exist.")
            continue
        
        # 캐시 매니저 사용
        cache_manager = get_cache_manager(directory)
        
        # 기존 캐시 로드
        existing_cache = get_cached_analysis_data(directory, "xai_analysis") or {}
        
        print(f"\nAnalyzing directory: {directory}")
        
        # XAI 분석기 초기화
        analyzer = XAIAnalyzer(device=device)
        analyzer.load_model(model_path)
        
        # 타겟 레이어 확인
        target_layers = analyzer.get_target_layers()
        if target_layers is None:
            print("❌ Error: Could not find target layers for CAM analysis")
            continue
        else:
            print(f"✅ Target layers ready: {len(target_layers)} layer(s)")
        
        new_cache = {}
        total_files = 0
        processed_files = 0
        skipped_files = 0
        failed_files = 0
        
        # 디렉토리 탐색하면서 바로 처리
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(tuple(formats)):
                    file_path = os.path.join(root, file)
                    total_files += 1
                    
                    # 캐시 확인
                    need_analysis = False
                    if file not in existing_cache:
                        need_analysis = True
                    else:
                        # 해시 확인
                        current_hash = analyzer.calculate_file_hash(file_path)
                        if existing_cache[file].get('hash') != current_hash:
                            need_analysis = True
                    
                    if need_analysis:
                        print(f"Processing {file}...")
                        
                        # 포괄적인 CAM 분석 수행
                        try:
                            comprehensive_result = analyzer.comprehensive_cam_analysis(
                                file_path, 
                                target_layers=target_layers,
                                save_visualizations=False,
                                output_dir=None
                            )
                            if comprehensive_result:
                                processed_files += 1
                                print(f"Comprehensive CAM analysis completed for {file}")
                                
                                # 해시 정보 추가
                                comprehensive_result['hash'] = analyzer.calculate_file_hash(file_path)
                                
                                new_cache[file] = comprehensive_result
                                
                                # 주요 결과 요약 출력
                                cam_stats = comprehensive_result['cam_stats']
                                print(f"  - CAM 통계: 평균={cam_stats['mean'][1]:.4f}, 최대={cam_stats['max'][1]:.4f}")
                                
                                components = comprehensive_result['components_analysis']
                                print(f"  - 연결된 영역: {components['num_components']}개, 활성 비율={components['active_ratio']:.2f}%")
                                
                                entropy = comprehensive_result['entropy_results']
                                if 'shannon' in entropy:
                                    print(f"  - Shannon 엔트로피: {entropy['shannon']:.4f}")
                                
                                if comprehensive_result['overlap_results']:
                                    overlap = comprehensive_result['overlap_results']
                                    print(f"  - IoU: {overlap['iou']:.4f}, 클래스: {overlap['largest_class_name']}")
                            else:
                                failed_files += 1
                                print(f"Failed to generate comprehensive CAM analysis for {file}")
                        except Exception as e:
                            failed_files += 1
                            print(f"Error in comprehensive CAM analysis for {file}: {e}")
                    else:
                        skipped_files += 1
                        print(f"Skipping unchanged file: {file}")
        
        print(f"Total files found: {total_files}")
        print(f"Files processed: {processed_files}")
        print(f"Files skipped (cached): {skipped_files}")
        print(f"Files failed: {failed_files}")
        print(f"Files in cache: {len(existing_cache)}")
        
        if not new_cache:
            print("All files are up to date in cache.")
        else:
            # 기존 캐시와 새 캐시 병합
            existing_cache.update(new_cache)
            
            # 캐시에 저장
            save_analysis_data(directory, existing_cache, "xai_analysis")
            
            print(f"\nSaved XAI analysis cache to {directory}")
            print(f"Cache location: {cache_manager.cache_dir}")
            print(f"Total files in cache: {len(existing_cache)}")
            print(f"New files processed: {len(new_cache)}")
        
        # 처리 통계 저장
        processing_stats[directory] = {
            'total_files': total_files,
            'processed_files': processed_files,
            'skipped_files': skipped_files,
            'failed_files': failed_files
        }
    
    return processing_stats

def run_comparison(directories, mode):
    pass


def run_report(directory, mode):
    """디렉토리에 대한 보고서를 생성합니다."""
    if not os.path.exists(directory):
        print(f"Error: Directory {directory} does not exist.")
        sys.exit(1)
    
    report_start_time = time.time()
    print(f"Generating report for directory: {directory}")
    
    if mode == 'html':
        try:
            # 데이터셋 이름 추출 (디렉토리명 사용)
            dataset_name = os.path.basename(directory)
            
            # 1. 이미지 분석 HTML 본문 생성 (실시간 생성)
            from report_generator.create_report import create_report_body
            html_body = create_report_body(directory)
            if not html_body:
                print("❌ Failed to generate HTML body content")
                return
            
            # 2. report_layout.py의 generate_combined_html 함수 사용하여 완전한 HTML 생성
            from report_generator.report_layout import generate_combined_html
            
            # 완전한 HTML 보고서 생성
            complete_html = generate_combined_html(
                dataset_name=dataset_name,
                dataset_directory=directory
            )
            
            if complete_html:
                print(f"Complete HTML report generated successfully for: {directory}")
                
                # HTML 파일로 저장 (캐시 없이)
                cache_manager = get_cache_manager(directory)
                output_file = os.path.join(cache_manager.cache_dir, f'{dataset_name}_complete_report.html')
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(complete_html)
                print(f"Complete HTML report saved to: {output_file}")
                
            else:
                print("❌ Failed to generate complete HTML report.")
                return
            
            # 보고서 생성 시간 출력
            report_time = time.time() - report_start_time
            print("\n" + "=" * 50)
            print("📄 REPORT GENERATION SUMMARY")
            print("=" * 50)
            print(f"⏱️  Report Generation Time: {format_time(report_time)}")
            print(f"📅 Completed at:           {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 50)
                
        except ImportError as e:
            print(f"❌ Error importing report module: {e}")
            print("💡 Make sure report_generator module exists and all dependencies are installed.")
            print(f"📁 Expected path: report_generator/create_report.py")
            return
        except Exception as e:
            print(f"❌ Error generating report: {e}")
            print("💡 Please check the analysis data and try again.")
            return
    
    elif mode == 'streamlit':
        print("Streamlit mode is not implemented yet.")
        print("Please use 'html' mode for now.")
    
    elif mode == 'pdf':
        print("PDF mode is not implemented yet.")
        print("Please use 'html' mode for now.")


def run_webapp(port=5555, debug=False):
    """Flask 웹앱을 실행합니다."""
    try:
        # flask_webapp 디렉토리를 Python 경로에 추가
        flask_webapp_dir = os.path.join(os.path.dirname(__file__), 'flask_webapp')
        sys.path.insert(0, flask_webapp_dir)
        
        # app.py에서 Flask 앱 import
        import app
        
        print(f"Starting Flask web application on port {port}...")
        print(f"Debug mode: {debug}")
        print(f"Access the application at: http://localhost:{port}")
        
        # Flask 앱 실행
        app.socketio.run(app.app, port=port, debug=debug)
        
    except ImportError as e:
        print(f"Error importing Flask app: {e}")
        print("Make sure flask_webapp/app.py exists and all dependencies are installed.")
        print(f"Flask webapp directory: {flask_webapp_dir}")
        print(f"Available files in flask_webapp: {os.listdir(flask_webapp_dir) if os.path.exists(flask_webapp_dir) else 'Directory not found'}")
    except Exception as e:
        print(f"Error starting Flask web application: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze images, compare datasets, create reports, or run web application.',
        usage='ddoc <command> [options]',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis without XAI
  ddoc analysis /path/to/dataset
  
  # Analysis with XAI (requires YOLO model)
  ddoc analysis /path/to/dataset --model-path /path/to/yolo_model.pt
  
  # Analysis with custom device and formats
  ddoc analysis /path/to/dataset --model-path /path/to/yolo_model.pt --device cuda --format jpg png
  
  # Analysis of multiple directories
  ddoc analysis /path/to/dataset1 /path/to/dataset2 --model-path /path/to/yolo_model.pt
  
  # Generate report
  ddoc report /path/to/dataset
  
  # Run web application
  ddoc app --port 5555 --debug
        """
    )
    subparsers = parser.add_subparsers(dest='command', help='')

    # Analysis sub-command
    parser_analysis = subparsers.add_parser('analysis', help='Analyze images in directories.')
    parser_analysis.add_argument('-r', '--root', help='Root directory to search for datasets.')
    parser_analysis.add_argument('directories', nargs='*', help='Directory to analyze.')
    parser_analysis.add_argument('--format', nargs='+', default=['jpg', 'jpeg', 'png'], help='Image formats to include. (jpg | jpeg | png etc.)')
    parser_analysis.add_argument('--model-path', help='Path to YOLO model for XAI analysis (optional).')
    parser_analysis.add_argument('--device', default=None, help='Device to use for XAI analysis (cuda/cpu, default: auto).')

    # Compare sub-command
    parser_compare = subparsers.add_parser('compare', help='Compare datasets in directories.')
    parser_compare.add_argument('-r', '--root', help='Root directory to search for datasets.')
    parser_compare.add_argument('directories', nargs='*', help='Directories to compare.')
    parser_compare.add_argument('--mode', choices=['streamlit', 'pdf'], default='streamlit', help='Mode to generate report: streamlit or pdf.')

    # Report sub-command
    parser_report = subparsers.add_parser('report', help='Create a report for a directory.')
    parser_report.add_argument('-r', '--root', help='Root directory to search for datasets.')
    parser_report.add_argument('directory', help='Directory to create a report for.')
    parser_report.add_argument('--mode', choices=['streamlit', 'pdf', 'html'], default='html', help='Mode to generate report: streamlit, pdf, or html.')

    # Webapp sub-command
    parser_app = subparsers.add_parser('app', help='Run the Flask web application.')
    parser_app.add_argument('--port', type=int, default=5555, help='Port to run the web application on.')
    parser_app.add_argument('--debug', action='store_true', help='Run in debug mode.')
    
    # Cache sub-command
    parser_cache = subparsers.add_parser('cache', help='Manage cache files.')
    parser_cache.add_argument('--info', action='store_true', help='Show cache information.')
    parser_cache.add_argument('--clear', action='store_true', help='Clear all cache files.')
    parser_cache.add_argument('--directory', help='Directory to show cache info for.')
    


    args = parser.parse_args()

    if args.command == 'analysis':
        if args.root:
            if not args.directories:
                # If only -r is provided, analyze all subdirectories
                args.directories = [os.path.join(args.root, d) for d in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, d))]
            else:
                # Combine -r with directories
                args.directories = [os.path.join(args.root, d) for d in args.directories]
        
        # 분석 시작 시간 기록
        analysis_start_time = time.time()
        
        # 분석 단계별 시간 측정
        analysis_times = {}
        
        # 기본 속성 분석 및 드리프트 분석
        print("=" * 60)
        print("🔍 Starting Attribute Analysis and Drift Analysis...")
        print("=" * 60)
        
        attribute_start = time.time()
        attribute_stats = run_attribute_analysis_wrapper(args.directories, args.format)
        attribute_end = time.time()
        analysis_times['attribute_analysis'] = attribute_end - attribute_start
        
        drift_start = time.time()
        drift_stats = run_drift_analysis(args.directories, args.format)
        drift_end = time.time()
        analysis_times['drift_analysis'] = drift_end - drift_start
        
        # XAI 분석 (모델 경로가 제공된 경우)
        xai_stats = {}
        if args.model_path:
            if not os.path.exists(args.model_path):
                print(f"❌ Error: Model file not found: {args.model_path}")
                return
            
            print("=" * 60)
            print("🧠 Starting XAI (Explainable AI) Analysis...")
            print(f"Model: {args.model_path}")
            print(f"Device: {args.device or 'auto'}")
            print("=" * 60)
            
            xai_start = time.time()
            try:
                xai_stats = run_xai_analysis(args.directories, args.format, args.model_path, args.device)
                print("✅ XAI analysis completed successfully!")
                analysis_times['xai_analysis'] = time.time() - xai_start
            except Exception as e:
                print(f"❌ Error during XAI analysis: {e}")
                print("Continuing with report generation...")
                analysis_times['xai_analysis'] = time.time() - xai_start
        else:
            print("ℹ️  Skipping XAI analysis (no model path provided)")
            print("   Use --model-path to enable XAI analysis")
            analysis_times['xai_analysis'] = 0
        
        # 전체 분석 완료 시간 계산
        total_analysis_time = time.time() - analysis_start_time
        
        # 실제 처리된 파일 수 계산 (중복 제거)
        total_processed_files = 0
        total_skipped_files = 0
        
        # 각 디렉토리별로 실제 파일 수 계산 (중복 제거)
        for directory in args.directories:
            # 각 분석 단계에서 처리된 파일 수를 합산하되, 실제로는 같은 파일들이므로
            # attribute 분석의 파일 수를 기준으로 사용
            if directory in attribute_stats:
                total_processed_files += attribute_stats[directory]['processed_files']
                total_skipped_files += attribute_stats[directory]['skipped_files']
            elif directory in drift_stats:
                total_processed_files += drift_stats[directory]['processed_files']
                total_skipped_files += drift_stats[directory]['skipped_files']
            elif directory in xai_stats:
                total_processed_files += xai_stats[directory]['processed_files']
                total_skipped_files += xai_stats[directory]['skipped_files']
        
        # 분석 시간 요약 출력
        print("\n" + "=" * 60)
        print("⏱️  ANALYSIS TIME SUMMARY")
        print("=" * 60)
        print(f"📊 Attribute Analysis:     {format_time(analysis_times['attribute_analysis'])}")
        print(f"📈 Drift Analysis:         {format_time(analysis_times['drift_analysis'])}")
        if args.model_path:
            print(f"🧠 XAI Analysis:           {format_time(analysis_times['xai_analysis'])}")
        else:
            print(f"🧠 XAI Analysis:           Skipped")
        
        print("-" * 60)
        print(f"🎯 Total Analysis Time:    {format_time(total_analysis_time)}")
        print(f"📅 Completed at:           {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # 성능 통계 (실제 처리된 파일만 계산)
        if total_analysis_time > 0 and total_processed_files > 0:
            files_per_second = total_processed_files / total_analysis_time
            print(f"📁 Files processed:        {total_processed_files}")
            print(f"📁 Files skipped (cached): {total_skipped_files}")
            print(f"⚡ Processing speed:       {files_per_second:.2f} files/second")
        else:
            print(f"📁 Files processed:        {total_processed_files}")
            print(f"📁 Files skipped (cached): {total_skipped_files}")
            if total_processed_files == 0:
                print("ℹ️  No new files were processed (all files were cached)")
        
        # 분석 단계별 파일 수 요약
        print("\n📊 Analysis Summary by Stage:")
        for directory in args.directories:
            print(f"  📁 {os.path.basename(directory)}:")
            if directory in attribute_stats:
                print(f"    📊 Attribute: {attribute_stats[directory]['processed_files']} processed, {attribute_stats[directory]['skipped_files']} skipped")
            if directory in drift_stats:
                print(f"    📈 Drift: {drift_stats[directory]['processed_files']} processed, {drift_stats[directory]['skipped_files']} skipped")
            if directory in xai_stats:
                print(f"    🧠 XAI: {xai_stats[directory]['processed_files']} processed, {xai_stats[directory]['skipped_files']} skipped")
        
        print("\n✅ All analysis completed successfully!")
        print("💡 Use 'ddoc report <directory>' to generate HTML reports")
        print("💡 Use 'ddoc cache --info' to view cache information")
        
        # 캐시 저장 위치 정보 출력
        if args.directories:
            cache_manager = get_cache_manager(args.directories[0])
            print(f"💾 Cache location: {cache_manager.cache_dir}")

    elif args.command == 'compare':
        if args.root:
            if not args.directories:
                # If only -r is provided, analyze all subdirectories
                args.directories = [os.path.join(args.root, d) for d in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, d))]
            else:
                # Combine -r with directories
                args.directories = [os.path.join(args.root, d) for d in args.directories]
        run_comparison(args.directories, args.mode)

    elif args.command == 'report':
        if args.root:
            args.directory = os.path.join(args.root, args.directory)
        run_report(args.directory, args.mode)

    elif args.command == 'app':
        run_webapp(args.port, args.debug)
    
    elif args.command == 'cache':
        if args.info:
            if args.directory:
                cache_manager = get_cache_manager(args.directory)
            else:
                cache_manager = get_cache_manager()
            
            cache_info = cache_manager.get_cache_info()
            print("=" * 60)
            print("📁 Cache Information")
            print("=" * 60)
            print(f"Cache Directory: {cache_info['cache_dir']}")
            print(f"Total Files: {cache_info['total_files']}")
            print(f"Total Size: {cache_info['total_size_mb']} MB")
            print(f"Max Size: {cache_info['max_size_mb']} MB")
            print(f"Expiry Days: {cache_info['expiry_days']}")
            
            if cache_info['cache_details']:
                print("\n📋 Cache Files:")
                print("-" * 60)
                for file_info in cache_info['cache_details']:
                    created_time = file_info['created_time']
                    if created_time:
                        # ISO 형식을 읽기 쉬운 형식으로 변환
                        try:
                            dt = datetime.fromisoformat(created_time.replace('Z', '+00:00'))
                            created_time = dt.strftime("%Y-%m-%d %H:%M:%S")
                        except:
                            pass
                    
                    print(f"📄 {file_info['filename']}")
                    print(f"   Size: {file_info['size_mb']} MB")
                    print(f"   Type: {file_info['content_type'] or 'Unknown'}")
                    print(f"   Created: {created_time or 'Unknown'}")
                    print()
        
        elif args.clear:
            if args.directory:
                cache_manager = get_cache_manager(args.directory)
            else:
                cache_manager = get_cache_manager()
            
            print("🗑️  Clearing cache files...")
            if cache_manager.clear_all_cache():
                print("✅ Cache cleared successfully!")
            else:
                print("❌ Failed to clear cache.")




if __name__ == '__main__':
    main()