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

# 캐시 매니저 import
from cache_utils.cache_manager import get_cache_manager, save_analysis_data, get_cached_analysis_data

# 데이터 유틸리티 import
from data_utils import run_attribute_analysis


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
    """
    print("Starting attribute analysis...")
    
    # 분석 모듈 한 번만 로드
    from data_utils import AttributeAnalyzer
    analyzer = AttributeAnalyzer()
    
    for directory in directories:
        if not os.path.exists(directory):
            print(f"Error: Directory {directory} does not exist.")
            continue
        
        # 캐시 매니저 사용
        cache_manager = get_cache_manager(directory)
        
        # 기존 캐시 로드
        existing_cache = get_cached_analysis_data(directory, "image_analysis") or {}
        
        print(f"\nAnalyzing directory: {directory}")
        
        new_cache = {}
        total_files = 0
        processed_files = 0
        
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
                        print(f"Skipping unchanged file: {file}")
        
        print(f"Total files found: {total_files}")
        print(f"Files processed: {processed_files}")
        print(f"Files in cache: {len(existing_cache)}")
        
        if not new_cache:
            print("All files are up to date in cache.")
            continue
        
        # 기존 캐시와 새 캐시 병합
        existing_cache.update(new_cache)
        
        # 캐시에 저장
        save_analysis_data(directory, existing_cache, "image_analysis")
        
        print(f"\nSaved attribute analysis cache to {directory}")
        print(f"Cache location: {cache_manager.cache_dir}")
        print(f"Total files in cache: {len(existing_cache)}")
        print(f"New files processed: {len(new_cache)}")


def run_drift_analysis(directories, formats, model=None, device=None, n_clusters=None, method='kmeans'):
    """
    임베딩 추출과 클러스터링 분석을 수행하고 캐시에 저장합니다.
    디렉토리 탐색하면서 파일을 바로 처리합니다.
    """
    print("Starting drift analysis (embedding extraction and clustering)...")
    
    # 분석 모듈 한 번만 로드
    from data_utils import EmbeddingManager
    manager = EmbeddingManager(device)
    manager.load_model("ViT-B/16")
    
    for directory in directories:
        if not os.path.exists(directory):
            print(f"Error: Directory {directory} does not exist.")
            continue
        
        # 캐시 매니저 사용
        cache_manager = get_cache_manager(directory)
        
        # 기존 캐시 로드
        existing_cache = get_cached_analysis_data(directory, "image_drift_content") or {}
        
        print(f"\nExtracting embeddings for directory: {directory}")
        
        new_cache = {}
        total_files = 0
        processed_files = 0
        
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
                        print(f"Skipping unchanged file: {file}")
        
        print(f"Total files found: {total_files}")
        print(f"Files processed: {processed_files}")
        print(f"Files in cache: {len(existing_cache)}")
        
        if not new_cache:
            print("All files are up to date in cache.")
        else:
            # 기존 캐시와 새 캐시 병합
            existing_cache.update(new_cache)
            
            # 캐시에 저장
            save_analysis_data(directory, existing_cache, "image_drift_content")
            
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


def run_comparison(directories, mode):
    pass


def run_report(directory, mode):
    """디렉토리에 대한 보고서를 생성합니다."""
    if not os.path.exists(directory):
        print(f"Error: Directory {directory} does not exist.")
        sys.exit(1)
    
    print(f"Generating report for directory: {directory}")
    
    if mode == 'html':
        try:
            # 데이터셋 이름 추출 (디렉토리명 사용)
            dataset_name = os.path.basename(directory)
            
            # 1. 이미지 분석 HTML 본문 생성 및 캐시에 저장
            from report_generator.create_report import create_report_body
            html_body = create_report_body(directory)
            if html_body:
                # 캐시 매니저를 사용하여 HTML 본문을 캐시에 저장
                cache_manager = get_cache_manager(directory)
                body_cache_key = "image_analysis_html_body"
                
                # HTML 본문을 캐시에 저장
                cache_manager.save_cached_content(body_cache_key, html_body, "html")
                print(f"Image analysis HTML body cached successfully")
                print(f"Cache key: {body_cache_key}")
            
            # 2. report_layout.py의 generate_combined_html 함수 사용하여 완전한 HTML 생성
            from report_generator.report_layout import generate_combined_html
            
            # 완전한 HTML 보고서 생성
            complete_html = generate_combined_html(
                dataset_name=dataset_name,
                dataset_directory=directory
            )
            
            if complete_html:
                print(f"Complete HTML report generated successfully for: {directory}")
                
                # 완전한 HTML을 캐시에 저장
                complete_cache_key = "complete_html_report"
                cache_manager.save_cached_content(complete_cache_key, complete_html, "html")
                
                print(f"Complete HTML report cached successfully")
                print(f"Cache key: {complete_cache_key}")
                
                # 캐시 폴더에 완전한 HTML 파일로 저장
                output_file = os.path.join(cache_manager.cache_dir, f'{dataset_name}_complete_report.html')
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(complete_html)
                print(f"Complete HTML report saved to: {output_file}")
                
            else:
                print("Failed to generate complete HTML report.")
                
        except ImportError as e:
            print(f"Error importing report module: {e}")
            print("Make sure report_gen/report_layout.py exists and all dependencies are installed.")
        except Exception as e:
            print(f"Error generating report: {e}")
    
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
        add_help=False,
    )
    subparsers = parser.add_subparsers(dest='command', help='')

    # Analysis sub-command
    parser_analysis = subparsers.add_parser('analysis', help='Analyze images in directories.')
    parser_analysis.add_argument('-r', '--root', help='Root directory to search for datasets.')
    parser_analysis.add_argument('directories', nargs='*', help='Directory to analyze.')
    parser_analysis.add_argument('--format', nargs='+', default=['jpg', 'jpeg', 'png'], help='Image formats to include. (jpg | jpeg | png etc.)')
    parser_analysis.add_argument('--method', choices=['kmeans', 'dbscan', 'hierarchical'], default='kmeans', help='Clustering method to use.')
    parser_analysis.add_argument('--n-clusters', type=int, help='Number of clusters (auto-determined if not specified).')

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

    args = parser.parse_args()

    if args.command == 'analysis':
        if args.root:
            if not args.directories:
                # If only -r is provided, analyze all subdirectories
                args.directories = [os.path.join(args.root, d) for d in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, d))]
            else:
                # Combine -r with directories
                args.directories = [os.path.join(args.root, d) for d in args.directories]
        run_attribute_analysis_wrapper(args.directories, args.format)
        run_drift_analysis(args.directories, args.format, args.n_clusters, args.method)

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


if __name__ == '__main__':
    main()