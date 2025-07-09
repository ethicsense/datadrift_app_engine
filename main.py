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

def calculate_hash(file_path):
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)

    return hasher.hexdigest()

def run_analysis(directories, formats):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Loading Embedding Model...")
    model, preprocess = clip.load("ViT-B/16", device=device)

    for directory in directories:
        if not os.path.exists(directory):
            print(f"Error: Directory {directory} does not exist.")
            sys.exit(1)
        
        print(f"\nAnalyzing images in directory: {directory}\n")
        
        # 캐시 매니저 사용
        cache_manager = get_cache_manager(directory)
        
        # 기존 캐시 로드
        existing_cache = get_cached_analysis_data(directory, "image_analysis") or {}
        new_cache = {}
        format_count = 0

        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(tuple(formats)):
                    format_count += 1
        print(f"Found {format_count} datas in {directory}\n")

        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(tuple(formats)):
                    file_path = os.path.join(root, file)
                    file_hash = calculate_hash(file_path)

                    if file in existing_cache and existing_cache[file]['hash'] == file_hash:
                        print(f"Skipping unchanged file: {file_path}")
                        continue

                    try:
                        with Image.open(file_path) as img:
                            # 임베딩 추출
                            with torch.no_grad():
                                print(f"Embedding {file_path}...")
                                input = preprocess(img).unsqueeze(0).to(device)
                                embedding = model.encode_image(input).cpu().numpy().flatten()

                            # 기본 메타데이터
                            file_size_bytes = os.path.getsize(file_path)
                            file_size_mb = file_size_bytes / (1024 * 1024)  # Convert to MB
                            image_format = img.format
                            width, height = img.size
                            resolution = f"{width}x{height}"

                            # 이미지 데이터 분석
                            image_array = img_as_float(io.imread(file_path, as_gray=True))
                            noise_level = np.std(image_array)
                            sharpness = filters.sobel(image_array).mean()

                            # 결과 저장
                            new_cache[file] = {
                                'hash': file_hash,
                                'path': os.path.abspath(file_path),
                                'size': file_size_mb,
                                'format': image_format,
                                'resolution': resolution,
                                'noise_level': noise_level,
                                'sharpness': sharpness,
                                'embedding': embedding.tolist()
                            }
                            print(f"Processed {file}")

                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")

        # 기존 캐시와 새 캐시 병합
        existing_cache.update(new_cache)
        
        # 캐시에 저장
        save_analysis_data(directory, existing_cache, "image_analysis")
        
        print(f"\nSaved cache to {directory}")
        print(f"Cache location: {cache_manager.cache_dir}")


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
        run_analysis(args.directories, args.format)

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