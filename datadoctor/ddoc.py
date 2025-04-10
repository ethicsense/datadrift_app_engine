#!/Users/bhc/opt/anaconda3/envs/datadoctor/bin/python

import torch
import clip
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from yellowbrick.cluster import KElbowVisualizer

from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import io, filters, img_as_float
import json
from fpdf import FPDF

import numpy as np
import hashlib
import random

import argparse
import os
import subprocess

def calculate_hash(file_path):
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)

    return hasher.hexdigest()

def load_cache(directory):
    save_path = os.path.join(directory, 'logs')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cache_file = os.path.join(save_path, 'data_cache.json')
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return json.load(f)
        
    return {}

def save_cache(directory, cache):
    save_path = os.path.join(directory, 'logs')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cache_file = os.path.join(save_path, 'data_cache.json')
    with open(cache_file, 'w') as f:
        json.dump(cache, f, indent=4)

def analyze_images(directories, formats):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Loading Embedding Model...")
    model, preprocess = clip.load("ViT-B/16", device=device)

    for directory in directories:
        print(f"\nAnalyzing images in directory: {directory}\n")
        cache = load_cache(directory)
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

                    if file in cache and cache[file]['hash'] == file_hash:
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

        cache.update(new_cache)
        save_cache(directory, cache)
        print(f"\nSaved cache to {directory}")

def create_report(directory, mode):
    cache = load_cache(directory)
    
    if not cache:
        print("No data found in cache.")
        return
    
    # 임베딩 벡터 수집
    embeddings = []
    file_paths = []
    for file_name, data in cache.items():
        embeddings.append(data['embedding'])
        file_paths.append(data['path'])
    
    # 임베딩 벡터 분포 시각화
    flattened_embeddings = np.concatenate(embeddings)
    
    # 클러스터링
    embeddings_array = np.array(embeddings)
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings_array)

    # 자동으로 최적의 클러스터 수 결정
    model = KMeans(random_state=42)
    visualizer = KElbowVisualizer(model, k=(1, 10))
    visualizer.fit(reduced_embeddings)
    num_clusters = visualizer.elbow_value_

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(reduced_embeddings)

    # 유사도 히트맵
    similarity_matrix = cosine_similarity(embeddings_array)

    # 유사도가 0.8 이상인 이미지 쌍 중 최대 3쌍 선택
    threshold = 0.8
    similar_pairs = []

    for i in range(len(similarity_matrix)):
        for j in range(i + 1, len(similarity_matrix)):
            if similarity_matrix[i, j] >= threshold:
                similar_pairs.append((i, j, similarity_matrix[i, j]))
                if len(similar_pairs) == 3:
                    break
        if len(similar_pairs) == 3:
            break
    
    save_path = os.path.join(directory, 'logs')
    # Streamlit 모드
    if mode == 'streamlit':
        subprocess.run(["streamlit", "run", "streamlit_app.py", "--", directory])

    # PDF 모드
    elif mode == 'pdf':
        pdf = FPDF()
        pdf.add_page()
        pdf.add_font("NotoSans", "", "fonts/NotoSansKR-Regular.ttf", uni=True)
        pdf.set_font("NotoSans", size=12)
        
        for file_name, data in cache.items():
            pdf.cell(200, 10, txt=f"File Name: {file_name}", ln=True)
            pdf.cell(200, 10, txt=f"Path: {data['path']}", ln=True)
            pdf.cell(200, 10, txt=f"Size: {data['size']} bytes", ln=True)
            pdf.cell(200, 10, txt=f"Format: {data['format']}", ln=True)
            pdf.cell(200, 10, txt=f"Resolution: {data['resolution']}", ln=True)
            pdf.cell(200, 10, txt=f"Noise Level: {data['noise_level']:.2f}", ln=True)
            pdf.cell(200, 10, txt=f"Sharpness: {data['sharpness']:.2f}", ln=True)
            pdf.cell(200, 10, txt="---", ln=True)

        # 그래프 저장 및 PDF에 삽입
        def save_and_insert_plot(pdf, fig, title):
            image_path = os.path.join(save_path, f"{title}.png")
            fig.savefig(image_path, bbox_inches='tight')
            pdf.add_page()
            pdf.image(image_path, x=10, y=10, w=180)
            os.remove(image_path)

        # 임베딩 벡터 분포 그래프
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(flattened_embeddings, bins=50, color='blue', alpha=0.7, kde=True, ax=ax)
        ax.set_title('Embedding Vector Distribution with KDE')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.grid(True)
        save_and_insert_plot(pdf, fig, "Embedding_Vector_Distribution")

        # 클러스터링 결과 그래프
        fig, ax = plt.subplots(figsize=(10, 6))
        for i in range(num_clusters):
            cluster_points = reduced_embeddings[labels == i]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i}')
        ax.set_title('2D Scatter Plot of Clustering Results')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.legend()
        ax.grid(True)
        save_and_insert_plot(pdf, fig, "Clustering_Results")

        # 유사도 히트맵 그래프
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(similarity_matrix, cmap='viridis', cbar=True, annot=True, fmt=".2f", linewidths=0.5, ax=ax)
        ax.set_title('Detailed Heatmap of Image Similarities')
        ax.set_xlabel('Image Index')
        ax.set_ylabel('Image Index')
        save_and_insert_plot(pdf, fig, "Similarity_Heatmap")

        # PDF 저장
        pdf.output(os.path.join(save_path, "report.pdf"))

def compare_datasets(directories):
    print(f"\nComparing datasets in directories: {directories}\n")
    # 데이터셋 비교 로직을 여기에 추가하세요.
    # 예시: 각 디렉토리의 파일 목록을 비교
    for directory in directories:
        print(f"\nAnalyzing directory: {directory}\n")
        # 디렉토리 내 파일 목록 출력
        for root, _, files in os.walk(directory):
            for file in files:
                print(f"Found file: {file}")

def main():
    parser = argparse.ArgumentParser(
        description='Analyze images, compare datasets, or create reports in directories.',
        usage='ddoc <command> [options]',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
    )
    subparsers = parser.add_subparsers(dest='command', help='')

    # Analysis sub-command
    parser_analysis = subparsers.add_parser('analysis', help='Analyze images in directories.')
    parser_analysis.add_argument('directories', nargs='+', help='Directory to analyze.')
    parser_analysis.add_argument('--format', nargs='+', default=['jpg', 'jpeg', 'png'], help='Image formats to include. (jpg | jpeg | png etc.)')

    # Compare sub-command
    parser_compare = subparsers.add_parser('compare', help='Compare datasets in directories.')
    parser_compare.add_argument('directories', nargs='+', help='Directories to compare.')

    # Report sub-command
    parser_report = subparsers.add_parser('report', help='Create a report for a directory.')
    parser_report.add_argument('directory', help='Directory to create a report for.')
    parser_report.add_argument('--mode', choices=['streamlit', 'pdf'], default='streamlit', help='Mode to generate report: streamlit or pdf.')

    args, unknown = parser.parse_known_args()

    if '-h' in unknown or '--help' in unknown:
        print("""
        usage: ddoc <command> [options]

        Analyze images, compare datasets, or create reports in directories.

        Commands:
          analysis    Analyze images in directories.
          compare     Compare datasets in directories.
          report      Create a report for a directory.

        Options:
          -h, --help  show this help message and exit
        """)
        return

    if args.command == 'analysis':
        analyze_images(args.directories, args.format)
    elif args.command == 'compare':
        compare_datasets(args.directories)
    elif args.command == 'report':
        create_report(args.directory, args.mode)


if __name__ == '__main__':
    main()