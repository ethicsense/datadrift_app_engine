#!/Users/bhc/opt/anaconda3/envs/datadoctor/bin/python

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
from fpdf import FPDF

import numpy as np
import hashlib
import random

import argparse
import os
import sys
import subprocess

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
        if not os.path.exists(directory):
            print(f"Error: Directory {directory} does not exist.")
            sys.exit(1)
        
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
    # Streamlit 모드
    if mode == 'streamlit':
        subprocess.run(["streamlit", "run", "streamlit_app.py", "--", "report", directory])

    # PDF 모드
    elif mode == 'pdf':
        save_path = os.path.join(directory, 'logs')
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

        pdf = FPDF()
        pdf.add_page()
        pdf.add_font("NotoSans", "", "fonts/NotoSansKR-Regular.ttf")
        pdf.set_font("NotoSans", size=12)
        
        for file_name, data in cache.items():
            pdf.cell(200, 10, text=f"File Name: {file_name}")
            pdf.cell(200, 10, text=f"Path: {data['path']}")
            pdf.cell(200, 10, text=f"Size: {data['size']} bytes")
            pdf.cell(200, 10, text=f"Format: {data['format']}")
            pdf.cell(200, 10, text=f"Resolution: {data['resolution']}")
            pdf.cell(200, 10, text=f"Noise Level: {data['noise_level']:.2f}")
            pdf.cell(200, 10, text=f"Sharpness: {data['sharpness']:.2f}")
            pdf.cell(200, 10, text="---")

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

def compare_datasets(directories, mode):
    print(f"\nComparing datasets in directories: {directories}\n")
    if mode == 'streamlit':
        subprocess.run(["streamlit", "run", "streamlit_app.py", "--", "compare"] + directories)

    elif mode == 'pdf':
        save_path = os.path.dirname(directories[0])
        embeddings_list = []
        sizes_list = []
        resolutions_list = []
        noise_levels_list = []
        sharpness_list = []
        
        for directory in directories:
            cache_file = os.path.join(directory, 'logs', 'data_cache.json')
            if not os.path.exists(cache_file):
                print(f"No cache file found in {directory}")
                continue
            
            with open(cache_file, 'r') as f:
                cache = json.load(f)
            
            embeddings = [data['embedding'] for data in cache.values()]
            sizes = [data['size'] for data in cache.values()]
            resolutions = [data['resolution'] for data in cache.values()]
            noise_levels = [data['noise_level'] for data in cache.values()]
            sharpness = [data['sharpness'] for data in cache.values()]
            
            embeddings_list.append(np.array(embeddings))
            sizes_list.append(sizes)
            resolutions_list.append(resolutions)
            noise_levels_list.append(noise_levels)
            sharpness_list.append(sharpness)
        
        if len(embeddings_list) < 2:
            print("Not enough datasets to compare.")
            return
        
        # MMD 계산
        for i in range(len(embeddings_list) - 1):
            for j in range(i + 1, len(embeddings_list)):
                mmd_score = calculate_mmd_batch(embeddings_list[i], embeddings_list[j])
                print(f"MMD between dataset {i+1} and dataset {j+1}: {mmd_score:.4f}")
                pdf.cell(200, 10, txt=f"MMD between dataset {i+1} and dataset {j+1}: {mmd_score:.4f}", ln=True)
        
        # 임베딩 시각화
        pca = PCA(n_components=2)
        pca.fit(np.vstack(embeddings_list))
        
        transformed_data = [pca.transform(embeddings) for embeddings in embeddings_list]

        # PDF settings
        pdf_filename = "_".join([os.path.basename(d) for d in directories]) + "_compare.pdf"
        pdf = FPDF()
        pdf.add_page()
        pdf.add_font("NotoSans", "", "fonts/NotoSansKR-Regular.ttf")
        pdf.set_font("NotoSans", size=12)

        # 그래프 저장 및 PDF에 삽입
        def save_and_insert_plot(pdf, fig, title):
            image_path = os.path.join(save_path, f"{title}.png")
            fig.savefig(image_path, bbox_inches='tight')
            pdf.add_page()
            pdf.image(image_path, x=10, y=10, w=180)
            os.remove(image_path)

        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ['blue', 'red', 'green', 'purple', 'orange']
        labels = [f'Dataset {i+1}' for i in range(len(directories))]
        
        for i, data in enumerate(transformed_data):
            ax.scatter(data[:, 0], data[:, 1], alpha=0.5, label=labels[i], color=colors[i])
        
        ax.set_title('PCA of Embeddings')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.legend()
        save_and_insert_plot(pdf, fig, "PCA_of_Embeddings")

        # 등고선 그래프
        for i, data in enumerate(transformed_data):
            fig, ax = plt.subplots(figsize=(8, 6))
            # Calculate the point density
            x = data[:, 0]
            y = data[:, 1]
            xy = np.vstack([x, y])
            
            # Create a grid for contour plotting
            margin = 0.7  # 10% margin
            x_min, x_max = x.min() - margin * (x.max() - x.min()), x.max() + margin * (x.max() - x.min())
            y_min, y_max = y.min() - margin * (y.max() - y.min()), y.max() + margin * (y.max() - y.min())
            xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            
            # Calculate the density on the grid
            z = gaussian_kde(xy)(positions).reshape(xx.shape)
            
            ax.contourf(xx, yy, z, levels=20, cmap='YlGn', alpha=1)
            ax.contour(xx, yy, z, levels=20, colors='k', linewidths=0.5)
            ax.scatter(x, y, c='black', s=10, alpha=0.5)
            ax.set_title(f'Contour Plot of {labels[i]}')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            
            save_and_insert_plot(pdf, fig, f"Contour_Plot_of_Embeddings_{i+1}")
        
        # 크기 분포
        fig, ax = plt.subplots(figsize=(12, 6))
        for i, sizes in enumerate(sizes_list):
            sns.histplot(sizes, bins=30, label=f'Dataset {i+1}', color=colors[i], alpha=0.5, ax=ax)
        ax.set_title('Size Distribution (MB)')
        ax.set_xlabel('Size (MB)')
        ax.set_ylabel('Frequency')
        ax.legend()
        save_and_insert_plot(pdf, fig, "Size_Distribution")
        
        # 해상도 분포
        fig, ax = plt.subplots(figsize=(12, 6))
        for i, resolutions in enumerate(resolutions_list):
            unique_res, counts = np.unique(resolutions, return_counts=True)
            plt.bar(unique_res, counts, label=f'Dataset {i+1}', alpha=0.5, color=colors[i])
        ax.set_title('Resolution Distribution')
        ax.set_xlabel('Resolution')
        ax.set_ylabel('Count')
        ax.legend()
        save_and_insert_plot(pdf, fig, "Resolution_Distribution")
        
        # 노이즈 레벨 분포
        fig, ax = plt.subplots(figsize=(12, 6))
        for i, noise_levels in enumerate(noise_levels_list):
            sns.histplot(noise_levels, bins=30, label=f'Dataset {i+1}', color=colors[i], alpha=0.5, ax=ax)
        ax.set_title('Noise Level Distribution')
        ax.set_xlabel('Noise Level')
        ax.set_ylabel('Frequency')
        ax.legend()
        save_and_insert_plot(pdf, fig, "Noise_Level_Distribution")
        
        # 선명도 분포
        fig, ax = plt.subplots(figsize=(12, 6))
        for i, sharpness in enumerate(sharpness_list):
            sns.histplot(sharpness, bins=30, label=f'Dataset {i+1}', color=colors[i], alpha=0.5, ax=ax)
        ax.set_title('Sharpness Distribution')
        ax.set_xlabel('Sharpness')
        ax.set_ylabel('Frequency')
        ax.legend()
        save_and_insert_plot(pdf, fig, "Sharpness_Distribution")

        # PDF 저장
        pdf.output(os.path.join(save_path, pdf_filename))
        
        

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
    parser_compare.add_argument('--mode', choices=['streamlit', 'pdf'], default='streamlit', help='Mode to generate report: streamlit or pdf.')

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
        compare_datasets(args.directories, args.mode)
    elif args.command == 'report':
        create_report(args.directory, args.mode)


if __name__ == '__main__':
    main()