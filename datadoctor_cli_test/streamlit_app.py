import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import json
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
from sklearn.metrics.pairwise import cosine_similarity
from yellowbrick.cluster import KElbowVisualizer

import numpy as np
import os
import sys
import argparse

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

def load_cache(directory):
    cache_file = os.path.join(directory, 'logs', 'data_cache.json')
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return json.load(f)
    return {}

def run_report(directory):
    print(directory)
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

    st.title("Image Data Analysis Report")

    for file_name, data in cache.items():
        st.write(f"**File Name:** {file_name}")
        st.write(f"**Path:** {data['path']}")
        st.write(f"**Size:** {data['size']} bytes")
        st.write(f"**Format:** {data['format']}")
        st.write(f"**Resolution:** {data['resolution']}")
        st.write(f"**Noise Level:** {data['noise_level']:.2f}")
        st.write(f"**Sharpness:** {data['sharpness']:.2f}")
        st.write("---")

    st.subheader("Embedding Vector Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(flattened_embeddings, bins=50, color='blue', alpha=0.7, kde=True, ax=ax)
    ax.set_title('Embedding Vector Distribution with KDE')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("Clustering Results")
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(num_clusters):
        cluster_points = reduced_embeddings[labels == i]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i}')
    ax.set_title('2D Scatter Plot of Clustering Results')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("Similarity Heatmap")
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(similarity_matrix, cmap='viridis', cbar=True, annot=False, fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title('Detailed Heatmap of Image Similarities')
    ax.set_xlabel('Image Index')
    ax.set_ylabel('Image Index')
    st.pyplot(fig)

    st.subheader("Similar Image Pairs")
    for pair in similar_pairs:
        idx1, idx2, sim = pair
        st.write(f"**Image {idx1} and Image {idx2} have similarity: {sim:.2f}**")
        
        img1 = Image.open(file_paths[idx1])
        img2 = Image.open(file_paths[idx2])
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(img1, caption=f"Image {idx1}", use_container_width=True)
        with col2:
            st.image(img2, caption=f"Image {idx2}", use_container_width=True)

def run_compare(*directories):
    st.title("Dataset Comparison")

    # 데이터 로드 및 전처리
    embeddings_list = []
    sizes_list = []
    resolutions_list = []
    noise_levels_list = []
    sharpness_list = []
    
    for directory in directories:
        cache_file = os.path.join(directory, 'logs', 'data_cache.json')
        if not os.path.exists(cache_file):
            st.write(f"No cache file found in {directory}")
            return
        
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
        st.write("Not enough datasets to compare.")
        return
    
    # MMD 계산
    for i in range(len(embeddings_list) - 1):
        for j in range(i + 1, len(embeddings_list)):
            mmd_score = calculate_mmd_batch(embeddings_list[i], embeddings_list[j])
            st.write(f"MMD between dataset {i+1} and dataset {j+1}: {mmd_score:.4f}")
    
    # 임베딩 시각화
    pca = PCA(n_components=2)
    pca.fit(np.vstack(embeddings_list))
    
    transformed_data = [pca.transform(embeddings) for embeddings in embeddings_list]
    
    # PCA of Embeddings
    st.subheader('PCA of Embeddings')
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    labels = [f'Dataset {i+1}' for i in range(len(directories))]
    
    for i, data in enumerate(transformed_data):
        ax.scatter(data[:, 0], data[:, 1], alpha=0.5, label=labels[i], color=colors[i % len(colors)])
    
    ax.set_title('PCA of Embeddings')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend()
    st.pyplot(fig)

    # 등고선 그래프
    st.subheader('Contour Plots')
    for i, data in enumerate(transformed_data):
        fig, ax = plt.subplots(figsize=(8, 6))
        x = data[:, 0]
        y = data[:, 1]
        xy = np.vstack([x, y])
        
        margin = 0.7
        x_min, x_max = x.min() - margin * (x.max() - x.min()), x.max() + margin * (x.max() - x.min())
        y_min, y_max = y.min() - margin * (y.max() - y.min()), y.max() + margin * (y.max() - y.min())
        xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        
        z = gaussian_kde(xy)(positions).reshape(xx.shape)
        
        ax.contourf(xx, yy, z, levels=20, cmap='YlGn', alpha=1)
        ax.contour(xx, yy, z, levels=20, colors='k', linewidths=0.5)
        ax.scatter(x, y, c='black', s=10, alpha=0.5)
        ax.set_title(f'Contour Plot of Dataset {i+1}')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        st.pyplot(fig)
    
    # 크기 분포
    st.subheader('Size Distribution (MB)')
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, sizes in enumerate(sizes_list):
        sns.histplot(sizes, bins=30, label=f'Dataset {i+1}', color=colors[i % len(colors)], alpha=0.5, ax=ax)
    ax.set_title('Size Distribution (MB)')
    ax.set_xlabel('Size (MB)')
    ax.set_ylabel('Frequency')
    ax.legend()
    st.pyplot(fig)
    
    # 해상도 분포
    st.subheader('Resolution Distribution')
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, resolutions in enumerate(resolutions_list):
        unique_res, counts = np.unique(resolutions, return_counts=True)
        ax.bar(unique_res, counts, label=f'Dataset {i+1}', alpha=0.5, color=colors[i % len(colors)])
    ax.set_title('Resolution Distribution')
    ax.set_xlabel('Resolution')
    ax.set_ylabel('Count')
    ax.legend()
    plt.xticks(rotation=30)
    st.pyplot(fig)
    
    # 노이즈 레벨 분포
    st.subheader('Noise Level Distribution')
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, noise_levels in enumerate(noise_levels_list):
        sns.histplot(noise_levels, bins=30, label=f'Dataset {i+1}', color=colors[i % len(colors)], alpha=0.5, ax=ax)
    ax.set_title('Noise Level Distribution')
    ax.set_xlabel('Noise Level')
    ax.set_ylabel('Frequency')
    ax.legend()
    st.pyplot(fig)
    
    # 선명도 분포
    st.subheader('Sharpness Distribution')
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, sharpness in enumerate(sharpness_list):
        sns.histplot(sharpness, bins=30, label=f'Dataset {i+1}', color=colors[i % len(colors)], alpha=0.5, ax=ax)
    ax.set_title('Sharpness Distribution')
    ax.set_xlabel('Sharpness')
    ax.set_ylabel('Frequency')
    ax.legend()
    st.pyplot(fig)

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', help='')

    parser_report = subparsers.add_parser('report', help='Create a report for a directory.')
    parser_report.add_argument('directory', help='Directory to create a report for.')

    parser_compare = subparsers.add_parser('compare', help='Compare two directories.')
    parser_compare.add_argument('directories', nargs='+', help='Directories to compare.')

    args = parser.parse_args()

    if args.command == 'report':
        run_report(args.directory)
    elif args.command == 'compare':
        run_compare(*args.directories)


if __name__ == "__main__":
    main()