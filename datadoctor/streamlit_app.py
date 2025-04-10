import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import json
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from yellowbrick.cluster import KElbowVisualizer

import numpy as np
import os
import sys

def load_cache(directory):
    cache_file = os.path.join(directory, 'logs', 'data_cache.json')
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return json.load(f)
    return {}

def run_streamlit_app(directory):
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

if __name__ == "__main__":
    run_streamlit_app(sys.argv[1])