#!/usr/bin/env python3
"""
Data Analysis Example Script

이 스크립트는 datadrift_app_engine의 데이터 분석 기능을 사용하는 예시를 보여줍니다.
"""

import sys
import os
import numpy as np

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from data_utils import AttributeAnalyzer, EmbeddingManager
from cache_utils.cache_manager import get_cache_manager, get_cached_analysis_data


def example_attribute_analysis():
    """속성 분석 예시"""
    print("=== Attribute Analysis Example ===")
    
    # 분석할 디렉토리와 형식 설정
    directories = ['/path/to/your/images']  # 실제 경로로 변경하세요
    formats = ['jpg', 'jpeg', 'png']
    
    # AttributeAnalyzer 초기화
    analyzer = AttributeAnalyzer()
    
    # 디렉토리 분석
    for directory in directories:
        if not os.path.exists(directory):
            print(f"Directory not found: {directory}")
            continue
        
        print(f"\nAnalyzing directory: {directory}")
        
        # 속성 분석 수행
        analysis_results = analyzer.analyze_directory(directory, formats)
        
        if analysis_results:
            # 요약 통계 계산
            summary_stats = analyzer.get_summary_statistics(analysis_results)
            
            print(f"Analysis completed!")
            print(f"Total files: {len(analysis_results)}")
            print(f"Total size: {summary_stats['total_size_mb']:.2f} MB")
            print(f"Average size: {summary_stats['avg_size_mb']:.2f} MB")
            print(f"Formats: {summary_stats['formats']}")
            
            # 첫 번째 파일의 상세 정보 출력
            if analysis_results:
                first_file = list(analysis_results.keys())[0]
                first_data = analysis_results[first_file]
                print(f"\nSample file analysis ({first_file}):")
                print(f"  Size: {first_data['size']:.2f} MB")
                print(f"  Format: {first_data['format']}")
                print(f"  Resolution: {first_data['resolution']}")
                print(f"  Noise level: {first_data['noise_level']:.4f}")
                print(f"  Sharpness: {first_data['sharpness']:.4f}")


def example_embedding_extraction():
    """임베딩 추출 예시"""
    print("\n=== Embedding Extraction Example ===")
    
    # 분석할 디렉토리와 형식 설정
    directories = ['/path/to/your/images']  # 실제 경로로 변경하세요
    formats = ['jpg', 'jpeg', 'png']
    
    # EmbeddingManager 초기화
    manager = EmbeddingManager(device='cuda')  # GPU 사용 (CPU: 'cpu')
    manager.load_model("ViT-B/16")
    
    # 임베딩 추출
    for directory in directories:
        if not os.path.exists(directory):
            print(f"Directory not found: {directory}")
            continue
        
        print(f"\nExtracting embeddings from: {directory}")
        
        # 임베딩 추출 수행
        results = manager.extract_embeddings([directory], formats)
        
        if directory in results:
            embeddings_data = results[directory]['embeddings_data']
            print(f"Embedding extraction completed!")
            print(f"Total files processed: {len(embeddings_data)}")
            
            # 첫 번째 파일의 임베딩 정보 출력
            if embeddings_data:
                first_file = list(embeddings_data.keys())[0]
                first_data = embeddings_data[first_file]
                embedding = first_data['embedding']
                print(f"\nSample embedding ({first_file}):")
                print(f"  Embedding shape: {len(embedding)} dimensions")
                print(f"  First 5 values: {embedding[:5]}")
                print(f"  File path: {first_data['path']}")


def example_clustering_analysis():
    """클러스터링 분석 예시"""
    print("\n=== Clustering Analysis Example ===")
    
    # 분석할 디렉토리와 형식 설정
    directories = ['/path/to/your/images']  # 실제 경로로 변경하세요
    formats = ['jpg', 'jpeg', 'png']
    
    # EmbeddingManager 초기화
    manager = EmbeddingManager(device='cuda')
    
    # 클러스터링 분석
    for directory in directories:
        if not os.path.exists(directory):
            print(f"Directory not found: {directory}")
            continue
        
        print(f"\nPerforming clustering analysis on: {directory}")
        
        # 캐시에서 임베딩 데이터 로드
        embedding_cache = get_cached_analysis_data(directory, "image_drift_content")
        
        if not embedding_cache:
            print("No embedding data found. Please run embedding extraction first.")
            continue
        
        # 임베딩 데이터 수집
        embeddings = []
        file_paths = []
        file_names = []
        
        for file_name, data in embedding_cache.items():
            if 'embedding' in data:
                embeddings.append(data['embedding'])
                file_paths.append(data['path'])
                file_names.append(file_name)
        
        if len(embeddings) < 2:
            print("Not enough data for clustering")
            continue
        
        # 클러스터링 분석 수행
        clustering_result = manager.perform_clustering(
            embeddings, file_names, file_paths, 
            n_clusters=3, method='kmeans'
        )
        
        if clustering_result:
            print("Clustering analysis completed!")
            print(f"Method: {clustering_result['method']}")
            print(f"Number of clusters: {clustering_result['n_clusters']}")
            print(f"Total samples: {clustering_result['total_samples']}")
            
            # 클러스터별 통계 출력
            cluster_stats = clustering_result['cluster_stats']
            for cluster_id, stats in cluster_stats.items():
                print(f"\n{cluster_id}: {stats['size']} samples")
                print(f"  Average similarity to centroid: {stats['avg_similarity']:.4f}")
                print(f"  Similarity range: {stats['min_similarity']:.4f} - {stats['max_similarity']:.4f}")
                
                # 상위 유사도 파일들 출력
                if stats['top_similar_files']:
                    print(f"  Top similar files:")
                    for i, file_info in enumerate(stats['top_similar_files'], 1):
                        print(f"    {i}. {file_info['file']} (similarity: {file_info['similarity']:.4f})")


def example_similarity_search():
    """유사도 검색 예시"""
    print("\n=== Similarity Search Example ===")
    
    # 분석할 디렉토리와 형식 설정
    directories = ['/path/to/your/images']  # 실제 경로로 변경하세요
    formats = ['jpg', 'jpeg', 'png']
    
    # EmbeddingManager 초기화
    manager = EmbeddingManager(device='cuda')
    
    for directory in directories:
        if not os.path.exists(directory):
            print(f"Directory not found: {directory}")
            continue
        
        print(f"\nPerforming similarity search on: {directory}")
        
        # 캐시에서 임베딩 데이터 로드
        embedding_cache = get_cached_analysis_data(directory, "image_drift_content")
        
        if not embedding_cache:
            print("No embedding data found. Please run embedding extraction first.")
            continue
        
        # 임베딩 데이터 수집
        embeddings = []
        file_names = []
        
        for file_name, data in embedding_cache.items():
            if 'embedding' in data:
                embeddings.append(data['embedding'])
                file_names.append(file_name)
        
        if len(embeddings) < 2:
            print("Not enough data for similarity search")
            continue
        
        # 유사도 행렬 계산
        embeddings_array = np.array(embeddings)
        similarity_matrix = manager.calculate_similarity_matrix(embeddings_array, method='cosine')
        
        # 유사한 이미지 쌍 찾기
        similar_pairs = manager.find_similar_pairs(similarity_matrix, threshold=0.8, max_pairs=5)
        
        print(f"Found {len(similar_pairs)} similar image pairs:")
        for i, j, similarity in similar_pairs:
            print(f"  {file_names[i]} <-> {file_names[j]}: {similarity:.4f}")


def example_using_cli_commands():
    """CLI 명령어 사용 예시"""
    print("\n=== CLI Commands Example ===")
    print("You can also use the CLI commands for analysis:")
    print()
    print("# 1. 속성 분석 + 임베딩 추출 (효율적인 캐시 시스템)")
    print("python main.py analysis /path/to/your/images")
    print("# - 개별 파일 단위로 캐시를 점검하여 변경된 파일만 분석")
    print()
    print("# 2. 클러스터링 분석 (임베딩 데이터 변경 시에만 재계산)")
    print("python main.py clustering /path/to/your/images --method kmeans --n-clusters 5")
    print("# - 임베딩 데이터가 변경된 경우에만 클러스터링 재수행")
    print()
    print("# 3. HTML 보고서 생성")
    print("python main.py report /path/to/your/images --mode html")
    print()
    print("# 4. 웹 애플리케이션 실행")
    print("python main.py app --port 5555")
    print()
    print("# 캐시 시스템 특징:")
    print("# - 파일 해시 기반 변경 감지")
    print("# - 개별 파일 단위 처리")
    print("# - 중복 분석 방지")
    print("# - 증분 업데이트 지원")


if __name__ == "__main__":
    print("Data Analysis Example Script")
    print("=" * 50)
    
    # 예시 실행
    try:
        example_attribute_analysis()
        example_embedding_extraction()
        example_clustering_analysis()
        example_similarity_search()
        example_using_cli_commands()
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure to update the directory paths in the script to point to your actual image directories.")
    
    print("\n" + "=" * 50)
    print("Example script completed!") 