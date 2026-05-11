import os
import hashlib
import numpy as np
from PIL import Image
from skimage import io, filters, img_as_float
import sys


class AttributeAnalyzer:
    """이미지 속성 분석을 관리하는 클래스"""
    
    def __init__(self):
        """AttributeAnalyzer 초기화"""
        pass
    
    def analyze_image_attributes(self, file_path):
        """
        단일 이미지의 속성을 분석하고 해시와 함께 결과를 반환합니다.
        
        Args:
            file_path: 분석할 이미지 파일 경로
        
        Returns:
            dict: 이미지 속성 정보 (해시 포함) 또는 None
        """
        try:
            with Image.open(file_path) as img:
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

                # 해시 계산
                import hashlib
                hasher = hashlib.md5()
                with open(file_path, 'rb') as f:
                    buf = f.read()
                    hasher.update(buf)
                file_hash = hasher.hexdigest()

                return {
                    'hash': file_hash,
                    'path': os.path.abspath(file_path),
                    'size': file_size_mb,
                    'format': image_format,
                    'resolution': resolution,
                    'width': width,
                    'height': height,
                    'noise_level': noise_level,
                    'sharpness': sharpness
                }
        
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return None
    
    def analyze_directory(self, directory, formats):
        """
        디렉토리의 모든 이미지 속성을 분석합니다.
        
        Args:
            directory: 분석할 디렉토리 경로
            formats: 분석할 이미지 포맷 리스트
        
        Returns:
            dict: 파일별 속성 분석 결과
        """
        if not os.path.exists(directory):
            print(f"Error: Directory {directory} does not exist.")
            return {}
        
        print(f"\nAnalyzing images in directory: {directory}\n")
        
        results = {}
        format_count = 0

        # 파일 수 계산
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(tuple(formats)):
                    format_count += 1
        print(f"Found {format_count} files in {directory}\n")

        # 속성 분석
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(tuple(formats)):
                    file_path = os.path.join(root, file)

                    # 속성 분석 수행
                    attributes = self.analyze_image_attributes(file_path)
                    
                    if attributes:
                        results[file] = attributes
                        print(f"Processed {file}")
                    else:
                        print(f"Failed to process {file}")

        return results
    
    def get_summary_statistics(self, analysis_results):
        """
        분석 결과의 요약 통계를 계산합니다.
        
        Args:
            analysis_results: 속성 분석 결과 딕셔너리
        
        Returns:
            dict: 요약 통계 정보
        """
        if not analysis_results:
            return {}
        
        # 데이터 수집
        sizes = [item['size'] for item in analysis_results.values()]
        noise_levels = [item['noise_level'] for item in analysis_results.values()]
        sharpness_values = [item['sharpness'] for item in analysis_results.values()]
        
        # 형식별 통계
        formats = {}
        resolutions = {}
        
        for item in analysis_results.values():
            fmt = item['format']
            formats[fmt] = formats.get(fmt, 0) + 1
            
            res = item['resolution']
            resolutions[res] = resolutions.get(res, 0) + 1
        
        return {
            'total_images': len(analysis_results),
            'total_size_mb': sum(sizes),
            'avg_size_mb': np.mean(sizes),
            'formats': formats,
            'resolutions': resolutions,
            'size_stats': {
                'min': np.min(sizes),
                'max': np.max(sizes),
                'mean': np.mean(sizes),
                'std': np.std(sizes)
            },
            'noise_stats': {
                'min': np.min(noise_levels),
                'max': np.max(noise_levels),
                'mean': np.mean(noise_levels),
                'std': np.std(noise_levels)
            },
            'sharpness_stats': {
                'min': np.min(sharpness_values),
                'max': np.max(sharpness_values),
                'mean': np.mean(sharpness_values),
                'std': np.std(sharpness_values)
            }
        }


def run_attribute_analysis(directories, formats):
    """
    속성 분석을 실행하는 편의 함수
    
    Args:
        directories: 분석할 디렉토리 리스트
        formats: 이미지 포맷 리스트
    
    Returns:
        dict: 각 디렉토리별 분석 결과
    """
    analyzer = AttributeAnalyzer()
    results = {}
    
    for directory in directories:
        if not os.path.exists(directory):
            print(f"Error: Directory {directory} does not exist.")
            continue
        
        # 속성 분석 수행
        analysis_results = analyzer.analyze_directory(directory, formats)
        
        if analysis_results:
            # 요약 통계 계산
            summary_stats = analyzer.get_summary_statistics(analysis_results)
            
            results[directory] = {
                'analysis_results': analysis_results,
                'summary_stats': summary_stats,
                'total_files': len(analysis_results)
            }
            
            print(f"\nAnalysis completed for {directory}")
            print(f"Total files processed: {len(analysis_results)}")
            print(f"Total size: {summary_stats['total_size_mb']:.2f} MB")
            print(f"Average size: {summary_stats['avg_size_mb']:.2f} MB")
        else:
            print(f"No valid files found in {directory}")
    
    return results 