#!/Users/bhc/opt/anaconda3/envs/datadoctor/bin/python

import argparse
import os
from PIL import Image
from skimage import io, filters, img_as_float
import numpy as np

def analyze_images(directories, formats):

    for directory in directories:
        print(f"Analyzing images in directory: {directory}")

        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(tuple(formats)):
                    file_path = os.path.join(root, file)

                    try:
                        with Image.open(file_path) as img:
                            # 기본 메타데이터
                            file_name = file
                            file_size = os.path.getsize(file_path)
                            image_format = img.format
                            width, height = img.size
                            resolution = f"{width}x{height}"

                            # 이미지 데이터 분석
                            image_array = img_as_float(io.imread(file_path, as_gray=True))
                            noise_level = np.std(image_array)
                            sharpness = filters.sobel(image_array).mean()

                            # 결과 출력
                            print(f"Image Name: {file_name}")
                            print(f"Path: {file_path}")
                            print(f"Size: {file_size} bytes")
                            print(f"Format: {image_format}")
                            print(f"Resolution: {resolution}")
                            print(f"Noise Level: {noise_level:.4f}")
                            print(f"Sharpness: {sharpness:.4f}")
                            print("-" * 40)

                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")

def create_report(directory):
    pass

def compare_datasets(directories):
    print(f"Comparing datasets in directories: {directories}")
    # 데이터셋 비교 로직을 여기에 추가하세요.
    # 예시: 각 디렉토리의 파일 목록을 비교
    for directory in directories:
        print(f"Analyzing directory: {directory}")
        # 디렉토리 내 파일 목록 출력
        for root, _, files in os.walk(directory):
            for file in files:
                print(f"Found file: {file}")

def main():
    parser = argparse.ArgumentParser(
        description='Analyze images, compare datasets, or create reports in directories.',
        usage='dd <command> [options]',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', help='')

    ## Analysis sub-command
    parser_analysis = subparsers.add_parser(
        'analysis',
        help='Analyze images in directories.'
    )
    parser_analysis.add_argument(
        'directories',
        nargs='+',
        help='Directories to analyze.'
    )
    parser_analysis.add_argument(
        '--format',
        nargs='+',
        default=['jpg', 'jpeg', 'png'], 
        help='Image formats to include.'
    )
    ## Report sub-command
    parser_report = subparsers.add_parser(
        'report',
        help='Create a report for a directory.'
    )
    parser_report.add_argument(
        'directory', 
        help='Directory to create a report for.'
    )
    ## Compare sub-command
    parser_compare = subparsers.add_parser(
        'compare', 
        help='Compare datasets in directories.'
    )
    parser_compare.add_argument(
        'directories', 
        nargs='+', 
        help='Directories to compare.'
    )

    args = parser.parse_args()

    if args.command == 'analysis':
        analyze_images(args.directories, args.format)
    elif args.command == 'compare':
        compare_datasets(args.directories)
    elif args.command == 'report':
        create_report(args.directory)

if __name__ == '__main__':
    main()