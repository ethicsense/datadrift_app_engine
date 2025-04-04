#!/Users/bhc/opt/anaconda3/envs/datadoctor/bin/python

import torch
import clip

from PIL import Image
from skimage import io, filters, img_as_float
import json

import numpy as np
import hashlib

import argparse
import os

def calculate_hash(file_path):
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)

    return hasher.hexdigest()

def load_cache(directory):
    cache_file = os.path.join(directory, 'data_cache.json')
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return json.load(f)
        
    return {}

def save_cache(directory, cache):
    cache_file = os.path.join(directory, 'data_cache.json')
    with open(cache_file, 'w') as f:
        json.dump(cache, f, indent=4)

def analyze_images(directories, formats):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Loading Embedding Model...")
    model, preprocess = clip.load("ViT-B/16", device=device)

    for directory in directories:
        print(f"Analyzing images in directory: {directory}")
        cache = load_cache(directory)
        new_cache = {}

        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(tuple(formats)):
                    file_path = os.path.join(root, file)
                    file_hash = calculate_hash(file_path)

                    if file_path in cache and cache[file_path]['hash'] == file_hash:
                        print(f"Skipping unchanged file: {file_path}")
                        new_cache[file_path] = cache[file_path]
                        continue

                    try:
                        with Image.open(file_path) as img:
                            # 임베딩 추출
                            with torch.no_grad():
                                print(f"Embedding {file_path}...")
                                input = preprocess(img).unsqueeze(0).to(device)
                                embedding = model.encode_image(input).cpu().numpy().flatten()

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

                            # 결과 저장
                            new_cache[file_name] = {
                                'hash': file_hash,
                                'path': os.path.abspath(file_path),
                                'size': file_size,
                                'format': image_format,
                                'resolution': resolution,
                                'noise_level': noise_level,
                                'sharpness': sharpness,
                                'embedding': embedding.tolist()
                            }

                            print(f"Processed {file_name}")

                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")

        save_cache(directory, new_cache)

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
        usage='ddoc <command> [options]',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
    )
    subparsers = parser.add_subparsers(dest='command', help='')

    # Analysis sub-command
    parser_analysis = subparsers.add_parser('analysis', help='Analyze images in directories.')
    parser_analysis.add_argument('directories', nargs='+', help='Directories to analyze.')
    parser_analysis.add_argument('--format', nargs='+', default=['jpg', 'jpeg', 'png'], help='Image formats to include.')

    # Compare sub-command
    parser_compare = subparsers.add_parser('compare', help='Compare datasets in directories.')
    parser_compare.add_argument('directories', nargs='+', help='Directories to compare.')

    # Report sub-command
    parser_report = subparsers.add_parser('report', help='Create a report for a directory.')
    parser_report.add_argument('directory', help='Directory to create a report for.')

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
        create_report(args.directory)


if __name__ == '__main__':
    main()