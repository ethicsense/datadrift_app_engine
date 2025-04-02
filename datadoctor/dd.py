#!/usr/bin/env python3

import argparse
import os

def analyze_images(directories, formats):
    for directory in directories:
        print(f"Analyzing images in directory: {directory}")
        print(f"Formats: {formats}")
        # 예시: 디렉토리 내의 파일을 순회하며 포맷에 맞는 파일을 찾기
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(tuple(formats)):
                    print(f"Found image: {file}")
                    # 이미지 분석 코드 추가

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