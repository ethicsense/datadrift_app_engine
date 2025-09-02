from bs4 import BeautifulSoup
from datetime import datetime
import os

# 캐시 매니저 import (분석 데이터용)
from cache_utils.cache_manager import get_cache_manager, get_cached_html_content, get_cached_analysis_data, save_analysis_data

# 차트 설명 생성기 import
from .guideline_generator import ChartDescriptionGenerator

# 차트 설명 생성기 인스턴스 생성
chart_description_generator = ChartDescriptionGenerator()

def add_chart_description(chart_key, chart_html):
    """차트에 설명을 추가하는 함수"""
    description_html = chart_description_generator.generate_description_html(chart_key)
    if description_html:
        return f"""
        <div class="chart-with-description">
            {chart_html}
            {description_html}
        </div>
        """
    return chart_html

def add_chart_descriptions_to_section(section_html, chart_keys):
    """섹션 내의 차트들에 설명을 추가하는 함수"""
    if not chart_keys:
        return section_html
    
    # 각 차트 키에 대해 설명 추가
    for chart_key in chart_keys:
        if chart_key in chart_description_generator.descriptions:
            # 차트 이미지나 차트 관련 HTML을 찾아서 설명 추가
            # 여기서는 간단하게 차트 제목을 찾아서 설명을 추가하는 방식 사용
            title_pattern = f'<h[1-6][^>]*>{chart_description_generator.descriptions[chart_key]["title"]}</h[1-6]>'
            import re
            if re.search(title_pattern, section_html):
                # 차트 제목 다음에 설명 추가
                section_html = re.sub(
                    title_pattern,
                    f'\\g<0>{chart_description_generator.generate_description_html(chart_key)}',
                    section_html
                )
    
    return section_html

def generate_chart_description_css():
    """차트 설명을 위한 CSS 스타일 생성"""
    return """
    <style>
        .chart-description {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 15px;
            margin: 15px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .description-header {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
            padding-bottom: 8px;
            border-bottom: 2px solid #007bff;
        }
        
        .description-icon {
            font-size: 1.2em;
            margin-right: 8px;
        }
        
        .description-title {
            font-weight: bold;
            color: #495057;
            font-size: 1.1em;
        }
        
        .description-content {
            color: #6c757d;
            line-height: 1.6;
            font-size: 0.95em;
        }
        
        .chart-with-description {
            margin: 20px 0;
        }
        
        .chart-description-summary {
            background: #ffffff;
            border: 1px solid #e9ecef;
            border-radius: 6px;
            padding: 12px;
            margin: 10px 0;
        }
        
        .chart-description-summary h4 {
            color: #495057;
            margin-bottom: 8px;
            font-size: 1.1em;
        }
        
        .chart-description-summary p {
            color: #6c757d;
            margin: 0;
            line-height: 1.5;
        }
    </style>
    """

# HTML에서 <body> 태그만 추출하고 h1 태그 제거
def get_html_body(html):
    if not html:
        return ''
    if BeautifulSoup:
        soup = BeautifulSoup(html, "html.parser")
        body = soup.find('body')
        if body:
            for h1 in body.find_all('h1'):
                h1.decompose()
            return str(body)
        else:
            return str(soup)
    else:
        import re
        return re.sub(r'<h1[^>]*>.*?</h1>', '', html, flags=re.DOTALL)

# 캐시된 HTML 가져오기 또는 생성하기 (기존 함수 유지)
def get_cached_html(cache_key, generator_func, *args, dataset_directory=None):
    """기존 호환성을 위한 함수 (내부적으로 새로운 캐시 매니저 사용)"""
    return get_cached_html_content(cache_key, generator_func, *args, dataset_directory=dataset_directory)

# main HTML 생성 함수
def generate_combined_html(dataset_name=None, database_export_report=None, drift_export_report=None, dataset_directory=None):
    """최적화된 HTML 생성 (파일 기반 캐시 활용)"""
    timestamp = datetime.now().strftime('%Y년 %m월 %d일 %H시 %M분')
    
    # 컨텐츠 초기화
    database_content = ''
    image_drift_content = ''
    image_analysis_content = ''
    
    # 데이터베이스 리포트 생성
    if database_export_report:
        database_content = get_cached_html_content(
            f"db_html_{dataset_name}",
            database_export_report.generate_html_from_session,
            dataset_name,
            dataset_directory=dataset_directory
        )
    
    # 드리프트 분석 리포트 생성
    if drift_export_report:
        image_drift_content = get_cached_html_content(
            f"drift_html_{dataset_name}",
            drift_export_report.generate_html_from_session,
            dataset_directory=dataset_directory
        )
    
    # 이미지 분석 리포트 생성 (실시간 생성)
    if dataset_directory:
        try:
            from report_generator.create_report import create_report_body
            image_analysis_content = create_report_body(dataset_directory)
            if not image_analysis_content or image_analysis_content.strip() == '':
                image_analysis_content = '<p>이미지 분석 데이터가 없습니다. 먼저 이미지 분석을 실행해주세요.</p>'
        except Exception as e:
            image_analysis_content = f'<p>이미지 분석 로드 중 오류: {e}</p>'
    
    # 섹션별 컨텐츠 구성
    sections = []
    
    # 데이터베이스 정보 섹션
    if database_content:
        sections.append(f"""
        <div class="section">
            <div class="section-title">📊 Dataset Information & Statistics</div>
            {database_content}
        </div>
        """)
    
    # 이미지 분석 섹션 (항상 표시, 데이터가 없으면 안내 메시지)
    sections.append(f"""
    <div class="section" style="margin-bottom: 40px;">
        {image_analysis_content}
    </div>
    """)
    
    # 드리프트 분석 섹션
    if image_drift_content:
        sections.append(f"""
        <div class="section">
            <div class="section-title">🔍 Data Drift Analysis Results</div>
            {image_drift_content}
        </div>
        """)
    
    # 섹션이 없는 경우 기본 메시지
    if not sections:
        sections.append("""
        <div class="section">
            <div class="section-title">📊 Analysis Results</div>
            <p>분석 데이터가 없습니다. 먼저 데이터베이스 분석, 이미지 분석, 또는 드리프트 분석을 실행해주세요.</p>
        </div>
        """)
    
    # 모든 섹션을 하나로 결합
    all_sections = '\n'.join(sections)
    
    combined_html = f"""<!DOCTYPE html>
                        <html lang="ko">
                        <head>
                            <meta charset="utf-8">
                            <title>{dataset_name} - 통합 분석 리포트</title>
                            <style>
                                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                                body {{ 
                                    font-family: 'Malgun Gothic', sans-serif; 
                                    line-height: 1.6; color: #2c3e50; 
                                    background: #f8f9fa; padding: 30px;
                                }}
                                .container {{ 
                                    max-width: 1200px; margin: 0 auto; 
                                    background: white; padding: 30px; 
                                    border-radius: 10px; box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                                }}
                                .header {{ 
                                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                    color: white; padding: 25px; border-radius: 8px; 
                                    margin-bottom: 25px; text-align: center;
                                }}
                                .title {{ font-size: 2em; margin-bottom: 5px; }}
                                .subtitle {{ font-size: 1.1em; opacity: 0.9; }}
                                .section {{ 
                                    margin: 30px 0; padding: 25px; 
                                    border: 2px solid #e9ecef; border-radius: 12px;
                                    background: white; box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                                }}
                                .section-title {{ 
                                    font-size: 1.5em; color: #495057; 
                                    margin-bottom: 20px; padding-bottom: 12px;
                                    border-bottom: 3px solid #007bff; font-weight: bold;
                                }}
                                table {{ 
                                    width: 100%; border-collapse: collapse; margin: 15px 0;
                                    border-radius: 5px; overflow: hidden;
                                }}
                                th {{ 
                                    background: #6c757d; color: white; 
                                    padding: 10px; text-align: left;
                                }}
                                td {{ padding: 8px; border-bottom: 1px solid #dee2e6; }}
                                img {{ max-width: 100%; height: auto; margin: 10px 0; }}
                                pre {{ 
                                    background: #f8f9fa; padding: 15px; 
                                    border-radius: 5px; overflow-x: auto;
                                }}
                                .footer {{ 
                                    text-align: center; margin-top: 30px; 
                                    padding: 15px; background: #f8f9fa; 
                                    border-radius: 5px; color: #6c757d;
                                }}
                            </style>
                            {generate_chart_description_css()}
                        </head>
                        <body>
                            <div class="container">
                                <div class="header">
                                    <div class="title">{dataset_name} 통합 분석 리포트</div>
                                    <div class="subtitle">데이터 드리프트 분석 보고서</div>
                                    <div style="margin-top: 10px; font-size: 0.9em;">생성일시: {timestamp}</div>
                                </div>
                                
                                {all_sections}
                                
                                <div class="footer">
                                    <strong>
                                        <a href="https://github.com/keti-datadrift/datadrift_dataclinic" target="_blank" style="color: #3498db; text-decoration: none;">DataDrift Dataclinic System</a>
                                    </strong><br>
                                    @2025 KETI, Korea Electronics Technology Institute<br>
                                </div>
                            </div>
                        </body>
                        </html>"""
    return combined_html

# 캐시 관리 유틸리티 함수들
def get_cache_info(dataset_directory=None):
    """캐시 정보 반환"""
    cache_manager = get_cache_manager(dataset_directory)
    return cache_manager.get_cache_info()

def clear_all_cache(dataset_directory=None):
    """모든 캐시 정리"""
    cache_manager = get_cache_manager(dataset_directory)
    return cache_manager.clear_all_cache()

def invalidate_cache(identifier, content_type="html", dataset_directory=None):
    """특정 캐시 무효화"""
    cache_manager = get_cache_manager(dataset_directory)
    return cache_manager.invalidate_cache(identifier, content_type)


def generate_individual_xai_html(filename, xai_result, visualizer):
    """개별 XAI 분석 결과에 대한 HTML 보고서를 생성합니다."""
    timestamp = datetime.now().strftime('%Y년 %m월 %d일 %H시 %M분')
    
    # XAI 시각화 생성
    try:
        visualizations = visualizer.create_comprehensive_visualization(xai_result)
    except Exception as e:
        print(f"    ⚠️  Error creating visualizations for {filename}: {e}")
        visualizations = {}
    
    # 개별 보고서용 XAI 요약 정보 생성
    summary_info = generate_xai_summary_for_individual_report(xai_result)
    
    # 공통 XAI 시각화 컨테이너 사용
    viz_container = generate_xai_visualizations_container(visualizations, f"🔬 XAI Analysis Results - {filename}")
    
    # 완전한 HTML 생성
    html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="utf-8">
    <title>XAI Analysis Report - {filename}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Malgun Gothic', sans-serif; 
            line-height: 1.6; color: #2c3e50; 
            background: #f8f9fa; padding: 30px;
        }}
        .container {{ 
            max-width: 1200px; margin: 0 auto; 
            background: white; padding: 30px; 
            border-radius: 10px; box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        .header {{ 
            background: linear-gradient(135deg, #ffc107 0%, #ff8c00 100%);
            color: white; padding: 25px; border-radius: 8px; 
            margin-bottom: 25px; text-align: center;
        }}
        .title {{ font-size: 2em; margin-bottom: 5px; }}
        .subtitle {{ font-size: 1.1em; opacity: 0.9; }}
        .summary-section {{
            margin: 25px 0; padding: 20px; 
            border: 2px solid #ffc107; border-radius: 12px;
            background: #fff3cd; box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}
        .summary-title {{ 
            font-size: 1.5em; color: #495057; 
            margin-bottom: 20px; padding-bottom: 12px;
            border-bottom: 3px solid #ffc107; font-weight: bold;
        }}
        .summary-grid {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
            gap: 15px; margin-bottom: 20px;
        }}
        .summary-item {{
            background: white; padding: 15px; border-radius: 8px; 
            text-align: center; border: 1px solid #e9ecef;
        }}
        .summary-label {{ 
            color: #6c757d; margin: 0 0 8px 0; font-size: 0.9em; 
        }}
        .summary-value {{ 
            font-size: 1.8em; font-weight: bold; color: #495057; 
        }}
        .footer {{ 
            text-align: center; margin-top: 30px; 
            padding: 15px; background: #f8f9fa; 
            border-radius: 5px; color: #6c757d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="title">🧠 XAI Analysis Report</div>
            <div class="subtitle">Explainable AI Analysis Results</div>
            <div style="margin-top: 10px; font-size: 0.9em;">File: {filename}</div>
            <div style="margin-top: 5px; font-size: 0.9em;">Generated: {timestamp}</div>
        </div>
        
        <div class="summary-section">
            <div class="summary-title">📊 Analysis Summary</div>
            <div class="summary-grid">
                {summary_info}
            </div>
        </div>
        
        <div class="summary-section">
            <div class="summary-title">🔬 Detailed Analysis Results</div>
            {viz_container}
        </div>
        
        <div class="footer">
            <strong>
                <a href="https://github.com/keti-datadrift/datadrift_dataclinic" target="_blank" style="color: #3498db; text-decoration: none;">DataDrift Dataclinic System</a>
            </strong><br>
            @2025 KETI, Korea Electronics Technology Institute<br>
        </div>
    </div>
</body>
</html>"""
    
    return html_content


# ===== 섹션별 HTML 생성 함수들 =====

def generate_summary_statistics_section(summary_data, size_chart=None):
    """요약 통계 섹션 HTML 생성"""
    if not summary_data:
        return ""
    
    chart_html = ""
    if size_chart:
        chart_html = f"""
        <div style="text-align: center; margin: 20px 0;">
            <img src="data:image/png;base64,{size_chart}" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
        </div>
        """
    
    # 차트 설명 추가
    chart_description = ""
    if size_chart and chart_description_generator:
        chart_description = chart_description_generator.generate_description_html('file_size_distribution')
    
    return f"""
    <div style="margin-bottom: 20px;">
        <h3 style="color: #495057; margin-bottom: 15px; padding-bottom: 8px; border-bottom: 2px solid #dee2e6;">📈 Summary Statistics</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px;">
            <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #e9ecef;">
                <h4 style="color: #6c757d; margin: 0 0 8px 0; font-size: 0.9em;">Total Images</h4>
                <div style="font-size: 1.8em; font-weight: bold; color: #495057;">{summary_data.get('total_images', 0):,}</div>
            </div>
            <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #e9ecef;">
                <h4 style="color: #6c757d; margin: 0 0 8px 0; font-size: 0.9em;">Total Size</h4>
                <div style="font-size: 1.8em; font-weight: bold; color: #495057;">{summary_data.get('total_size_mb', 0):.2f} MB</div>
            </div>
            <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #e9ecef;">
                <h4 style="color: #6c757d; margin: 0 0 8px 0; font-size: 0.9em;">Average Size</h4>
                <div style="font-size: 1.8em; font-weight: bold; color: #495057;">{summary_data.get('avg_size_mb', 0):.2f} MB</div>
            </div>
            <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #e9ecef;">
                <h4 style="color: #6c757d; margin: 0 0 8px 0; font-size: 0.9em;">Unique Formats</h4>
                <div style="font-size: 1.8em; font-weight: bold; color: #495057;">{len(summary_data.get('formats', {}))}</div>
            </div>
        </div>
        {chart_html}
        {chart_description}
    </div>
    """


def generate_format_distribution_section(summary_data, format_chart=None):
    """형식별 분포 섹션 HTML 생성"""
    if not summary_data.get('formats'):
        return ""
    
    format_items = []
    for fmt, count in summary_data.get('formats', {}).items():
        percentage = (count / summary_data['total_images']) * 100
        format_items.append(f"""
        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #e9ecef;">
            <h4 style="color: #6c757d; margin: 0 0 8px 0; font-size: 0.9em;">
                <span style="background: #e74c3c; color: white; padding: 3px 6px; border-radius: 8px; font-size: 0.7em; font-weight: bold;">{fmt.upper()}</span>
            </h4>
            <div style="font-size: 1.5em; font-weight: bold; color: #495057;">{count:,}</div>
            <p style="margin: 5px 0 0 0; color: #6c757d; font-size: 0.9em;">({percentage:.1f}%)</p>
        </div>
        """)
    
    chart_html = ""
    if format_chart:
        chart_html = f"""
        <div style="text-align: center; margin: 20px 0;">
            <img src="data:image/png;base64,{format_chart}" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
        </div>
        """
    
    # 차트 설명 추가
    chart_description = ""
    if format_chart and chart_description_generator:
        chart_description = chart_description_generator.generate_description_html('image_format_distribution')
    
    return f"""
    <div style="margin-bottom: 20px;">
        <h3 style="color: #495057; margin-bottom: 15px; padding-bottom: 8px; border-bottom: 2px solid #dee2e6;">📋 Format Distribution</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px;">
            {''.join(format_items)}
        </div>
        {chart_html}
        {chart_description}
    </div>
    """


def generate_visualizations_section(charts_data):
    """시각화 섹션 HTML 생성"""
    if not charts_data:
        return ""
    
    chart_items = []
    
    # 파일 크기 분포
    if 'size_distribution' in charts_data:
        chart_items.append(f"""
        <div style="text-align: center; margin: 20px 0;">
            <img src="data:image/png;base64,{charts_data['size_distribution']}" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
        </div>
        """)
    
    # 형식별 분포
    if 'format_distribution' in charts_data:
        chart_items.append(f"""
        <div style="text-align: center; margin: 20px 0;">
            <img src="data:image/png;base64,{charts_data['format_distribution']}" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
        </div>
        """)
    
    # 노이즈 vs 선명도
    if 'noise_vs_sharpness' in charts_data:
        chart_items.append(f"""
        <div style="text-align: center; margin: 20px 0;">
            <img src="data:image/png;base64,{charts_data['noise_vs_sharpness']}" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
        </div>
        """)
    
    # 해상도별 분포
    if 'resolution_distribution' in charts_data:
        chart_items.append(f"""
        <div style="text-align: center; margin: 20px 0;">
            <img src="data:image/png;base64,{charts_data['resolution_distribution']}" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
        </div>
        """)
    
    # 임베딩 PCA
    if 'embeddings_pca' in charts_data:
        chart_items.append(f"""
        <div style="text-align: center; margin: 20px 0;">
            <img src="data:image/png;base64,{charts_data['embeddings_pca']}" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
        </div>
        """)
    
    # 클러스터링 결과
    if 'clustering_results' in charts_data:
        chart_items.append(f"""
        <div style="text-align: center; margin: 20px 0;">
            <img src="data:image/png;base64,{charts_data['clustering_results']}" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
        </div>
        """)
    
    # 클러스터 크기 분포
    if 'cluster_size_distribution' in charts_data:
        chart_items.append(f"""
        <div style="text-align: center; margin: 20px 0;">
            <img src="data:image/png;base64,{charts_data['cluster_size_distribution']}" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
        </div>
        """)
    
    return f"""
    <div style="margin-bottom: 20px;">
        <h3 style="color: #495057; margin-bottom: 15px; padding-bottom: 8px; border-bottom: 2px solid #dee2e6;">📊 Visualizations</h3>
        {''.join(chart_items)}
    </div>
    """


def generate_sample_images_section(samples_data):
    """샘플 이미지 테이블 섹션 HTML 생성"""
    if not samples_data:
        return ""
    
    sample_rows = []
    for sample in samples_data:
        sample_rows.append(f"""
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #dee2e6;">{sample['filename']}</td>
                <td style="padding: 8px; border-bottom: 1px solid #dee2e6;">{sample['size_mb']}</td>
                <td style="padding: 8px; border-bottom: 1px solid #dee2e6;">
                    <span style="background: #e74c3c; color: white; padding: 2px 6px; border-radius: 8px; font-size: 0.8em; font-weight: bold;">{sample['format'].upper()}</span>
                </td>
                <td style="padding: 8px; border-bottom: 1px solid #dee2e6;">{sample['resolution']}</td>
                <td style="padding: 8px; border-bottom: 1px solid #dee2e6;">{sample['noise_level']}</td>
                <td style="padding: 8px; border-bottom: 1px solid #dee2e6;">{sample['sharpness']}</td>
            </tr>
        """)
    
    return f"""
    <div style="margin-bottom: 20px;">
        <h3 style="color: #495057; margin-bottom: 15px; padding-bottom: 8px; border-bottom: 2px solid #dee2e6;">🖼️ Sample Images (Top 10 by Size)</h3>
        <table style="width: 100%; border-collapse: collapse; margin: 15px 0; border-radius: 5px; overflow: hidden; background: white;">
            <thead>
                <tr>
                    <th style="background: #6c757d; color: white; padding: 10px; text-align: left;">Filename</th>
                    <th style="background: #6c757d; color: white; padding: 10px; text-align: left;">Size (MB)</th>
                    <th style="background: #6c757d; color: white; padding: 10px; text-align: left;">Format</th>
                    <th style="background: #6c757d; color: white; padding: 10px; text-align: left;">Resolution</th>
                    <th style="background: #6c757d; color: white; padding: 10px; text-align: left;">Noise Level</th>
                    <th style="background: #6c757d; color: white; padding: 10px; text-align: left;">Sharpness</th>
                </tr>
            </thead>
            <tbody>
                {''.join(sample_rows)}
            </tbody>
        </table>
    </div>
    """


def generate_detailed_statistics_section(summary_data, quality_chart=None):
    """상세 통계 섹션 HTML 생성"""
    if not summary_data:
        return ""
    
    # 노이즈 통계
    noise_stats = summary_data.get('noise_stats', {})
    noise_html = f"""
    <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #e9ecef;">
        <h4 style="color: #6c757d; margin: 0 0 10px 0;">📊 Noise Level Statistics</h4>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;">
            <div style="text-align: center;">
                <div style="font-size: 1.2em; font-weight: bold; color: #495057;">{noise_stats.get('mean', 0):.4f}</div>
                <div style="font-size: 0.8em; color: #6c757d;">Mean</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.2em; font-weight: bold; color: #495057;">{noise_stats.get('std', 0):.4f}</div>
                <div style="font-size: 0.8em; color: #6c757d;">Std Dev</div>
            </div>
        </div>
    </div>
    """
    
    # 선명도 통계
    sharpness_stats = summary_data.get('sharpness_stats', {})
    sharpness_html = f"""
    <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #e9ecef;">
        <h4 style="color: #6c757d; margin: 0 0 10px 0;">🔍 Edgeness Statistics</h4>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;">
            <div style="text-align: center;">
                <div style="font-size: 1.2em; font-weight: bold; color: #495057;">{sharpness_stats.get('mean', 0):.4f}</div>
                <div style="font-size: 0.8em; color: #6c757d;">Mean</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.2em; font-weight: bold; color: #495057;">{sharpness_stats.get('std', 0):.4f}</div>
                <div style="font-size: 0.8em; color: #6c757d;">Std Dev</div>
            </div>
        </div>
    </div>
    """
    
    chart_html = ""
    if quality_chart:
        chart_html = f"""
        <div style="text-align: center; margin: 20px 0;">
            <img src="data:image/png;base64,{quality_chart}" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
        </div>
        """
    
    # 차트 설명 추가
    chart_description = ""
    if quality_chart and chart_description_generator:
        chart_description = chart_description_generator.generate_description_html('noise_vs_sharpness')
    
    return f"""
    <div style="margin-bottom: 20px;">
        <h3 style="color: #495057; margin-bottom: 15px; padding-bottom: 8px; border-bottom: 2px solid #dee2e6;">📊 Detailed Statistics</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin-bottom: 20px;">
            {noise_html}
            {sharpness_html}
        </div>
        {chart_html}
        {chart_description}
    </div>
    """


def generate_embedding_info_section(embed_data, embedding_chart=None):
    """임베딩 정보 섹션 HTML 생성"""
    if not embed_data:
        return ""
    
    total_embeddings = len(embed_data)
    embedding_dim = len(list(embed_data.values())[0]['embedding']) if embed_data else 0
    
    chart_html = ""
    if embedding_chart:
        chart_html = f"""
        <div style="text-align:center; margin: 20px 0;">
            <img src="data:image/png;base64,{embedding_chart}" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
        </div>
        """
    
    # 차트 설명 추가
    chart_description = ""
    if embedding_chart and chart_description_generator:
        chart_description = chart_description_generator.generate_description_html('embeddings_pca')
    
    return f"""
    <div style="margin-bottom: 20px;">
        <h3 style="color: #495057; margin-bottom: 15px; padding-bottom: 8px; border-bottom: 2px solid #dee2e6;">🧠 Embedding Information</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
            <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #e9ecef;">
                <h4 style="color: #6c757d; margin: 0 0 8px 0; font-size: 0.9em;">Total Embeddings</h4>
                <div style="font-size: 1.8em; font-weight: bold; color: #495057;">{total_embeddings:,}</div>
            </div>
            <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #e9ecef;">
                <h4 style="color: #6c757d; margin: 0 0 8px 0; font-size: 0.9em;">Embedding Dimension</h4>
                <div style="font-size: 1.8em; font-weight: bold; color: #495057;">{embedding_dim}</div>
            </div>
        </div>
        {chart_html}
        {chart_description}
    </div>
    """


def generate_resolution_info_section(summary_data, resolution_chart=None):
    """해상도 정보 섹션 HTML 생성"""
    if not summary_data.get('resolutions'):
        return ""
    
    top_resolutions = dict(sorted(summary_data['resolutions'].items(), key=lambda x: x[1], reverse=True)[:5])
    
    resolution_items = []
    for res, count in top_resolutions.items():
        percentage = (count / summary_data['total_images']) * 100
        resolution_items.append(f"""
        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #e9ecef;">
            <h4 style="color: #6c757d; margin: 0 0 8px 0; font-size: 0.9em;">{res}</h4>
            <div style="font-size: 1.5em; font-weight: bold; color: #495057;">{count:,}</div>
            <p style="margin: 5px 0 0 0; color: #6c757d; font-size: 0.9em;">({percentage:.1f}%)</p>
        </div>
        """)
    
    chart_html = ""
    if resolution_chart:
        chart_html = f"""
        <div style="text-align: center; margin: 20px 0;">
            <img src="data:image/png;base64,{resolution_chart}" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
        </div>
        """
    
    # 차트 설명 추가
    chart_description = ""
    if resolution_chart and chart_description_generator:
        chart_description = chart_description_generator.generate_description_html('resolution_distribution')
    
    return f"""
    <div style="margin-bottom: 20px;">
        <h3 style="color: #495057; margin-bottom: 15px; padding-bottom: 8px; border-bottom: 2px solid #dee2e6;">📐 Resolution Information</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px;">
            {''.join(resolution_items)}
        </div>
        {chart_html}
        {chart_description}
    </div>
    """


def generate_clustering_summary_section(clustering_summary, clustering_charts=None):
    """클러스터링 요약 섹션 HTML 생성"""
    if not clustering_summary:
        return ""
    
    # 클러스터 요약 정보
    summary_html = f"""
    <div style="margin-bottom: 20px;">
        <h3 style="color: #495057; margin-bottom: 15px; padding-bottom: 8px; border-bottom: 2px solid #dee2e6;">🧠 Clustering Summary</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
            <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #e9ecef;">
                <h4 style="color: #6c757d; margin: 0 0 8px 0; font-size: 0.9em;">Method</h4>
                <div style="font-size: 1.8em; font-weight: bold; color: #495057;">{clustering_summary.get('method', 'N/A').upper()}</div>
            </div>
            <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #e9ecef;">
                <h4 style="color: #6c757d; margin: 0 0 8px 0; font-size: 0.9em;">Total Samples</h4>
                <div style="font-size: 1.8em; font-weight: bold; color: #495057;">{clustering_summary.get('total_samples', 0):,}</div>
            </div>
            <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #e9ecef;">
                <h4 style="color: #6c757d; margin: 0 0 8px 0; font-size: 0.9em;">Clusters</h4>
                <div style="font-size: 1.8em; font-weight: bold; color: #495057;">{clustering_summary.get('n_clusters', 0)}</div>
            </div>
        </div>
    </div>
    """
    
    # 클러스터링 차트들 추가 (각 차트 아래에 설명 포함)
    charts_html = ""
    if clustering_charts:
        if 'clustering_results' in clustering_charts:
            chart_description = ""
            if chart_description_generator:
                chart_description = chart_description_generator.generate_description_html('clustering_results')
            charts_html += f"""
            <div style="text-align: center; margin: 20px 0;">
                <img src="data:image/png;base64,{clustering_charts['clustering_results']}" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
            </div>
            {chart_description}
            """
        if 'cluster_size_distribution' in clustering_charts:
            chart_description = ""
            if chart_description_generator:
                chart_description = chart_description_generator.generate_description_html('cluster_size_distribution')
            charts_html += f"""
            <div style="text-align: center; margin: 20px 0;">
                <img src="data:image/png;base64,{clustering_charts['cluster_size_distribution']}" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
            </div>
            {chart_description}
            """
    
    # 클러스터 상세 테이블
    if clustering_summary.get('cluster_summary'):
        cluster_rows = []
        for cluster in clustering_summary['cluster_summary']:
            cluster_rows.append(f"""
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #dee2e6;">{cluster['cluster_id']}</td>
                <td style="padding: 8px; border-bottom: 1px solid #dee2e6;">{cluster['size']:,}</td>
                <td style="padding: 8px; border-bottom: 1px solid #dee2e6;">{cluster['percentage']:.1f}%</td>
                <td style="padding: 8px; border-bottom: 1px solid #dee2e6;">
                    {''.join([f'{f}<br>' for f in cluster['sample_files']])}
                </td>
            </tr>
            """)
        
        table_html = f"""
        <div style="margin-top: 20px;">
            <h4 style="color: #495057; margin-bottom: 10px;">Cluster Details</h4>
            <table style="width: 100%; border-collapse: collapse; margin: 15px 0; border-radius: 5px; overflow: hidden; background: white;">
                <thead>
                    <tr>
                        <th style="background: #6c757d; color: white; padding: 10px; text-align: left;">Cluster ID</th>
                        <th style="background: #6c757d; color: white; padding: 10px; text-align: left;">Size</th>
                        <th style="background: #6c757d; color: white; padding: 10px; text-align: left;">Percentage</th>
                        <th style="background: #6c757d; color: white; padding: 10px; text-align: left;">Sample Files</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(cluster_rows)}
                </tbody>
            </table>
        </div>
        """
        
        return summary_html + charts_html + table_html
    
    return summary_html + charts_html


def generate_xai_visualizations_container(xai_charts, title="🔬 Representative Sample Report"):
    """XAI 시각화 컨테이너 HTML 생성 (공통 함수)"""
    if not xai_charts:
        return ""
    
    # XAI 시각화들을 시각화 타입별로 그룹화
    viz_types = {}
    for key, viz_data in xai_charts.items():
        # 개별 보고서에서는 키가 직접 시각화 타입 (예: "cam_heatmap")
        # 통합 보고서에서는 키가 "filename_viztype" 형식 (예: "image1_cam_heatmap")
        known_viz_types = [
            'cam_heatmap', 'cam_threshold_analysis', 'cam_distribution_analysis',
            'cam_statistics', 'connected_components', 'entropy_analysis',
            'centroid_analysis', 'overlap_analysis', 'overlap_statistics'
        ]
        
        viz_type = None
        # 먼저 직접 매칭 시도 (개별 보고서용)
        if key in known_viz_types:
            viz_type = key
        else:
            # 통합 보고서용: "filename_viztype" 형식에서 추출
            for viz_type_name in known_viz_types:
                if key.endswith(f'_{viz_type_name}'):
                    viz_type = viz_type_name
                    break
            
            # 알려진 타입이 없으면 기본 분리 방식 사용
            if viz_type is None:
                parts = key.split('_', 1)
                if len(parts) == 2:
                    viz_type = parts[1]
                else:
                    viz_type = 'unknown'
        
        viz_types[viz_type] = viz_data
    
    # XAI 분석 결과 컨테이너 시작
    html_parts = [f"""
    <div style="margin-bottom: 25px; padding: 20px; background: white; border-radius: 10px; border: 2px solid #ffc107;">
        <h4 style="color: #495057; margin-bottom: 15px; border-bottom: 2px solid #ffc107; padding-bottom: 8px;">
            {title}
        </h4>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px;">
    """]
    
    # 시각화 타입별 제목 매핑
    viz_titles = {
        'cam_heatmap': '🔥 CAM Heatmap',
        'cam_threshold_analysis': '🎯 Threshold Analysis',
        'cam_distribution_analysis': '📊 Distribution Analysis',
        'cam_statistics': '📈 CAM Statistics',
        'connected_components': '🔗 Connected Components Analysis',
        'entropy_analysis': '📊 Entropy Analysis',
        'centroid_analysis': '🎯 Centroid Analysis',
        'overlap_analysis': '🔍 Overlap Analysis',
        'overlap_statistics': '📊 Overlap Statistics'
    }
    
    # 원하는 순서로 시각화 타입 정렬
    desired_order = [
        'cam_heatmap',
        'cam_statistics',
        'cam_threshold_analysis',
        'connected_components',
        'entropy_analysis',
        'centroid_analysis',
        'overlap_analysis',
        'overlap_statistics',
        'cam_distribution_analysis'
    ]
    
    # 정렬된 순서로 시각화 출력
    for viz_type in desired_order:
        if viz_type in viz_types:
            viz_data = viz_types[viz_type]
            title = viz_titles.get(viz_type, viz_type.replace('_', ' ').title())
            
            # 차트 설명 추가
            chart_description = ""
            if chart_description_generator:
                # XAI 차트 키 매핑
                xai_chart_keys = {
                    'cam_heatmap': 'cam_distribution',
                    'cam_statistics': 'cam_statistics',
                    'cam_threshold_analysis': 'adaptive_cam_analysis',
                    'connected_components': 'connected_components_analysis',
                    'entropy_analysis': 'cam_entropy_analysis',
                    'centroid_analysis': 'cam_centroid_analysis',
                    'overlap_analysis': 'object_detection_cam_overlay',
                    'overlap_statistics': 'object_detection_performance_summary',
                    'cam_distribution_analysis': 'cam_distribution'
                }
                chart_key = xai_chart_keys.get(viz_type)
                if chart_key:
                    chart_description = chart_description_generator.generate_description_html(chart_key)
            
            html_parts.append(f"""
        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #dee2e6;">
            <h5 style="color: #495057; margin-bottom: 10px; font-size: 1.1em;">{title}</h5>
            <div style="text-align: center;">
                <img src="data:image/png;base64,{viz_data}" 
                     style="max-width: 100%; height: auto; border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
            </div>
            {chart_description}
        </div>
            """)
    
    # 정렬되지 않은 기타 시각화 타입들도 출력
    for viz_type, viz_data in viz_types.items():
        if viz_type not in desired_order:
            title = viz_titles.get(viz_type, viz_type.replace('_', ' ').title())
            
            # 차트 설명 추가
            chart_description = ""
            if chart_description_generator:
                # XAI 차트 키 매핑
                xai_chart_keys = {
                    'cam_heatmap': 'cam_distribution',
                    'cam_statistics': 'cam_statistics',
                    'cam_threshold_analysis': 'adaptive_cam_analysis',
                    'connected_components': 'connected_components_analysis',
                    'entropy_analysis': 'cam_entropy_analysis',
                    'centroid_analysis': 'cam_centroid_analysis',
                    'overlap_analysis': 'object_detection_cam_overlay',
                    'overlap_statistics': 'object_detection_performance_summary',
                    'cam_distribution_analysis': 'cam_distribution'
                }
                chart_key = xai_chart_keys.get(viz_type)
                if chart_key:
                    chart_description = chart_description_generator.generate_description_html(chart_key)
            
            html_parts.append(f"""
        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #dee2e6;">
            <h5 style="color: #495057; margin-bottom: 10px; font-size: 1.1em;">{title}</h5>
            <div style="text-align: center;">
                <img src="data:image/png;base64,{viz_data}" 
                     style="max-width: 100%; height: auto; border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
            </div>
            {chart_description}
        </div>
            """)
    
    # XAI 분석 결과 컨테이너 종료
    html_parts.append("""
        </div>
    </div>
    """)
    
    return ''.join(html_parts)


def generate_xai_summary_for_integrated_report(xai_summary):
    """통합 보고서용 XAI 요약 정보 HTML 생성"""
    if not xai_summary:
        return ""
    
    return f"""
    <div style="margin-bottom: 25px; padding: 20px; background: white; border-radius: 8px; border: 1px solid #dee2e6;">
        <h4 style="color: #495057; margin-bottom: 15px; padding-bottom: 8px; border-bottom: 2px solid #ffc107;">📊 XAI Analysis Summary</h4>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px;">
            <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #e9ecef;">
                <h5 style="color: #6c757d; margin: 0 0 8px 0; font-size: 0.9em;">Total Files Analyzed</h5>
                <div style="font-size: 1.8em; font-weight: bold; color: #495057;">{xai_summary.get('total_files', 0):,}</div>
            </div>
            <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #e9ecef;">
                <h5 style="color: #6c757d; margin: 0 0 8px 0; font-size: 0.9em;">High Quality Analyses</h5>
                <div style="font-size: 1.8em; font-weight: bold; color: #28a745;">{xai_summary.get('quality_summary', {}).get('excellent', 0):,}</div>
                <small style="color: #6c757d;">IoU > 0.5</small>
            </div>
            <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #e9ecef;">
                <h5 style="color: #6c757d; margin: 0 0 8px 0; font-size: 0.9em;">Complex Patterns</h5>
                <div style="font-size: 1.8em; font-weight: bold; color: #ffc107;">{xai_summary.get('analysis_coverage', {}).get('complex_patterns', 0):,}</div>
                <small style="color: #6c757d;">> 5 components</small>
            </div>
            <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #e9ecef;">
                <h5 style="color: #6c757d; margin: 0 0 8px 0; font-size: 0.9em;">Representative Image</h5>
                <div style="font-size: 1.8em; font-weight: bold; color: #17a2b8;">{xai_summary.get('representative_info', {}).get('representative_images', 0):,}</div>
                <small style="color: #6c757d;">from {xai_summary.get('representative_info', {}).get('total_samples', 0):,} total samples</small>
            </div>
        </div>
    </div>
    """


def generate_xai_summary_for_individual_report(xai_result):
    """개별 보고서용 XAI 요약 정보 HTML 생성 (일단 비워둠)"""
    # TODO: 개별 데이터에 특화된 요약 정보 구현
    return """
    <div style="margin-bottom: 25px; padding: 20px; background: white; border-radius: 8px; border: 1px solid #dee2e6;">
        <h4 style="color: #495057; margin-bottom: 15px; padding-bottom: 8px; border-bottom: 2px solid #ffc107;">📊 Individual Analysis Summary</h4>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px;">
            <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #e9ecef;">
                <h5 style="color: #6c757d; margin: 0 0 8px 0; font-size: 0.9em;">Analysis Status</h5>
                <div style="font-size: 1.8em; font-weight: bold; color: #495057;">Ready</div>
                <small style="color: #6c757d;">Individual Report</small>
            </div>
        </div>
    </div>
    """


def generate_xai_analysis_section(xai_summary, xai_charts):
    """XAI 분석 결과 섹션 HTML 생성 (통합 보고서용)"""
    if not xai_summary and not xai_charts:
        return ""
    
    html_parts = []
    
    # XAI 요약 통계 추가 (통합 보고서용)
    if xai_summary:
        html_parts.append(generate_xai_summary_for_integrated_report(xai_summary))
    
    # XAI 시각화 컨테이너 추가
    if xai_charts:
        html_parts.append(generate_xai_visualizations_container(xai_charts))
    
    return ''.join(html_parts)


