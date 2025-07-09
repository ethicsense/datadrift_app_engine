from bs4 import BeautifulSoup
from datetime import datetime
import os

# 캐시 매니저 import
from cache_utils.cache_manager import get_cache_manager, get_cached_html_content, get_cached_image_analysis_html, get_cached_analysis_data, save_analysis_data

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
    
    # 이미지 분석 리포트 생성
    if dataset_directory:
        try:
            image_analysis_content = get_cached_image_analysis_html(dataset_directory)
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
    <div class="section">
        <div class="section-title">🖼️ Image Attributes Analysis Results</div>
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
                                    margin: 25px 0; padding: 20px; 
                                    border: 1px solid #e9ecef; border-radius: 8px;
                                    background: white;
                                }}
                                .section-title {{ 
                                    font-size: 1.4em; color: #495057; 
                                    margin-bottom: 15px; padding-bottom: 8px;
                                    border-bottom: 2px solid #dee2e6;
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