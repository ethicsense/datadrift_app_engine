import os
import re
from datetime import datetime

class XAIGuidelineGenerator:
    def __init__(self):
        self.guideline_data = self._load_guideline_content()
    
    def _load_guideline_content(self):
        """XAI 가이드라인 내용을 구조화된 데이터로 로드합니다."""
        return {
            'title': 'XAI Report Guideline',
            'sections': [
                {
                    'title': 'CAM 분석 결과 해석',
                    'subsections': [
                        {
                            'title': 'Heatmap',
                            'items': [
                                {
                                    'name': 'original image',
                                    'description': '분석 대상 이미지 원본입니다.'
                                },
                                {
                                    'name': 'CAM overlay image',
                                    'description': '전체 활성화 데이터 heatmap을 원본에 겹쳐서 표현한 이미지입니다.\n\n이미지 내 활성 영역의 정확한 위치를 확인할 수 있습니다.'
                                },
                                {
                                    'name': 'activation distribution',
                                    'description': '전체 활성화 데이터의 수치와 빈도에 따른 분포 히스토그램입니다.'
                                }
                            ]
                        },
                        {
                            'title': 'Statistics',
                            'items': [
                                {
                                    'name': 'cam value distribution',
                                    'description': '활성화 수치에 대한 분포 히스토그램입니다.',
                                    'sub_description': '하늘색 막대 그래프 : q75 - q25 범위에 해당하는 값\n\n주황색 선 : 중앙값\n\n흰색 점 : 이상치 (최대 활성화 값)\n\n평균, 표준편차, 범위 표기'
                                },
                                {
                                    'name': 'cam activation percentile',
                                    'description': '활성화된 CAM 값의 백분위수를 측정합니다. (0 값 제외)\n전체 활성 분포 백분위에 해당하는 값 표기\n\n분포의 치우침 정도를 알 수 있습니다.'
                                },
                                {
                                    'name': 'activation ratio w/ threshold',
                                    'description': '임계값 별 활성화 비율을 나타냅니다.\n\n임계값에 따른 변화율을 관측할 수 있습니다.'
                                },
                                {
                                    'name': 'skewness analysis',
                                    'description': '비대칭도를 분석합니다. (정규분포 기준 얼마나 벗어났는지 측정)\n\n수식 : `평균((값 - 평균) / 표준편차)³`',
                                    'sub_items': [
                                        {
                                            'name': 'skewness 해석',
                                            'items': [
                                                '오른꼬리 분포 (양수) : 낮은 활성치의 비율이 더 높음',
                                                '왼꼬리 분포 (음수) : 높은 활성치의 비율이 더 높음',
                                                '0 : 대칭 (정규분포)'
                                            ]
                                        },
                                        {
                                            'name': 'percentile skewness',
                                            'description': 'q1, q2, q3 값을 활용한 비대칭도 측정 (백분위수 기준 25% ~ 75% 값)\n\n극단값에 덜 민감하고 안정적인 분포 특성을 보여줍니다.'
                                        }
                                    ]
                                },
                                {
                                    'name': 'quality metrics',
                                    'description': '데이터 품질 관련 지표를 측정합니다.',
                                    'sub_items': [
                                        {
                                            'name': 'concentration (집중도)',
                                            'description': '`(max_val - mean_val) / (max_val - min_val)`\n\n수치가 낮을수록 분산적, 높을수록 집중적'
                                        },
                                        {
                                            'name': 'uniformity (균등성)',
                                            'description': '`1 - (std_val / max_val)`\n\n낮을수록 불규칙적, 높을수록 활성화 값이 균등하게 분포'
                                        },
                                        {
                                            'name': 'confidence (신뢰도)',
                                            'description': '`(high_activation_ratio / 100) * (mean_val / max_val)`'
                                        }
                                    ]
                                }
                            ]
                        },
                        {
                            'title': 'Threshold',
                            'description': '임계값 기반으로 활성화 데이터에 대한 스레스홀드 적용 후 활성화 영역을 관찰할 수 있습니다.\n\n활성화가 강하게 일어나는 영역을 특정할 수 있습니다.',
                            'items': [
                                {
                                    'name': 'thresholds range',
                                    'description': 'p80, p85, p90, p95'
                                }
                            ]
                        },
                        {
                            'title': 'Clustering',
                            'description': '임계값 기반 스레스홀드 적용 후 활성화 영역에 대한 클러스터링을 수행합니다.\n\n가장 많은 클러스터가 관측되는 임계값에 대한 결과를 출력합니다.'
                        },
                        {
                            'title': 'Entropy',
                            'description': '활성화 데이터에 대한 엔트로피를 측정합니다.',
                            'items': [
                                {
                                    'name': 'histogram entropy',
                                    'code': 'hist, _ = np.histogram(non_zero_values, bins=50, density=True)\nentropy_val = entropy(hist + 1e-10)  # 0 방지',
                                    'description': '엔트로피가 낮을수록 집중적, 높을수록 분산적인 분포\n\n최소 활성값이 높을수록 노이즈 작다고 판단할 수 있음\n\n분포 모양에 따라 오른꼬리/왼꼬리 분포로 나누어 판단할 수 있음'
                                },
                                {
                                    'name': 'spatial entropy',
                                    'code': '# 그래디언트 계산 (중앙 차분 방식)\ngx = np.zeros_like(cam)\ngy = np.zeros_like(cam)\n\n# 수평 그래디언트 (Gx) - 중앙 차분\ngx[:, 1:-1] = (cam[:, 2:] - cam[:, :-2]) / 2.0\n\n# 수직 그래디언트 (Gy) - 중앙 차분\ngy[1:-1, :] = (cam[2:, :] - cam[:-2, :]) / 2.0\n\n# 그래디언트 크기 (magnitude)\ngradient_magnitude = np.sqrt(gx**2 + gy**2)\n\n# 각 방향별 엔트로피 계산\ngx_ent = shannon_entropy((gx * 255).astype(np.uint8))\ngy_ent = shannon_entropy((gy * 255).astype(np.uint8))\nmagnitude_ent = shannon_entropy((gradient_magnitude * 255).astype(np.uint8))\n\n# 평균 공간 엔트로피 (그래디언트 크기에 더 높은 가중치)\nspatial_ent = 0.4 * gx_ent + 0.4 * gy_ent + 0.2 * magnitude_ent',
                                    'description': '엔트로피가 낮을수록 단순한 패턴과 특정 값 집중, 높을수록 균등한 활성 분포와 복잡한 패턴\n\n붉은색 부분이 활성화 값이 급격히 변하는 영역\n\n패턴의 종류 : 중앙 집중형(핫스팟) 패턴, 선형 패턴, 대칭 패턴'
                                },
                                {
                                    'name': 'conditional entropy',
                                    'description': '임계값 이상 활성치 데이터에 대한 조건부 엔트로피 계산\n\n영역이 좁아짐에 따라 자연스럽게 엔트로피 상승',
                                    'code': 'threshold_val = np.percentile(cam, thresh)\nactive_mask = cam > threshold_val\ninactive_mask = cam <= threshold_val\n\nactive_ratio = np.sum(active_mask) / cam.size\n\nconditional_ent = active_ratio * active_ent + (1 - active_ratio) * inactive_ent',
                                    'sub_description': '임계값이 높을수록 비활성 영역 커짐, 범위 증가에 따른 가중 비율 상승 ⇒ 엔트로피 상승\n\n활성화 수치의 분포, 영역의 분포 특성에 따라 엔트로피가 오히려 감소하거나 변화율이 급변할 수 있음'
                                }
                            ]
                        },
                        {
                            'title': 'Centroid',
                            'items': [
                                {
                                    'name': 'max centroid',
                                    'code': 'confidence = max_activation / theoretical_max\n# 신뢰도 수치 자체는 대체로 높을 수 밖에 없음',
                                    'description': '가장 높은 활성화 값을 가진 단일 픽셀 위치\n\nspurious correlation 에 취약함'
                                },
                                {
                                    'name': 'components centroid',
                                    'code': 'confidence = component_strength * component_size\n# 연결된 컴포넌트의 크기와 강도',
                                    'description': '여러 활성화 영역을 고려한 중심점\n\n안정적이고 강건한 방식'
                                },
                                {
                                    'name': 'weighted centroid',
                                    'code': 'confidence = mean_activation / max_activation if max_activation > 0 else 0\n# 전체 활성화 강도 대비 평균 활성화 강도',
                                    'description': '가중 평균 기반 중심점\n\n활성화 패턴이 분산적일 경우 신뢰도 하락'
                                },
                                {
                                    'name': 'threshold centroid',
                                    'code': 'confidence = active_ratio * active_intensity\n# 임계값 이상 활성화된 영역의 비율과 강도',
                                    'description': '임계값 기반 중심점\n\n영역 위치에 대한 고려 없음'
                                },
                                {
                                    'name': 'coordinate distribution',
                                    'description': '계산 방법에 따른 중심 좌표의 분포 분석',
                                    'sub_description': '하늘색 막대 (x coordinate value) : x 좌표의 범위\n\n분홍색 막대 (y coordinate value) : y 좌표의 범위\n\n빈도 (frequency) : 각 범위에 해당하는 중심 좌표의 갯수 (최대 4)'
                                }
                            ]
                        },
                        {
                            'title': 'Overlap',
                            'items': [
                                {
                                    'name': 'IoU',
                                    'description': 'Intersection / Union\n\n활성 영역과 GT bbox의 겹침 정도'
                                },
                                {
                                    'name': 'CAM coverage',
                                    'code': 'cam_coverage = intersection_area / bbox_area if bbox_area > 0 else 0\n# gt bbox 영역 중 cam 영역과 겹치는 비율',
                                    'description': 'bbox 내에서 cam 활성 영역이 차지하는 비율'
                                },
                                {
                                    'name': 'BBOX coverage',
                                    'code': 'bbox_coverage = intersection_area / cam_active_area if cam_active_area > 0 else 0\n# cam 영역 중 gt bbox와 겹치는 비율',
                                    'description': 'cam 영역 내에서 bbox가 차지하는 비율'
                                }
                            ]
                        }
                    ]
                }
            ]
        }
    
    def generate_html_guideline(self):
        """XAI 가이드라인을 HTML 형태로 생성합니다."""
        html_parts = []
        
        # 헤더
        html_parts.append(f"""
        <div style="max-width: 1200px; margin: 0 auto; padding: 20px; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 15px; margin-bottom: 30px; text-align: center;">
                <h1 style="margin: 0; font-size: 2.5em; font-weight: 300;">{self.guideline_data['title']}</h1>
            </div>
        """)
        
        # 목차
        html_parts.append("""
        <div style="background: #f8f9fa; padding: 25px; border-radius: 10px; margin-bottom: 30px; border-left: 5px solid #667eea;">
            <h2 style="color: #495057; margin-top: 0; margin-bottom: 20px;">📚 목차</h2>
            <ul style="list-style: none; padding: 0; margin: 0;">
        """)
        
        for section in self.guideline_data['sections']:
            html_parts.append(f"""
                <li style="margin-bottom: 10px;">
                    <a href="#section-{section['title'].replace(' ', '-').lower()}" 
                       style="color: #667eea; text-decoration: none; font-weight: 500; display: block; padding: 8px 15px; border-radius: 5px; transition: background-color 0.3s;"
                       onmouseover="this.style.backgroundColor='#e9ecef'" 
                       onmouseout="this.style.backgroundColor='transparent'">
                        📊 {section['title']}
                    </a>
                </li>
            """)
            
            if 'subsections' in section:
                for i, subsection in enumerate(section['subsections']):
                    html_parts.append(f"""
                        <li style="margin-bottom: 5px; margin-left: 20px;">
                            <a href="#subsection-{section['title'].replace(' ', '-').lower()}-{i}" 
                               style="color: #6c757d; text-decoration: none; font-size: 0.9em; display: block; padding: 5px 15px; border-radius: 5px; transition: background-color 0.3s;"
                               onmouseover="this.style.backgroundColor='#e9ecef'" 
                               onmouseout="this.style.backgroundColor='transparent'">
                                🔍 {subsection['title']}
                            </a>
                        </li>
                    """)
        
        html_parts.append("""
            </ul>
        </div>
        """)
        
        # 메인 콘텐츠
        for section in self.guideline_data['sections']:
            html_parts.append(f"""
            <div id="section-{section['title'].replace(' ', '-').lower()}" style="margin-bottom: 40px;">
                <h2 style="color: #495057; margin-bottom: 25px; padding-bottom: 10px; border-bottom: 3px solid #667eea; font-size: 2em;">
                    📊 {section['title']}
                </h2>
            """)
            
            if 'subsections' in section:
                for i, subsection in enumerate(section['subsections']):
                    html_parts.append(f"""
                    <div id="subsection-{section['title'].replace(' ', '-').lower()}-{i}" style="margin-bottom: 30px; padding: 25px; background: white; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); border-left: 5px solid #28a745;">
                        <h3 style="color: #495057; margin-top: 0; margin-bottom: 20px; font-size: 1.5em;">
                            🔍 {subsection['title']}
                        </h3>
                    """)
                    
                    # 설명이 있는 경우
                    if 'description' in subsection:
                        # 줄바꿈을 <br> 태그로 변환하고 코드 블록 처리
                        section_description_html = subsection['description'].replace('\n\n', '</p><p style="margin: 0; color: #495057; font-size: 1.1em;">').replace('\n', '<br>')
                        # 백틱으로 감싸진 코드를 인라인 코드 스타일로 변환
                        section_description_html = re.sub(r'`([^`]+)`', r'<code style="background: #f8f9fa; padding: 2px 4px; border-radius: 3px; font-family: monospace; color: #e83e8c;">\1</code>', section_description_html)
                        html_parts.append(f"""
                        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 4px solid #17a2b8;">
                            <p style="margin: 0; color: #495057; font-size: 1.1em;">{section_description_html}</p>
                        </div>
                        """)
                    
                    # 아이템들이 있는 경우
                    if 'items' in subsection:
                        for item in subsection['items']:
                            html_parts.append(f"""
                            <div style="margin-bottom: 20px; padding: 20px; background: #f8f9fa; border-radius: 8px; border: 1px solid #e9ecef;">
                                <h4 style="color: #495057; margin-top: 0; margin-bottom: 15px; font-size: 1.2em; display: flex; align-items: center;">
                                    <span style="background: #667eea; color: white; padding: 3px 8px; border-radius: 15px; font-size: 0.8em; margin-right: 10px;">📌</span>
                                    {item['name']}
                                </h4>
                            """)
                            
                            # 설명
                            if 'description' in item:
                                # 줄바꿈을 <br> 태그로 변환하고 코드 블록 처리
                                description_html = item['description'].replace('\n\n', '</p><p style="color: #6c757d; margin-bottom: 15px; line-height: 1.6;">').replace('\n', '<br>')
                                # 백틱으로 감싸진 코드를 인라인 코드 스타일로 변환
                                description_html = re.sub(r'`([^`]+)`', r'<code style="background: #f8f9fa; padding: 2px 4px; border-radius: 3px; font-family: monospace; color: #e83e8c;">\1</code>', description_html)
                                html_parts.append(f"""
                                <p style="color: #6c757d; margin-bottom: 15px; line-height: 1.6;">{description_html}</p>
                                """)
                            
                            # 코드가 있는 경우
                            if 'code' in item:
                                html_parts.append(f"""
                                <div style="background: #2d3748; color: #e2e8f0; padding: 15px; border-radius: 8px; margin: 15px 0; font-family: 'Courier New', monospace; font-size: 0.9em; overflow-x: auto;">
                                    <pre style="margin: 0; white-space: pre-wrap;">{item['code']}</pre>
                                </div>
                                """)
                            
                            # 서브 설명이 있는 경우
                            if 'sub_description' in item:
                                # 줄바꿈을 <br> 태그로 변환
                                sub_desc_html = item['sub_description'].replace('\n\n', '</p><p style="margin: 0; color: #856404; font-style: italic;">💡 ').replace('\n', '<br>')
                                html_parts.append(f"""
                                <div style="background: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 8px; margin-top: 15px;">
                                    <p style="margin: 0; color: #856404; font-style: italic;">💡 {sub_desc_html}</p>
                                </div>
                                """)
                            
                            # 서브 아이템들이 있는 경우
                            if 'sub_items' in item:
                                html_parts.append("""
                                <div style="margin-top: 15px;">
                                """)
                                
                                for sub_item in item['sub_items']:
                                    if isinstance(sub_item, dict):
                                        if 'name' in sub_item:
                                            html_parts.append(f"""
                                            <div style="margin-bottom: 10px;">
                                                <h5 style="color: #495057; margin-bottom: 8px; font-size: 1em;">
                                                    <span style="background: #ffc107; color: #212529; padding: 2px 6px; border-radius: 10px; font-size: 0.8em; margin-right: 8px;">🔹</span>
                                                    {sub_item['name']}
                                                </h5>
                                            """)
                                            
                                            if 'description' in sub_item:
                                                # 줄바꿈을 <br> 태그로 변환하고 코드 블록 처리
                                                sub_description_html = sub_item['description'].replace('\n\n', '</p><p style="color: #6c757d; margin-bottom: 10px; margin-left: 20px; font-size: 0.95em;">').replace('\n', '<br>')
                                                # 백틱으로 감싸진 코드를 인라인 코드 스타일로 변환
                                                sub_description_html = re.sub(r'`([^`]+)`', r'<code style="background: #f8f9fa; padding: 2px 4px; border-radius: 3px; font-family: monospace; color: #e83e8c;">\1</code>', sub_description_html)
                                                html_parts.append(f"""
                                                <p style="color: #6c757d; margin-bottom: 10px; margin-left: 20px; font-size: 0.95em;">{sub_description_html}</p>
                                                """)
                                            
                                            if 'items' in sub_item:
                                                html_parts.append("""
                                                <ul style="margin-left: 20px; color: #6c757d;">
                                                """)
                                                for sub_sub_item in sub_item['items']:
                                                    html_parts.append(f"""
                                                    <li style="margin-bottom: 5px;">{sub_sub_item}</li>
                                                    """)
                                                html_parts.append("</ul>")
                                            
                                            html_parts.append("</div>")
                                    else:
                                        # 단순 문자열인 경우
                                        html_parts.append(f"""
                                        <div style="margin-bottom: 8px; margin-left: 20px;">
                                            <span style="color: #6c757d;">• {sub_item}</span>
                                        </div>
                                        """)
                                
                                html_parts.append("</div>")
                            
                            html_parts.append("</div>")
                    
                    html_parts.append("</div>")
            
            html_parts.append("</div>")
        
        # 푸터
        html_parts.append(f"""
        <div style="background: #f8f9fa; padding: 30px; border-radius: 10px; text-align: center; margin-top: 40px; border-top: 3px solid #667eea;">
            <p style="margin: 0; color: #6c757d; font-size: 0.9em;">
                📄 이 가이드라인은 XAI 분석 결과를 해석하는 데 도움을 주기 위해 작성되었습니다.<br>
                🔄 최종 업데이트: {datetime.now().strftime('%Y년 %m월 %d일 %H:%M')}
            </p>
        </div>
        </div>
        """)
        
        return ''.join(html_parts)
    
    def save_guideline_html(self, output_path):
        """가이드라인을 HTML 파일로 저장합니다."""
        html_content = self.generate_html_guideline()
        
        # 완전한 HTML 문서 구조
        full_html = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>XAI Report Guideline</title>
    <style>
        body {{
            margin: 0;
            padding: 0;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 30px;
            text-align: center;
        }}
        
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        
        .header .meta {{
            margin-top: 15px;
            opacity: 0.9;
        }}
        
        .header .meta span {{
            background: rgba(255,255,255,0.2);
            padding: 5px 15px;
            border-radius: 20px;
            margin-right: 15px;
        }}
        
        .toc {{
            background: #f8f9fa;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 30px;
            border-left: 5px solid #667eea;
        }}
        
        .toc h2 {{
            color: #495057;
            margin-top: 0;
            margin-bottom: 20px;
        }}
        
        .toc ul {{
            list-style: none;
            padding: 0;
            margin: 0;
        }}
        
        .toc li {{
            margin-bottom: 10px;
        }}
        
        .toc a {{
            color: #667eea;
            text-decoration: none;
            font-weight: 500;
            display: block;
            padding: 8px 15px;
            border-radius: 5px;
            transition: background-color 0.3s;
        }}
        
        .toc a:hover {{
            background-color: #e9ecef;
        }}
        
        .toc li li {{
            margin-bottom: 5px;
            margin-left: 20px;
        }}
        
        .toc li li a {{
            color: #6c757d;
            font-size: 0.9em;
            padding: 5px 15px;
        }}
        
        .section {{
            margin-bottom: 40px;
        }}
        
        .section h2 {{
            color: #495057;
            margin-bottom: 25px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
            font-size: 2em;
        }}
        
        .subsection {{
            margin-bottom: 30px;
            padding: 25px;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-left: 5px solid #28a745;
        }}
        
        .subsection h3 {{
            color: #495057;
            margin-top: 0;
            margin-bottom: 20px;
            font-size: 1.5em;
        }}
        
        .description {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 4px solid #17a2b8;
        }}
        
        .description p {{
            margin: 0;
            color: #495057;
            font-size: 1.1em;
        }}
        
        .item {{
            margin-bottom: 20px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
            border: 1px solid #e9ecef;
        }}
        
        .item h4 {{
            color: #495057;
            margin-top: 0;
            margin-bottom: 15px;
            font-size: 1.2em;
            display: flex;
            align-items: center;
        }}
        
        .item h4 span {{
            background: #667eea;
            color: white;
            padding: 3px 8px;
            border-radius: 15px;
            font-size: 0.8em;
            margin-right: 10px;
        }}
        
        .item p {{
            color: #6c757d;
            margin-bottom: 15px;
            line-height: 1.6;
        }}
        
        .code-block {{
            background: #2d3748;
            color: #e2e8f0;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            overflow-x: auto;
        }}
        
        .code-block pre {{
            margin: 0;
            white-space: pre-wrap;
        }}
        
        .note {{
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
        }}
        
        .note p {{
            margin: 0;
            color: #856404;
            font-style: italic;
        }}
        
        .sub-item {{
            margin-bottom: 10px;
        }}
        
        .sub-item h5 {{
            color: #495057;
            margin-bottom: 8px;
            font-size: 1em;
        }}
        
        .sub-item h5 span {{
            background: #ffc107;
            color: #212529;
            padding: 2px 6px;
            border-radius: 10px;
            font-size: 0.8em;
            margin-right: 8px;
        }}
        
        .sub-item p {{
            color: #6c757d;
            margin-bottom: 10px;
            margin-left: 20px;
            font-size: 0.95em;
        }}
        
        .sub-item ul {{
            margin-left: 20px;
            color: #6c757d;
        }}
        
        .sub-item li {{
            margin-bottom: 5px;
        }}
        
        .simple-item {{
            margin-bottom: 8px;
            margin-left: 20px;
        }}
        
        .simple-item span {{
            color: #6c757d;
        }}
        
        .footer {{
            background: #f8f9fa;
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            margin-top: 40px;
            border-top: 3px solid #667eea;
        }}
        
        .footer p {{
            margin: 0;
            color: #6c757d;
            font-size: 0.9em;
        }}
        
        @media (max-width: 768px) {{
            .container {{
                padding: 10px;
            }}
            
            .header h1 {{
                font-size: 2em;
            }}
            
            .header .meta span {{
                display: block;
                margin-bottom: 10px;
                margin-right: 0;
            }}
            
            .subsection {{
                padding: 15px;
            }}
            
            .item {{
                padding: 15px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        {html_content}
    </div>
</body>
</html>
        """
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_html)
        
        print(f"✅ XAI 가이드라인이 성공적으로 생성되었습니다: {output_path}")

def create_xai_guideline(output_path="xai_guideline.html"):
    """XAI 가이드라인을 생성하고 저장합니다."""
    generator = XAIGuidelineGenerator()
    generator.save_guideline_html(output_path)
    return output_path

if __name__ == "__main__":
    # 현재 디렉토리에 가이드라인 생성
    output_file = create_xai_guideline()
    print(f"📄 가이드라인 파일이 생성되었습니다: {output_file}")
