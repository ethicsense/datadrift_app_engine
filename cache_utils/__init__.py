"""
캐시 관리 유틸리티 패키지

이 패키지는 데이터셋별 캐시 관리를 위한 유틸리티들을 제공합니다.
"""

from .cache_manager import (
    CacheManager,
    get_cache_manager,
    get_cached_html_content,
    get_cached_analysis_data,
    save_analysis_data,
    global_cache_manager
)

__all__ = [
    'CacheManager',
    'get_cache_manager',
    'get_cached_html_content',
    'get_cached_analysis_data',
    'save_analysis_data',
    'global_cache_manager'
] 