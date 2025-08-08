import os
import json
import hashlib
import pickle
from datetime import datetime, timedelta
from pathlib import Path

'''
dataset_folder/
├── cache/
│   ├── analysis_image_analysis_test_data2.cache
│   ├── analysis_image_analysis_test_data2_meta.json
│   ├── analysis_image_drift_content_test_data2.cache
│   ├── analysis_image_drift_content_test_data2_meta.json
│   ├── analysis_xai_analysis_test_data2.cache
│   └── analysis_xai_analysis_test_data2_meta.json
'''

class CacheManager:
    def __init__(self, dataset_directory=None):
        """캐시 매니저 초기화"""
        if dataset_directory is None:
            # 기본 캐시 디렉토리 (전역 캐시용)
            cache_dir = os.getenv('DATADRIFT_CACHE_DIR', 'cache')
            self.cache_dir = Path(cache_dir)
        else:
            # 데이터셋별 캐시 디렉토리
            self.cache_dir = Path(dataset_directory) / 'cache'
        
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # 캐시 설정
        self.cache_expiry_days = int(os.getenv('DATADRIFT_CACHE_EXPIRY_DAYS', '30'))
        self.max_cache_size_mb = int(os.getenv('DATADRIFT_CACHE_MAX_SIZE_MB', '999999'))
    
    def _get_cache_key(self, identifier, content_type="html"):
        """캐시 키 생성"""
        # 식별자를 안전한 파일명으로 변환
        safe_identifier = self._make_safe_filename(identifier)
        
        # 파일명 길이 제한 (너무 길어지지 않도록)
        if len(safe_identifier) > 50:
            # 해시를 사용하여 짧게 만들기
            hash_obj = hashlib.md5(identifier.encode())
            safe_identifier = f"{safe_identifier[:30]}_{hash_obj.hexdigest()[:8]}"
        
        return f"{content_type}_{safe_identifier}"
    
    def _make_safe_filename(self, filename):
        """파일명을 안전하게 만듭니다."""
        # 특수문자 제거 및 공백을 언더스코어로 변경
        import re
        safe_name = re.sub(r'[^\w\-_.]', '_', filename)
        # 연속된 언더스코어를 하나로 변경
        safe_name = re.sub(r'_+', '_', safe_name)
        # 앞뒤 언더스코어 제거
        safe_name = safe_name.strip('_')
        return safe_name
    
    def _get_cache_path(self, cache_key):
        """캐시 파일 경로 반환"""
        return self.cache_dir / f"{cache_key}.cache"
    
    def _get_metadata_path(self, cache_key):
        """메타데이터 파일 경로 반환"""
        return self.cache_dir / f"{cache_key}_meta.json"
    
    def _is_cache_valid(self, metadata_path):
        """캐시가 유효한지 확인"""
        if not metadata_path.exists():
            return False
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # 만료 시간 확인
            created_time = datetime.fromisoformat(metadata['created_time'])
            expiry_time = created_time + timedelta(days=self.cache_expiry_days)
            
            return datetime.now() < expiry_time
        except Exception:
            return False
    
    def _save_metadata(self, metadata_path, content_size, content_type="html"):
        """메타데이터 저장"""
        metadata = {
            'created_time': datetime.now().isoformat(),
            'content_size': content_size,
            'content_type': content_type,
            'expiry_days': self.cache_expiry_days
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _cleanup_expired_cache(self):
        """만료된 캐시 정리"""
        for cache_file in self.cache_dir.glob("*.cache"):
            metadata_path = cache_file.with_name(f"{cache_file.stem}_meta.json")
            if not self._is_cache_valid(metadata_path):
                try:
                    cache_file.unlink()
                    if metadata_path.exists():
                        metadata_path.unlink()
                except Exception as e:
                    print(f"캐시 정리 중 오류: {e}")
    
    def _check_cache_size(self):
        """캐시 크기 확인 및 정리"""
        total_size = 0
        cache_files = []
        
        for cache_file in self.cache_dir.glob("*.cache"):
            size = cache_file.stat().st_size
            total_size += size
            cache_files.append((cache_file, size))
        
        # MB로 변환
        total_size_mb = total_size / (1024 * 1024)
        
        if total_size_mb > self.max_cache_size_mb:
            print(f"⚠️  캐시 크기 초과: {total_size_mb:.2f}MB > {self.max_cache_size_mb}MB")
            print(f"📁 캐시 파일들:")
            for cache_file, size in cache_files:
                try:
                    metadata_path = cache_file.with_name(f"{cache_file.stem}_meta.json")
                    cache_file.unlink()
                    if metadata_path.exists():
                        metadata_path.unlink()
                    
                    total_size_mb -= size / (1024 * 1024)
                    if total_size_mb <= self.max_cache_size_mb * 0.8:  # 80%까지 정리
                        break
                except Exception as e:
                    print(f"캐시 크기 정리 중 오류: {e}")
    
    def get_cached_content(self, identifier, content_type="html"):
        """캐시된 컨텐츠 가져오기"""
        cache_key = self._get_cache_key(identifier, content_type)
        cache_path = self._get_cache_path(cache_key)
        metadata_path = self._get_metadata_path(cache_key)
        
        # 캐시가 존재하고 유효한지 확인
        if cache_path.exists() and self._is_cache_valid(metadata_path):
            try:
                with open(cache_path, 'rb') as f:
                    content = pickle.load(f)
                return content
            except Exception as e:
                print(f"캐시 읽기 오류: {e}")
                return None
        
        return None
    
    def save_cached_content(self, identifier, content, content_type="html"):
        """컨텐츠를 캐시에 저장"""
        cache_key = self._get_cache_key(identifier, content_type)
        cache_path = self._get_cache_path(cache_key)
        metadata_path = self._get_metadata_path(cache_key)
        
        try:
            print(f"💾 캐시 저장 중: {cache_key}")
            
            # 컨텐츠를 pickle로 저장
            with open(cache_path, 'wb') as f:
                pickle.dump(content, f)
            
            # 저장된 파일 크기 확인
            file_size = cache_path.stat().st_size
            file_size_mb = file_size / (1024 * 1024)
            print(f"📁 저장된 파일 크기: {file_size_mb:.2f}MB")
            
            # 메타데이터 저장
            content_size = len(str(content).encode('utf-8'))
            self._save_metadata(metadata_path, content_size, content_type)
            
            # 캐시 크기 확인 및 정리 (저장 후에만)
            self._check_cache_size()
            
            return True
        except Exception as e:
            print(f"❌ 캐시 저장 오류: {e}")
            return False
    
    def invalidate_cache(self, identifier, content_type="html"):
        """특정 캐시 무효화"""
        cache_key = self._get_cache_key(identifier, content_type)
        cache_path = self._get_cache_path(cache_key)
        metadata_path = self._get_metadata_path(cache_key)
        
        try:
            if cache_path.exists():
                cache_path.unlink()
            if metadata_path.exists():
                metadata_path.unlink()
            return True
        except Exception as e:
            print(f"캐시 무효화 오류: {e}")
            return False
    
    def clear_all_cache(self):
        """모든 캐시 정리"""
        try:
            for cache_file in self.cache_dir.glob("*"):
                cache_file.unlink()
            return True
        except Exception as e:
            print(f"전체 캐시 정리 오류: {e}")
            return False
    
    def get_cache_info(self):
        """캐시 정보 반환"""
        cache_files = list(self.cache_dir.glob("*.cache"))
        total_size = sum(f.stat().st_size for f in cache_files)
        total_size_mb = total_size / (1024 * 1024)
        
        # 캐시 파일 상세 정보
        cache_details = []
        for cache_file in cache_files:
            metadata_path = cache_file.with_name(f"{cache_file.stem}_meta.json")
            file_info = {
                'filename': cache_file.name,
                'size_mb': round(cache_file.stat().st_size / (1024 * 1024), 3),
                'created_time': None,
                'content_type': None
            }
            
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    file_info['created_time'] = metadata.get('created_time')
                    file_info['content_type'] = metadata.get('content_type')
                except Exception:
                    pass
            
            cache_details.append(file_info)
        
        return {
            'total_files': len(cache_files),
            'total_size_mb': round(total_size_mb, 2),
            'max_size_mb': self.max_cache_size_mb,
            'expiry_days': self.cache_expiry_days,
            'cache_dir': str(self.cache_dir.absolute()),
            'cache_details': cache_details
        }

# 전역 캐시 매니저 인스턴스 (기본 캐시용)
global_cache_manager = CacheManager()

def get_cache_manager(dataset_directory=None):
    """데이터셋별 캐시 매니저 반환"""
    if dataset_directory:
        return CacheManager(dataset_directory)
    else:
        return global_cache_manager

def get_cached_html_content(identifier, generator_func, *args, dataset_directory=None):
    """HTML 컨텐츠를 캐시에서 가져오거나 생성"""
    cache_manager = get_cache_manager(dataset_directory)
    
    # 캐시에서 확인
    cached_content = cache_manager.get_cached_content(identifier, "html")
    if cached_content is not None:
        return cached_content
    
    # 캐시에 없으면 생성
    try:
        content = generator_func(*args)
        cache_manager.save_cached_content(identifier, content, "html")
        return content
    except Exception as e:
        # 에러가 발생하면 캐시에 저장하지 않고 예외를 다시 발생시킴
        raise e


def get_cached_analysis_data(directory, data_type="image_analysis"):
    """분석 데이터를 캐시에서 가져오거나 생성"""
    cache_manager = get_cache_manager(directory)
    
    # 디렉토리명을 포함한 더 구체적인 식별자 생성
    dir_name = os.path.basename(directory)
    identifier = f"{data_type}_{dir_name}"
    
    # 캐시에서 확인
    cached_data = cache_manager.get_cached_content(identifier, "analysis")
    if cached_data is not None:
        return cached_data
    
    return None

def save_analysis_data(directory, data, data_type="image_analysis"):
    """분석 데이터를 캐시에 저장"""
    cache_manager = get_cache_manager(directory)
    
    # 디렉토리명을 포함한 더 구체적인 식별자 생성
    dir_name = os.path.basename(directory)
    identifier = f"{data_type}_{dir_name}"
    
    return cache_manager.save_cached_content(identifier, data, "analysis")

 