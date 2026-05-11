"""
Audio Analysis Plugin Implementation for ddoc

Provides hookimpl for:
- eda_run: Audio attribute analysis
- drift_detect: Drift detection between baseline and current audio datasets
"""
import os
import yaml
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np

try:
    from ddoc.plugins.hookspecs import hookimpl
except ImportError:
    def hookimpl(func):
        return func

try:
    import librosa
except ImportError:
    librosa = None


class DOCAudioPlugin:
    """Audio Analysis Plugin for ddoc"""

    def _histogram(self, values: list[float], bins: int = 20, max_samples: int = 2000) -> dict[str, Any] | None:
        if not values:
            return None
        counts, edges = np.histogram(values, bins=bins)
        samples = values
        if len(values) > max_samples:
            idx = np.random.choice(len(values), size=max_samples, replace=False)
            samples = [values[i] for i in idx]
        return {"bins": edges.tolist(), "counts": counts.tolist(), "samples": samples}
    
    def _load_ddoc_yaml(self, dataset_path: Path) -> Dict[str, Any]:
        """Load and validate ddoc.yaml from dataset directory"""
        yaml_path = dataset_path / "ddoc.yaml"
        if not yaml_path.exists():
            raise ValueError(f"ddoc.yaml not found in {dataset_path}")
        
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if config.get("modality") != "audio_wave":
            raise ValueError(f"Dataset {dataset_path} is not configured as audio_wave modality")
        
        return config
    
    def _get_audio_files(self, directory: Path) -> list:
        """Get all audio files in directory"""
        audio_extensions = ('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac')
        audio_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(audio_extensions):
                    audio_files.append(Path(root) / file)
        return audio_files
    
    def _analyze_audio_attributes(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Calculate physical-based audio features"""
        if librosa is None:
            print("Warning: librosa not available")
            return None
        
        try:
            y, sr = librosa.load(str(file_path), sr=None, duration=30.0)  # Limit to 30s for speed
            
            # RMS energy
            rms = librosa.feature.rms(y=y)[0]
            rms_mean = float(np.mean(rms))
            rms_std = float(np.std(rms))
            
            # Zero Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            zcr_mean = float(np.mean(zcr))
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            
            # MFCC statistics
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = [float(x) for x in np.mean(mfccs, axis=1)]
            mfcc_std = [float(x) for x in np.std(mfccs, axis=1)]
            
            return {
                'rms_energy_mean': rms_mean,
                'rms_energy_std': rms_std,
                'zcr_mean': zcr_mean,
                'spectral_centroid_mean': float(np.mean(spectral_centroids)),
                'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
                'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
                'mfcc_mean': mfcc_mean,
                'mfcc_std': mfcc_std
            }
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return None
    
    @hookimpl
    def eda_run(self, snapshot_id, data_path, data_hash, output_path, invalidate_cache=False):
        """Run EDA for audio_wave datasets"""
        from ddoc.core.cache_service import get_cache_service
        
        cache_service = get_cache_service()
        input_path = Path(data_path)
        output_path = Path(output_path)
        
        print(f"🚀 Audio EDA Analysis Started")
        print(f"=" * 80)
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        metrics = {
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'snapshot_id': snapshot_id,
            'data_hash': data_hash,
            'modality': 'audio_wave'
        }

        # drift_studio(v2): data_path는 "개별 데이터셋 디렉토리"를 가리킵니다.
        # 따라서 현재 디렉토리의 ddoc.yaml을 기준으로 data_dir에서 오디오 파일을 찾습니다.
        config = self._load_ddoc_yaml(input_path)
        data_dir = config.get("data", {}).get("data_dir", ".")
        dataset_audio_root = (input_path / data_dir).resolve()
        if not dataset_audio_root.exists() or not dataset_audio_root.is_dir():
            raise ValueError(f"audio_wave data_dir not found: {data_dir}")
        
        # Load cache
        attr_cache = {}
        if not invalidate_cache:
            attr_cache_data = cache_service.load_analysis_cache(
                snapshot_id=snapshot_id,
                data_hash=data_hash,
                cache_type="attributes_audio_wave"
            )
            if attr_cache_data:
                attr_cache = attr_cache_data
        
        # input_path를 절대 경로로 변환하여 일관성 유지
        input_path_abs = Path(input_path).resolve()
        
        all_attributes = {}

        audio_files = self._get_audio_files(dataset_audio_root)
        print(f"   Found {len(audio_files)} audio files (root={dataset_audio_root})")

        for audio_file in audio_files:
            # audio_file과 input_path 모두 절대 경로로 변환하여 relative_to 사용
            audio_file_abs = Path(audio_file).resolve()
            rel_path = str(audio_file_abs.relative_to(input_path_abs))
            attrs = self._analyze_audio_attributes(audio_file)
            if attrs:
                all_attributes[rel_path] = attrs
        
        # Save cache
        if all_attributes:
            cache_service.save_analysis_cache(
                snapshot_id=snapshot_id,
                data_hash=data_hash,
                cache_type="attributes_audio_wave",
                data=all_attributes
            )
        
        metrics['num_files'] = len(all_attributes)
        distributions = {}
        if all_attributes:
            for key in ['rms_energy_mean', 'rms_energy_std', 'zcr_mean',
                        'spectral_centroid_mean', 'spectral_bandwidth_mean', 'spectral_rolloff_mean']:
                vals = [a.get(key, 0) for a in all_attributes.values() if key in a]
                hist = self._histogram(vals)
                if hist:
                    distributions[key] = hist
        metrics_file = output_path / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n✅ Audio Analysis Complete")
        
        return {
            "status": "success",
            "modality": "audio_wave",
            "files_analyzed": len(all_attributes),
            "metrics_file": str(metrics_file),
            "summary": metrics,
            "distributions": distributions or None,
        }
    
    @hookimpl
    def drift_detect(
        self,
        snapshot_id_ref: str,
        snapshot_id_cur: str,
        data_path_ref: str,
        data_path_cur: str,
        data_hash_ref: str,
        data_hash_cur: str,
        detector: str,
        cfg: Dict[str, Any],
        output_path: str
    ) -> Optional[Dict[str, Any]]:
        """Detect drift between two audio snapshots"""
        from ddoc.core.cache_service import get_cache_service
        
        cache_service = get_cache_service()
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load caches
        baseline_attr = cfg.get('baseline_cache') or cache_service.load_analysis_cache(
            snapshot_id=snapshot_id_ref,
            data_hash=data_hash_ref,
            cache_type="attributes_audio_wave"
        )
        
        current_attr = cfg.get('current_cache') or cache_service.load_analysis_cache(
            snapshot_id=snapshot_id_cur,
            data_hash=data_hash_cur,
            cache_type="attributes_audio_wave"
        )
        
        if not baseline_attr or not current_attr:
            return None
        
        drift_metrics = {
            'modality': 'audio_wave',
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
        }
        
        # Calculate drift for each metric
        metric_names = ['rms_energy_mean', 'zcr_mean', 'spectral_centroid_mean']
        drift_scores = []
        
        for metric in metric_names:
            ref_values = [a.get(metric, 0) for a in baseline_attr.values() if metric in a]
            cur_values = [a.get(metric, 0) for a in current_attr.values() if metric in a]
            
            if ref_values and cur_values:
                from scipy.stats import wasserstein_distance
                drift = wasserstein_distance(ref_values, cur_values)
                drift_scores.append(drift)
        
        drift_metrics['overall_score'] = float(np.mean(drift_scores)) if drift_scores else 0.0
        
        metrics_file = output_path / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(drift_metrics, f, indent=2)
        
        return drift_metrics

