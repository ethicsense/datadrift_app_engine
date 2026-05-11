"""
Text Analysis Plugin Implementation for ddoc

Provides hookimpl for:
- eda_run: Text attribute analysis, CLIP text embedding extraction
- drift_detect: Drift detection between baseline and current text datasets
"""
import os
import yaml
import json
import re
import zipfile
import shutil
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd

from ddoc.core.embedding_store import load_embeddings, save_embeddings
try:
    from ddoc.plugins.hookspecs import hookimpl
except ImportError:
    def hookimpl(func):
        return func

try:
    import torch
    import clip
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    try:
        from kiwipiepy import Kiwi
    except ImportError:
        Kiwi = None
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    # NLTK>=3.8에서 분리된 리소스(환경에 따라 punkt_tab이 필요)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        try:
            nltk.download('punkt_tab', quiet=True)
        except Exception:
            # 빌드/런타임에서 네트워크가 막혀있을 수 있으므로 무시(폴백 토크나이저 사용)
            pass
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
except ImportError as e:
    print(f"Warning: Some dependencies not available: {e}")


class DOCTextPlugin:
    """Text Analysis Plugin for ddoc"""
    
    def __init__(self):
        self.clip_model = None
        self.clip_tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._kiwi = None
    
    def _load_clip_model(self):
        """Load CLIP model for text encoding"""
        if self.clip_model is None:
            print(f"Loading CLIP model (device: {self.device})...")
            self.clip_model, _ = clip.load("ViT-B/16", device=self.device)
            self.clip_tokenizer = clip.tokenize
            print("CLIP model loaded")

    def _load_kiwi(self):
        """Load Kiwi tokenizer for Korean tokenization (best-effort)."""
        if self._kiwi is not None:
            return self._kiwi
        if Kiwi is None:
            return None
        try:
            self._kiwi = Kiwi()
        except Exception:
            self._kiwi = None
        return self._kiwi
    
    def _load_ddoc_yaml(self, dataset_path: Path) -> Dict[str, Any]:
        """Load and validate ddoc.yaml from dataset directory"""
        yaml_path = dataset_path / "ddoc.yaml"
        if not yaml_path.exists():
            raise ValueError(f"ddoc.yaml not found in {dataset_path}. Text datasets require ddoc.yaml with modality and schema definition.")
        
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        if not isinstance(config, dict):
            raise ValueError("ddoc.yaml must be a mapping (YAML object)")
        
        if config.get('modality') != 'text':
            raise ValueError(f"Dataset {dataset_path} is not configured as text modality")
        
        # ------------------------------------------------------------------
        # DriftStudio v2 ddoc.yaml 호환
        # - v1(legacy): text_column, id_column, language
        # - v2(spec):   data: { csv, columns }
        # ------------------------------------------------------------------
        if 'text_column' not in config:
            data = config.get("data") or {}
            cols = data.get("columns")
            if isinstance(cols, str) and cols.strip():
                config["text_column"] = cols.strip()
                config["_text_columns"] = [cols.strip()]
            elif isinstance(cols, list):
                cols2 = [str(c).strip() for c in cols if c and str(c).strip()]
                if cols2:
                    config["text_column"] = cols2[0]
                    config["_text_columns"] = cols2
            if 'text_column' not in config:
                raise ValueError("ddoc.yaml must specify legacy 'text_column' or v2 'data.columns'")

        # v2에서 CSV가 명시되면 힌트로 보관(직접 로드 우선)
        data = config.get("data") or {}
        if isinstance(data, dict) and data.get("csv"):
            config["_csv"] = data.get("csv")
        if isinstance(data, dict) and data.get("language") and not config.get("language"):
            config["language"] = data.get("language")
        if isinstance(data, dict) and data.get("id_column") and not config.get("id_column"):
            config["id_column"] = data.get("id_column")
        
        return config
    
    def _find_csv_files(self, dataset_path: Path) -> Tuple[List[Path], Optional[Path]]:
        """
        Find CSV files in dataset directory.
        - If single CSV file exists, return it
        - If ZIP file exists, extract to temp directory and recursively find all CSV files
        - Otherwise, recursively find all CSV files in directory
        
        Returns:
            (csv_files, temp_extract_dir): List of CSV file paths and optional temp directory for ZIP extraction
        """
        csv_files = []
        temp_extract_dir = None
        
        # Check for ZIP files first
        zip_files = list(dataset_path.glob("*.zip"))
        if zip_files:
            # Extract ZIP and find CSV files recursively
            for zip_file in zip_files:
                print(f"   Found ZIP file: {zip_file.name}")
                # Create temporary directory for extraction (will be cleaned up later)
                temp_extract_dir = Path(tempfile.mkdtemp(prefix="ddoc_text_"))
                try:
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        zip_ref.extractall(temp_extract_dir)
                        # Remove macOS metadata if present
                        macosx_dir = temp_extract_dir / "__MACOSX"
                        if macosx_dir.exists():
                            shutil.rmtree(macosx_dir)
                        
                        # Find all CSV files recursively in extracted directory
                        extracted_csvs = list(temp_extract_dir.rglob("*.csv"))
                        if extracted_csvs:
                            csv_files.extend(extracted_csvs)
                            print(f"   Found {len(extracted_csvs)} CSV files in ZIP")
                            for csv_path in extracted_csvs:
                                rel_path = csv_path.relative_to(temp_extract_dir)
                                print(f"      - {rel_path}")
                        else:
                            print(f"   ⚠️ No CSV files found in ZIP")
                except Exception as e:
                    print(f"   ⚠️ Error extracting {zip_file.name}: {e}")
                    if temp_extract_dir and temp_extract_dir.exists():
                        shutil.rmtree(temp_extract_dir)
                    temp_extract_dir = None
                    continue
        
        # If no ZIP files or no CSV found in ZIP, look for CSV files directly
        if not csv_files:
            # Check for single CSV file in root
            root_csvs = list(dataset_path.glob("*.csv"))
            if root_csvs:
                csv_files.extend(root_csvs)
                print(f"   Found {len(root_csvs)} CSV file(s) in root")
            else:
                # Recursively find all CSV files
                csv_files = list(dataset_path.rglob("*.csv"))
                if csv_files:
                    print(f"   Found {len(csv_files)} CSV file(s) recursively")
        
        return sorted(csv_files), temp_extract_dir
    
    def _load_and_combine_csvs(self, csv_files: List[Path], text_column: str, id_column: Optional[str] = None, base_path: Optional[Path] = None) -> pd.DataFrame:
        """Load and combine multiple CSV files into a single DataFrame"""
        dfs = []
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                if text_column not in df.columns:
                    # Try to get relative path for display
                    if base_path and csv_file.is_relative_to(base_path):
                        rel_path = csv_file.relative_to(base_path)
                    else:
                        rel_path = csv_file.name
                    print(f"   ⚠️ Text column '{text_column}' not found in {rel_path}")
                    continue
                
                # Add source file info for tracking
                if base_path and csv_file.is_relative_to(base_path):
                    rel_path = csv_file.relative_to(base_path)
                else:
                    rel_path = Path(csv_file.name)
                df['_source_file'] = str(rel_path)
                
                # Ensure ID column exists or create one
                if id_column and id_column in df.columns:
                    pass  # Use existing ID column
                elif id_column:
                    # ID column specified but not found, create sequential IDs
                    df[id_column] = [f"{rel_path.stem}_{i}" for i in range(len(df))]
                else:
                    # No ID column specified, try to find common ID columns
                    common_id_cols = ['id', 'ID', 'index', 'INDEX', 'idx']
                    found_id = None
                    for col in common_id_cols:
                        if col in df.columns:
                            found_id = col
                            break
                    if found_id:
                        id_column = found_id
                    else:
                        # Create sequential IDs
                        df['_auto_id'] = [f"{rel_path.stem}_{i}" for i in range(len(df))]
                        id_column = '_auto_id'
                
                dfs.append(df)
                print(f"   Loaded {len(df)} rows from {rel_path}")
                
            except Exception as e:
                print(f"   ⚠️ Error loading {csv_file}: {e}")
                continue
        
        if not dfs:
            return pd.DataFrame()
        
        # Combine all DataFrames
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"   ✅ Total: {len(combined_df)} rows from {len(dfs)} files")
        
        return combined_df
    
    def _analyze_text_attributes(self, text: str, language: str = 'english') -> Dict[str, Any]:
        """Calculate physical-based text metrics"""
        if not text or pd.isna(text):
            return {
                'length_chars': 0,
                'length_words': 0,
                'whitespace_ratio': 0,
                'special_char_ratio': 0,
                'stopword_ratio': 0,
                'vocab_diversity': 0,
                'readability': 0
            }
        
        text_str = str(text)
        
        # Length metrics
        length_chars = len(text_str)
        words = self._safe_word_tokenize(text_str, language=language)
        length_words = len(words)
        
        # Whitespace ratio
        whitespace_count = sum(1 for c in text_str if c.isspace())
        whitespace_ratio = whitespace_count / length_chars if length_chars > 0 else 0
        
        # Special character ratio
        special_char_count = sum(1 for c in text_str if not c.isalnum() and not c.isspace())
        special_char_ratio = special_char_count / length_chars if length_chars > 0 else 0
        
        # Stopword ratio
        stopword_ratio = 0
        # stopwords 코퍼스는 언어별 지원이 제한적이므로, 기본은 english만 의미있게 계산
        if language in ['english', 'en'] and length_words > 0:
            try:
                stop_words = set(stopwords.words('english'))
                stopword_count = sum(1 for w in words if w in stop_words)
                stopword_ratio = stopword_count / length_words
            except Exception:
                stopword_ratio = 0
        
        # Vocabulary diversity (Type-Token Ratio)
        unique_words = len(set(words))
        vocab_diversity = unique_words / length_words if length_words > 0 else 0
        
        # Simple readability (Flesch-like approximation for English)
        readability = 0.0
        if language == 'english' and length_words > 0:
            sentences = re.split(r'[.!?]+', text_str)
            sentences = [s for s in sentences if s.strip()]
            avg_sentence_length = length_words / len(sentences) if sentences else 0
            # Simplified readability score (higher = easier)
            readability = max(0, min(100, 100 - (avg_sentence_length * 1.5)))
        
        return {
            'length_chars': length_chars,
            'length_words': length_words,
            'whitespace_ratio': whitespace_ratio,
            'special_char_ratio': special_char_ratio,
            'stopword_ratio': stopword_ratio,
            'vocab_diversity': vocab_diversity,
            'readability': readability
        }

    def _safe_word_tokenize(self, text: str, *, language: str = "english") -> List[str]:
        """
        NLTK 리소스(punkt/punkt_tab)가 없거나, 언어가 영어가 아닐 때도 안전하게 토큰화합니다.
        - english: NLTK word_tokenize 시도, 실패 시 정규식 폴백
        - korean 등: 공백/유니코드 기반의 단순 토큰화(통계용)
        """
        s = str(text or "").strip()
        if not s:
            return []

        lang = (language or "english").lower()

        # 한국어: Kiwi 사용(가능하면), 실패 시 단순 토큰화
        if lang in ["korean", "ko", "kr"]:
            kiwi = self._load_kiwi()
            if kiwi is not None:
                try:
                    toks = kiwi.tokenize(s)
                    # 기호/공백 토큰은 제외 (EDA 통계 목적)
                    out = []
                    for t in toks:
                        form = getattr(t, "form", None) or ""
                        tag = getattr(t, "tag", "") or ""
                        if not form.strip():
                            continue
                        # SF/SP/SS/SE/SO/SW 등 기호류 제외
                        if tag.startswith("S"):
                            continue
                        out.append(form)
                    return out
                except Exception:
                    pass
            # 폴백: 한글/영문/숫자 덩어리 위주로 추출 (통계 목적)
            return re.findall(r"[가-힣]+|[a-zA-Z]+|[0-9]+", s)

        # 영어: 가능한 경우 NLTK 사용
        if lang in ["english", "en"]:
            try:
                return word_tokenize(s.lower())
            except Exception:
                # 리소스 누락/다운로드 불가 시 폴백
                return re.findall(r"[a-zA-Z]+|[0-9]+", s.lower())

        # 기타 언어: 최소한의 토큰화(공백/문자 덩어리)
        return re.findall(r"[\\w]+", s, flags=re.UNICODE)

    def _histogram(self, values: list[float], bins: int = 20, max_samples: int = 2000) -> dict[str, Any] | None:
        if not values:
            return None
        counts, edges = np.histogram(values, bins=bins)
        samples = values
        if len(values) > max_samples:
            idx = np.random.choice(len(values), size=max_samples, replace=False)
            samples = [values[i] for i in idx]
        return {"bins": edges.tolist(), "counts": counts.tolist(), "samples": samples}

    def _pca_projection(self, points: list[list[float]], max_points: int = 2000) -> list[dict[str, float]]:
        if not points:
            return []
        if len(points) > max_points:
            idx = np.random.choice(len(points), size=max_points, replace=False)
            points = [points[i] for i in idx]
        X = np.array(points, dtype=np.float32)
        X = X - X.mean(axis=0, keepdims=True)
        try:
            _, _, vt = np.linalg.svd(X, full_matrices=False)
            coords = X @ vt[:2].T
        except Exception:
            return []
        return [{"x": float(c[0]), "y": float(c[1])} for c in coords]
    
    def _extract_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """Extract CLIP text embedding"""
        if not text or pd.isna(text):
            return None
        
        self._load_clip_model()
        
        try:
            text_tokens = self.clip_tokenizer([str(text)], truncate=True).to(self.device)
            with torch.no_grad():
                text_features = self.clip_model.encode_text(text_tokens)
                # Normalize
                text_features = text_features / text_features.norm(dim=1, keepdim=True)
                return text_features.cpu().numpy().flatten()
        except Exception as e:
            print(f"Error extracting embedding: {e}")
            return None
    
    @hookimpl
    def eda_run(self, snapshot_id, data_path, data_hash, output_path, invalidate_cache=False):
        """Run EDA for text datasets"""
        from ddoc.core.cache_service import get_cache_service
        from ddoc.core.schemas import FileMetadata
        
        cache_service = get_cache_service()
        input_path = Path(data_path)
        output_path = Path(output_path)
        
        print(f"🚀 Text EDA Analysis Started")
        print(f"=" * 80)
        print(f"Snapshot: {snapshot_id}")
        print(f"Data Hash: {data_hash[:8] if data_hash != 'unknown' else 'unknown'}")
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")
        print()
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        metrics = {
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'snapshot_id': snapshot_id,
            'data_hash': data_hash,
            'modality': 'text'
        }
        
        # Find text datasets (directories with ddoc.yaml).
        # DriftStudio API는 보통 "단일 데이터셋 디렉토리"를 data_path로 넘기므로,
        # input_path 자체도 후보로 포함해야 합니다.
        text_datasets: list[tuple[Path, dict[str, Any]]] = []

        def _try_add_dataset_dir(p: Path) -> None:
            yaml_path = p / "ddoc.yaml"
            if not yaml_path.exists():
                return
            try:
                cfg = self._load_ddoc_yaml(p)
                if cfg.get("modality") == "text":
                    text_datasets.append((p, cfg))
            except Exception as e:
                print(f"⚠️ Skipping {p}: {e}")

        if input_path.is_dir():
            _try_add_dataset_dir(input_path)
        for item in input_path.iterdir():
            if item.is_dir():
                    _try_add_dataset_dir(item)
        
        if not text_datasets:
            print("⚠️ No text datasets found (directories with ddoc.yaml and modality=text)")
            return None
        
        # Load caches
        attr_cache = {}
        emb_cache = {}
        
        if not invalidate_cache:
            attr_cache_data = cache_service.load_analysis_cache(
                snapshot_id=snapshot_id,
                data_hash=data_hash,
                cache_type="attributes_text"
            )
            if attr_cache_data:
                attr_cache = attr_cache_data
            
            emb_cache_data = cache_service.load_analysis_cache(
                snapshot_id=snapshot_id,
                data_hash=data_hash,
                cache_type="embedding_text"
            )
            if emb_cache_data:
                emb_cache = emb_cache_data
        
        # Process each text dataset
        all_attributes = {}
        all_embeddings = {}
        
        for dataset_path, config in text_datasets:
            print(f"\n📊 Processing dataset: {dataset_path.name}")
            print("-" * 80)
            
            text_column = config['text_column']
            id_column = config.get('id_column', None)
            language = config.get('language', 'english')
            
            # Find CSV files:
            # - v2(spec)에서는 data.csv를 우선 사용
            # - 그 외에는 기존 로직(디렉토리/ZIP/재귀 탐색)
            csv_files: list[Path] = []
            temp_extract_dir = None
            csv_hint = config.get("_csv")
            if csv_hint:
                candidate = (dataset_path / str(csv_hint)).resolve()
                if candidate.exists() and candidate.is_file():
                    csv_files = [candidate]
                    print(f"   Using CSV from ddoc.yaml: {csv_hint}")

            if not csv_files:
                csv_files, temp_extract_dir = self._find_csv_files(dataset_path)
            
            if not csv_files:
                print(f"⚠️ No CSV files found in {dataset_path}")
                if temp_extract_dir and temp_extract_dir.exists():
                    shutil.rmtree(temp_extract_dir)
                continue
            
            # Determine base path for relative path calculation
            base_path = temp_extract_dir if temp_extract_dir else dataset_path
            
            # Load and combine all CSV files
            df = self._load_and_combine_csvs(csv_files, text_column, id_column, base_path)
            
            # Clean up temporary directory if created
            if temp_extract_dir and temp_extract_dir.exists():
                try:
                    shutil.rmtree(temp_extract_dir)
                except:
                    pass
            
            if df.empty:
                print(f"⚠️ No valid data loaded from {dataset_path}")
                continue
            
            # Determine actual ID column to use
            actual_id_column = id_column
            if not actual_id_column:
                # Try to find common ID columns
                common_id_cols = ['id', 'ID', 'index', 'INDEX', 'idx', '_auto_id']
                for col in common_id_cols:
                    if col in df.columns:
                        actual_id_column = col
                        break
            
            # Analyze attributes
            print(f"   Analyzing {len(df)} texts...")
            for idx, row in df.iterrows():
                text = row[text_column]
                
                # Generate row ID
                if actual_id_column and actual_id_column in df.columns:
                    row_id = str(row[actual_id_column])
                else:
                    row_id = f"row_{idx}"
                
                # Include source file in cache key for better tracking
                source_file = row.get('_source_file', dataset_path.name)
                cache_key = f"{dataset_path.name}/{source_file}/{row_id}"
                
                # Attributes
                attrs = self._analyze_text_attributes(text, language)
                all_attributes[cache_key] = attrs
                
                # Embedding
                embedding = self._extract_text_embedding(text)
                if embedding is not None:
                    all_embeddings[cache_key] = {
                        'embedding': embedding.tolist(),
                        'text_length': len(str(text))
                    }
            
            print(f"   ✅ Analyzed {len(df)} texts")
        
        # Save caches
        if all_attributes:
            cache_service.save_analysis_cache(
                snapshot_id=snapshot_id,
                data_hash=data_hash,
                cache_type="attributes_text",
                data=all_attributes
            )
            print(f"💾 Saved {len(all_attributes)} attribute records")
        
        embedding_index = None
        if all_embeddings:
            embedding_index = save_embeddings(
                modality="text",
                data_hash=data_hash,
                embeddings={k: v.get("embedding") for k, v in all_embeddings.items() if v.get("embedding") is not None},
            )
            cache_service.save_analysis_cache(
                snapshot_id=snapshot_id,
                data_hash=data_hash,
                cache_type="embedding_index_text",
                data=embedding_index,
            )
            print(f"💾 Saved embedding index: {embedding_index.get('count', 0)} vectors")
        
        # Calculate summary statistics
        if all_attributes:
            metrics['num_texts'] = len(all_attributes)
            metrics['avg_length_chars'] = np.mean([a['length_chars'] for a in all_attributes.values()])
            metrics['avg_length_words'] = np.mean([a['length_words'] for a in all_attributes.values()])
            metrics['avg_vocab_diversity'] = np.mean([a['vocab_diversity'] for a in all_attributes.values()])

        distributions = {}
        if all_attributes:
            for key in ['length_chars', 'length_words', 'whitespace_ratio', 'special_char_ratio',
                        'stopword_ratio', 'vocab_diversity', 'readability']:
                vals = [a.get(key, 0) for a in all_attributes.values()]
                hist = self._histogram(vals)
                if hist:
                    distributions[key] = hist

        embedding_projection = None
        if all_embeddings:
            emb_list = [v.get("embedding") for v in all_embeddings.values() if v.get("embedding") is not None]
            embedding_projection = {
                "method": "pca",
                "points": self._pca_projection(emb_list),
            }
        
        # Save metrics
        metrics_file = output_path / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n✅ Text Analysis Complete")
        print(f"   📄 Metrics: {metrics_file}")
        
        if embedding_index is None:
            embedding_index = cache_service.load_analysis_cache(
                snapshot_id=snapshot_id,
                data_hash=data_hash,
                cache_type="embedding_index_text",
            )

        return {
            "status": "success",
            "modality": "text",
            "texts_analyzed": metrics.get('num_texts', 0),
            "metrics_file": str(metrics_file),
            "output_path": str(output_path),
            "summary": metrics,
            "distributions": distributions or None,
            "embedding_projection": embedding_projection,
            "embedding_index": embedding_index,
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
        """Detect drift between two text snapshots"""
        from ddoc.core.cache_service import get_cache_service
        
        cache_service = get_cache_service()
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"🔍 Text Drift Detection Started")
        print(f"=" * 80)
        
        # Load caches
        baseline_attr = cfg.get('baseline_cache') or cache_service.load_analysis_cache(
            snapshot_id=snapshot_id_ref,
            data_hash=data_hash_ref,
            cache_type="attributes_text"
        )
        baseline_emb_index = cache_service.load_analysis_cache(
            snapshot_id=snapshot_id_ref,
            data_hash=data_hash_ref,
            cache_type="embedding_index_text"
        )
        baseline_emb_legacy = cache_service.load_analysis_cache(
            snapshot_id=snapshot_id_ref,
            data_hash=data_hash_ref,
            cache_type="embedding_text"
        )
        
        current_attr = cfg.get('current_cache') or cache_service.load_analysis_cache(
            snapshot_id=snapshot_id_cur,
            data_hash=data_hash_cur,
            cache_type="attributes_text"
        )
        current_emb_index = cache_service.load_analysis_cache(
            snapshot_id=snapshot_id_cur,
            data_hash=data_hash_cur,
            cache_type="embedding_index_text"
        )
        current_emb_legacy = cache_service.load_analysis_cache(
            snapshot_id=snapshot_id_cur,
            data_hash=data_hash_cur,
            cache_type="embedding_text"
        )
        
        if not baseline_attr or not current_attr:
            print("❌ Missing baseline or current data")
            return None
        
        drift_metrics = {
            'modality': 'text',
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
        }
        drift_metrics["embedding_index_ref"] = baseline_emb_index
        drift_metrics["embedding_index_cur"] = current_emb_index
        drift_metrics["data_hash_ref"] = data_hash_ref
        drift_metrics["data_hash_cur"] = data_hash_cur
        
        # Attribute drift (PSI for each metric)
        print("📈 Attribute Drift:")
        print("-" * 80)
        
        attribute_names = ['length_chars', 'length_words', 'whitespace_ratio', 
                          'special_char_ratio', 'stopword_ratio', 'vocab_diversity', 'readability']
        
        attribute_drifts = {}
        for attr_name in attribute_names:
            ref_values = [a.get(attr_name, 0) for a in baseline_attr.values()]
            cur_values = [a.get(attr_name, 0) for a in current_attr.values()]
            
            if ref_values and cur_values:
                try:
                    # Use PSI for drift
                    from scipy.stats import wasserstein_distance
                    psi = wasserstein_distance(ref_values, cur_values) / (np.mean(ref_values) + 1e-10)
                    attribute_drifts[attr_name] = float(psi)
                    print(f"   {attr_name:20s} Drift: {psi:.4f}")
                except:
                    attribute_drifts[attr_name] = 0.0
        
        drift_metrics['attribute_drifts'] = attribute_drifts
        drift_metrics['attribute_drift_overall'] = np.mean(list(attribute_drifts.values())) if attribute_drifts else 0.0
        
        # Embedding drift
        baseline_emb = None
        current_emb = None
        if baseline_emb_index:
            ref_vectors, ref_ids, _ = load_embeddings(embedding_index=baseline_emb_index)
            baseline_emb = {ref_ids[i]: {"embedding": ref_vectors[i].tolist()} for i in range(len(ref_ids))}
        elif baseline_emb_legacy:
            baseline_emb = baseline_emb_legacy
        if current_emb_index:
            cur_vectors, cur_ids, _ = load_embeddings(embedding_index=current_emb_index)
            current_emb = {cur_ids[i]: {"embedding": cur_vectors[i].tolist()} for i in range(len(cur_ids))}
        elif current_emb_legacy:
            current_emb = current_emb_legacy

        if baseline_emb and current_emb:
            print("\n🧠 Embedding Drift:")
            print("-" * 80)
            
            ref_emb_list = [np.array(e['embedding']) for e in baseline_emb.values()]
            cur_emb_list = [np.array(e['embedding']) for e in current_emb.values()]
            
            if ref_emb_list and cur_emb_list:
                ref_emb_array = np.array(ref_emb_list)
                cur_emb_array = np.array(cur_emb_list)
                
                # Cosine distance between mean embeddings
                ref_mean = ref_emb_array.mean(axis=0)
                cur_mean = cur_emb_array.mean(axis=0)
                cosine_sim = np.dot(ref_mean, cur_mean) / (np.linalg.norm(ref_mean) * np.linalg.norm(cur_mean) + 1e-10)
                embedding_drift = 1.0 - cosine_sim
                
                drift_metrics['embedding_drift'] = float(embedding_drift)
                print(f"   Embedding Drift (cosine): {embedding_drift:.4f}")
            else:
                drift_metrics['embedding_drift'] = 0.0
        
        # Overall score
        attr_score = drift_metrics.get('attribute_drift_overall', 0)
        emb_score = drift_metrics.get('embedding_drift', 0)
        overall = 0.5 * attr_score + 0.5 * emb_score
        drift_metrics['overall_score'] = float(overall)
        
        # Save metrics
        metrics_file = output_path / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(drift_metrics, f, indent=2)
        
        print(f"\n✅ Text Drift Detection Complete")
        print(f"   📄 Metrics: {metrics_file}")
        
        return drift_metrics

