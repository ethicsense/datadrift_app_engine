# ddoc/plugins/hookspecs.py
"""
Hook specifications for ddoc plugin system with version tag.
"""
import pluggy
from typing import Any, Dict, Optional, List # List 타입 임포트 추가

hookspec = pluggy.HookspecMarker("ddoc")
hookimpl = pluggy.HookimplMarker("ddoc") 

# Bump this if you change HookSpec signatures in a breaking way.
HOOKSPEC_VERSION = "1.0.0"

# --- MLOps Core Operations (Implemented by ddoc/core/ops.py) ---

@hookspec
def data_add(name: str, config: str) -> Optional[Dict[str, Any]]:
    """Registers a new dataset version (dvc add, git branch/commit, params update)."""
    
@hookspec
def exp_run(name: str, params: str, dry_run: bool) -> Optional[Dict[str, Any]]:
    """Updates params.yaml and executes dvc exp run."""
    
@hookspec
def exp_show(name: Optional[str], baseline: Optional[str]) -> Optional[Dict[str, Any]]:
    """Shows DVC experiment results, optionally comparing two versions."""

# --- Analytical Operations (Implemented by external plugins) ---

@hookspec
def eda_run(
    snapshot_id: str,
    data_path: str,
    data_hash: str,
    output_path: str,
    invalidate_cache: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Run EDA analysis on a snapshot.
    
    Args:
        snapshot_id: Snapshot ID (or "workspace" for current workspace)
        data_path: Path to data directory
        data_hash: DVC hash of the data
        output_path: Path to save analysis results
        invalidate_cache: Whether to invalidate existing cache
        
    Returns:
        Dictionary with analysis summary
    """

@hookspec
def transform_apply(input_path: str, transform: str, args: Dict[str, Any], output_path: str) -> Optional[Dict[str, Any]]:
    """Apply a named transform and write result to output_path."""

@hookspec
def drift_detect(
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
    """
    Detect drift between two snapshots.
    
    Args:
        snapshot_id_ref: Reference snapshot ID
        snapshot_id_cur: Current snapshot ID
        data_path_ref: Reference snapshot data path
        data_path_cur: Current snapshot data path
        data_hash_ref: Reference snapshot data hash
        data_hash_cur: Current snapshot data hash
        detector: Drift detector method (e.g., "mmd")
        cfg: Configuration dictionary
        output_path: Path to save drift analysis results
        
    Returns:
        Dictionary with drift metrics
    """

@hookspec
def reconstruct_apply(input_path: str, method: str, args: Dict[str, Any], output_path: str) -> Optional[Dict[str, Any]]:
    """Reconstruct/Impute/Resample data and write result."""

@hookspec
def retrain_run(train_path: str, trainer: str, params: Dict[str, Any], model_out: str) -> Optional[Dict[str, Any]]:
    """Retrain model and write model artifact."""

@hookspec
def monitor_run(source: str, mode: str, schedule: Optional[str]) -> Optional[Dict[str, Any]]:
    """Run monitors once or on schedule."""

@hookspec
def vis_run() -> Optional[Dict[str, Any]]:
    """Run monitors once or on schedule."""
    
# --- Plugin Metadata Hook (NEW) ---
@hookspec(firstresult=True)
def ddoc_get_metadata() -> Optional[Dict[str, Any]]:
    """
    Returns structured metadata (name, description, implemented hooks) 
    about the plugin.
    
    Note: firstresult=True is typically not used for gathering, but here 
    we assume the list of results is gathered elsewhere. Since the external 
    plugin returns a full dict, we keep it simple for now. 
    For listing, we gather all results manually (see cli/commands.py).
    """

# --- Plugin Metadata Hook (NEW) ---
@hookspec(firstresult=True)
def ddoc_get_metadata2() -> Optional[Dict[str, Any]]:
    """
    Returns structured metadata (name, description, implemented hooks) 
    about the plugin.
    
    Note: firstresult=True is typically not used for gathering, but here 
    we assume the list of results is gathered elsewhere. Since the external 
    plugin returns a full dict, we keep it simple for now. 
    For listing, we gather all results manually (see cli/commands.py).
    """

