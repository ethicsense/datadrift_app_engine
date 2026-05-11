from __future__ import annotations
import pluggy
import importlib
import importlib.metadata # <--- 추가: 엔트리 포인트를 직접 로드하기 위해 필요
import logging
from typing import Any, Dict, Optional, Iterable

from ddoc.plugins.hookspecs import HOOKSPEC_VERSION

log = logging.getLogger(__name__)
GROUP = "ddoc" # U+00A0 제거됨


# PluginManager 인스턴스를 저장하는 전역 변수 (싱글톤 패턴)
_PLUGIN_MANAGER: Optional['PluginManager'] = None

class PluginManager:
    def __init__(self) -> None:
        # pluggy PluginManager 초기화
        self.pm = pluggy.PluginManager("ddoc")

        # hookspecs 등록
        import ddoc.plugins.hookspecs as hookspecs
        self.pm.add_hookspecs(hookspecs)
        
    # --- 핵심: .hook 속성 위임 (Delegation) ---
    @property
    def hook(self):
        """플러그인 훅 호출을 위해 pluggy.PluginManager.hook을 위임합니다."""
        return self.pm.hook
        
    # --- pluggy API 위임 (Delegation) ---
    def add_hookspecs(self, hookspecs: Any) -> None:
        """pluggy.PluginManager.add_hookspecs() 호출을 위임합니다."""
        self.pm.add_hookspecs(hookspecs)

    def load_setuptools_entrypoints(self, group: str = GROUP) -> None:
        """pluggy.PluginManager.load_setuptools_entrypoints() 호출을 위임합니다."""
        # pluggy 자체의 함수를 사용하도록 수정
        self.pm.load_setuptools_entrypoints(group)
        
    # -----------------------------------
    
    # NOTE: _is_version_compatible 함수는 버전 문자열 비교 로직을 수행한다고 가정하고 생략
    def _is_version_compatible(self, current: str, min_v: Optional[str], max_v: Optional[str]) -> bool:
        """버전 호환성 확인 로직 (실제 구현 생략)"""
        return True

    def _check_and_register(self, plugin_obj: Any, name: Optional[str] = None) -> None:
        """버전 호환성을 확인하고 플러그인을 등록합니다."""
        pmin = getattr(plugin_obj, "DDOC_HOOKSPEC_MIN", None)
        pmax = getattr(plugin_obj, "DDOC_HOOKSPEC_MAX", None)
        
        if not self._is_version_compatible(HOOKSPEC_VERSION, pmin, pmax):
            log.warning("Plugin %s incompatible with HookSpec %s (min=%s, max=%s). Skipped.",
                        name or getattr(plugin_obj, "__name__", plugin_obj), HOOKSPEC_VERSION, pmin, pmax)
            return

        # 중복 등록 방지 로직 (생략된 부분 포함)

        if isinstance(plugin_obj, dict):
            log.warning("Dict plugin object detected for %s; refusing to register. "
                        "Plugins must register a module or instance (not dict).", name or plugin_obj)
            return
            
        try:
            self.pm.register(plugin_obj, name=name)
        except ValueError as e:
            log.info("Duplicate plugin %s detected. Skipping. (%s)", name or plugin_obj, e)
        
    def load_entrypoints(self, group: str = GROUP) -> None:
        """setuptools entry points를 로드하고 등록합니다. (클래스 객체를 인스턴스화하여 등록)"""
        log.debug("Loading setuptools entry points...")
        
        # pluggy의 기본 메서드 대신, 직접 entry_points를 로드하고 인스턴스화합니다.
        try:
            eps = importlib.metadata.entry_points()
            # py>=3.10: EntryPoints(select=...), old: dict-like
            if hasattr(eps, "select"):
                entry_points = eps.select(group=group)
            else:
                entry_points = eps.get(group, [])
        except Exception as e:
            # fallback이 필요하다면 이 부분을 수정해야 하지만, 현재는 인스턴스화 로직이 핵심입니다.
            log.error("Failed to load entry points list: %s", e)
            return

        for entry_point in entry_points:
            try:
                # 1. Entry Point에서 클래스 객체(예: DDOCNlpPlugin)를 로드
                plugin_cls_or_obj = entry_point.load()
                plugin_name = entry_point.name
                
                # 2. 로드된 객체가 클래스(type)라면, 반드시 인스턴스화합니다.
                if isinstance(plugin_cls_or_obj, type):
                    plugin_obj = plugin_cls_or_obj() # <--- 핵심 수정: 클래스 인스턴스화
                else:
                    # 이미 인스턴스이거나 모듈이라면 그대로 사용 (예: ddoc_builtins)
                    plugin_obj = plugin_cls_or_obj
                
                # 3. _check_and_register를 통해 인스턴스를 등록
                self._check_and_register(plugin_obj, name=plugin_name)
                log.debug("Registered external plugin: %s", plugin_name)
                
            except Exception as e:
                log.error("Failed to load or register external plugin '%s': %s", entry_point.name, e)


    def get_plugins(self) -> Iterable[object]:
        return self.pm.get_plugins()

    def list_plugins(self) -> Dict[str, object]:
        """Get all registered plugins as a dictionary"""
        plugins = {}
        for plugin in self.pm.get_plugins():
            name = self.pm.get_name(plugin)
            if name:
                plugins[name] = plugin
        return plugins

    def get_name(self, plugin: object) -> Optional[str]:
        return self.pm.get_name(plugin)

    def call_hook(self, hook_name: str, provider: Optional[str] = None, first_non_none: bool = True, **kwargs) -> Any:
        """
        Hook 호출.

        IMPORTANT:
        - pluggy의 기본 hook 호출(hook(**kwargs))은 "등록된 모든 플러그인 구현"을 실행합니다.
          provider를 지정하더라도 실행 자체는 모두 발생하고, 우리는 결과만 필터링했었습니다.
          이 때문에 timeseries EDA를 실행해도 vision/text/audio가 같이 실행(모델 다운로드 등)되는 문제가 생깁니다.
        - provider가 지정된 경우에는 해당 provider의 hookimpl만 "직접" 호출하여 부작용을 방지합니다.
        """
        hook = getattr(self.pm.hook, hook_name)

        # provider 지정 시: 전체 실행 금지(직접 1개만 호출)
        if provider:
            found_impls = []
            for impl in hook.get_hookimpls():
                # impl.plugin은 플러그인 객체, pm.get_name()은 entry point 이름을 반환
                plugin_obj = impl.plugin
                plugin_name = self.pm.get_name(plugin_obj) if plugin_obj else None
                found_impls.append((plugin_name, plugin_obj))
                if plugin_name == provider:
                    try:
                        return impl.function(**kwargs)  # type: ignore[attr-defined]
                    except Exception:
                        # 기존 동작과 동일하게 예외는 상위로 전달
                        raise
            # 디버깅: 매칭 실패 시 로그 출력
            log.warning(
                f"No hook implementation found for provider '{provider}' (hook={hook_name}). "
                f"Available implementations: {[name for name, _ in found_impls]}"
            )
            return None

        # provider 미지정 시: 기존처럼 전체 실행 후 first_non_none 옵션 적용
        results = hook(**kwargs)
        if first_non_none:
            for res in results:
                if res is not None:
                    return res
            return None
        return results

def get_plugin_manager() -> PluginManager:
    """PluginManager의 싱글톤 인스턴스를 반환합니다."""
    global _PLUGIN_MANAGER
    if _PLUGIN_MANAGER is None:
        _PLUGIN_MANAGER = PluginManager()
        _PLUGIN_MANAGER.load_entrypoints()
    return _PLUGIN_MANAGER
