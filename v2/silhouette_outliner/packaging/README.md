# Silhouette Outliner 데모 패키지 빌드

작성일: 2026-06-05

Windows·macOS 설치형 데모 패키지는 **GitHub Actions**에서 OS별로 빌드합니다. 로컬 Windows/Mac 없이도 CI에서 산출물을 받을 수 있습니다.

## GitHub Actions로 빌드 (권장)

저장소: `ethicsense/datadrift_app_engine`

### 1. 수동 실행 (학습·테스트용)

1. GitHub → **Actions** → **Silhouette Outliner Demo Package**
2. **Run workflow** → 브랜치 선택 → Run
3. 완료 후 워크플로 run 페이지 하단 **Artifacts** 에서 다운로드:
   - `silhouette-outliner-windows-x64`
   - `silhouette-outliner-macos`

### 2. 태그로 실행 (Artifacts 생성용)

```bash
git tag demo-v0.1.0
git push origin demo-v0.1.0
```

`demo-v*` 태그 push 시 동일 워크플로가 실행되고, 성공 시 Actions Artifacts만 생성됩니다.

## OS별 테스트

| OS | 받은 파일 | 테스트 방법 |
|----|-----------|-------------|
| **Windows** | `SilhouetteOutliner-win-x64.zip` | 압축 해제 → `SilhouetteOutliner/SilhouetteOutliner.exe` 실행 |
| **macOS** | `Silhouette Outliner-macos.zip` | 압축 해제 → `Silhouette Outliner.app` 실행 (우클릭 → 열기) |

보고서·실행 결과는 사용자 데이터 폴더에 저장됩니다.

- **Windows:** `%LOCALAPPDATA%\SilhouetteOutliner\runs\`
- **macOS:** `~/Library/Application Support/SilhouetteOutliner/runs/`

### 보안 안내 (데모용, 서명 없음)

- **macOS:** Gatekeeper 경고 시 **우클릭 → 열기** 또는 시스템 설정에서 허용
- **Windows:** SmartScreen 경고 시 **추가 정보 → 실행**

## 로컬 빌드 (선택)

해당 OS 머신에서만 가능합니다.

```bash
cd v2/silhouette_outliner
source .venv/bin/activate
pip install -e ".[gui,packaging]"
export PLAYWRIGHT_BROWSERS_PATH="$PWD/packaging/playwright-browsers"
playwright install chromium
pyinstaller packaging/silhouette_outliner.spec --noconfirm
```

산출물: `dist/SilhouetteOutliner/` (Windows), `dist/Silhouette Outliner.app` (macOS)

## 포함 내용

- GUI 앱 (`Silhouette Outliner`)
- `configs/periodic-multag.json` (전체 분석 프리셋)
- **Playwright Chromium 번들 포함 (Windows·macOS 공통)** — 앱에 동봉되므로 별도 브라우저 설치가 필요 없습니다.
- 분석 시 **네트워크 연결 필요** (데이터 수집 대상 사이트 접근용)

### Chromium 동봉 방식 (빌드 관점)

수집은 Playwright(Chromium)로 페이지를 직접 열어 네트워크 JSON·DOM을 가져오는 구조이며, 패키지에는 항상 Chromium이 포함됩니다.

- **Windows:** PyInstaller `datas`로 번들 → `_internal/playwright-browsers/`
- **macOS:** PyInstaller가 Chromium의 중첩 `.app/.framework`를 per-binary로 코드사인하려다 실패(`bundle format unrecognized`)하므로, **빌드 후** CI가 Chromium을 `Silhouette Outliner.app/Contents/Frameworks/playwright-browsers/`로 복사하고 번들 전체를 `codesign --deep`로 한 번에 ad-hoc 서명합니다.

런타임에서는 `runtime_paths.configure_playwright_browsers()`가 동봉된 Chromium 경로를 `PLAYWRIGHT_BROWSERS_PATH`로 지정합니다. 다운로드나 API 폴백 같은 동작 변경은 없습니다.
