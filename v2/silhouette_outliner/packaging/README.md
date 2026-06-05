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
- Playwright Chromium (CI 빌드 시 번들)
- 분석 시 **네트워크 연결 필요**
