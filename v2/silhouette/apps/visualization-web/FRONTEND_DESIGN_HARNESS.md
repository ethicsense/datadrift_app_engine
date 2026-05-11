# Frontend Design Harness

이 문서는 `apps/visualization-web`의 화면 구조/문체/컴포넌트 패턴을 일관되게 유지하기 위한 실행용 하네스입니다.

## 목적

- 페이지별 UI를 수정할 때 구조, 카드 패턴, 제목/설명 톤을 일관되게 유지
- 디자인 변경 전에 현재 기준선을 빠르게 진단
- 변경 후 품질 점검(문구, 구조, 접근성, 상호작용)을 같은 체크리스트로 검증

## 이 프로젝트의 UI 기준선

### 1) 페이지 골격

- 페이지 루트: `PageContainer`
  - `title`은 짧은 명사형
  - `description`은 1문장 핵심 설명
- 상단 맥락: `PageContextBar`
  - `title`: 현재 화면의 읽기 기준
  - `summary`: 기준 1~2문장
  - `badges`: 현재 필터/선택 상태
  - `notes`: 읽는 순서, 주의사항

### 2) 본문 패널 패턴

- 기본 패널: `SectionCard`
  - `section`: `summary | input | formula | result | interpretation | examples`
  - `title`, `description`, `takeaway`, `explainability`
- 위젯 렌더: `WidgetRenderer` + `widgetRegistry`
  - 차트: `EChartPanel`
  - 표: `DataTable`
  - 애니메이션: `RankRaceChart`, `AnimatedScatterChart`
  - 특수: `ProductRankTrajectoriesChart`, `LocationMapPanel`

### 3) 문구 톤 규칙

- 한국어 우선, 혼용 최소화
- 제목은 짧고 직관적으로
- 설명은 짧은 현재형 문장
- `takeaway`는 해석 포인트 1문장
- 필요할 때만 식별자 유지:
  - 예: `nameItem`, `rank_velocity`, `momentum_score`, `new/retained/exited`

### 4) 스타일 기준

- 공통 스타일 파일: `src/styles.css`
- 카드/패널 클래스:
  - `.page-context-bar*`, `.section-card*`, `.legend-filter*`
- 신규 UI를 만들 때 기존 클래스군 우선 재사용

## Cursor에서 사용하는 방법

### 빠른 사용 절차

1. Cursor 채팅에서 아래 파일을 컨텍스트에 포함
   - `@apps/visualization-web/FRONTEND_DESIGN_HARNESS.md`
2. 수정하려는 페이지 파일도 함께 포함
   - 예: `@apps/visualization-web/src/pages/PricePage.tsx`
3. 아래 템플릿으로 요청

### 요청 템플릿 (복붙용)

```text
이 하네스를 기준으로 아래 페이지를 정리해줘.

목표:
- 제목/설명/takeaway 톤 통일
- 한글/영문 혼용 최소화
- PageContainer > PageContextBar > SectionCard 구조 일관성 확인
- 기존 상호작용(필터/선택/하이라이트)은 유지

대상:
- <페이지 파일 경로>

출력 방식:
1) 변경 요약
2) 적용한 문구 규칙
3) 남은 후속 제안(선택)
```

## 실행 체크리스트

수정 시 아래를 순서대로 확인합니다.

1. 구조 점검
   - `PageContainer`/`PageContextBar` 누락 없는가
   - 카드가 `SectionCard` 패턴을 따르는가
2. 문구 점검
   - 제목이 짧은가
   - 설명/요약이 1~2문장으로 간결한가
   - 불필요한 영문 혼용이 없는가
3. 상호작용 점검
   - 필터/검색/선택/링크 동작이 기존과 동일한가
4. 기술 점검
   - 린트 오류 없는가
   - 타입 오류 없는가

## 금지/주의

- 시각 구조를 크게 바꾸는 리팩터링을 문구 정리 작업과 한 번에 섞지 않기
- 데이터 키/쿼리 파라미터 이름을 문구 정리 중 임의 변경하지 않기
- 사용자 노출 텍스트와 내부 식별자를 혼동하지 않기

## 결과물 기대 형태

- PR/커밋 없이도 로컬에서 바로 반영 가능한 파일 수정
- 변경 파일마다:
  - 제목/설명/takeaway 정리
  - 한글/영문 혼용 정리
  - 스타일 클래스 재사용 여부 확인

