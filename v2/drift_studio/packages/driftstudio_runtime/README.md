# driftstudio_runtime

drift_studio의 런타임 실행기입니다.

## 입력 데이터셋 규약

- 런타임은 데이터셋을 **압축 해제된 디렉토리**로 처리합니다.
- 해당 디렉토리 루트에 `ddoc.yaml`이 반드시 있어야 하며, 스키마/구성 검증에 실패하면 예외로 중단됩니다.

상세 스키마/템플릿은 `packages/driftstudio_spec/README.md` 참고.
