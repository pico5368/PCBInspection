# 데이터 릴리스 (Data Releases)

> `data/` 트리를 GitHub Releases에 버전 단위로 올려, 신규 클론에서도 동일한
> 학습/검사 결과를 재현할 수 있게 한다. DATA_POLICY.md가 데이터의 *위치/명명*
> 규칙이라면, 이 문서는 그 스냅샷을 *어떻게 패키징·게시·복원*할지 정한다.

| 항목 | 값 |
|------|-----|
| 버전 | 0.1 (초안) |
| 도구 | `scripts/data_release.py` (pack / publish / fetch) |
| 저장소 | GitHub Releases (`github.com/d3draw/PCBInspection`) |
| 태그 규칙 | `data-v<MAJOR>.<MINOR>.<PATCH>` |
| 자산 단위 | `data/` 최상위 디렉토리별 tar 한 개 |
| 무결성 | SHA-256 (MANIFEST.json) |

---

## 1. 왜 Releases인가

`data/`는 약 4 GB이고 일부 ckpt 파일은 100 MB를 넘는다. Git 본문에는
올릴 수 없고 (`.gitignore`로 제외), Git LFS는 무료 한도 1 GB / 1 GB 월
대역폭이라 부족하다. **GitHub Releases**는 파일당 2 GB까지 무료로 첨부할 수
있고 코드 히스토리와 분리된 버전 축으로 관리된다 — 데이터 스냅샷의 보관처로
이 프로젝트 규모에 가장 잘 맞는다.

대안 비교:
- **Git LFS** — 4 GB면 유료 데이터팩 필수, 대역폭 비용도 누적.
- **외부 스토리지(S3/GCS)** — 가장 견고하지만 계정·권한 셋업 부담.
- **DVC + S3** — ML 표준이나 POC 단계엔 과함. 운영 단계에서 재검토.

---

## 2. 자산 구성

태그 하나에 다음 tar들을 첨부한다 (`data/` 최상위 디렉토리 각각):

| 자산 | 원본 경로 | 대략 크기 | 필요 시점 |
|------|----------|---------|----------|
| `datasets.tar` | `data/datasets/` | ~1.0 GB | 학습 재현 |
| `golden.tar`   | `data/golden/`   | ~1.4 GB | 정렬/검사 파이프라인 재현 |
| `models.tar`   | `data/models/`   | ~725 MB | 학습 없이 추론만 할 때 |
| `captures.tar` | `data/captures/` | ~842 MB | 재학습 / DOE 재현 |
| `MANIFEST.json` | (자동 생성) | <1 KB | 무결성 검증 |

tar 내부는 `data/` 기준 상대 경로를 보존하므로, 해제 시 그대로 `data/`
아래로 복원된다. **무압축 tar**를 쓴다 — PNG·ckpt는 사실상 압축되지
않아 패킹 시간을 줄였고, tar는 zip과 달리 **심볼릭 링크를 메타로 보존**
하므로 `poc_panel_B1`/`B2`가 `poc_panel/good/`을 가리키는 520여 개 링크가
중복 저장되지 않는다 (zip이면 같은 파일을 세 번 저장해 2 GiB 한도를 위협).

### MANIFEST.json 형식

```json
{
  "version": "0.1.0",
  "tag": "data-v0.1.0",
  "created_at": "2026-06-11T05:00:00+00:00",
  "data_root": "data/",
  "assets": [
    {"name": "datasets.tar", "size": 1073741824, "sha256": "..."},
    {"name": "golden.tar",   "size": 1503238553, "sha256": "..."}
  ]
}
```

---

## 3. 버전 태그 규칙

`data-v<MAJOR>.<MINOR>.<PATCH>` — 코드 릴리스 태그(`v0.1.0`)와 충돌하지
않도록 `data-` 접두사를 둔다.

- **MAJOR** — 디렉토리 레이아웃이 바뀌어 기존 fetch 스크립트가 깨질 때
- **MINOR** — 새 보드/패널 데이터셋 추가 등 의미 있는 확장
- **PATCH** — 잘못된 라벨 정정, 누락 파일 보충 같은 소규모 수정

코드와 데이터 버전은 **독립**이다. 예를 들어 `v0.3.0` 코드가
`data-v0.1.0`을 그대로 쓸 수 있다. 호환 매트릭스가 필요해지면 이 표에
"코드 ↔ 데이터" 칼럼을 추가한다.

---

## 4. 워크플로 1 — 릴리스 게시 (data 보유 머신에서)

```bash
# 1) data/ 하위를 패킹. dist/releases/data-v0.1.0/ 아래에 tar + MANIFEST.json 생성
.venv/bin/python scripts/data_release.py pack --version 0.1.0

# 일부만 다시 만들고 싶다면:
.venv/bin/python scripts/data_release.py pack --version 0.1.0 \
  --dirs models --force

# 2) GitHub Releases에 업로드 (gh CLI 인증 필요)
.venv/bin/python scripts/data_release.py publish --version 0.1.0 \
  --notes "B1/B2 패널 추가, padim ckpt 재학습"

# 초안으로 먼저 올리고 검토하려면 --draft 사용
```

전제 조건:
- `gh auth status`가 통과해야 한다 (`d3draw` 계정).
- `data/` 아래에 4개 디렉토리(`datasets`, `golden`, `models`, `captures`)가
  모두 존재해야 한다. 빠진 게 있으면 `--dirs`로 추리거나 비어 있는
  디렉토리를 만들고 다시 시도.

---

## 5. 워크플로 2 — 신규 클론에서 복원

```bash
git clone git@github.com:d3draw/PCBInspection.git
cd PCBInspection
python -m venv .venv && .venv/bin/pip install -e .[ml]

# 전체 복원
.venv/bin/python scripts/data_release.py fetch --version 0.1.0

# 추론만 할 거라 모델·골든만 필요할 때
.venv/bin/python scripts/data_release.py fetch --version 0.1.0 \
  --dirs models golden
```

`fetch`는 다음 순서로 동작한다:

1. `dist/releases/data-v0.1.0/`에 tar과 `MANIFEST.json`을 내려받는다.
2. 각 tar의 SHA-256을 manifest와 대조한다 — 불일치면 즉시 비정상 종료.
3. `data/` 아래로 압축을 풀고, 검증 완료된 tar은 삭제한다
   (`--keep-tars`로 보존 가능).

---

## 6. 운영 주의사항

- **2 GB 자산 한도**: `pack` 시 한 자산이 2 GB를 넘으면 경고와 함께
  종료 코드 2를 반환한다. 디렉토리가 그만큼 커지면 패널 단위 같은 더 작은
  단위로 쪼개 다시 설계해야 한다 (예: `golden_B1.tar` / `golden_B2.tar`).
- **원본 데이터 보존**: `publish` 직후라도 로컬 `data/`를 절대 삭제하지
  않는다 — 다음 패치 릴리스 때 baseline으로 필요하다.
- **민감 데이터**: `data/captures/`에 외부 공개가 곤란한 보드가 섞일 수
  있다. 게시 전에 한 번 훑어보고, 필요하면 그 보드만 빼고 패킹한다.
- **release를 잘못 올렸을 때**: 같은 태그로 덮어쓰지 말고
  `gh release delete data-v0.1.0 --yes`로 지운 뒤 PATCH 번호를 올려
  재게시한다. 이미 동기화한 클론의 무결성을 깨지 않으려면 이게 안전하다.

---

## 7. TODO

- [ ] 첫 게시 후 README에 "데이터 복원" 한 줄 추가
- [ ] CI에서 `fetch --dirs golden models`만으로 추론 스모크 테스트 통과 검증
- [ ] 데이터셋 라벨 변경 시 SCHEMA 버전과 `data-v*` 버전 매핑 표 작성
