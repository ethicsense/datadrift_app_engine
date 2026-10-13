"""Build static attribute label review HTML for Cohen κ dual labeling."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]

ATTR_ORDER = ("pattern", "sleeve", "neck", "color", "fit")
ATTR_LABELS_KO = {
    "pattern": "무늬",
    "sleeve": "소매",
    "neck": "넥라인",
    "color": "주색",
    "fit": "핏",
}
CLASS_LABELS_KO = {
    "pattern": {
        "solid": "무지",
        "stripe": "스트라이프",
        "check": "체크",
        "graphic_logo": "그래픽·로고",
        "other": "기타",
        "uncertain": "판단불가",
    },
    "sleeve": {
        "sleeveless": "민소매",
        "short": "반팔",
        "long": "긴팔",
        "uncertain": "판단불가",
    },
    "neck": {
        "round": "라운드",
        "vneck": "V넥",
        "collar": "셔츠칼라",
        "hood": "후드",
        "turtleneck": "터틀·하이넥",
        "uncertain": "판단불가",
    },
    "color": {
        "white_ivory": "흰·아이보리",
        "black": "검정",
        "gray": "회색",
        "navy_blue": "네이비·블루",
        "beige_brown": "베이지·브라운",
        "chromatic": "유채색",
        "uncertain": "판단불가",
    },
    "fit": {
        "slim": "슬림",
        "regular": "레귤러",
        "oversized": "오버사이즈",
        "uncertain": "판단불가",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cohort-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/garment_segmentation/attribute_cluster_eval-20261002",
    )
    p.add_argument(
        "--agent-labels",
        type=Path,
        default=None,
        help="labels.jsonl from agent; defaults to cohort-dir/labels.jsonl",
    )
    return p.parse_args()


def _load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[str(row["item_id"])] = row
    return out


def main() -> None:
    args = parse_args()
    root = args.cohort_dir
    cohort = json.loads((root / "cohort.json").read_text(encoding="utf-8"))
    review_ids = list(cohort["review_item_ids"])
    by_id = {row["item_id"]: row for row in cohort["items"]}
    agent_path = args.agent_labels or (root / "labels.jsonl")
    agent = _load_jsonl(agent_path)

    review_items: list[dict[str, Any]] = []
    missing_agent = 0
    for iid in review_ids:
        row = by_id[iid]
        agent_row = agent.get(iid) or {}
        labels = agent_row.get("labels") or {}
        if not labels:
            missing_agent += 1
        review_items.append(
            {
                "item_id": iid,
                "product": row.get("product"),
                "subtype": row.get("subtype"),
                "origin": row.get("origin"),
                "thumb": f"thumbs/{iid}.jpg",
                "agent_labels": labels,
                "provisional": row.get("provisional") or {},
            }
        )

    attributes = {attr: list(cohort["attributes"][attr]) + ["uncertain"] for attr in ATTR_ORDER}
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "agent_labels_path": str(agent_path),
        "missing_agent_labels": missing_agent,
        "attributes": attributes,
        "attr_order": list(ATTR_ORDER),
        "attr_labels_ko": ATTR_LABELS_KO,
        "class_labels_ko": CLASS_LABELS_KO,
        "items": review_items,
    }
    data_js = root / "review_data.js"
    data_js.write_text(
        "window.REVIEW_DATA = " + json.dumps(payload, ensure_ascii=False, indent=2) + ";\n",
        encoding="utf-8",
    )

    html = """<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>속성 라벨 검수 · 50장</title>
<style>
:root{--ink:#182825;--muted:#64746f;--line:#dce4df;--green:#17634e;--bg:#f5f7f4;--warn:#8a4b00}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,sans-serif}
header{max-width:1100px;margin:auto;padding:28px 24px 12px}
h1{font-size:26px;letter-spacing:-.03em;margin:6px 0}
p{color:var(--muted);margin:6px 0}
.badge{display:inline-block;background:#e5ede7;border-radius:5px;padding:3px 8px;font-size:12px;margin:8px 6px 0 0}
.toolbar{position:sticky;top:0;z-index:2;background:#f5f7f4f2;backdrop-filter:blur(10px);border-bottom:1px solid var(--line)}
.bar{max-width:1100px;margin:auto;padding:12px 24px;display:flex;gap:12px;flex-wrap:wrap;align-items:center;justify-content:space-between}
button{border:1px solid var(--line);background:white;border-radius:8px;padding:8px 14px;cursor:pointer}
button.primary{background:var(--green);color:white;border-color:var(--green)}
button:disabled{opacity:.4;cursor:default}
main{max-width:1100px;margin:auto;padding:20px 24px 60px}
.card{display:grid;grid-template-columns:minmax(240px,360px) 1fr;gap:22px;background:white;border:1px solid var(--line);border-radius:14px;overflow:hidden}
.photo{background:#eef1ec;display:flex;align-items:center;justify-content:center;padding:16px;min-height:360px}
.photo img{max-width:100%;max-height:520px;object-fit:contain}
.meta{padding:18px 20px 22px}
.meta h2{font-size:16px;margin:0 0 6px}
.meta .sub{font-size:12px;color:var(--muted);margin-bottom:14px}
.attr{margin-bottom:14px}
.attr h3{font-size:13px;margin:0 0 6px;color:var(--muted)}
.choices{display:flex;flex-wrap:wrap;gap:6px}
.choices button{font-size:13px;padding:7px 10px}
.choices button[aria-pressed=true]{background:var(--green);color:white;border-color:var(--green)}
.agent{font-size:12px;color:var(--muted);margin-top:4px}
.agent.warn{color:var(--warn)}
.progress{font-size:13px}
.nav{display:flex;gap:8px}
textarea{width:100%;min-height:60px;margin-top:8px;border:1px solid var(--line);border-radius:8px;padding:8px;font:inherit}
@media(max-width:800px){.card{grid-template-columns:1fr}.photo{min-height:240px}}
</style>
</head>
<body>
<header>
  <div class="badge">ATTRIBUTE CLUSTER EVAL</div>
  <h1>속성 라벨 검수 (50장)</h1>
  <p>에이전트 1차 라벨과 독립적으로 선택하세요. 완료 후 Export로 human_labels.jsonl을 저장합니다. Cohen κ는 두 JSONL을 비교해 계산합니다.</p>
  <span class="badge" id="agent-status"></span>
</header>
<div class="toolbar"><div class="bar">
  <div class="progress" id="progress"></div>
  <div class="nav">
    <button id="prev">← 이전</button>
    <button id="next">다음 →</button>
    <button id="export" class="primary">Export human JSONL</button>
  </div>
</div></div>
<main>
  <article class="card">
    <div class="photo"><img id="img" alt=""/></div>
    <div class="meta">
      <h2 id="title"></h2>
      <div class="sub" id="sub"></div>
      <div id="attrs"></div>
      <label class="notes-label">메모
        <textarea id="notes" placeholder="애매한 점 메모"></textarea>
      </label>
    </div>
  </article>
</main>
<script src="review_data.js"></script>
<script>
const DATA = window.REVIEW_DATA;
if(!DATA || !DATA.items || !DATA.items.length){
  document.getElementById('agent-status').textContent = 'review_data.js 로드 실패';
  throw new Error('REVIEW_DATA missing');
}
const storageKey = 'attr_label_review_v1';
let idx = 0;
const human = JSON.parse(localStorage.getItem(storageKey) || '{}');

function ensure(itemId){
  if(!human[itemId]) human[itemId] = {item_id:itemId, labeler:'human', labels:{}, notes:''};
  return human[itemId];
}
function save(){ localStorage.setItem(storageKey, JSON.stringify(human)); updateProgress(); }

function updateProgress(){
  const done = DATA.items.filter(it => {
    const h = human[it.item_id];
    if(!h) return false;
    return DATA.attr_order.every(a => h.labels[a]);
  }).length;
  document.getElementById('progress').textContent = `${idx+1} / ${DATA.items.length} · 완료 ${done}장`;
  document.getElementById('agent-status').textContent =
    DATA.missing_agent_labels ? `에이전트 라벨 미완 ${DATA.missing_agent_labels}장` : '에이전트 라벨 로드됨';
}

function render(){
  const it = DATA.items[idx];
  const h = ensure(it.item_id);
  document.getElementById('img').src = it.thumb;
  document.getElementById('img').alt = it.product || it.item_id;
  document.getElementById('title').textContent = it.product || it.item_id;
  document.getElementById('sub').textContent = `${it.item_id} · ${it.subtype||''} · ${it.origin||''}`;
  document.getElementById('notes').value = h.notes || '';
  const root = document.getElementById('attrs');
  root.innerHTML = '';
  for(const attr of DATA.attr_order){
    const wrap = document.createElement('div');
    wrap.className = 'attr';
    const title = document.createElement('h3');
    title.textContent = DATA.attr_labels_ko[attr] || attr;
    wrap.appendChild(title);
    const choices = document.createElement('div');
    choices.className = 'choices';
    for(const cls of DATA.attributes[attr]){
      const btn = document.createElement('button');
      const ko = (DATA.class_labels_ko[attr]||{})[cls] || cls;
      btn.type = 'button';
      btn.textContent = ko;
      btn.setAttribute('aria-pressed', h.labels[attr]===cls ? 'true':'false');
      btn.onclick = () => { h.labels[attr]=cls; save(); render(); };
      choices.appendChild(btn);
    }
    wrap.appendChild(choices);
    const agent = document.createElement('div');
    const av = (it.agent_labels||{})[attr];
    agent.className = 'agent' + (av ? '' : ' warn');
    const ako = av ? ((DATA.class_labels_ko[attr]||{})[av]||av) : '없음';
    agent.textContent = `에이전트: ${ako}`;
    wrap.appendChild(agent);
    root.appendChild(wrap);
  }
  document.getElementById('prev').disabled = idx===0;
  document.getElementById('next').disabled = idx===DATA.items.length-1;
  updateProgress();
}

document.getElementById('prev').onclick = () => { idx=Math.max(0,idx-1); render(); };
document.getElementById('next').onclick = () => { idx=Math.min(DATA.items.length-1,idx+1); render(); };
document.getElementById('notes').oninput = (e) => { ensure(DATA.items[idx].item_id).notes=e.target.value; save(); };
document.getElementById('export').onclick = () => {
  const lines = DATA.items.map(it => {
    const h = ensure(it.item_id);
    return JSON.stringify({
      item_id: it.item_id,
      labeler: 'human',
      labels: h.labels,
      notes: h.notes||'',
      labeled_at: new Date().toISOString(),
    });
  }).join('\\n') + '\\n';
  const blob = new Blob([lines], {type:'application/jsonl'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'human_labels_review50.jsonl';
  a.click();
};
document.addEventListener('keydown', (e) => {
  if(e.target && e.target.id === 'notes') return;
  if(e.key==='ArrowLeft') document.getElementById('prev').click();
  if(e.key==='ArrowRight') document.getElementById('next').click();
});
render();
</script>
</body>
</html>
"""
    out_html = root / "label_review.html"
    out_html.write_text(html, encoding="utf-8")
    print(f"wrote {out_html}")
    print(f"wrote {data_js}")
    print(f"review_items={len(review_items)} missing_agent={missing_agent}")


if __name__ == "__main__":
    main()
