# utils/molstar_embed.py
from __future__ import annotations

import os, json, base64
import streamlit as st

try:
    import requests  # URL에서 PDB/JSON 로딩
except Exception:
    requests = None


# ===== 공통 유틸 =====
def _sites_to_list(sites: list[dict]) -> list[dict]:
    out = []
    for s in sites or []:
        if "resi" in s:
            out.append({
                "chain": str(s.get("chain", "A")).strip().upper(),
                "resi": int(s["resi"]),
                "ins": ("" if s.get("ins") in (None, "nan") else str(s.get("ins")).strip().upper())
            })
    return out

def _have_local_molstar_assets() -> tuple[str, str]:
    js_text, css_text = "", ""
    for p in ("molstar_3.46.0.js", "./molstar_3.46.0.js"):
        if os.path.exists(p):
            try: js_text = open(p, "r", encoding="utf-8", errors="ignore").read(); break
            except Exception: pass
    for p in ("molstar_3.46.0.css", "./molstar_3.46.0.css"):
        if os.path.exists(p):
            try: css_text = open(p, "r", encoding="utf-8", errors="ignore").read(); break
            except Exception: pass
    return js_text or "", css_text or ""

def _hex_to_jsint(hx: str) -> str:
    h = hx.lstrip("#")
    if len(h) == 3: h = "".join([c*2 for c in h])
    return f"0x{h.lower()}"


# ===== Mol* (인라인) =====
def _molstar_html_from_text(div_id: str, pdb_text: str, sites_json: str,
                            pos_hex: str, neg_hex: str, height: int,
                            pdb_format: str = "pdb",
                            show_left_panel: bool = False) -> str:
    """Mol* 로컬 자산 필요. show_left_panel=False로 초기 카메라 중앙 정렬."""
    js_text, css_text = _have_local_molstar_assets()
    if not js_text:
        raise RuntimeError("Molstar assets not found")

    js_b64  = base64.b64encode(js_text.encode("utf-8")).decode("ascii")
    css_b64 = base64.b64encode((css_text or "").encode("utf-8")).decode("ascii")
    pdb_b64 = base64.b64encode(pdb_text.encode("utf-8")).decode("ascii")

    pos_js = _hex_to_jsint(pos_hex)
    neg_js = _hex_to_jsint(neg_hex)

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"/>
<style>
  html,body{{margin:0;height:100%}}
  #{div_id}{{width:100%;height:{height}px}}
</style></head>
<body><div id="{div_id}"></div>
<script>
const MOLSTAR_JS_B64="{js_b64}", MOLSTAR_CSS_B64="{css_b64}";
(function(){{
  if(MOLSTAR_CSS_B64){{const s=document.createElement('style');s.textContent=atob(MOLSTAR_CSS_B64);document.head.appendChild(s)}}
  const js=document.createElement('script');js.textContent=atob(MOLSTAR_JS_B64);document.head.appendChild(js);
}})();

const PDB_B64="{pdb_b64}";
const SITES={sites_json};
const COLOR_POS={pos_js};
const COLOR_NEG={neg_js};

function atobSafe(b64){{try{{return atob(b64)}}catch(e){{return decodeURIComponent(escape(window.atob(b64)))}}}}

(async () => {{
  if(!window.molstar) throw new Error("molstar global not available");

  // 좌측 패널 숨김 → 뷰포트 정중앙
  const v = new molstar.Viewer("{div_id}", {{
    layoutIsExpanded: true,
    layoutShowControls: true,
    layoutShowSequence: true,
    layoutShowLog: false,
    layoutShowLeftPanel: {str(show_left_panel).lower()},
    viewportShowExpand: true,
    showImportControls: false,
    backgroundColor: 0xFFFFFF
  }});

  const pdbText = atobSafe(PDB_B64);
  const structure = await v.loadStructureFromData(pdbText, "{pdb_format}");

  // 전체 회색
  await v.plugin.builders.structure.representation.addRepresentation(structure,{{
    type:'cartoon', color:'uniform', size:'uniform',
    colorParams:{{value:COLOR_NEG}}
  }});

  if (Array.isArray(SITES) && SITES.length>0) {{
    function emptyIns(x) {{
      if (x==null) return true;
      const s=String(x).trim().toLowerCase();
      return !s || s==='nan' || s==='none' || s==='null' || s==='-' || s==='없음';
    }}
    const sel = molstar.Script.getStructureSelection(q => {{
      return q.struct.modifier.union(SITES.map(s => {{
        const tests = [
          q.core.rel.eq([ q.struct.atomProperty.macromolecular.auth_asym_id(), q.core.type.str(String(s.chain)) ]),
          q.core.rel.eq([ q.struct.atomProperty.macromolecular.auth_seq_id(),  q.core.type.int(parseInt(s.resi)) ])
        ];
        if (!emptyIns(s.ins)) {{
          tests.push(q.core.rel.eq([ q.struct.atomProperty.macromolecular.label_ins_code(), q.core.type.str(String(s.ins)) ]));
        }}
        return q.struct.generator.atomGroups({{'residue-test': q.core.logic.and(tests)}});
      }}));
    }}, structure.data);

    const comp = await v.plugin.builders.structure.tryCreateComponentFromSelection(structure, sel, 'epitope');
    if (comp) {{
      await v.plugin.builders.structure.representation.addRepresentation(comp,{{
        type:'cartoon', color:'uniform', size:'uniform',
        colorParams:{{value:COLOR_POS}}
      }});
    }}
  }}

  v.plugin.managers.camera.focusRigorous();
}})();
</script></body></html>"""


# ===== py3Dmol (폴백) =====
def _py3dmol_html_from_text(pdb_text: str, sites: list[dict],
                            pos_hex: str, neg_hex: str, height: int,
                            fmt: str = "pdb") -> str:
    import py3Dmol
    # 정사각형 캔버스 → 시각적으로 중앙감 ↑
    width = height
    view = py3Dmol.view(width=width, height=height)
    view.addModel(pdb_text, "cif" if fmt in ("cif", "bcif") else "pdb")
    view.setStyle({"cartoon": {"color": neg_hex}})
    by_chain = {}
    for s in _sites_to_list(sites):
        by_chain.setdefault(s["chain"], set()).add(int(s["resi"]))
    for ch, resis in by_chain.items():
        view.setStyle({"chain": ch, "resi": sorted(list(resis))}, {"cartoon": {"color": pos_hex}})
    view.zoomTo()
    return view._make_html()


# ===== 공개 API =====
def molstar_paint_truth(struct_url: str, true_sites: list[dict],
                        pos_hex: str = "#ff0000", neg_hex: str = "#c0c0c0", height: int = 640) -> None:
    js_text, _ = _have_local_molstar_assets()

    pdb_text = None
    if requests is not None:
        try:
            r = requests.get(struct_url, timeout=20)
            if r.ok: pdb_text = r.text
        except Exception:
            pass

    if js_text and pdb_text:
        html = _molstar_html_from_text("molstar-truth", pdb_text, json.dumps(_sites_to_list(true_sites)),
                                       pos_hex, neg_hex, height, pdb_format="pdb",
                                       show_left_panel=False)
        st.components.v1.html(html, height=height, scrolling=False)
    elif pdb_text:
        html = _py3dmol_html_from_text(pdb_text, true_sites, pos_hex, neg_hex, height)
        st.components.v1.html(html, height=height, scrolling=False)
    else:
        st.error("PDB를 불러올 수 없습니다(네트워크/권한).")


def molstar_paint_truth_from_text(pdb_text: str, true_sites: list[dict],
                                  pos_hex: str = "#ff0000", neg_hex: str = "#c0c0c0",
                                  height: int = 640, pdb_format: str = "pdb") -> None:
    js_text, _ = _have_local_molstar_assets()
    if js_text:
        html = _molstar_html_from_text("molstar-truth-text", pdb_text, json.dumps(_sites_to_list(true_sites)),
                                       pos_hex, neg_hex, height, pdb_format=pdb_format,
                                       show_left_panel=False)
        st.components.v1.html(html, height=height, scrolling=False)
    else:
        html = _py3dmol_html_from_text(pdb_text, true_sites, pos_hex, neg_hex, height, fmt=pdb_format)
        st.components.v1.html(html, height=height, scrolling=False)


def molstar_paint_pred_binary_from_url(struct_url: str, pred_url: str,
                                       threshold: float = 0.5,
                                       pos_hex: str = "#ff0000", neg_hex: str = "#c0c0c0",
                                       height: int = 640) -> None:
    # 1) 예측 JSON → sites
    sites = []
    if requests is not None:
        try:
            r = requests.get(pred_url, timeout=20)
            if r.ok:
                data = r.json()
                arr = data.get("predictions") or data.get("data") or []
                for p in arr:
                    score = float(p.get("score") or p.get("prob") or p.get("probability") or 0.0)
                    if score >= float(threshold) and (p.get("is_epitope") is None or float(p.get("is_epitope")) > 0):
                        resi = p.get("pdb_resseq") or p.get("resi") or p.get("resid") or p.get("position")
                        try:
                            rint = int(str(resi).strip().split()[0])
                            sites.append({"chain": str(p.get("chain") or "A"), "resi": rint})
                        except Exception:
                            continue
        except Exception:
            pass

    # 2) PDB 텍스트
    pdb_text = None
    if requests is not None:
        try:
            rr = requests.get(struct_url, timeout=20)
            if rr.ok: pdb_text = rr.text
        except Exception:
            pass

    js_text, _ = _have_local_molstar_assets()
    if js_text and pdb_text:
        html = _molstar_html_from_text("molstar-pred", pdb_text, json.dumps(_sites_to_list(sites)),
                                       pos_hex, neg_hex, height, pdb_format="pdb",
                                       show_left_panel=False)
        st.components.v1.html(html, height=height, scrolling=False)
    elif pdb_text:
        html = _py3dmol_html_from_text(pdb_text, sites, pos_hex, neg_hex, height)
        st.components.v1.html(html, height=height, scrolling=False)
    else:
        st.error("시각화를 위해 PDB/예측 JSON을 불러오지 못했습니다.")


# ---- 단일 색상 뷰어 (Structures 탭용) ----
def show_structure_from_text(pdb_text: str, color: str = "#8a8a8a",
                             height: int = 600, pdb_format: str = "pdb") -> None:
    """단일 색으로 전체를 보여주는 간단 뷰어 (Mol* 우선, py3Dmol 폴백)."""
    js_text, _ = _have_local_molstar_assets()
    if js_text:
        html = _molstar_html_from_text("molstar-uniform", pdb_text, "[]",
                                       pos_hex=color,  # 사용되지 않음
                                       neg_hex=color,
                                       height=height, pdb_format=pdb_format,
                                       show_left_panel=False)
        st.components.v1.html(html, height=height, scrolling=False)
    else:
        html = _py3dmol_html_from_text(pdb_text, [], pos_hex=color, neg_hex=color, height=height, fmt=pdb_format)
        st.components.v1.html(html, height=height, scrolling=False)


def show_structure_from_url(url: str, color: str = "#8a8a8a",
                            height: int = 600, pdb_format: str = "pdb") -> None:
    """URL(Presigned S3 등)에서 받아와 단일 색으로 렌더."""
    if requests is None:
        st.error("네트워크 요청 모듈이 비활성화되어 URL을 읽을 수 없습니다.")
        return
    pdb_text = None
    try:
        r = requests.get(url, timeout=20)
        if r.ok:
            pdb_text = r.text
    except Exception as e:
        st.error(f"PDB URL 로딩 실패: {e}")
        return
    if not pdb_text:
        st.error("PDB URL에서 데이터를 읽지 못했습니다.")
        return
    show_structure_from_text(pdb_text, color=color, height=height, pdb_format=pdb_format)
    
def show_structure_confidence_from_text(pdb_text: str,
                                        height: int = 460,
                                        pdb_format: str = "pdb",
                                        vmin: float = 0.0,
                                        vmax: float = 100.0) -> None:
    """
    pLDDT가 B-factor(TempFactor)에 들어있는 PDB를 pLDDT 그라디언트로 컬러링해 보여줍니다.
    Mol* 대신 py3Dmol을 사용해 호환성 확보.
    """
    import py3Dmol
    view = py3Dmol.view(width=height, height=height)  # 정사각 캔버스 → 중앙감 ↑
    view.addModel(pdb_text, "cif" if pdb_format in ("cif", "bcif") else "pdb")
    view.setStyle({
        "cartoon": {
            "colorscheme": {
                "prop": "b",         # B-factor
                "gradient": "roygb", # 0-100에 잘 맞는 스킴
                "min": vmin,
                "max": vmax
            }
        }
    })
    view.zoomTo()
    html = view._make_html()
    st.components.v1.html(html, height=height, scrolling=False)


def show_structure_confidence_from_url(url: str,
                                       height: int = 460,
                                       pdb_format: str = "pdb",
                                       vmin: float = 0.0,
                                       vmax: float = 100.0) -> None:
    """URL에서 PDB 텍스트를 읽어 pLDDT 컬러링 표시."""
    if requests is None:
        st.error("requests 모듈이 비활성화되어 URL을 읽을 수 없습니다.")
        return
    try:
        r = requests.get(url, timeout=20)
        if not r.ok:
            st.error(f"PDB 로딩 실패(status={r.status_code})"); return
        show_structure_confidence_from_text(r.text, height=height, pdb_format=pdb_format,
                                            vmin=vmin, vmax=vmax)
    except Exception as e:
        st.error(f"PDB URL 로딩 실패: {e}")
        
def molstar_paint_pred_gradient_from_url(struct_url: str, pred_url: str,
                                         min_score: float = 0.0, max_score: float = 1.0,
                                         neg_hex: str = "#c0c0c0", pos_hex: str = "#ff0000",
                                         height: int = 640) -> None:
    """
    예측 확률(0-1)을 회색→빨강 그라데이션으로 강조. (py3Dmol 사용)
    - 구조 전체는 회색(neg_hex)으로 표시
    - 예측 값 있는 잔기는 값에 비례해 pos_hex 쪽으로 보간
    JSON 형식 예:
    { "predictions": [{"chain":"A","pdb_resseq":42,"score":0.87}, ...] }
    """
    try:
        import py3Dmol, requests
    except Exception:
        st.error("py3Dmol/requests 모듈이 필요합니다."); return

    # 1) 예측 파싱
    mapping = {}  # (chain, resi) -> score
    try:
        r = requests.get(pred_url, timeout=20)
        if r.ok:
            data = r.json()
            arr = data.get("predictions") or data.get("data") or []
            for p in arr:
                score = p.get("score", p.get("prob", p.get("probability", None)))
                resi  = p.get("pdb_resseq", p.get("resi", p.get("resid", p.get("position", None))))
                chain = p.get("chain", "A")
                try:
                    score = float(score); resi = int(str(resi).strip().split()[0])
                except Exception:
                    continue
                mapping[(str(chain), int(resi))] = score
    except Exception as e:
        st.error(f"Prediction JSON 로딩 실패: {e}"); return

    # 2) PDB 불러오기
    pdb_text = None
    try:
        rr = requests.get(struct_url, timeout=20)
        if rr.ok: pdb_text = rr.text
    except Exception as e:
        st.error(f"PDB URL 로딩 실패: {e}"); return
    if not pdb_text:
        st.error("PDB를 불러오지 못했습니다."); return

    # 3) 색 보간 함수
    def _hex_to_rgb(h): h=h.lstrip("#"); 
    # 3-digit -> 6-digit
    def _hex_to_rgb(h): 
        h=h.lstrip("#"); 
        if len(h)==3: h="".join([c*2 for c in h]); 
        return tuple(int(h[i:i+2],16) for i in (0,2,4))
    def _rgb_to_hex(rgb): return "#{:02x}{:02x}{:02x}".format(*rgb)
    import math
    def _lerp(a,b,t): return int(round(a+(b-a)*t))
    c0=_hex_to_rgb(neg_hex); c1=_hex_to_rgb(pos_hex)
    def _mix_color(t):
        t=0.0 if math.isnan(t) else max(0.0, min(1.0, t))
        return _rgb_to_hex((_lerp(c0[0],c1[0],t), _lerp(c0[1],c1[1],t), _lerp(c0[2],c1[2],t)))

    # 4) 렌더
    view = py3Dmol.view(width=height, height=height)
    view.addModel(pdb_text, "pdb")
    view.setStyle({"cartoon": {"color": neg_hex}})  # 전체 회색

    if max_score <= min_score: max_score = min_score + 1e-6
    for (chain, resi), sc in mapping.items():
        t = (float(sc) - float(min_score)) / (float(max_score) - float(min_score))
        color = _mix_color(t)
        view.setStyle({"chain": str(chain), "resi": int(resi)}, {"cartoon": {"color": color}})

    view.zoomTo()
    st.components.v1.html(view._make_html(), height=height, scrolling=False)
