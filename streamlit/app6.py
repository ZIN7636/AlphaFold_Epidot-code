# app6.py — Epitope Platform (S3 input/output split; imports from utils/)
# - Sidebar: 알레르겐 이름으로 과거 run 검색/로드 (run_id 직접 입력 UI 없음)
# - New Run: FASTA + Allergen name → Start Pipeline 시 meta.json 자동 저장
# - Monitor: status.json 기반 진행률과 단계 상태 표시
# - Structures: AF2/ESMFold pLDDT/PAE/pTM(ipTM), 사용자 PDB는 단일색, Mol* height 고정(460)
# - Feature Extraction: ESM2/DSSP/Protein Graph를 feature_io 로더로 읽어 표준화, 시각화
# - Final Epitope: Binary/Gradient 컬러링, Ground Truth 업로드, Feature Importance 막대그래프

from __future__ import annotations
import io, os, re, json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from streamlit_autorefresh import st_autorefresh
from PIL import Image

# ====== local utils (utils/ 폴더) ======
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from utils.config import (
    S3_INPUT_BUCKET, S3_OUTPUT_BUCKET,
    RUNS_PREFIX, INPUT_PREFIX
)
from utils.aws_io import (
    s3, presign_get, s3_exists, s3_read_json, s3_upload_fileobj, s3_put_json,
    kjoin, gen_run_id, lambda_invoke, s3_list_runs
)
from utils.molstar_embed import (
    molstar_paint_pred_binary_from_url,
    molstar_paint_pred_gradient_from_url,
    molstar_paint_truth_from_text,
    show_structure_from_text, show_structure_from_url,
    show_structure_confidence_from_url,
)
from utils.feature_io import (
    load_esm2_array, load_dssp, load_protein_graph, summarise_graph
)

st.set_page_config(page_title="Epitope Pipeline", layout="wide")

import base64
import os

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

current_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(current_dir, "logo.png")
logo_base64 = get_base64_image(logo_path)

st.markdown(f"""
<div style="background: linear-gradient(135deg, #232526 0%, #414345 100%); padding: 2rem; border-radius: 10px; text-align: center; margin-bottom: 2rem;">
<h1 style="color: white; margin: 0;"><img src="data:image/png;base64,{logo_base64}" style="height: 60px; margin-bottom: 1rem;" alt="Logo">
 EpiDot AI</h1>
<h3 style="color: white; margin: 0.5rem 0;">Food Allergen Protein B-cell Epitope Prediction AI Solution</h3>
</div>
""", unsafe_allow_html=True)

#st_autorefresh(interval=5000, key="monitor_autorefresh")

# ====== Session state defaults ======
if "run_id" not in st.session_state: st.session_state.run_id = None
if "q_name" not in st.session_state: st.session_state.q_name = ""       # sidebar search
if "struct_color" not in st.session_state: st.session_state.struct_color = "#8a8a8a"
if "pred_thr" not in st.session_state: st.session_state.pred_thr = 0.5

# ====== Helpers ======
def _render_html_table(df: pd.DataFrame, height: int = 320):
    html = f"""
    <style>
      .tbl-wrap {{max-height:{height}px;overflow:auto;border:1px solid #e8e8e8;border-radius:8px;}}
      .tbl-wrap table {{width:100%;border-collapse:collapse;font-size:14px}}
      .tbl-wrap th,.tbl-wrap td {{padding:6px 10px;border-bottom:1px solid #f0f0f0;white-space:nowrap;text-align:left}}
      .tbl-wrap thead th {{position:sticky;top:0;background:#fafafa;z-index:1}}
    </style>
    <div class="tbl-wrap">{df.to_html(index=False, escape=False, border=0)}</div>
    """
    st.components.v1.html(html, height=height+30, scrolling=True)

def _display_grid_or_table(df: pd.DataFrame, height: int = 320):
    try:
        st.dataframe(df, use_container_width=True, height=height, hide_index=True)
    except Exception:
        _render_html_table(df, height=height)

# ---- pLDDT (AF2/ESM 전용) ----
def _extract_plddt_df(pdb_text: str) -> pd.DataFrame:
    rows = []
    for ln in pdb_text.splitlines():
        if not (ln.startswith("ATOM") or ln.startswith("HETATM")):
            continue
        try:
            if ln[12:16].strip() != "CA":  # CA 원자만 사용
                continue
            chain = (ln[21].strip() or "A")
            resi = int(ln[22:26].strip())
            icode = ln[26].strip() or None
            bfac = float(ln[60:66].strip())
            rows.append((chain, resi, icode, bfac))
        except Exception:
            continue
    if not rows:
        return pd.DataFrame(columns=["idx","chain","resi","icode","plddt"])
    df = pd.DataFrame(rows, columns=["chain","resi","icode","plddt"])
    df.insert(0, "idx", range(1, len(df)+1))
    return df

def _plddt_summary_block(df: pd.DataFrame):
    if df.empty:
        st.caption("pLDDT를 추출할 수 없습니다.")
        return
    import altair as alt
    bins = [-1,50,70,90,101]
    labels = ["very low (<50)","low (50–70)","confident (70–90)","very high (≥90)"]
    s = pd.cut(df["plddt"], bins=bins, labels=labels).value_counts(normalize=True)\
        .reindex(labels, fill_value=0)*100
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Mean pLDDT", f"{df['plddt'].mean():.1f}")
    c2.metric("Median", f"{df['plddt'].median():.1f}")
    c3.metric("≥90 (%)", f"{s.get('very high (≥90)', 0):.1f}")
    c4.metric("50–70 (%)", f"{s.get('low (50–70)', 0):.1f}")

    chart = alt.Chart(df).mark_line().encode(
        x=alt.X("idx:Q", title="Residue index (CA)"),
        y=alt.Y("plddt:Q", title="pLDDT (0–100)", scale=alt.Scale(domain=[0,100])),
        tooltip=["idx","chain","resi","plddt"]
    ).properties(height=180)
    rules = alt.Chart(pd.DataFrame({"y":[50,70,90]})).mark_rule(strokeDash=[4,4]).encode(y="y:Q")
    st.altair_chart(chart + rules, use_container_width=True)

# ---- pTM/ipTM & PAE helpers ----
def _first_existing_key(cands: list[str]) -> str|None:
    for k in cands:
        if s3_exists(k):
            return k
    return None

def _read_json_from_outputs(key: str) -> dict|None:
    try:
        obj = s3.get_object(Bucket=S3_OUTPUT_BUCKET, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))
    except Exception:
        return None

def _extract_ptm_like_scores(js: dict) -> dict:
    if not isinstance(js, dict):
        return {}
    low = {k.lower(): k for k in js.keys()}
    out = {}
    for cand in ["ptm","ptm_score","predicted_tm_score","avg_ptm"]:
        if cand in low:
            out["pTM"] = js[low[cand]]
            break
    for cand in ["iptm","iptm_score","predicted_interface_tm_score"]:
        if cand in low:
            out["ipTM"] = js[low[cand]]
            break
    if not out and "ranking_debug" in low:
        rd = js[low["ranking_debug"]]
        if isinstance(rd, dict):
            for k, v in rd.items():
                if isinstance(v, (float, int)) and "ptm" in k.lower():
                    out["pTM"] = v
                    break
    return out

    
def _render_pae_heatmap(pae_matrix: list[list[float]], title: str = "PAE heatmap"):
    import altair as alt
    import numpy as np

    A = pd.DataFrame(pae_matrix, dtype=float).values
    n = A.shape[0]
    original_n = n  
    
    # PAE 통계 정보 표시
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Matrix Size", f"{n}×{n}")
    with col2:
        st.metric("Min PAE", f"{A.min():.1f}Å")
    with col3:
        st.metric("Max PAE", f"{A.max():.1f}Å") 
    with col4:
        st.metric("Avg PAE", f"{A.mean():.1f}Å")
    
    # 신뢰도 분석
    confident_pairs = (A < 5.0).sum()
    total_pairs = A.size
    confidence_ratio = confident_pairs / total_pairs
    
    # 색상 구분 기준 설명
    st.write("**PAE 해석**: Lower values = Better confidence")
    st.write(f"- 🟢 **Very High** (< 5Å): {((A < 5.0).sum()/total_pairs*100):.1f}%")
    st.write(f"- 🟡 **Confident** (5-10Å): {(((A >= 5.0) & (A < 10.0)).sum()/total_pairs*100):.1f}%")
    st.write(f"- 🟠 **Low** (10-15Å): {(((A >= 10.0) & (A < 15.0)).sum()/total_pairs*100):.1f}%")
    st.write(f"- 🔴 **Very Low** (≥15Å): {((A >= 15.0).sum()/total_pairs*100):.1f}%")
    
    # 샘플링을 더 강하게
    if n > 100:  # 400에서 100으로 줄임
        stride = max(2, int(np.ceil(n/100)))
        A = A[::stride, ::stride]
        n = A.shape[0]
        st.info(f"매트릭스를 {original_n}×{original_n} → {n}×{n}로 샘플링")
    
    # DataFrame을 더 간단하게 생성
    df_data = []
    for i in range(n):
        for j in range(n):
            df_data.append([i+1, j+1, float(A[i, j])])
    
    df = pd.DataFrame(df_data, columns=['i', 'j', 'pae'])
    
    # Plotly로 대체 (Altair 대신)
    import plotly.express as px
    
    # 2D 배열을 다시 만들기
    pivot_df = df.pivot(index='j', columns='i', values='pae')
    
    fig = px.imshow(
        pivot_df,
        labels=dict(x="Residue i", y="Residue j", color="PAE (Å)"),
        title=f"{title} - Overall Confidence: {confidence_ratio:.1%}",
        color_continuous_scale="RdYlBu_r",
        aspect="equal"
    )
    
    fig.update_layout(
    height=600,  # 400에서 600으로 증가
    width=600,   # 400에서 600으로 증가
    margin=dict(l=50, r=50, t=80, b=50)  # 여백도 조정
)
    st.plotly_chart(fig, use_container_width=True)
    
    # 도메인 분석 (선택적)
    if n <= 200:  # 너무 크지 않은 경우만
        
        # 대각선 근처의 평균 PAE (로컬 구조 신뢰도)
        local_pae = []
        for i in range(min(50, n-10)):
            diag_region = A[i:i+10, i:i+10]
            local_pae.append(diag_region.mean())
        
        if local_pae:
            local_df = pd.DataFrame({
                "position": range(len(local_pae)),
                "local_confidence": local_pae
            })
            
            local_chart = alt.Chart(local_df).mark_line(point=True).encode(
                x=alt.X("position:Q", title="Residue Position"),
                y=alt.Y("local_confidence:Q", title="Local PAE (Å)"),
                tooltip=["position", "local_confidence"]
            ).properties(
                title="Local Structure Confidence Along Sequence",
                height=200
            )
            
            st.altair_chart(local_chart, use_container_width=True)
# ========================= Sidebar (Name search only) =========================


with st.sidebar:
    st.markdown("### Load Run by Allergen name")
    query = st.text_input("e.g. Ara h 2", key="q_name")

    colR1, colR2 = st.columns([1,1])
    with colR1:
        refresh = st.button("🔄 Refresh index", use_container_width=True)
    with colR2:
        show_all = st.checkbox("Show all runs", value=False)

    @st.cache_data
    def _runs_index_df() -> pd.DataFrame:
        rows = []
        for rid in s3_list_runs():
            meta = s3_read_json(kjoin(RUNS_PREFIX, rid, "meta.json")) or {}
            rows.append({
                "run_id": rid,
                "allergen_name": (meta.get("allergen_name") or meta.get("target_name") or "").strip(),
                "created_utc": meta.get("created_utc") or ""
            })
        return pd.DataFrame(rows)

    if refresh:
        _runs_index_df.clear()

    dfidx = _runs_index_df()
    if show_all and not dfidx.empty:
        _display_grid_or_table(dfidx, height=220)

    def _norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", str(s).lower())

    if query:
        qn = _norm(query)
        matches_df = dfidx[
        dfidx["allergen_name"].apply(lambda x: qn in _norm(x)) |
        dfidx["run_id"].apply(lambda x: qn in _norm(x))
    ]
    else:
        matches_df = dfidx.iloc[0:0]

    match_ids = list(matches_df["run_id"])
    def _fmt_run(rid: str) -> str:
        row = matches_df[matches_df["run_id"] == rid].iloc[0]
        nm = row["allergen_name"] or "(no name)"
        ct = row["created_utc"] or "(unknown)"
        return f"{nm} — {rid} ({ct})"

    sel_by_name = st.selectbox(
        "Matches", match_ids, format_func=_fmt_run,
        index=0 if match_ids else None,
        placeholder="Type a name & Enter",
        disabled=(len(match_ids) == 0),
    )
    if st.button("Load", use_container_width=True, disabled=(len(match_ids) == 0)):
        st.session_state.run_id = sel_by_name
        st.success(f"Loaded run: {sel_by_name}")

# =========================== NAV ===========================
nav = st.sidebar.radio("Tab", [
    "0) New Run",
    "1) Monitor",
    "2) Structures",
    "3) Features",
    "4) Final Epitope"
], index=0)

# ------------------------ 0) New Run ------------------------
if nav.startswith("0)"):
    st.header("New Run")
    fasta = st.file_uploader("Upload FASTA", type=["fasta","fa","txt"])
    prefill = st.session_state.get("q_name", "")
    allergen = st.text_input(
        "Allergen name (e.g., Ara h 2)",
        value=prefill, placeholder="required for easy lookup",
        key="newrun_allergen"
    )
    

    if st.button("▶ Start Pipeline"):
        if not fasta:
            st.error("FASTA를 업로드하세요.")
        elif not allergen.strip():
            st.error("Allergen name을 입력해주세요.")
        else:
            # run_id는 추적용으로만 사용
            run_id = gen_run_id()
            in_key = kjoin(INPUT_PREFIX, fasta.name)
            
            # FASTA 업로드만 수행
            s3_upload_fileobj(fasta, in_key)
            st.success(f"Uploaded FASTA → s3://{S3_INPUT_BUCKET}/{in_key}")
            
            # 메타데이터는 로컬 모니터링 시스템이 생성하도록 함
            st.info("Pipeline이 자동으로 시작됩니다. 결과는 파일명 기반으로 저장됩니다.")
            
            # session_state는 임시 추적용
            st.session_state.run_id = fasta.name.split('.')[0]  # 파일명 기반

    rid = st.session_state.get("run_id")
    if rid:
        st.success(f"Current run_id: {rid}")
    
    st.markdown(f"""<h3 style="color: black; margin: 0.5rem 0;">EpiDot Model Architecture</h3>""", unsafe_allow_html=True)
    architecture_path = os.path.join(current_dir, "architecture.png")
    architecture_base64 = get_base64_image(architecture_path)   
    st.markdown(
        f'<img src="data:image/png;base64,{architecture_base64}" width="1300">',
        unsafe_allow_html=True
    )

    

# ------------------------ 1) Monitor ------------------------
elif nav.startswith("1)"):
    st.header("Monitor")
    rid = st.session_state.get("run_id")
    if not rid:
        st.warning("먼저 New Run에서 run을 시작/선택하세요.")
    else:
      # 메타데이터 파일로 완료 여부 확인
        meta_key = kjoin(RUNS_PREFIX, rid, "metadata.json")
        
        if s3_exists(meta_key):
            meta = s3_read_json(meta_key)
            status = meta.get("status", "unknown")
            
            if status == "completed":
                st.success("파이프라인 완료!")
                st.progress(1.0, text="100%")
            elif status == "failed":
                st.error("파이프라인 실패")
                error_msg = meta.get("error", "Unknown error")
                st.error(f"오류: {error_msg}")
            else:
                st.info("처리 중...")
                st.progress(0.5, text="50%")
        else:
            st.info("처리 대기 중...")
            st.progress(0.1, text="10%")
        
        if st.button("🔄 Refresh"):
            st.experimental_rerun()
# ------------------------ 2) Structures ------------------------
elif nav.startswith("2)"):
    st.header("Structures")
    rid = st.session_state.get("run_id")
    af_key  = kjoin(RUNS_PREFIX, rid, "alphafold_result.pdb") if rid else None
    esm_key = kjoin(RUNS_PREFIX, rid, "esmfold_result.pdb")   if rid else None

    height = 460
    color  = st.color_picker("Uniform color", st.session_state.struct_color, key="struct_color")
    color_conf = st.toggle("Color AlphaFold/ESMFold by confidence (pLDDT)", value=False)

    col_left, col_mid, col_right = st.columns(3)

    # AF2
    with col_left:
        st.subheader("AlphaFold")
        if rid and s3_exists(af_key):
            url = presign_get(af_key)
            
            if color_conf:
                show_structure_confidence_from_url(url, height=height)
            else:
                show_structure_from_url(url, color=color, height=height)
            
            # pLDDT
            obj = s3.get_object(Bucket=S3_OUTPUT_BUCKET, Key=af_key)
            df = _extract_plddt_df(obj["Body"].read().decode("utf-8", errors="ignore"))
            with st.expander("Confidence (pLDDT) summary", expanded=False):
                _plddt_summary_block(df)
            # pTM/ipTM
            af2_score_key = _first_existing_key([
                kjoin(RUNS_PREFIX, rid, "af2_confidence.json"),
                kjoin(RUNS_PREFIX, rid, "ptm.json"),
                kjoin(RUNS_PREFIX, rid, "ranking_debug.json"),
            ])
            if af2_score_key:
                js = _read_json_from_outputs(af2_score_key)
                scores = _extract_ptm_like_scores(js or {})
                if scores:
                    with st.expander("Model-level confidence (pTM / ipTM)", expanded=False):
                        cols = st.columns(len(scores))
                        for (name,val), c in zip(scores.items(), cols):
                            try:
                                num = float(val)
                                c.metric(name, f"{num*100:.1f}%" if 0<=num<=1.5 else f"{num:.3f}")
                            except Exception:
                                c.metric(name, str(val))
            # PAE
            af2_pae_key = _first_existing_key([
                kjoin(RUNS_PREFIX, rid, "af2_pae.json"),
                kjoin(RUNS_PREFIX, rid, "predicted_aligned_error.json"),
                kjoin(RUNS_PREFIX, rid, "pae.json"),
            ])
            if af2_pae_key:
                js = _read_json_from_outputs(af2_pae_key) or {}
                pae = js.get("pae") or js.get("predicted_aligned_error")
                if isinstance(pae, list) and pae and isinstance(pae[0], list):
                    with st.expander("PAE heatmap", expanded=False):
                        _render_pae_heatmap(pae, title="AF2 PAE")
        else:
            st.info("alphafold_result.pdb 대기 중…")

    # ESMFold
    with col_mid:
        st.subheader("ESMFold")
        if rid and s3_exists(esm_key):
            url = presign_get(esm_key)
            if color_conf:
                show_structure_confidence_from_url(url, height=height)
            else:
                show_structure_from_url(url, color=color, height=height)
            obj = s3.get_object(Bucket=S3_OUTPUT_BUCKET, Key=esm_key)
            df = _extract_plddt_df(obj["Body"].read().decode("utf-8", errors="ignore"))
            with st.expander("Confidence (pLDDT) summary", expanded=False):
                _plddt_summary_block(df)
            # pTM/ipTM
            esm_score_key = _first_existing_key([
                kjoin(RUNS_PREFIX, rid, "esmfold_confidence.json"),
                kjoin(RUNS_PREFIX, rid, "esm_ptm.json"),
                kjoin(RUNS_PREFIX, rid, "ranking_debug.json"),
            ])
            if esm_score_key:
                js = _read_json_from_outputs(esm_score_key)
                scores = _extract_ptm_like_scores(js or {})
                if scores:
                    with st.expander("Model-level confidence (pTM / ipTM)", expanded=False):
                        cols = st.columns(len(scores))
                        for (name,val), c in zip(scores.items(), cols):
                            try:
                                num = float(val)
                                c.metric(name, f"{num*100:.1f}%" if 0<=num<=1.5 else f"{num:.3f}")
                            except Exception:
                                c.metric(name, str(val))
            # PAE
            esm_pae_key = _first_existing_key([
                kjoin(RUNS_PREFIX, rid,  "esmfold_pae.json"),
                kjoin(RUNS_PREFIX, rid,  "pae.json"),
            ])
            if esm_pae_key:
                js = _read_json_from_outputs(esm_pae_key) or {}
                pae = js.get("pae") or js.get("predicted_aligned_error")
                if isinstance(pae, list) and pae and isinstance(pae[0], list):
                    with st.expander("PAE heatmap", expanded=False):
                        _render_pae_heatmap(pae, title="ESMFold PAE")
        else:
            st.info("esmfold_result.pdb 대기 중…")

    # User upload (단일색, 지표 없음)
    with col_right:
        st.subheader("PDB (Upload)")
        upl = st.file_uploader("Upload a PDB", type=["pdb"], key="user_pdb")
        if upl:
            pdb_text = upl.read().decode("utf-8", errors="ignore")
            show_structure_from_text(pdb_text, color=color, height=height)
            st.caption("※ Uploaded PDB: confidence metrics are not shown for user files.")
        else:
            st.caption("PDB를 올리면 단일 색으로 즉시 미리보기됩니다.")

# ------------------------ 3) Feature Extraction ------------------------
elif nav.startswith("3)"):
    st.header("Features")

    rid = st.session_state.get("run_id")
    if not rid:
        st.info("왼쪽 사이드바에서 알레르겐 이름으로 실행(run)을 먼저 불러와 주세요.")
        st.stop()

    out_prefix = kjoin(RUNS_PREFIX, rid)

    # ---------- 1) ESM2 embedding ----------
    st.subheader("ESM2 embedding (per-residue)")
    esm_keys = [
        kjoin(out_prefix, "esm2_embedding.npz"),
        kjoin(out_prefix, "esm2_embedding.npy"),
        kjoin(out_prefix, "esm2_embedding.json"),
        kjoin(out_prefix, "esm2.pt"),
        kjoin(out_prefix, "esm2_embedding.pt"),
    ]
    esm_key = _first_existing_key(esm_keys)
    if esm_key and s3_exists(esm_key):
        obj = s3.get_object(Bucket=S3_OUTPUT_BUCKET, Key=esm_key)
        by = obj["Body"].read()
        try:
            arr = load_esm2_array(by)  # (N,D)
            N, D = arr.shape
            st.write(f"**Shape**: {N} residues × {D} dims  (예: ESM2 3B 모델은 잔기당 2560차원)")
            # PCA 2D
            try:
                from sklearn.decomposition import PCA
                import altair as alt
                X2 = PCA(n_components=2).fit_transform(arr)
                df = pd.DataFrame({"x": X2[:,0], "y": X2[:,1], "idx": np.arange(N)+1})
                chart = alt.Chart(df).mark_circle().encode(
                    x="x", y="y", tooltip=["idx"]
                ).properties(height=260)
                st.altair_chart(chart, use_container_width=True)
            except Exception as e:
                st.warning(f"PCA 시각화는 건너뜀: {e}")
            st.download_button("Download embedding", data=by, file_name=os.path.basename(esm_key))
        except Exception as e:
            st.error(f"ESM2 로드 실패: {e}")
    else:
        st.info("ESM2 임베딩 파일 대기 중… (esm2_embedding.[npy|npz|json|pt])")

    st.divider()

    # ---------- 2) DSSP features ----------
    st.subheader("DSSP features")
    dssp_keys = [
        kjoin(out_prefix, "dssp_features.npy"),
        kjoin(out_prefix, "dssp_features.npz"),
    ]
    dssp_key = _first_existing_key(dssp_keys)
    if dssp_key and s3_exists(dssp_key):
        obj = s3.get_object(Bucket=S3_OUTPUT_BUCKET, Key=dssp_key)
        by = obj["Body"].read()
        try:
            df_dssp = load_dssp(by)
            st.write("**정의**: SS(one-hot) + RSA + (φ,ψ의 sin/cos 4D) → 총 13D")
            st.dataframe(df_dssp.head(20), use_container_width=True)
            n_rows, n_cols = df_dssp.shape
            st.caption(f"shape = {n_rows}×{n_cols}")
            if "rsa" in df_dssp:
                mean_rsa = float(pd.to_numeric(df_dssp["rsa"], errors="coerce").fillna(0).mean())
                st.progress(min(1.0, max(0.0, mean_rsa)))
            st.download_button("Download DSSP", data=by, file_name=os.path.basename(dssp_key))
        except Exception as e:
            st.error(f"DSSP 로드 실패: {e}")
    else:
        st.info("DSSP 파일 대기 중… (dssp_features.[npy|npz|csv|json])")

    st.divider()

    # ---------- 3) Protein graph ----------
    st.subheader("Protein graph (residue-level)")
    graph_keys = [
        kjoin(out_prefix, "protein_graph.json"),
        kjoin(out_prefix, "protein_graph.npz"),
    ]
    graph_key = _first_existing_key(graph_keys)
    if graph_key and s3_exists(graph_key):
        obj = s3.get_object(Bucket=S3_OUTPUT_BUCKET, Key=graph_key)
        by = obj["Body"].read()
        try:
            g = load_protein_graph(by)
            summ = summarise_graph(g)
            
            # 1. 그래프 통계 요약
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Nodes (residues)", f"{summ['num_nodes']:,}")
            with c2:
                st.metric("Edges", f"{summ['num_edges']:,}")
            with c3:
                avg_degree = summ['num_edges'] * 2 / summ['num_nodes'] if summ['num_nodes'] > 0 else 0
                st.metric("Avg Degree", f"{avg_degree:.1f}")
            with c4:
                density = summ['num_edges'] / (summ['num_nodes'] * (summ['num_nodes'] - 1) / 2) if summ['num_nodes'] > 1 else 0
                st.metric("Density", f"{density:.3f}")
            
            st.write("**Edge types**: KNN / Radius / Sequential(|i−j|≤3), 파라미터 예시 dseq=3, dradius=10Å, K=10")
            
            # 2. 연결 타입별 통계 시각화
            if "edges_by_type" in summ and summ["edges_by_type"]:
                edge_types_df = pd.DataFrame([
                    {"Type": k, "Count": v} for k, v in summ["edges_by_type"].items()
                ])
                if not edge_types_df.empty:
                    fig_bar = px.bar(edge_types_df, x="Type", y="Count", 
                                    title="Edge Distribution by Type")
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.json(summ["edges_by_type"])
            else:
                st.json(summ.get("edges_by_type", {}))
            
            # 3. 연결성 히트맵 (노드가 적당한 수일 때만)
            if g.get("adjacency_matrix") is not None and summ['num_nodes'] <= 200:
                adj_matrix = np.array(g["adjacency_matrix"])
                fig_heatmap = px.imshow(adj_matrix, 
                                    title="Residue-Residue Connectivity Matrix",
                                    labels=dict(x="Residue Index", y="Residue Index"),
                                    color_continuous_scale="Viridis")
                fig_heatmap.update_layout(height=400)
                st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # 4. 네트워크 그래프 시각화 (작은 그래프일 때만)
            if summ['num_nodes'] <= 100 and g.get("adjacency_matrix") is not None:
                try:
                    adj_matrix = np.array(g["adjacency_matrix"])
                    G = nx.from_numpy_array(adj_matrix)
                    pos = nx.spring_layout(G, k=1, iterations=50)
                    
                    # 엣지 좌표 생성
                    edge_x = []
                    edge_y = []
                    for edge in G.edges():
                        x0, y0 = pos[edge[0]]
                        x1, y1 = pos[edge[1]]
                        edge_x.extend([x0, x1, None])
                        edge_y.extend([y0, y1, None])
                    
                    # 노드 좌표
                    node_x = [pos[node][0] for node in G.nodes()]
                    node_y = [pos[node][1] for node in G.nodes()]
                    node_text = [f"Residue {i}" for i in G.nodes()]
                    
                    # 플롯 생성
                    fig_network = go.Figure()
                    
                    # 엣지 추가
                    fig_network.add_trace(go.Scatter(x=edge_x, y=edge_y,
                                                mode='lines',
                                                line=dict(width=0.5, color='#888'),
                                                hoverinfo='none',
                                                showlegend=False))
                    
                    # 노드 추가
                    fig_network.add_trace(go.Scatter(x=node_x, y=node_y,
                                                mode='markers',
                                                marker=dict(size=8,
                                                            color=[G.degree(node) for node in G.nodes()],
                                                            colorscale='Viridis',
                                                            colorbar=dict(title="Degree")),
                                                text=node_text,
                                                hoverinfo='text',
                                                showlegend=False))
                    
                    fig_network.update_layout(title="Protein Graph Network",
                                            showlegend=False,
                                            hovermode='closest',
                                            margin=dict(b=20,l=5,r=5,t=40),
                                            annotations=[ dict(text="Node size = degree",
                                                            showarrow=False,
                                                            xref="paper", yref="paper",
                                                            x=0.005, y=-0.002 )],
                                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                            height=500)
                    
                    st.plotly_chart(fig_network, use_container_width=True)
                except Exception as e:
                    st.warning(f"네트워크 시각화 생성 실패: {e}")
            
            # 5. 엣지 특징 분석 (특징이 있는 경우)
            if g.get("edges") and len(g["edges"]) > 0:
                edges_df = pd.DataFrame(g["edges"])
                if "features" in edges_df.columns:
                    st.write("**Edge Feature Analysis**")
                    
                    # 특징 차원별 통계
                    features_array = np.array([f for f in edges_df["features"] if f is not None])
                    if len(features_array) > 0:
                        feature_stats = pd.DataFrame({
                            "Feature_Dim": range(features_array.shape[1]),
                            "Mean": np.mean(features_array, axis=0),
                            "Std": np.std(features_array, axis=0),
                            "Non_Zero_Ratio": np.mean(features_array != 0, axis=0)
                        })
                        
                        # 특징 카테고리 정의 (GraphBepi 기준)
                        if features_array.shape[1] >= 47:  # 최소 47차원
                            st.write("- **Amino Acid Pairs** (0-41): 21×2 one-hot encoding")
                            st.write("- **Connection Type** (42-44): Sequential/Spatial/K-NN flags") 
                            st.write("- **Geometric** (45-46): Sequential distance, Euclidean distance")
                        
                        st.dataframe(feature_stats.head(10), use_container_width=True)
            
            # 6. 데이터 미리보기 (기존 유지)
            with st.expander("Raw Data Preview", expanded=False):
                if g.get("nodes"):
                    st.caption("nodes preview")
                    st.dataframe(pd.DataFrame(g["nodes"]).head(10), use_container_width=True)
                if g.get("edges"):
                    st.caption("edges preview") 
                    st.dataframe(pd.DataFrame(g["edges"]).head(20), use_container_width=True)
            
            st.download_button(
                "Download graph", 
                data=by, 
                file_name=os.path.basename(graph_key),
                key="download_graph_button"  # 고유 키 추가
            )
            
        except Exception as e:
            st.error(f"Protein graph 로드 실패: {e}")
    else:
        st.info("Protein graph 파일 대기 중… (protein_graph.[json|npz])")

# ------------------------ 4) Final Epitope ------------------------
else:
    st.header("Final Epitope")
    rid = st.session_state.get("run_id")
    if not rid:
        st.warning("run_id가 없습니다.")
    else:
        # ✅ AF2 우선 사용, 없으면 ESMFold 폴백
        af_key  = kjoin(RUNS_PREFIX, rid,  "alphafold_result.pdb")
        esm_key = kjoin(RUNS_PREFIX, rid,  "esmfold_result.pdb")
        epi_key = kjoin(RUNS_PREFIX, rid,  "epitope_predictions.json")

        tab_viz, tab_imp = st.tabs(["Epitope Visualization", "Results"])

        with tab_viz:
            left, right = st.columns(2)

            # Prediction (pipeline)
            with left:
                st.subheader("Prediction")

                # ---- 구조 소스 선택 (AF2 > ESMFold) ----
                af_exists  = s3_exists(af_key)
                esm_exists = s3_exists(esm_key)
                if af_exists:
                    struct_url = presign_get(af_key)
                    struct_label = "AlphaFold2"
                elif esm_exists:
                    struct_url = presign_get(esm_key)
                    struct_label = "ESMFold (fallback)"
                else:
                    struct_url = None
                    struct_label = None

                mode = st.radio("Coloring mode", ["Binary (threshold)", "Gradient (by likelihood)"], horizontal=True)
                if mode.startswith("Binary"):
                    thr = st.slider("Score threshold", 0.0, 1.0, 0.57, 0.01, key="pred_thr")
                else:
                    smin, smax = st.slider("Score range for gradient", 0.0, 1.0, (0.0, 1.0), 0.01, key="pred_minmax")
                
                st.info("EpiDot Best Threshold : 0.57")
                
                have_pdb  = struct_url is not None
                have_pred = s3_exists(epi_key)

                if have_pdb and have_pred:
                    if mode.startswith("Binary"):
                        molstar_paint_pred_binary_from_url(
                            struct_url=struct_url,
                            pred_url=presign_get(epi_key),
                            threshold=thr,
                            pos_hex="#ff0000", neg_hex="#c0c0c0", height=640
                        )
                    else:
                        molstar_paint_pred_gradient_from_url(
                            struct_url=struct_url,
                            pred_url=presign_get(epi_key),
                            min_score=smin, max_score=smax,
                            neg_hex="#c0c0c0", pos_hex="#ff0000", height=640
                        )

                    # 어떤 구조를 기반으로 칠했는지 명시
                    st.caption(f"Structure base: **{struct_label}**")
                    # 예측 파일 다운로드
                    obj = s3.get_object(Bucket=S3_OUTPUT_BUCKET, Key=epi_key)
                    st.download_button("⬇ Download predictions (JSON)",
                                       obj["Body"].read(), file_name="epitope_predictions.json")
                else:
                    missing = []
                    if not have_pdb:
                        if not af_exists and not esm_exists:
                            missing.append("alphafold_result.pdb / esmfold_result.pdb")
                        elif not af_exists and esm_exists:
                            # 이 경우 have_pdb=True가 되어야 하므로 실제로는 안 옵니다.
                            pass
                    if not have_pred:
                        missing.append("epitope_predictions.json")
                    st.info("다음 산출물이 아직 없어 시각화를 건너뜁니다: " + ", ".join(missing))

            # Ground Truth (upload)
            with right:
                st.subheader("Ground Truth (Upload)")


                def _parse_truth_sites(file, default_chain="A") -> list[dict]:
                    if file.name.lower().endswith(".json"):
                        df = pd.DataFrame(json.loads(file.read().decode("utf-8")))
                    else:
                        df = pd.read_csv(io.BytesIO(file.read()))
                    if df.empty:
                        return []
                    norm = {c.strip().lower(): c for c in df.columns}
                    def pick(cands):
                        for k in cands:
                            k = k.strip().lower()
                            if k in norm:
                                return norm[k]
                        return None
                    chain_col = pick(["epitope_chain","pdb_chain","chain","chain_id"])
                    resi_col  = pick(["pdb_resseq","epitope_resseq","epitope_resi","pdb_resi","resi","resid","position","auth_seq_id"])
                    icode_col = pick(["pdb_icode","icode","inscode","insertion_code"])
                    label_col = pick(["epitope_label","label","is_epitope","epitope","binary_label"])
                    if resi_col is None:
                        return []
                    if label_col and label_col in df.columns:
                        s = df[label_col].astype(str).str.strip().str.lower()
                        mask = s.isin(["1","true","yes","y","pos","epitope"]) | \
                               pd.to_numeric(df[label_col], errors="coerce").fillna(0).gt(0)
                        df = df[mask]
                    sites = []
                    for _, r in df.iterrows():
                        chain = (str(r[chain_col]).strip() if (chain_col and pd.notna(r.get(chain_col))) else default_chain).upper()
                        raw = r.get(resi_col)
                        if pd.isna(raw):
                            continue
                        raw = str(raw).strip()
                        m = re.match(r"^(\d+)\s*([A-Za-z]?)$", raw)
                        if m:
                            resi = int(m.group(1)); ins = m.group(2).upper() if m.group(2) else None
                        else:
                            try:
                                resi = int(float(raw)); ins = None
                            except Exception:
                                continue
                        if icode_col and pd.notna(r.get(icode_col)):
                            ins = str(r.get(icode_col)).strip().upper() or None
                        site = {"chain": chain, "resi": resi}
                        if ins:
                            site["ins"] = ins
                        sites.append(site)
                    return sites
                truth_pdb = st.file_uploader("Ground-truth PDB", type=["pdb"], key="truth_pdb")
                truth_ann = st.file_uploader("Epitope labels (CSV/JSON)", type=["csv","json"], key="truth_labels")
                sites = []
                if truth_ann is not None:
                    try:
                        sites = _parse_truth_sites(truth_ann, default_chain="A")
                    except Exception as e:
                        st.error(f"Truth 파일 파싱 실패: {e}"); sites = []

                st.caption(f"Parsed truth residues: {len(sites)}개")
                if sites[:10]:
                    st.code(sites[:10], language="json")

                if truth_pdb is not None and sites:
                    pdb_text = truth_pdb.read().decode("utf-8", errors="ignore")
                    molstar_paint_truth_from_text(pdb_text, sites, pos_hex="#ff0000", neg_hex="#c0c0c0", height=640)
                else:
                    st.info("PDB와 CSV/JSON(라벨=1) 둘 다 업로드해야 표시됩니다.")
                    

        # Feature importance (pipeline) — 그대로 유지
        with tab_imp:
            st.subheader("Epitope Prediction Results")
            rep_key = kjoin(RUNS_PREFIX, rid,  "final_analysis_report.csv")
            if not s3_exists(rep_key):
                st.info("파이프라인 산출물(final_analysis_report.csv)이 아직 준비되지 않았습니다.")
            else:
                obj = s3.get_object(Bucket=S3_OUTPUT_BUCKET, Key=rep_key)
                by = obj["Body"].read()
                try:
                    df = pd.read_csv(io.BytesIO(by))
                except Exception:
                    st.error("CSV 파싱 실패"); df = None
                if df is not None:
                    cols = {c.lower(): c for c in df.columns}
                    def pick(cands):
                        for k in cands:
                            if k in cols:
                                return cols[k]
                        return None
                    feat_col = pick(["feature","name","input_feature","variable","feature_name"])
                    imp_col  = pick(["importance","gain","weight","shap_value","mean_abs_shap_value",
                                     "mean(|shap|)","mean_abs_shap","mean_shap_abs"])
                    if feat_col and imp_col:
                        dd = df[[feat_col, imp_col]].copy()
                        dd.columns = ["Feature","Importance"]
                        dd["Importance"] = pd.to_numeric(dd["Importance"], errors="coerce").fillna(0)
                        dd["AbsImportance"] = dd["Importance"].abs()
                        dd = dd.sort_values("AbsImportance", ascending=False)
                        total = dd["AbsImportance"].sum()
                        dd["Share(%)"] = (dd["AbsImportance"]/total*100.0) if total>0 else 0.0
                        topN = st.slider("Top-N features", 5, 30, 12, 1)
                        top = dd.head(topN)
                        try:
                            import altair as alt
                            chart = alt.Chart(top).mark_bar().encode(
                                x=alt.X("AbsImportance:Q", title="Importance (|value|)"),
                                y=alt.Y("Feature:N", sort="-x", title="Feature"),
                                tooltip=["Feature", alt.Tooltip("Importance:Q", format=".4f"),
                                         alt.Tooltip("Share(%):Q", title="Share (%)", format=".1f")],
                            ).properties(height=360)
                            st.altair_chart(chart, use_container_width=True)
                        except Exception:
                            _display_grid_or_table(top[["Feature","Importance","Share(%)"]], height=380)
                        st.divider()
                        _display_grid_or_table(dd[["Feature","Importance","Share(%)"]], height=420)
                    else:
                        # st.warning("final_analysis_report.csv에서 feature/importance 컬럼을 찾지 못했습니다.")
                        _display_grid_or_table(df, height=420)