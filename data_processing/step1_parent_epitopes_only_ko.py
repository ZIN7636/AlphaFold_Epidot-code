r"""
STEP 1: IEDB epitope_table_export CSV → per_parent_epitopes_clean.csv
-------------------------------------------------------------------------
경로 설정 방법:
- (A) 아래 DEFAULT_* 상수 값을 코드에서 직접 수정하거나
- (B) 실행 시 CLI 인자로 덮어쓰기 (--iedb-csv, --out-csv, --canon-parent)

사용 예시:
python step1_parent_epitopes_only_ko.py \
  --iedb-csv "d:/final/aws/epitope_table_export_1757479618.csv" \
  --out-csv  "d:/final/aws/per_parent_epitopes_clean.csv" \
  --canon-parent
"""

import re
import sys
import json
import argparse
from pathlib import Path

import pandas as pd

# ===== 기본 경로 =====
DEFAULT_IEDB_CSV = "./iedb_epitope_table_export.csv"
DEFAULT_OUT_CSV  = "./per_parent_epitopes_clean.csv"
# ===========================================

def parse_args():
    """명령행 인자 파싱"""
    p = argparse.ArgumentParser(description="IEDB CSV → per_parent_epitopes_clean.csv (STEP1 최종)")
    p.add_argument("--iedb-csv", default=DEFAULT_IEDB_CSV, help="IEDB epitope_table_export CSV 경로")
    p.add_argument("--out-csv",  default=DEFAULT_OUT_CSV,  help="출력 CSV 경로 (per_parent_epitopes_clean.csv)")
    p.add_argument("--canon-parent", action="store_true",
                   help="Molecule Parent 기본형으로 보수적 정규화 (예: 'Ara h 2.0101'→'Ara h 2')")
    return p.parse_args()

def normalize_colnames(cols):
    """원본 헤더를 소문자+언더스코어 키로 정규화 → 원래 이름 매핑"""
    norm = {}
    for c in cols:
        key = re.sub(r"[^a-z0-9]+","_", c.lower()).strip("_")
        norm[key] = c
    return norm

def canon_parent_name(name: str) -> str:
    """'Ara h 2.0101' → 'Ara h 2' (보수적 간이 규칙)"""
    if not isinstance(name, str):
        return name
    s = name.strip()
    m = re.match(r"^\s*([A-Za-z]{3}\s+[a-z]\s+\d+)", s)
    return m.group(1) if m else s

def has_url_in_column(series: pd.Series) -> bool:
    """해당 컬럼 내 값 중 URL 패턴이 하나라도 있으면 True"""
    URL_RE = re.compile(r"https?://|www\.", re.I)
    try:
        return series.astype(str).str.contains(URL_RE).any()
    except Exception:
        return False

def main():
    args = parse_args()

    in_csv = Path(args.iedb_csv)
    out_csv = Path(args.out_csv)

    if not in_csv.exists():
        print(f"[ERROR] 입력 CSV 없음: {in_csv}", file=sys.stderr)
        sys.exit(1)

    # 1) CSV 로드 + 컬럼명 정규화(유연 매핑)
    df = pd.read_csv(in_csv)
    norm = normalize_colnames(df.columns)

    # 주요 컬럼 후보 지정
    col_parent = norm.get("epitope_molecule_parent") or norm.get("molecule_parent") \
                 or norm.get("epitope_source_molecule") or norm.get("source_molecule")
    col_org    = norm.get("epitope_source_organism") or norm.get("source_organism")
    col_type   = norm.get("epitope_object_type") or norm.get("object_type") or norm.get("epitope_-_object_type")
    col_name   = norm.get("epitope_name") or norm.get("name") or norm.get("epitope")
    col_start  = norm.get("epitope_starting_position") or norm.get("starting_position")
    col_end    = norm.get("epitope_ending_position") or norm.get("ending_position")
    col_id     = norm.get("epitope_id") or norm.get("epitope_id_-_iedb_iri") or norm.get("iedb_id")

    # 2) Molecule Parent 없으면 Source Molecule로 대체 시도
    if col_parent is None:
        print("[WARN] 'Epitope - Molecule Parent' 컬럼을 찾지 못했습니다. 'Source Molecule' 계열로 대체합니다.", file=sys.stderr)
        col_parent = norm.get("epitope_source_molecule") or norm.get("source_molecule")
        if col_parent is None:
            print("[ERROR] Molecule Parent/Source Molecule 컬럼을 찾을 수 없습니다.", file=sys.stderr)
            sys.exit(2)

    # 3) parent_name 결측 제거
    #    - NaN / 빈칸 / 'nan' 문자열 → 제거
    parent_raw = df[col_parent]
    mask_valid_parent = parent_raw.notna() \
        & (parent_raw.astype(str).str.strip() != "") \
        & (parent_raw.astype(str).str.strip().str.lower() != "nan")
    df = df[mask_valid_parent].copy()

    # 4) Non-peptidic 제거
    removed_nonpep = 0
    if col_type and df[col_type].notna().any():
        # 'non-peptidic' 또는 'Nonpeptidic' 등 다양한 표기를 견딜 수 있게 패턴 구성
        s = df[col_type].astype(str).str.lower()
        mask_nonpep = (s.str.contains("non") & s.str.contains("peptidic")) | (s.str.contains("nonpeptidic"))
        before = len(df)
        df = df[~mask_nonpep].copy()
        removed_nonpep = before - len(df)

    # 5) URL 포함 컬럼 전체 제거
    url_cols = [c for c in df.columns if has_url_in_column(df[c])]
    # 열 이름에 IRI/URL 힌트가 있는 경우도 제거
    name_hint_drop = [c for c in df.columns if re.search(r"(iri|url|http|https)", c, flags=re.I)]
    drop_cols = sorted(set(url_cols + name_hint_drop))
    if drop_cols:
        df = df[[c for c in df.columns if c not in drop_cols]]

    # 6) parent_name 생성(+정규화 옵션)
    df["parent_name"] = df[col_parent].astype(str).str.strip()
    if args.canon_parent:
        df["parent_name"] = df["parent_name"].apply(canon_parent_name)

    # 7) 펩타이드 서열 정규화(알파벳만 대문자)
    if col_name and col_name in df.columns:
        df["peptide"] = df[col_name].astype(str).str.replace(r"[^A-Za-z]", "", regex=True).str.upper()
    else:
        df["peptide"] = ""

    # 8) 선형 좌표 정제(start/end → Int64)
    if col_start and col_start in df.columns:
        df["start"] = pd.to_numeric(df[col_start], errors="coerce").astype("Int64")
    else:
        df["start"] = pd.Series([pd.NA]*len(df), dtype="Int64")

    if col_end and col_end in df.columns:
        df["end"] = pd.to_numeric(df[col_end], errors="coerce").astype("Int64")
    else:
        df["end"] = pd.Series([pd.NA]*len(df), dtype="Int64")

    # 9) 출력 컬럼 구성(존재하는 것만 포함)
    rename_map = {}
    if col_type and col_type in df.columns: rename_map[col_type] = "object_type"
    if col_org  and col_org  in df.columns: rename_map[col_org]  = "source_organism"
    if col_id   and col_id   in df.columns: rename_map[col_id]   = "epitope_id"

    df_out = df.rename(columns=rename_map)

    keep_order = ["parent_name","peptide","start","end","object_type","source_organism","epitope_id"]
    cols_final = [c for c in keep_order if c in df_out.columns]

    # 10) 중복 제거(같은 parent_name/peptide/start/end/object_type/epitope_id 조합)
    subset_cols = [c for c in ["parent_name","peptide","start","end","object_type","epitope_id"] if c in cols_final]
    if subset_cols:
        df_out = df_out.drop_duplicates(subset=subset_cols)

    # 11) 저장
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out[cols_final].to_csv(out_csv, index=False)

    print(json.dumps({
        "input_csv": str(in_csv),
        "out_csv": str(out_csv),
        "n_rows_out": int(df_out.shape[0]),
        "removed_nonpeptidic_rows": int(removed_nonpep),
        "dropped_url_like_columns": drop_cols,
        "columns_out": cols_final
    }, indent=2))

if __name__ == "__main__":
    main()
