r"""
step3_build_graphbepi_total_from_mapping_ko.py
==============================================
입력
----
1) --iedb-csv : IEDB epitope_table_export.csv (식품 알레르겐 필터 덤프)
   - 사용 컬럼(존재 시): 
     * 'Epitope - Molecule Parent'
     * 'Epitope - Object Type' (non-peptidic 제외)
     * 'Epitope - Starting Position', 'Epitope - Ending Position'  (parent 상 좌표, 1-based)
2) --per-parent : per_parent_epitopes_clean.csv/.xlsx (parent_name, full_fasta 포함)
3) --mapping : step2 결과 mapping_best.csv (필수 컬럼: 'PDB chain','parent_name','parent_start','chain_start','overlap_len')
4) --pdb-fasta : pdb_chains.fasta (체인 서열; resi→resn 변환에 사용)

출력
----
- --out-total : GraphBepi 호환 total.csv (컬럼: 'PDB chain','Epitopes (resi_resn)')

사용 예
-------
python step3_build_graphbepi_total_from_mapping_ko.py ^
  --iedb-csv "d:/final/aws/iedb_epitope_table_export.csv" ^
  --per-parent "d:/final/aws/per_parent_epitopes_clean_uniprot_label.xlsx" ^
  --mapping "d:/final/aws/mapping_best.csv" ^
  --pdb-fasta "d:/final/aws/pdb_chains.fasta" ^
  --out-total "d:/final/aws/total.csv" ^
  --exclude-parent "(?i)\\bAra\\s*h\\s*2\\b" ^
  --verbose
  
- 동일 parent가 여러 체인에 매핑된 경우, 매핑된 체인 각각에 대해 좌표를 투영합니다.
"""

import re
import csv
import sys
import argparse
from pathlib import Path
from collections import defaultdict

# 1-letter → 3-letter
AA3 = {
    "A":"ALA","R":"ARG","N":"ASN","D":"ASP","C":"CYS",
    "Q":"GLN","E":"GLU","G":"GLY","H":"HIS","I":"ILE",
    "L":"LEU","K":"LYS","M":"MET","F":"PHE","P":"PRO",
    "S":"SER","T":"THR","W":"TRP","Y":"TYR","V":"VAL",
    "U":"SEC","O":"PYL","B":"ASX","Z":"GLX","X":"UNK"
}

def clean_seq(s: str) -> str:
    return re.sub(r"[^A-Za-z]", "", (s or "")).upper()

def read_csv_rows(path: Path):
    last=None
    for enc in ("utf-8-sig","utf-8","cp949","latin-1"):
        try:
            with open(path, newline="", encoding=enc, errors="ignore") as f:
                rdr = csv.DictReader(f)
                return [r for r in rdr]
        except Exception as e:
            last=e
    raise RuntimeError(f"CSV 읽기 실패: {path} ({last})")

def read_xlsx_rows(path: Path):
    try:
        import openpyxl
    except Exception as e:
        raise RuntimeError("XLSX를 읽으려면 openpyxl이 필요합니다. "
                           "pip install openpyxl 또는 CSV로 저장해 주세요.") from e
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        return []
    header = [str(c).strip() if c is not None else "" for c in rows[0]]
    out = []
    for r in rows[1:]:
        d = {}
        for i, val in enumerate(r):
            key = header[i] if i < len(header) else f"col{i+1}"
            d[key] = val
        out.append(d)
    return out

def read_per_parent(path: Path):
    suf = path.suffix.lower()
    if suf==".csv":
        rows = read_csv_rows(path)
    elif suf in (".xlsx",".xlsm",".xltx",".xltm"):
        rows = read_xlsx_rows(path)
    else:
        raise RuntimeError(f"지원하지 않는 확장자: {suf}")
    # 컬럼 normalize
    def norm(s): return re.sub(r"[^a-z0-9]+","_", (s or "").lower()).strip("_")
    if not rows:
        return []
    colmap = {norm(k): k for k in rows[0].keys()}
    c_parent = None
    for k in ("parent_name","parent","molecule_parent","epitope_molecule_parent","epitope_-_molecule_parent"):
        if k in colmap: c_parent = colmap[k]; break
    c_fasta = None
    for k in ("full_fasta","fasta","sequence","full_sequence","uniprot_fasta"):
        if k in colmap: c_fasta = colmap[k]; break
    if not c_parent or not c_fasta:
        raise RuntimeError(f"per-parent 파일에 parent_name/full_fasta 컬럼이 필요합니다. (헤더: {list(rows[0].keys())})")
    out=[]
    for r in rows:
        pname=str(r.get(c_parent) or "").strip()
        fseq = clean_seq(r.get(c_fasta) or "")
        if pname and fseq:
            out.append((pname, fseq))
    return out

def read_fasta(path: Path):
    seqs = {}
    header=None
    buf=[]
    with open(path, encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            if line.startswith(">"):
                if header is not None:
                    seqs[header]="".join(buf)
                header=line[1:].strip()
                buf=[]
            else:
                buf.append(re.sub(r"\s+","",line))
        if header is not None:
            seqs[header]="".join(buf)
    # clean
    for k in list(seqs.keys()):
        seqs[k]=clean_seq(seqs[k])
    return seqs

def parse_int(x):
    try:
        if x is None or str(x).strip()=="" or str(x).strip().lower()=="nan":
            return None
        return int(float(str(x)))
    except:
        return None

def build_parent_epitope_positions(iedb_rows, *, parent_key, obj_type_key, start_key, end_key, exclude_non_peptidic=True):
    """
    parent별 {1-based 위치 집합} 생성
    """
    peps = defaultdict(set)
    for r in iedb_rows:
        parent = (r.get(parent_key) or "").strip()
        if not parent:
            continue
        if exclude_non_peptidic:
            ot = (r.get(obj_type_key) or "").strip().lower()
            if "non-peptidic" in ot:
                continue
        s = parse_int(r.get(start_key))
        e = parse_int(r.get(end_key))
        if s is None or e is None:  # 좌표 없는 항목은 건너뜀
            continue
        if e < s:
            s, e = e, s
        for pos in range(s, e+1):
            peps[parent].add(pos)
    return peps  # dict[parent] -> set(positions)

def main():
    ap = argparse.ArgumentParser(description="IEDB parent 좌표 → PDB 체인으로 투영하여 GraphBepi total.csv 생성")
    ap.add_argument("--iedb-csv", required=True, help="IEDB epitope_table_export.csv 경로")
    ap.add_argument("--per-parent", required=True, help="per_parent_epitopes_clean.csv/.xlsx (parent_name, full_fasta 포함)")
    ap.add_argument("--mapping", required=True, help="step2 mapping_best.csv")
    ap.add_argument("--pdb-fasta", required=True, help="pdb_chains.fasta (resn 변환용)")
    ap.add_argument("--out-total", default=None, help="total.csv 출력 경로(기본: mapping과 같은 폴더)")
    ap.add_argument("--exclude-parent", default=None, help="제외할 parent 정규식 (예: (?i)\\bAra\\s*h\\s*2\\b)")
    ap.add_argument("--include-parent", default=None, help="포함할 parent 정규식")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    iedb_path = Path(args.iedb_csv)
    per_parent_path = Path(args.per_parent)
    mapping_path = Path(args.mapping)
    pdb_fasta_path = Path(args.pdb_fasta)
    out_total = Path(args.out_total) if args.out_total else mapping_path.with_name("total.csv")

    # 0) per-parent 읽기(검증용: parent 존재 여부 확인에 사용)
    parents = read_per_parent(per_parent_path)
    parent_names = {p for p,_ in parents}

    # 1) IEDB 읽기 + parent별 epitope 위치 집합 빌드
    iedb_rows = read_csv_rows(iedb_path)
    # 필수 컬럼 추정
    def find_col(name_candidates):
        for c in iedb_rows[0].keys():
            for pat in name_candidates:
                if re.fullmatch(pat, c):
                    return c
        low = {c.lower(): c for c in iedb_rows[0].keys()}
        for pat in name_candidates:
            for k in low:
                if pat.lower() in k:
                    return low[k]
        return None

    col_parent = find_col(["Epitope - Molecule Parent"])
    col_objtype = find_col(["Epitope - Object Type"])
    col_start  = find_col(["Epitope - Starting Position"])
    col_end    = find_col(["Epitope - Ending Position"])
    if not (col_parent and col_objtype and col_start and col_end):
        raise RuntimeError("IEDB CSV에서 필요한 컬럼을 찾지 못했습니다. "
                           "필요: Epitope - Molecule Parent / Object Type / Starting Position / Ending Position")

    # parent 필터(포함/제외)
    inc_pat = re.compile(args.include_parent) if args.include_parent else None
    exc_pat = re.compile(args.exclude_parent) if args.exclude_parent else None
    if inc_pat or exc_pat:
        iedb_rows = [r for r in iedb_rows if (r.get(col_parent) or "").strip()]
        if inc_pat:
            iedb_rows = [r for r in iedb_rows if inc_pat.search((r.get(col_parent) or ""))]
        if exc_pat:
            iedb_rows = [r for r in iedb_rows if not exc_pat.search((r.get(col_parent) or ""))]

    peps_by_parent = build_parent_epitope_positions(
        iedb_rows, parent_key=col_parent, obj_type_key=col_objtype, start_key=col_start, end_key=col_end,
        exclude_non_peptidic=True
    )

    # 2) PDB 체인 서열 읽기(1-letter → 3-letter 변환용)
    chain_fasta = read_fasta(pdb_fasta_path)

    # 3) mapping_best 읽고 chain별로 parent 투영
    map_rows = read_csv_rows(mapping_path)
    # 헤더 확인
    needed = {"PDB chain","parent_name","parent_start","chain_start","overlap_len"}
    if not needed.issubset(set(map_rows[0].keys())):
        raise RuntimeError(f"mapping_best.csv에 필요한 컬럼이 없습니다. 필요: {sorted(needed)}")

    # chain → (parent, parent_start, chain_start, overlap_len)
    chain_map = defaultdict(list)
    for r in map_rows:
        chain_id = (r.get("PDB chain") or "").strip()
        pname = (r.get("parent_name") or "").strip()
        pstart = parse_int(r.get("parent_start")) or 0
        cstart = parse_int(r.get("chain_start")) or 0
        ovlen  = parse_int(r.get("overlap_len")) or 0
        if chain_id and pname and ovlen>0:
            # parent 존재 여부(옵션) 체크는 생략. (외부 parent도 허용)
            chain_map[chain_id].append((pname, pstart, cstart, ovlen))

    # 4) 체인별 epitope 좌표 생성
    def parent_to_chain_positions(pname, pstart, cstart, ovlen, epipos_set):
        """parent의 1-based epitope 좌표 집합(epipos_set)을 체인 1-based 좌표로 투영"""
        chain_positions = set()
        pend = pstart + ovlen - 1
        for pp in epipos_set:
            if pstart <= pp <= pend:
                cp = cstart + (pp - pstart)  # 1-based 유지
                chain_positions.add(cp)
        return chain_positions

    total_rows = []
    skipped_no_epitope = 0

    for chain_id, links in chain_map.items():
        # 이 체인을 구성하는 parent들이 여럿일 수 있어 모두 합산(중복 가능)
        chain_positions = set()
        for pname, pstart, cstart, ovlen in links:
            episet = peps_by_parent.get(pname)
            if not episet:
                continue
            chain_positions |= parent_to_chain_positions(pname, pstart, cstart, ovlen, episet)

        if not chain_positions:
            skipped_no_epitope += 1
            continue

        # 체인 서열에서 resn 구하기
        seq1 = chain_fasta.get(chain_id, "")
        if not seq1:
            continue
        items = []
        for cp in sorted(chain_positions):
            idx = cp - 1
            if idx < 0 or idx >= len(seq1):
                continue
            aa1 = seq1[idx]
            aa3 = AA3.get(aa1, "UNK")
            items.append(f"{cp}_{aa3}")

        if not items:
            skipped_no_epitope += 1
            continue

        total_rows.append({"PDB chain": chain_id, "Epitopes (resi_resn)": ", ".join(items)})

    # 5) 저장
    out_total.parent.mkdir(parents=True, exist_ok=True)
    with open(out_total, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["PDB chain","Epitopes (resi_resn)"])
        w.writeheader()
        w.writerows(total_rows)

    if args.verbose:
        print(f"[DONE] chains_out={len(total_rows)}  skipped_no_epitope={skipped_no_epitope}  out={out_total}")

if __name__ == "__main__":
    main()
