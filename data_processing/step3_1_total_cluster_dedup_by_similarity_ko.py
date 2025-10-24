r"""
step3_1_total_cluster_dedup_by_similarity_ko.py
==============================================
파이프라인
---------
1) (stage='cluster') 에피토프 유사도 기반 클러스터링 → 각 클러스터의 대표 1행만 보존
   - 유사도: |A∩B| / max(|A|, |B|)  (기본 --similarity 0.9)
   - 대표 선택: --keep longest(기본) / first / chainalpha

2) (stage='pdbid') 1단계에서 남은 대표행들 중 **같은 PDB ID(예: 1ABC_A → 1ABC)** 가 중복이면 추가로 정리
   - 대표 선택: 1단계와 동일 규칙 (--keep)
   - 범위: --pdbid-scope global(기본) / group
           (group이면 동일 그룹(예: parent_name)이면서 PDB ID가 같은 경우만 중복 제거)

입력 CSV
-----------------------
- 체인 컬럼: "PDB chain" (또는 chain 유사명)
- 에피토프 컬럼: "Epitopes (resi_resn)" (예: "18_HIS, 20_TYR, ...")
- (선택) 그룹 컬럼: 알레르겐/parent명 등. **같은 그룹 내에서만** 중복 제거하고 싶을 때 지정
  (예: --group-col parent_name)

출력
----
- --out-dedup : 최종 dedup CSV (대표행만, 원본의 다른 컬럼 보존)
- --out-map   : 매핑 CSV(각 원본 행이 어떤 대표로 묶였는지 / 어떤 단계에서 제외됐는지)
               - stage = "cluster" 또는 "pdbid" 로 표기

사용 예
------
# 90% 유사도 기준 + 같은 알레르겐 내부에서 비교 + PDB ID 전역 중복 제거
python step3c_total_cluster_dedup_by_similarity_ko.py \
  --in-csv "d:/final/aws/2nd_total.csv" \
  --out-dedup "d:/final/aws/2nd_total_dedup90.csv" \
  --out-map   "d:/final/aws/2nd_total_dedup90_map.csv" \
  --similarity 0.9 --keep longest --upper-chain \
  --group-col parent_name \
  --dedup-by-pdb-id --pdbid-scope global
"""

import re
import csv
import sys
import argparse
from pathlib import Path
from collections import defaultdict

# ------------------------------
# (공용) 헤더/키 정규화
# ------------------------------
def _normkey(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_")

def _detect_cols(header):
    """
    체인/에피토프/그룹 컬럼을 느슨하게 감지
    """
    nmap = {_normkey(h): h for h in header}

    # 체인
    col_chain = None
    for k in ("pdb_chain", "pdbchain", "chain", "pdb_id_chain", "pdbid_chain"):
        if k in nmap:
            col_chain = nmap[k]; break
    if col_chain is None and "PDB chain" in header:
        col_chain = "PDB chain"
    if col_chain is None:
        raise SystemExit("ERROR: 'PDB chain' 컬럼을 찾을 수 없습니다.")

    # 에피토프
    col_epi = None
    for k in ("epitopes_resi_resn", "epitopes", "epitope_list"):
        if k in nmap:
            col_epi = nmap[k]; break
    if col_epi is None and "Epitopes (resi_resn)" in header:
        col_epi = "Epitopes (resi_resn)"
    if col_epi is None:
        for h in header:
            if "epi" in h.lower():
                col_epi = h; break
    if col_epi is None:
        raise SystemExit("ERROR: 'Epitopes (resi_resn)' 컬럼을 찾을 수 없습니다.")

    # 그룹(선택)
    col_group = None
    for key in ("parent_name", "parent", "allergen", "molecule_parent", "epitope_molecule_parent"):
        if key in nmap:
            col_group = nmap[key]; break

    # 별도 PDB ID 컬럼(있으면 사용)
    col_pdb = None
    for key in ("pdb", "pdb_id", "pdbid"):
        if key in nmap:
            col_pdb = nmap[key]; break

    return col_chain, col_epi, col_group, col_pdb

# ---------------------------------------
# 에피토프 문자열을 (resi,resn) 집합으로 파싱
# ---------------------------------------
def _parse_epilist(s: str, ignore_resn: bool = False):
    parts = re.split(r"[;,]", str(s or ""))
    out = set()
    for p in parts:
        t = p.strip()
        if not t:
            continue
        m = re.match(r"^\s*(\d+)\s*[_\-\s]\s*([A-Za-z]{3})\s*$", t)
        if m:
            resi = int(m.group(1)); resn = m.group(2).upper()
            out.add((resi, None if ignore_resn else resn))
        else:
            # 숫자만 들어온 경우 허용
            m2 = re.match(r"^\s*(\d+)\s*$", t)
            if m2:
                resi = int(m2.group(1))
                out.add((resi, None if ignore_resn else "UNK"))
    return out

# ------------------------------
# 에피토프 유사도: |A∩B| / max(|A|, |B|)
# ------------------------------
def _epi_similarity(A: set, B: set) -> float:
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    inter = len(A & B)
    denom = max(len(A), len(B))
    return inter / denom if denom > 0 else 0.0

# ------------------------------
# 대표 선택 규칙
# ------------------------------
def _choose_rep(indices, entries, how="longest"):
    """
    indices: 후보 인덱스 리스트
    entries[i] = (chain, group, episet)
    """
    if how == "first":
        return indices[0]
    if how == "chainalpha":
        return sorted(indices, key=lambda i: entries[i][0])[0]
    # longest(기본): 에피토프 수가 가장 많은 행 → tie-break는 체인ID 사전순
    return sorted(indices, key=lambda i: (len(entries[i][2]), entries[i][0]), reverse=True)[0]

# ------------------------------
# PDB ID 추출
# ------------------------------
def _get_pdb_id(row, col_pdb, col_chain):
    """
    1) 별도 PDB 컬럼(col_pdb)이 있으면 그 값을 사용
    2) 없으면 'PDB chain'에서 언더스코어('_') 앞부분 사용 (예: 1ABC_A -> 1ABC)
    """
    if col_pdb:
        v = (row.get(col_pdb) or "").strip()
        if v:
            return v
    chain = (row.get(col_chain) or "").strip()
    if not chain:
        return ""
    return chain.split("_")[0] if "_" in chain else chain

# ------------------------------
# 메인
# ------------------------------
def main():
    ap = argparse.ArgumentParser(description="에피토프 유사도 + PDB ID 중복 제거기")
    ap.add_argument("--in-csv", required=True, help="입력 CSV 경로(예: 2nd_total.csv)")
    ap.add_argument("--out-dedup", required=True, help="최종 dedup CSV 출력 경로")
    ap.add_argument("--out-map", required=True, help="매핑 CSV(단계별 제거 내역 포함)")
    ap.add_argument("--similarity", type=float, default=0.9, help="에피토프 유사도 임계값(기본 0.9)")
    ap.add_argument("--keep", choices=["longest", "first", "chainalpha"], default="longest",
                    help="대표 선택 규칙(기본 longest)")
    ap.add_argument("--group-col", default=None,
                    help="중복 제거를 **이 컬럼 내부에서만** 수행(예: parent_name). 없으면 전체 비교")
    ap.add_argument("--upper-chain", action="store_true",
                    help="체인ID를 대문자로 통일(중복 검출에 유리)")
    ap.add_argument("--ignore-resn", action="store_true",
                    help="resn(아미노산 3문자) 무시하고 residue 번호만으로 유사도 계산")
    # 추가: PDB ID 중복 제거
    ap.add_argument("--dedup-by-pdb-id", action="store_true",
                    help="클러스터 대표들 중 PDB ID(언더스코어 앞) 중복도 제거")
    ap.add_argument("--pdbid-scope", choices=["global", "group"], default="global",
                    help="PDB ID 중복 제거 범위: global(기본) 또는 group(같은 그룹 내에서만)")
    ap.add_argument("--pdb-col", default=None,
                    help="PDB ID가 따로 있는 컬럼명(없으면 'PDB chain'에서 언더스코어 앞부분 사용)")
    args = ap.parse_args()

    in_path = Path(args.in_csv)
    out_dedup = Path(args.out_dedup)
    out_map = Path(args.out_map)
    out_dedup.parent.mkdir(parents=True, exist_ok=True)

    # 1) CSV 읽기
    with open(in_path, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        header = rdr.fieldnames or []
        if not header:
            raise SystemExit("ERROR: 입력 CSV 헤더를 읽지 못했습니다.")
        rows = list(rdr)

    # 2) 필수 컬럼 감지
    col_chain, col_epi, auto_group, auto_pdb = _detect_cols(header)
    group_col = args.group_col or auto_group
    col_pdb = args.pdb_col or auto_pdb

    # 3) 행 → (chain, group, episet) 파싱 및 보조 정보
    chains, groups, episets, pdbids = [], [], [], []
    for r in rows:
        chain = (r.get(col_chain) or "").strip()
        if args.upper_chain:
            chain = chain.upper()
        group = (r.get(group_col) or "").strip() if group_col else ""
        episet = _parse_epilist(r.get(col_epi), ignore_resn=args.ignore_resn)
        pdbid  = _get_pdb_id(r, col_pdb, col_chain)
        chains.append(chain); groups.append(group); episets.append(episet); pdbids.append(pdbid)
    entries = list(zip(chains, groups, episets, pdbids))  # (chain, group, episet, pdbid)

    # 4) 1단계: 그룹별 에피토프 유사도 클러스터링
    group_to_indices = defaultdict(list)
    for i, g in enumerate(groups):
        group_to_indices[g].append(i)

    kept_indices_stage1 = []   # 1단계 대표행 집합
    map_rows = []              # 단계별 매핑 기록
    cluster_id = 0

    def sim(i, j): 
        return _epi_similarity(entries[i][2], entries[j][2])

    for g, idxs in group_to_indices.items():
        clusters = []  # 각 요소: [행 idx...]
        reps = []      # 각 클러스터 대표 idx
        for i in idxs:
            placed = False
            for ci, rep_i in enumerate(reps):
                if sim(i, rep_i) >= args.similarity:
                    clusters[ci].append(i)
                    reps[ci] = _choose_rep(clusters[ci], [(e[0], e[1], e[2]) for e in entries], how=args.keep)
                    placed = True
                    break
            if not placed:
                clusters.append([i]); reps.append(i)
        for ci, members in enumerate(clusters):
            cluster_id += 1
            rep = reps[ci]
            kept_indices_stage1.append(rep)
            for m in members:
                map_rows.append({
                    "stage": "cluster",
                    "cluster_id": cluster_id,
                    "group": g,
                    "representative_row": rep + 1,
                    "representative_chain": entries[rep][0],
                    "row": m + 1,
                    "chain": entries[m][0],
                    "pdb_id": entries[m][3],
                    "similarity_to_rep": round(sim(m, rep), 6),
                    "kept": (m == rep)
                })

    kept_set = set(kept_indices_stage1)

    # 5) 2단계: PDB ID 중복 제거 (옵션)
    if args.dedup_by_pdb_id:
        # 범위가 group이면 (group, pdbid)로 묶고, global이면 (pdbid)로 묶음
        bucket = defaultdict(list)
        for i in kept_indices_stage1:
            key = (groups[i], pdbids[i]) if args.pdbid_scope == "group" else (pdbids[i],)
            bucket[key].append(i)

        final_kept = set()
        for key, idxs in bucket.items():
            if len(idxs) == 1:
                final_kept.add(idxs[0]); continue
            # 대표 선택
            rep = _choose_rep(idxs, [(e[0], e[1], e[2]) for e in entries], how=args.keep)
            final_kept.add(rep)
            # 매핑 기록(stage='pdbid')
            for m in idxs:
                map_rows.append({
                    "stage": "pdbid",
                    "cluster_id": None,
                    "group": groups[m] if args.pdbid_scope == "group" else "",
                    "representative_row": rep + 1,
                    "representative_chain": entries[rep][0],
                    "row": m + 1,
                    "chain": entries[m][0],
                    "pdb_id": entries[m][3],
                    "similarity_to_rep": None,
                    "kept": (m == rep)
                })
        kept_set = final_kept

    # 6) 최종 dedup CSV 저장(대표행만, 원본 컬럼 보존)
    with open(out_dedup, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for i, r in enumerate(rows):
            if i in kept_set:
                w.writerow(r)

    # 7) 매핑 CSV 저장
    with open(out_map, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "stage","cluster_id","group",
            "representative_row","representative_chain",
            "row","chain","pdb_id","similarity_to_rep","kept"
        ])
        w.writeheader()
        w.writerows(map_rows)

    print(f"[DONE] rows_in={len(rows)}  kept={len(kept_set)}  out='{out_dedup}'")
    if args.dedup_by_pdb_id:
        print(f"[INFO] PDB ID dedup applied (scope={args.pdbid_scope})")
    print(f"[INFO] map file: '{out_map}'")

if __name__ == "__main__":
    main()
