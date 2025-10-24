r"""
make_pdb_chains_from_pdb_seqres_ko.py
=====================================
기능
----
- RCSB/wwPDB에서 배포하는 **pdb_seqres.txt.gz**를 내려받아
  **pdb_chains.fasta**(헤더: `>PDBID_CHAIN`)로 변환합니다.
- 이미 gz/txt 파일이 있다면 경로만 지정해도 됩니다.
-------
# 1) 전체 파일 자동 다운로드 → d:/final/aws/pdb_chains.fasta 생성
python make_pdb_chains_from_pdb_seqres_ko.py \
  --out "d:/final/aws/pdb_chains.fasta"

# 2) 이미 받은 gz 파일이 있다면
python make_pdb_chains_from_pdb_seqres_ko.py \
  --pdb-seqres "d:/data/pdb_seqres.txt.gz" \
  --out "d:/final/aws/pdb_chains.fasta"

# 3) 특정 PDB/체인만 포함 (include 리스트)
#    include.txt 예시:
#      1ABC
#      2XYZ_A
python make_pdb_chains_from_pdb_seqres_ko.py \
  --out "d:/final/aws/pdb_chains.fasta" \
  --include-list "d:/final/aws/include.txt"
"""

import re
import os
import gzip
import argparse
from pathlib import Path
import requests

PDB_SEQRES_URL = "https://ftp.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt.gz"

def download_pdb_seqres(dest: Path, *, timeout=120):
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(PDB_SEQRES_URL, timeout=timeout, stream=True, headers={"User-Agent":"Mozilla/5.0 (pdb-seqres-fetch)"})
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024*256):
            if chunk:
                f.write(chunk)
    return dest

def parse_include_list(path: Path):
    if not path: return None, None
    if not path.exists(): raise FileNotFoundError(f"include-list not found: {path}")
    ids=set(); id_chains=set()
    with open(path, encoding="utf-8") as f:
        for ln in f:
            s=ln.strip()
            if not s: continue
            if "_" in s:
                id_chains.add(s.upper())
            else:
                ids.add(s.upper())
    return ids, id_chains

def want_entry(pdbid, chain, ids, id_chains):
    if ids is None and id_chains is None:
        return True
    key = f"{pdbid}_{chain}"
    if id_chains and key in id_chains: return True
    if ids and pdbid in ids: return True
    return False

def main():
    ap = argparse.ArgumentParser(description="pdb_seqres.txt.gz → pdb_chains.fasta 생성기")
    ap.add_argument("--pdb-seqres", default=None, help="로컬 pdb_seqres.txt 또는 pdb_seqres.txt.gz 경로 (미지정 시 자동 다운로드)")
    ap.add_argument("--out", required=True, help="출력 FASTA 경로 (예: d:/final/aws/pdb_chains.fasta)")
    ap.add_argument("--include-list", default=None, help="포함할 PDBID 또는 PDBID_CHAIN 목록 파일")
    ap.add_argument("--only-protein", dest="only_protein", action="store_true", help="단백질만 포함(기본)")
    ap.add_argument("--no-only-protein", dest="only_protein", action="store_false", help="단백질만 포함 옵션 해제")
    ap.set_defaults(only_protein=True)
    ap.add_argument("--limit", type=int, default=0, help="상위 N 엔트리만 처리(테스트용)")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 입력 파일 확보
    src_path = Path(args.pdb_seqres) if args.pdb_seqres else out_path.with_name("pdb_seqres.txt.gz")
    if not src_path.exists():
        print(f"[INFO] 다운로드 중: {PDB_SEQRES_URL} → {src_path}")
        download_pdb_seqres(src_path)

    # include 목록 파싱
    ids, id_chains = parse_include_list(Path(args.include_list)) if args.include_list else (None, None)

    # gz 여부 판단
    is_gz = src_path.suffix.lower()==".gz"
    opener = (lambda p: gzip.open(p, "rt", encoding="utf-8", errors="ignore")) if is_gz else (lambda p: open(p, "r", encoding="utf-8", errors="ignore"))

    # 파싱
    n_total=0; n_written=0
    with opener(src_path) as fh, open(out_path, "w", encoding="utf-8") as out:
        seq_buf=[]
        hdr=None
        for ln in fh:
            ln=ln.rstrip("\n")
            if ln.startswith(">"):
                # flush prev
                if hdr is not None and seq_buf:
                    seq="".join(seq_buf).replace(" ", "").upper()
                    if seq:
                        out.write(hdr+"\n")
                        for i in range(0,len(seq),60):
                            out.write(seq[i:i+60]+"\n")
                        n_written+=1
                        if args.limit and n_written>=args.limit:
                            break
                # new header
                # 예시: >1STP_A mol:protein length:96  ...
                # 또는   >1ABC:A ...
                m = re.match(r"^>(?P<pdb>[0-9A-Za-z]{4})[ _:]?(?P<chain>[A-Za-z0-9])\b", ln)
                if not m:
                    hdr=None; seq_buf=[]
                    continue
                pdbid=m.group("pdb").upper()
                chain=m.group("chain")
                if args.only_protein:
                    if ("mol:protein" not in ln.lower()) and ("mol:polypeptide" not in ln.lower()):
                        hdr=None; seq_buf=[]
                        continue
                if not want_entry(pdbid, chain, ids, id_chains):
                    hdr=None; seq_buf=[]
                    continue
                hdr=f">{pdbid}_{chain}"
                seq_buf=[]
                n_total+=1
            else:
                if hdr is not None:
                    seq_buf.append(ln.strip())
        # flush last
        if hdr is not None and seq_buf and (not args.limit or n_written<args.limit):
            seq="".join(seq_buf).replace(" ", "").upper()
            if seq:
                out.write(hdr+"\n")
                for i in range(0,len(seq),60):
                    out.write(seq[i:i+60]+"\n")
                n_written+=1

    print({"entries_seen": n_total, "entries_written": n_written, "out": str(out_path)})

if __name__ == "__main__":
    main()
