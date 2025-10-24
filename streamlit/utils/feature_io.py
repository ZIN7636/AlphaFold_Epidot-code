# utils/feature_io.py
from __future__ import annotations
import io, json, math, struct
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd

# ---------- Common helpers ----------
def _to_bytes(obj_or_bytes: bytes | str | io.BytesIO) -> bytes:
    if isinstance(obj_or_bytes, (bytes, bytearray)):
        return bytes(obj_or_bytes)
    if isinstance(obj_or_bytes, io.BytesIO):
        return obj_or_bytes.getvalue()
    if isinstance(obj_or_bytes, str):
        return obj_or_bytes.encode("utf-8")
    raise TypeError("Unsupported buffer type")

def _guess_text(b: bytes) -> str:
    try:
        return b.decode("utf-8")
    except Exception:
        return b.decode("latin-1", errors="ignore")

# ---------- ESM2 ----------
def load_esm2_array(buf: bytes) -> np.ndarray:
    """
    Accepts: .npy (NxD), .npz (keys: 'emb','array','X','repr'), .json (list[list[float]]),
             lightweight .pt saved via torch.save({'emb': ndarray})도 best-effort로 지원.
    Returns: float32 ndarray (N residues, D dims)
    """
    # npy
    if buf[:6] == b"\x93NUMPY":
        arr = np.load(io.BytesIO(buf), allow_pickle=False)
        return np.asarray(arr, dtype=np.float32)
    # npz
    if buf[:4] == b"PK\x03\x04":
        with np.load(io.BytesIO(buf), allow_pickle=True) as z:
            for k in ("emb","array","X","repr","representations","token_representations"):
                if k in z:
                    arr = z[k]
                    break
            else:
                # take the first numeric array
                k = [k for k in z.files if isinstance(z[k], np.ndarray)][0]
                arr = z[k]
        return np.asarray(arr, dtype=np.float32)
    # JSON
    txt = _guess_text(buf)
    if txt.strip().startswith("["):
        arr = np.asarray(json.loads(txt), dtype=np.float32)
        return arr
    # very lightweight PT (pickle) — optional
    try:
        import torch  # noqa
        from torch import load as torch_load
        by = io.BytesIO(buf)
        obj = torch_load(by, map_location="cpu")
        if isinstance(obj, dict):
            for k in ("emb","array","X","repr"):
                if k in obj:
                    data = obj[k]
                    if hasattr(data, "numpy"):
                        return data.numpy().astype(np.float32)
        if hasattr(obj, "numpy"):
            return obj.numpy().astype(np.float32)
    except Exception:
        pass
    raise ValueError("Unsupported ESM2 file format")

# ---------- DSSP ----------
def load_dssp(buf: bytes) -> pd.DataFrame:
    """
    Accepts: .npy/.npz (Nx13 or Nx14)
    Returns: DataFrame with columns:
      ss_* (one-hot), rsa, phi_sin, phi_cos, psi_sin, psi_cos
    """
    # NPY/NPZ
    if buf[:6] == b"\x93NUMPY" or buf[:4] == b"PK\x03\x04":
        if buf[:6] == b"\x93NUMPY":
            arr = np.load(io.BytesIO(buf), allow_pickle=False)
        else:
            with np.load(io.BytesIO(buf), allow_pickle=False) as z:
                # prefer explicit keys
                for k in ("dssp","array","X"):
                    if k in z:
                        arr = z[k]
                        break
                else:
                    arr = z[z.files[0]]
        arr = np.asarray(arr)
        n, d = arr.shape
        if d not in (13,14):
            raise ValueError(f"DSSP array must be Nx13 or Nx14, got Nx{d}")
        # Heuristic: if 14D, assume 9 one-hot + 1 RSA + 4 angles.
        # if 13D, assume 8 one-hot + 1 RSA + 4 angles.
        onehot_dim = 9 if d == 14 else 8
        cols = [f"ss_{i}" for i in range(onehot_dim)] + ["rsa","phi_sin","phi_cos","psi_sin","psi_cos"]
        return pd.DataFrame(arr, columns=cols)

    raise ValueError("Unsupported DSSP file format")

# ---------- Protein graph ----------
def load_protein_graph(buf: bytes) -> Dict[str, Any]:
    """
    Accepts:
      - JSON: {'nodes':[{'i':1,'chain':'A','resi':1,'resn':'ALA', ...}, ...],
               'edges':[{'i':1,'j':5,'type':'sequential','seq_offset':4},
                        {'i':2,'j':9,'type':'radius','dist':8.3}, {'i':..,'j':..,'type':'knn'}]}
      - NPZ:  edge_index (2,E), edge_type (E,), optional edge_dist (E,), node_chain/node_resi/resn
    Returns a normalized dict with 'nodes' and 'edges'.
    """
    # NPZ
    if buf[:4] == b"PK\x03\x04":
        with np.load(io.BytesIO(buf), allow_pickle=True) as z:
            nodes = []
            if "node_resi" in z:
                L = int(len(z["node_resi"]))
                for i in range(L):
                    nodes.append({
                        "i": int(i+1),
                        "chain": (z["node_chain"][i].item() if "node_chain" in z else "A"),
                        "resi": int(z["node_resi"][i]),
                        "resn": (str(z["node_resn"][i]) if "node_resn" in z else None),
                    })
            elif "L" in z:
                for i in range(int(z["L"].item())):
                    nodes.append({"i": i+1})
            ei = z["edge_index"] if "edge_index" in z else None
            et = z["edge_type"] if "edge_type" in z else None
            ed = z["edge_dist"] if "edge_dist" in z else None
            edges = []
            if ei is not None:
                E = ei.shape[1]
                for k in range(E):
                    i, j = int(ei[0,k]), int(ei[1,k])
                    rec = {"i": i, "j": j}
                    if et is not None:
                        tv = et[k].item() if hasattr(et[k],"item") else int(et[k])
                        rec["type_id"] = int(tv)
                    if ed is not None:
                        rec["dist"] = float(ed[k])
                    edges.append(rec)
            return {"nodes": nodes, "edges": edges}

    # JSON
    txt = _guess_text(buf)
    js = json.loads(txt)
    if isinstance(js, dict) and "nodes" in js and "edges" in js:
        return js

    raise ValueError("Unsupported protein-graph format")


def summarise_graph(g: Dict[str, Any]) -> Dict[str, Any]:
    edges = g.get("edges", [])
    nodes = g.get("nodes", [])
    
    # 기본 타입별 카운팅 (기존 방식)
    by_type: Dict[str, int] = {}
    for e in edges:
        t = e.get("type") or f"type_{e.get('type_id','?')}"
        by_type[t] = by_type.get(t, 0) + 1
    
    # GraphBepi 스타일의 엣지 특징 분석 (추가)
    connection_types = {"Sequential": 0, "Spatial": 0, "K-NN": 0}
    
    for e in edges:
        features = e.get("features")
        if features and len(features) >= 45:
            # GraphBepi의 엣지 특징 구조:
            # 0-41: 아미노산 페어 (21x2)
            # 42: Sequential connection flag
            # 43: Spatial/Radius connection flag  
            # 44: K-NN connection flag
            # 45-46: 거리값들
            
            if features[42] > 0:  # Sequential flag
                connection_types["Sequential"] += 1
            if features[43] > 0:  # Spatial/Radius flag
                connection_types["Spatial"] += 1
            if features[44] > 0:  # K-NN flag
                connection_types["K-NN"] += 1
    
    # 의미있는 연결 타입이 있으면 교체, 없으면 기존 사용
    if sum(connection_types.values()) > 0:
        by_type = connection_types
    
    return {
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "edges_by_type": by_type,
    }
