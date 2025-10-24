import os
import torch
import numpy as np
import pandas as pd
import pickle as pk
import requests as rq
from tqdm import tqdm
from graph_construction import calcPROgraph
from preprocess import DICT, judge, process_dssp, transform_dssp
from graph_construction import calcPROgraph

# 아미노산 코드를 숫자로 변환하는 테이블
amino2id = {
    '<null_0>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3,
    'L': 4, 'A': 5, 'G': 6, 'V': 7, 'S': 8, 'E': 9, 'R': 10, 
    'T': 11, 'I': 12, 'D': 13, 'P': 14, 'K': 15, 'Q': 16, 
    'N': 17, 'F': 18, 'Y': 19, 'M': 20, 'H': 21, 'W': 22, 
    'C': 23, 'X': 24, 'B': 25, 'U': 26, 'Z': 27, 'O': 28, 
    '.': 29, '-': 30, '<null_1>': 31, '<mask>': 32
}

# ---------------- PDB chain 클래스 ----------------
class chain:
    def __init__(self):
        self.sequence, self.amino, self.coord, self.site = [], [], [], {}
        self.date, self.length = '', 0
        self.label = None
        self.name, self.chain_name, self.protein_name = '', '', ''
        self.is_processed = False

    def add(self, amino, pos, coord):
        if amino not in DICT: return
        self.sequence.append(DICT[amino])
        self.amino.append(amino2id[DICT[amino]])
        self.coord.append(coord)
        self.site[pos] = self.length
        self.length += 1

    def process(self):
        if not self.amino: return
        self.amino = torch.LongTensor(self.amino)
        self.coord = torch.FloatTensor(self.coord)
        self.label = torch.zeros(self.length, dtype=torch.long)
        self.sequence = ''.join(self.sequence)

    def update(self, pos, amino):
        if amino not in DICT or self.label is None: return
        amino_id = amino2id[DICT[amino]]
        for i_site, i_idx in self.site.items():
            if i_site.startswith(pos) and i_idx < len(self.amino) and amino_id == self.amino[i_idx]:
                self.label[i_idx] = 1
                return
    
    def __len__(self):
        return self.length

# ---------------- PDB 다운로드 및 chain 추출 ----------------
def extract_chain(root, pid, chain, force=False):
    chain_pdb_file = f'{root}/purePDB/{pid}_{chain}.pdb'
    if not force and os.path.exists(chain_pdb_file): return True
    source_pdb_file = f'{root}/PDB/{pid}.pdb'
    if not os.path.exists(source_pdb_file):
        try:
            with rq.get(f'https://files.rcsb.org/download/{pid}.pdb', timeout=20) as f:
                if f.status_code == 200:
                    with open(source_pdb_file,'wb') as wf: wf.write(f.content)
                else: return False
        except Exception:
            return False
    lines=[]
    try:
        with open(source_pdb_file,'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith(('ATOM', 'HETATM')) and line[21] == chain: lines.append(line)
        with open(chain_pdb_file,'w') as f: f.writelines(lines)
        return True
    except Exception:
        return False

# ---------------- DSSP 처리 ----------------
def get_dssp_from_pdb(pid, root):
    pdb_file = f"{root}/purePDB/{pid}.pdb"
    dssp_raw_file = f"{root}/dssp/{pid}.dssp"
    os.system(f"./mkdssp/mkdssp -i {pdb_file} -o {dssp_raw_file}")
    if not os.path.exists(dssp_raw_file) or os.path.getsize(dssp_raw_file) == 0:
        return None, None
    try:
        _, dssp_matrix, position = process_dssp(dssp_raw_file)
        dssp_transformed = transform_dssp(dssp_matrix)
        return torch.FloatTensor(dssp_transformed), position
    except Exception:
        return None, None

# ---------------- chain feature 생성 (ESM embedding + DSSP + graph) ----------------
def process_chain(data, root, chain_name, model=None, device='cpu'):
    """
    data: chain 객체
    root: PDB root path
    chain_name: 'PDBID_CHAINID'
    model: ESM 모델 (cuda로 이미 이동됨)
    device: 'cuda:0' or 'cpu'
    
    Returns: chain 객체 (x, edge, y 포함)
    """
    pdb_file = f"{root}/purePDB/{chain_name}.pdb"
    if not os.path.exists(pdb_file):
        return None

    # PDB 파일 읽고 chain 데이터 생성
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('HEADER'):
                data.date = line[50:59].strip()
            feats = judge(line, 'CA')
            if feats:
                data.add(feats[0], feats[2], feats[3:])

    data.process()
    if len(data) == 0:
        return None

    try:
        # Graph 생성
        graph = calcPROgraph(data.sequence, data.coord)
        torch.save(graph, f'{root}/graph/{data.name}.graph')

        # ESM 임베딩 처리
        if model is not None:
            batch_converter = model.alphabet.get_batch_converter()
            _, _, tokens = batch_converter([(chain_name, data.sequence)])
            tokens = tokens.to(device)
            with torch.no_grad():
                repr_layer = 36  # <<< ESM2-t36-3B 모델의 경우 36층 >>>
                x = model(tokens, repr_layers=[repr_layer])['representations'][repr_layer].squeeze(0)
                
                # ESM 임베딩의 시작/끝 특수 토큰 제거
                # [1:-1] 슬라이싱으로 실제 아미노산 서열에 해당하는 임베딩만 남깁니다.
                x = x[1:-1, :]  # 차원: (sequence_length, 2560)
                
                torch.save(x.cpu(), f'{root}/feat/{data.name}_esm2.ts')

        # DSSP 처리
        dssp_tensor, pos_list = get_dssp_from_pdb(chain_name, root)
        data.dssp = torch.zeros(len(data), 13)
        if dssp_tensor is not None and pos_list is not None:
            for i, pos in enumerate(pos_list):
                if pos in data.site:
                    data.dssp[data.site[pos]] = dssp_tensor[i]

        # chain 객체 업데이트
        data.x = x if model is not None else None
        data.edge = graph
        data.y = data.label
        data.is_processed = True

    except Exception as e:
        print(f"[Warning] Feature extraction failed for {chain_name}: {e}")

    return data

# ---------------- 전체 CSV 파일 처리 ----------------
def initial(file, root, model, device, resume=True, skip_list=None):
    if skip_list is None: skip_list=[]
    try:
        df = pd.read_csv(f'{root}/{file}', header=0, index_col=0)
    except FileNotFoundError:
        print(f"[Error] CSV file not found: {root}/{file}")
        return []

    prefix = df.index.unique()
    labels = df['Epitopes (resi_resn)']
    samples = []

    with tqdm(prefix, desc=f"Processing {os.path.basename(file)}") as tbar:
        for i in tbar:
            tbar.set_postfix(protein=i)
            p, c = i.split('_')
            chain_name = f"{p}_{c}"

            data = chain()
            data.name = chain_name

            if i in skip_list:
                samples.append(data)
                continue

            if resume and os.path.exists(f'{root}/feat/{chain_name}_esm2.ts'):
                data.is_processed = True

            if not data.is_processed:
                if not extract_chain(root, p, c):
                    samples.append(data)
                    continue

                processed_data = process_chain(data, root, chain_name, model, device)
                if processed_data is None:
                    samples.append(data)
                    continue
                data = processed_data

            try:
                if data.length == 0 and os.path.exists(f'{root}/purePDB/{chain_name}.pdb'):
                    with open(f'{root}/purePDB/{chain_name}.pdb', 'r') as f:
                        for line in f:
                            feats = judge(line, 'CA')
                            if feats: data.add(feats[0], feats[2], feats[3:])
                    data.process()

                label_info = labels.loc[i]
                if isinstance(label_info, str) and pd.notna(label_info):
                    for j in label_info.split(', '):
                        data.update(*j.split('_'))
            except Exception:
                pass

            samples.append(data)

    print(f"\nTotal processed attempts for {file}: {len(samples)}")
    return samples

# ---------------- DataLoader를 위한 함수 ----------------
def collate_fn(batch):
    root = './data/DEDUP90_FINETUNE' 
    batch_data = {'feat': [], 'adj': [], 'edge': [], 'label': []}

    for seq in batch:
        try:
            feat_tensor = torch.load(f'{root}/feat/{seq.name}_esm2.ts')  # 차원: (seq_len, 2560)

            dssp_tensor, pos_list = get_dssp_from_pdb(seq.name, root)
            dssp = torch.zeros(len(seq), 13)  # 차원: (seq_len, 13)
            if dssp_tensor is not None and pos_list is not None:
                for i, pos in enumerate(pos_list):
                    if pos in seq.site: dssp[seq.site[pos]] = dssp_tensor[i]

            graph = torch.load(f'{root}/graph/{seq.name}.graph')
            adj = graph['adj'].to_dense()
            edge = graph['edge'].to_dense()

            if feat_tensor.shape[0] != dssp.shape[0] or feat_tensor.shape[0] != len(seq.label):
                continue

            # ESM 임베딩 (2560차원) + DSSP (13차원) = 2573차원
            combined_features = torch.cat([feat_tensor, dssp], 1)  # 차원: (seq_len, 2573)
            batch_data['feat'].append(combined_features)
            batch_data['adj'].append(adj)
            batch_data['edge'].append(edge)
            batch_data['label'].append(seq.label)

        except Exception:
            continue

    if not batch_data['feat']:
        return None

    max_len = max(f.shape[0] for f in batch_data['feat'])
    final_batch = {}
    
    # Feature padding: (batch_size, max_seq_len, 2573)
    final_batch['feat'] = torch.stack([torch.nn.functional.pad(f, (0, 0, 0, max_len - f.shape[0])) for f in batch_data['feat']])
    
    # Adjacency matrix padding: (batch_size, max_seq_len, max_seq_len)
    final_batch['adj'] = torch.stack([torch.nn.functional.pad(a, (0, max_len - a.shape[0], 0, max_len - a.shape[0])) for a in batch_data['adj']])
    
    # Edge feature padding: (batch_size, max_seq_len, max_seq_len, edge_dim)
    final_batch['edge'] = torch.stack([torch.nn.functional.pad(e, (0, 0, 0, max_len - e.shape[0], 0, max_len - e.shape[0])) for e in batch_data['edge']])
    
    # Label padding: (batch_size, max_seq_len)
    final_batch['label'] = torch.stack([torch.nn.functional.pad(l, (0, max_len - l.shape[0])) for l in batch_data['label']])

    return final_batch

def collate_wrapper(batch):
    batch = [item for item in batch if item is not None and len(item) > 0]
    if not batch: return None
    return collate_fn(batch)