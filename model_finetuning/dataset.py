# dataset.py - 메모리 최적화 버전

import os
import torch
import pickle as pk
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
import esm  # ESM 모델 사용
import gc

from utils import chain, extract_chain, process_chain  # 기존 utils 필요

# ---------------- Dataset 클래스 ----------------
class GraphBepiDataset(Dataset):
    def __init__(self, samples):
        self.samples = [s for s in samples if s is not None]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        if not s.is_processed:
            return None
        x = torch.load(f'./data/DEDUP90_FINETUNE/feat/{s.name}_esm2.ts')
        graph = torch.load(f'./data/DEDUP90_FINETUNE/graph/{s.name}.graph')
        edge = graph['edge'].to_dense()
        y = s.label
        return x, edge, y

# ---------------- Collate 함수 ----------------
def collate_wrapper(batch):
    x_list, edge_list, y_list = [], [], []

    for item in batch:
        if item is None:
            continue
        x, edge, y = item
        x_list.append(x)
        edge_list.append(edge)
        y_list.append(y)

    if len(x_list) == 0:
        return None

    max_len = max([x.shape[0] for x in x_list])

    x_padded = [torch.cat([x, torch.zeros(max_len - x.shape[0], x.shape[1])], dim=0) for x in x_list]
    y_padded = [torch.cat([y, torch.zeros(max_len - y.shape[0], dtype=torch.long)], dim=0) for y in y_list]

    edge_padded = []
    for edge in edge_list:
        pad_size = max_len - edge.shape[0]
        edge = torch.nn.functional.pad(edge, (0, 0, 0, pad_size, 0, pad_size))
        edge_padded.append(edge)

    return torch.stack(x_padded), torch.stack(edge_padded), torch.stack(y_padded)

# ---------------- CSV 처리 ----------------
def split_csv(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    np.random.seed(seed)
    idx = np.arange(len(df))
    np.random.shuffle(idx)
    n = len(df)
    train_end = int(n*train_ratio)
    val_end = int(n*(train_ratio+val_ratio))
    return df.iloc[idx[:train_end]], df.iloc[idx[train_end:val_end]], df.iloc[idx[val_end:]]

def process_csv_df(df, root, model, device, skip_list=[], batch_size=5):
    """배치 처리로 메모리 사용량 줄이기"""
    samples = []
    success_count = 0
    
    # 데이터프레임을 배치로 나누어 처리
    total_items = list(df.iterrows())
    
    for batch_start in range(0, len(total_items), batch_size):
        batch_end = min(batch_start + batch_size, len(total_items))
        batch_items = total_items[batch_start:batch_end]
        
        print(f"Processing batch {batch_start//batch_size + 1}/{(len(total_items)-1)//batch_size + 1} "
              f"(items {batch_start+1}-{batch_end})")
        
        for pdb_chain, row in tqdm(batch_items, desc=f"Batch {batch_start//batch_size + 1}"):
            pid, ch = pdb_chain.split('_')
            data = chain()
            data.name = pdb_chain

            if pdb_chain in skip_list:
                samples.append(data)
                continue

            if not extract_chain(root, pid, ch):
                samples.append(data)
                continue

            try:
                processed = process_chain(data, root, pdb_chain, model, device)
                if processed is None:
                    samples.append(data)
                    continue
                
                data = processed
                if data.is_processed:
                    success_count += 1

                # 레이블 업데이트
                epitopes = row['Epitopes (resi_resn)']
                if isinstance(epitopes, str) and pd.notna(epitopes):
                    for e in epitopes.split(', '):
                        try:
                            data.update(*e.split('_'))
                        except Exception:
                            pass
                            
            except Exception as e:
                print(f"[ERROR] {pdb_chain}: {e}")
            
            samples.append(data)
        
        # 배치 처리 후 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    print(f"Successfully processed: {success_count} / {len(df)}")
    return samples

# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./data/DEDUP90_FINETUNE', help='dataset path')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--csv', type=str, default='2nd_total_except_prup3.csv', help='input CSV file')
    parser.add_argument('--batch_size', type=int, default=5, help='batch size for processing')
    args = parser.parse_args()

    device = 'cpu' if args.gpu == -1 else f'cuda:{args.gpu}'
    root = args.root
    csv_file = os.path.join(root, args.csv)

    for d in ['PDB','purePDB','feat','dssp','graph']:
        os.makedirs(os.path.join(root,d), exist_ok=True)

    print("Loading ESM-2 model (3B parameters)...")
    model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    model = model.to(device)
    model.eval()
    
    # 메모리 사용량 확인
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")

    print(f"\nReading CSV: {csv_file}")
    df_total = pd.read_csv(csv_file)
    if 'PDB chain' not in df_total.columns:
        raise ValueError("CSV must contain 'PDB chain' column")
    df_total = df_total.set_index('PDB chain')

    df_train, df_val, df_test = split_csv(df_total)
    print(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

    skip_list = ['7UTI_W', '8UAI_D', '9IS2_C']

    print("\n--- Processing Train ---")
    train_samples = process_csv_df(df_train, root, model, device, skip_list, args.batch_size)
    with open(os.path.join(root,'train.pkl'),'wb') as f: pk.dump(train_samples, f)

    print("\n--- Processing Validation ---")
    val_samples = process_csv_df(df_val, root, model, device, skip_list, args.batch_size)
    with open(os.path.join(root,'val.pkl'),'wb') as f: pk.dump(val_samples, f)

    print("\n--- Processing Test ---")
    test_samples = process_csv_df(df_test, root, model, device, skip_list, args.batch_size)
    with open(os.path.join(root,'test.pkl'),'wb') as f: pk.dump(test_samples, f)

    print("\n--- Summary ---")
    print(f"Train processed: {sum(1 for s in train_samples if s.is_processed)} / {len(train_samples)}")
    print(f"Val processed: {sum(1 for s in val_samples if s.is_processed)} / {len(val_samples)}")
    print(f"Test processed: {sum(1 for s in test_samples if s.is_processed)} / {len(test_samples)}")

    if sum(1 for s in train_samples if s.is_processed) > 0:
        idx = np.arange(sum(1 for s in train_samples if s.is_processed))
        np.random.shuffle(idx)
        np.save(os.path.join(root,'cross-validation.npy'), idx)

    print("\n--- Finished ---")

if __name__ == "__main__":
    main()