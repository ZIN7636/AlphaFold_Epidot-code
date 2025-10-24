import os
import torch
import torch.multiprocessing as mp
import pickle as pk
import argparse
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from EGAT import EGAT
from tqdm import tqdm # <<< 이 줄을 추가해 주세요
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryAUROC

# mp.set_start_method('spawn', force=True)

# ------------------- Dataset -------------------
class GraphBepiDataset(Dataset):
    def __init__(self, samples, root_path):
        self.samples = [s for s in samples if getattr(s, 'is_processed', False)]
        self.root_path = root_path

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        try:
            feat_path = os.path.join(self.root_path, 'feat', f'{s.name}_esm2.ts')
            x_esm = torch.load(feat_path).cpu()
            x_dssp = s.dssp.cpu()
            graph_path = os.path.join(self.root_path, 'graph', f'{s.name}.graph')
            graph = torch.load(graph_path)
            edge = graph['adj']
            y = s.label.cpu()
            
            min_len = min(x_esm.shape[0], x_dssp.shape[0], edge.shape[0], y.shape[0])
            x_esm, x_dssp = x_esm[:min_len, :], x_dssp[:min_len, :]
            edge_final, y_final = edge[:min_len, :min_len], y[:min_len]
            x_final = torch.cat([x_esm, x_dssp], dim=1)
            
            # <<< 수정: 결과 저장을 위해 시퀀스와 이름도 반환 >>>
            return x_final, edge_final, y_final, s.sequence[:min_len], s.name
        except Exception:
            return None

# ------------------- Collate 함수 -------------------
def collate_wrapper(batch):
    batch = [item for item in batch if item is not None]
    if not batch: return None

    x_list, edge_list, y_list_original, seq_list, name_list = zip(*batch)

    y_list = []
    for y in y_list_original:
        if y.dim() == 1: y = y.unsqueeze(1)
        y_list.append(y)
    
    max_len = max([x.shape[0] for x in x_list])
    
    x_padded = [F.pad(x, (0, 0, 0, max_len - x.shape[0])) for x in x_list]
    y_padded = [F.pad(y, (0, 0, 0, max_len - y.shape[0]), value=-1) for y in y_list] # 패딩값 -1로 변경

    edge_padded = []
    for edge in edge_list:
        if edge.is_sparse: edge = edge.to_dense()
        pad_size = max_len - edge.shape[0]
        edge = F.pad(edge, (0, pad_size, 0, pad_size))
        edge_padded.append(edge)

    # <<< 수정: 시퀀스와 이름은 그대로 리스트로 반환 >>>
    return torch.stack(x_padded), torch.stack(edge_padded), torch.stack(y_padded), seq_list, name_list

# ------------------- Lightning Model -------------------
class GraphBepiLightning(pl.LightningModule):
    def __init__(self, hidden_dim=128, dropout=0.2, lr=1e-3, threshold=0.5):
        super().__init__()
        self.save_hyperparameters()
        self.W_v = nn.Linear(333, hidden_dim) 
        self.gat = EGAT(hidden_dim, hidden_dim, edge_dim=1)
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(hidden_dim*2, hidden_dim, batch_first=True, bidirectional=True)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        self.loss_fn = nn.BCELoss()
        
        # <<< 수정: 클래스 변수로 hparams 저장 >>>
        self.threshold = self.hparams.threshold
        self.output_path = None # main 함수에서 설정

        # 테스트 성능 지표
        self.test_accuracy = BinaryAccuracy(threshold=self.threshold)
        self.test_precision = BinaryPrecision(threshold=self.threshold)
        self.test_recall = BinaryRecall(threshold=self.threshold)
        self.test_f1 = BinaryF1Score(threshold=self.threshold)
        self.test_auroc = BinaryAUROC()

    def forward(self, x, edge_attr):
        batch_size, seq_len, _ = x.shape
        h = F.relu(self.W_v(x))
        node_features = h.view(-1, h.size(-1))
        edge_indices = []
        for i in range(batch_size):
            edge_index = edge_attr[i].nonzero().t()
            offset = i * seq_len
            edge_indices.append(edge_index + offset)
        
        batched_edge_index = torch.cat(edge_indices, dim=1) if edge_indices else torch.empty((2, 0), dtype=torch.long, device=x.device)
        h_gat, _ = self.gat(node_features, batched_edge_index)
        h = h_gat.view(batch_size, seq_len, -1)
        h, _ = self.lstm1(h)
        h, _ = self.lstm2(h)
        out = torch.sigmoid(self.mlp(h))
        return out

    def training_step(self, batch, batch_idx):
        if batch is None: return None
        x, edge_attr, y, _, _ = batch
        y_hat = self(x, edge_attr)
        mask = y != -1
        loss = self.loss_fn(y_hat[mask], y[mask].float())
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if batch is None: return None
        x, edge_attr, y, _, _ = batch
        y_hat = self(x, edge_attr)
        mask = y != -1
        loss = self.loss_fn(y_hat[mask], y[mask].float())
        self.log('val_loss', loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        if batch is None: return None
        x, edge_attr, y, sequences, names = batch
        y_hat = self(x, edge_attr).squeeze(-1)
        y_true = y.squeeze(-1)
        
        mask = y_true != -1
        if mask.sum() > 0:
            self.test_accuracy.update(y_hat[mask], y_true[mask].int())
            self.test_precision.update(y_hat[mask], y_true[mask].int())
            self.test_recall.update(y_hat[mask], y_true[mask].int())
            self.test_f1.update(y_hat[mask], y_true[mask].int())
            self.test_auroc.update(y_hat[mask], y_true[mask].int())
        
        # <<< 수정: CSV 저장을 위해 결과 반환 >>>
        return {'preds': y_hat, 'sequences': sequences, 'names': names}

    def on_test_epoch_end(self, outputs):
        # 성능 지표 출력
        print("\n--- Test Results (Metrics) ---")
        print(f"Threshold: {self.threshold:.4f}")
        print(f"Accuracy: {self.test_accuracy.compute():.4f}")
        print(f"Precision: {self.test_precision.compute():.4f}")
        print(f"Recall: {self.test_recall.compute():.4f}")
        print(f"F1 Score: {self.test_f1.compute():.4f}")
        print(f"AUROC: {self.test_auroc.compute():.4f}")
        
        # <<< 추가: CSV 파일 저장 로직 >>>
        print("\n--- Saving prediction files ---")
        if self.output_path and not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            
        for output in tqdm(outputs, desc="Saving CSVs"):
            preds, sequences, names = output['preds'], output['sequences'], output['names']
            for i in range(len(names)):
                name = names[i]
                seq = sequences[i]
                pred = preds[i][:len(seq)] # 패딩 제거
                
                is_epitope = (pred > self.threshold).bool()
                df = pd.DataFrame({'resn': list(seq), 'score': pred.cpu().numpy(), 'is epitope': is_epitope.cpu().numpy()})
                df.to_csv(os.path.join(self.output_path, f'{name}.csv'), index=False)
        print(f"Results saved to {self.output_path}")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

# ------------------- Main -------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./data/DEDUP90_FINETUNE')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_epochs', type=int, default=50)
    # <<< 추가: 결과 저장을 위한 인자 >>>
    parser.add_argument('-o', '--output', type=str, default='./output', help='Output path for prediction CSVs')
    parser.add_argument('-t', '--threshold', type=float, default=0.5, help='Prediction threshold')
    args = parser.parse_args()

    device = 'cuda' if args.gpu != -1 and torch.cuda.is_available() else 'cpu'
    
    with open(os.path.join(args.root, 'train.pkl'), 'rb') as f: train_samples = pk.load(f)
    with open(os.path.join(args.root, 'val.pkl'), 'rb') as f: val_samples = pk.load(f)
    with open(os.path.join(args.root, 'test.pkl'), 'rb') as f: test_samples = pk.load(f)

    train_loader = DataLoader(GraphBepiDataset(train_samples, args.root), batch_size=args.batch_size, shuffle=True, collate_fn=collate_wrapper, num_workers=4)
    val_loader = DataLoader(GraphBepiDataset(val_samples, args.root), batch_size=args.batch_size, shuffle=False, collate_fn=collate_wrapper, num_workers=4)
    test_loader = DataLoader(GraphBepiDataset(test_samples, args.root), batch_size=args.batch_size, shuffle=False, collate_fn=collate_wrapper, num_workers=4)

    # <<< 수정: 모델에 threshold 인자 전달 >>>
    model = GraphBepiLightning(threshold=args.threshold)
    model.output_path = args.output # 출력 경로 설정

    trainer = pl.Trainer(devices=[args.gpu] if device == 'cuda' else 1, accelerator='gpu' if device == 'cuda' else 'cpu',
                         max_epochs=args.max_epochs, check_val_every_n_epoch=1, log_every_n_steps=10)
    
    print("\n--- Starting Training ---")
    trainer.fit(model, train_loader, val_loader)

    print("\n--- Starting Test ---")
    trainer.test(model=model, dataloaders=test_loader, ckpt_path='best')

if __name__ == "__main__":
    main()