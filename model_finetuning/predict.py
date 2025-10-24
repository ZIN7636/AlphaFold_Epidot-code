import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import argparse
import pandas as pd
from tqdm import tqdm
import pickle as pk
from torch.utils.data import DataLoader, Dataset
from EGAT import EGAT

# ------------------- Dataset & Collate 함수 (이전과 동일) -------------------
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
            if not isinstance(graph, dict) or 'adj' not in graph:
                return None
            edge = graph['adj']
            if edge.is_sparse:
                edge = edge.to_dense()
            y = s.label.cpu()
            min_len = min(x_esm.shape[0], x_dssp.shape[0], edge.shape[0], y.shape[0])
            x_esm = x_esm[:min_len, :]
            x_dssp = x_dssp[:min_len, :]
            edge_final = edge[:min_len, :min_len]
            y_final = y[:min_len]
            x_final = torch.cat([x_esm, x_dssp], dim=1)
            return x_final, edge_final, y_final, s.sequence[:min_len], s.name
        except Exception as e:
            return None

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
    y_padded = [F.pad(y, (0, 0, 0, max_len - y.shape[0]), value=-1) for y in y_list]
    edge_padded = []
    for edge in edge_list:
        if edge.is_sparse: edge = edge.to_dense()
        pad_size = max_len - edge.shape[0]
        edge = F.pad(edge, (0, pad_size, 0, pad_size))
        edge_padded.append(edge)
    return torch.stack(x_padded), torch.stack(edge_padded), torch.stack(y_padded), seq_list, name_list

# ------------------- 모델 아키텍처 정의 (이전과 동일) -------------------
class GraphBepiLightning(pl.LightningModule):
    def __init__(self, hidden_dim=512, dropout=0.3, lr=1e-4, threshold=0.5, pos_weight=None, use_focal_loss=False, focal_alpha=0.25, focal_gamma=2.0, use_residual=True, use_attention=True, num_heads=8):
        super().__init__()
        self.save_hyperparameters()
        self.use_residual = use_residual
        self.use_attention = use_attention
        self.input_projection = nn.Linear(333, hidden_dim)
        self.W_v = nn.Sequential(nn.Linear(333, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout * 0.3))
        if use_attention:
            self.self_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout * 0.3, batch_first=True)
            self.attention_norm = nn.LayerNorm(hidden_dim)
        self.gat = EGAT(hidden_dim, hidden_dim, edge_dim=1)
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim // 2, batch_first=True, bidirectional=True, dropout=dropout * 0.5, num_layers=2)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim // 2, batch_first=True, bidirectional=True, dropout=dropout * 0.5, num_layers=2)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout * 0.7),
            nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.LayerNorm(hidden_dim // 4), nn.ReLU(), nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 4, hidden_dim // 8), nn.ReLU(), nn.Dropout(dropout * 0.3), nn.Linear(hidden_dim // 8, 1)
        )
        self.use_focal_loss = use_focal_loss
        if use_focal_loss:
            self.focal_alpha = focal_alpha
            self.focal_gamma = focal_gamma
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        self.threshold = threshold
        self.lr = lr

    def focal_loss(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.focal_alpha * (1 - pt)**self.focal_gamma * bce_loss
        return focal_loss.mean()

    def forward(self, x, edge_attr):
        batch_size, seq_len, _ = x.shape
        if self.use_residual:
            x_projected = self.input_projection(x)
            h = self.W_v[0](x)
            h = x_projected + h
            h = self.W_v[1:](h)
        else:
            h = self.W_v(x)
        if self.use_attention:
            h_att, _ = self.self_attention(h, h, h)
            h = self.attention_norm(h + h_att)
        node_features = h.view(-1, h.size(-1))
        edge_indices = []
        for i in range(batch_size):
            edge_index = edge_attr[i].nonzero().t()
            offset = i * seq_len
            edge_indices.append(edge_index + offset)
        batched_edge_index = torch.cat(edge_indices, dim=1) if edge_indices else torch.empty((2, 0), dtype=torch.long, device=x.device)
        num_edges = batched_edge_index.shape[1]
        batched_edge_attr = torch.ones(num_edges, 1, device=x.device).float()
        h_gat, _ = self.gat((node_features, batched_edge_index), batched_edge_attr)
        h_gat = h_gat.view(batch_size, seq_len, -1)
        if self.use_residual:
            h = h + h_gat
        else:
            h = h_gat
        h_lstm1, _ = self.lstm1(h)
        if self.use_residual and h_lstm1.shape == h.shape:
            h = h + h_lstm1
        else:
            h = h_lstm1
        h_lstm2, _ = self.lstm2(h)
        if self.use_residual and h_lstm2.shape == h.shape:
            h = h + h_lstm2
        else:
            h = h_lstm2
        out = self.mlp(h)
        return out

    def predict_step(self, batch, batch_idx):
        x, edge_attr, y, sequences, names = batch
        y_hat = self(x, edge_attr).squeeze(-1)
        # 🌟🌟🌟 예측 점수를 시그모이드 함수로 변환합니다.
        preds_prob = torch.sigmoid(y_hat)
        return {'preds': preds_prob, 'sequences': sequences, 'names': names}
    
    def configure_optimizers(self):
        return []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./data/DEDUP90_FINETUNE')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/best_model-epoch=04-val_f1=0.6164.ckpt', help='Path to the model checkpoint file')
    parser.add_argument('--output_path', type=str, default='./predictions', help='Output path for prediction CSVs')
    parser.add_argument('--threshold', type=float, default=0.40, help='Prediction threshold for binary classification')
    
    args = parser.parse_args()
    device = 'cuda' if args.gpu != -1 and torch.cuda.is_available() else 'cpu'
    
    print("--- Loading Model from Checkpoint ---")
    model = GraphBepiLightning.load_from_checkpoint(
        args.checkpoint_path,
        map_location=device,
        hidden_dim=512,
        dropout=0.3,
        lr=1e-4,
        use_focal_loss=True,
        use_residual=True,
        use_attention=True,
        num_heads=8
    )
    model.eval()
    
    print("\n--- Loading Test Data ---")
    with open(os.path.join(args.root, 'test.pkl'), 'rb') as f:
        test_samples = pk.load(f)
    test_loader = DataLoader(GraphBepiDataset(test_samples, args.root),
                             batch_size=args.batch_size, shuffle=False, collate_fn=collate_wrapper, num_workers=0)
    
    print("\n--- Starting Prediction ---")
    trainer = pl.Trainer(accelerator=device, devices=[args.gpu])
    
    all_outputs = trainer.predict(model, test_loader)
    
    print("\n--- Saving Prediction Results ---")
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    for output in tqdm(all_outputs, desc="Saving CSVs"):
        for i in range(len(output['names'])):
            name, seq, pred = output['names'][i], output['sequences'][i], output['preds'][i]
            pred = pred[:len(seq)].squeeze()
            is_epitope = (pred > args.threshold).bool()
            df = pd.DataFrame({'resn': list(seq), 'score': pred.cpu().numpy(), 'is_epitope': is_epitope.cpu().numpy()})
            df.to_csv(os.path.join(args.output_path, f'{name}.csv'), index=False)

    print(f"Prediction results saved to {args.output_path}")

if __name__ == "__main__":
    main()