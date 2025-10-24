# nextflow_compatible_train.py - Nextflow 파이프라인과 완전 호환되는 파인튜닝 코드

import os
import torch
import pickle as pk
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset, Subset
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm import tqdm
from torchmetrics.classification import BinaryF1Score, BinaryPrecision, BinaryRecall, BinaryAUROC
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
# =============== Nextflow GraphBepi 컴포넌트들 (정확한 구현) ===============

class AE(nn.Module):
    """Auto-Encoder layer from GraphBepi"""
    def __init__(self, dim_in, dim_out, hidden, dropout=0., bias=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, hidden, bias=bias),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim_out, bias=bias),
            nn.LayerNorm(dim_out),
        )
    
    def forward(self, x):
        return self.net(x)

class EGraphAttentionLayer(nn.Module):
    """Edge-aware Graph Attention Layer from GraphBepi EGAT.py"""
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super().__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, edge_attr):
        # 차원 체크 및 수정
        if h.dim() != 2:
            raise ValueError(f"Expected h to be 2D tensor, got {h.dim()}D with shape {h.shape}")
        if edge_attr.dim() not in [2, 3]:
            raise ValueError(f"Expected edge_attr to be 2D or 3D tensor, got {edge_attr.dim()}D")
        
        # edge_attr가 3차원인 경우 첫 번째 채널만 사용
        if edge_attr.dim() == 3:
            edge_attr = edge_attr[0]  # (seq_len, seq_len)
        
        Wh = torch.mm(h, self.W)  # (L, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)
        e = e * edge_attr
        
        zero_vec = -9e15 * torch.ones_like(e)
        e = torch.where(edge_attr > 0, e, zero_vec)
        e = F.softmax(e, dim=1)
        e = F.dropout(e, self.dropout, training=self.training)
        
        # 어텐션 가중 평균 계산
        h_prime = torch.matmul(e, Wh)  # (L, out_features)
        
        # concat 처리 수정
        if self.concat:
            return F.elu(h_prime), e
        else:
            return h_prime, e

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

class EGAT(nn.Module):
    """Edge-aware Graph Attention Network from GraphBepi"""
    def __init__(self, nfeat, nhid, efeat, dropout=0.2, alpha=0.2):
        super().__init__()
        self.dropout = dropout
        self.nfeat = nfeat
        self.nhid = nhid
        self.efeat = efeat
        
        # 수정: 올바른 차원으로 어텐션 레이어 구성
        self.in_att = EGraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True)
        self.out_att = EGraphAttentionLayer(nhid, nfeat, dropout=dropout, alpha=alpha, concat=False)
        
    def forward(self, x, edge_attr):
        x_cut = x  # (L, nfeat)
        x = F.dropout(x, self.dropout, training=self.training)
        
        # edge_attr가 3차원인 경우 처리
        if edge_attr.dim() == 3:
            # (efeat, L, L) -> (L, L) - 평균 또는 첫 번째 채널 사용
            edge_2d = edge_attr.mean(dim=0)  # 또는 edge_attr[0]
        else:
            edge_2d = edge_attr
            
        x, edge_2d = self.in_att(x, edge_2d)  # (L, nhid)
        x, edge_2d = self.out_att(x, edge_2d)  # (L, nfeat)
        
        return x + x_cut, edge_2d

# =============== Nextflow 호환 GraphBepi 모델 ===============

class NextflowGraphBepi(pl.LightningModule):
    """Nextflow 파이프라인과 완전 호환되는 GraphBepi 모델"""
    
    def __init__(self, feat_dim=2560, hidden_dim=512, exfeat_dim=13, edge_dim=51, 
                 augment_eps=0.05, dropout=0.4, lr=2.5e-5, threshold=0.5, pos_weight=None,
                 use_focal_loss=True, focal_alpha=0.8, focal_gamma=5.0):
        super().__init__()
        self.save_hyperparameters()
        
        self.exfeat_dim = exfeat_dim
        self.augment_eps = augment_eps
        bias = False
        
        # Nextflow와 동일한 입력 특성 변환 레이어들
        self.W_v = nn.Linear(feat_dim, hidden_dim, bias=bias)  # ESM-2 임베딩 (2560 -> 256)
        self.W_u1 = AE(exfeat_dim, hidden_dim, hidden_dim, bias=bias)  # DSSP 특성 (13 -> 256)
        self.edge_linear = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim // 4, bias=True),
            nn.ELU(),
        )
        
        # 이중 경로: 그래프 신경망 + 순차 신경망 (Nextflow와 동일)
        self.gat = EGAT(2 * hidden_dim, hidden_dim, hidden_dim // 4, dropout)
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim // 2, 3, batch_first=True, bidirectional=True, dropout=dropout)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim // 2, 3, batch_first=True, bidirectional=True, dropout=dropout)
        
        # 최종 예측 레이어 (Nextflow GRAPHBEPI_PREDICT_WITH_WEIGHTS와 동일)
        self.mlp = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1, bias=True),
            nn.Sigmoid()
        )
        
        # Loss 함수 설정
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        if not use_focal_loss:
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight if pos_weight else 8.0]))
        
        self.threshold = threshold
        self.lr = lr
        
        # Xavier 초기화
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, V, edge):
        """
        Nextflow GraphBepi와 동일한 forward pass
        
        Args:
            V: 결합된 특성 (batch_size, seq_len, 2573) 또는 (seq_len, 2573)
            edge: 그래프 엣지 정보 (batch_size, seq_len, seq_len, edge_dim) 또는 (seq_len, seq_len, edge_dim)
        """
        # 차원 조정
        original_batch_input = V.dim() == 3
        if V.dim() == 2:  # (L, feat_dim)
            V = V.unsqueeze(0)  # (1, L, feat_dim)
        
        batch_size, seq_len = V.shape[:2]
        mask = V.sum(-1) != 0
        actual_length = mask[0].sum().item() if batch_size == 1 else seq_len
        
        # 특성 분리 및 투영 (Nextflow와 동일)
        feats = self.W_v(V[:, :, :-self.exfeat_dim])    # ESM-2 (2560 -> 256)
        exfeats = self.W_u1(V[:, :, -self.exfeat_dim:]) # DSSP (13 -> 256)
        
        # 그래프 경로 (EGAT) - 단일 서열 처리
        x1, x2 = feats[0, :actual_length], exfeats[0, :actual_length]
        x_gcn = torch.cat([x1, x2], -1)  # (L, 512)
        
        # 엣지 특성 처리 - 차원 안전 처리
        if edge is not None:
            if edge.dim() == 4:  # (batch, seq_len, seq_len, edge_dim)
                edge_features = edge[0, :actual_length, :actual_length, :]
            elif edge.dim() == 3:  # (seq_len, seq_len, edge_dim)
                edge_features = edge[:actual_length, :actual_length, :]
            else:  # (seq_len, seq_len)
                edge_features = edge[:actual_length, :actual_length]
                # edge_dim 차원 추가
                edge_features = edge_features.unsqueeze(-1).expand(-1, -1, 51)
        else:
            # 기본 엣지 특성 생성
            edge_features = torch.ones((actual_length, actual_length, 51), device=V.device)
        
        E = self.edge_linear(edge_features).permute(2, 0, 1)  # (hidden_dim//4, L, L)
        
        # EGAT 처리 - 차원 체크
        try:
            x_gcn, E = self.gat(x_gcn, E)
        except Exception as e:
            print(f"Debug info - x_gcn shape: {x_gcn.shape}, E shape: {E.shape}")
            print(f"actual_length: {actual_length}")
            raise e
        
        # 순차 경로 (BiLSTM) - 안전한 패킹
        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
        
        try:
            feats_packed = pack_padded_sequence(feats, [actual_length], True, False)
            exfeats_packed = pack_padded_sequence(exfeats, [actual_length], True, False)
            
            feats_out = pad_packed_sequence(self.lstm1(feats_packed)[0], True)[0]
            exfeats_out = pad_packed_sequence(self.lstm2(exfeats_packed)[0], True)[0]
            
            x_attn = torch.cat([feats_out[0, :actual_length], exfeats_out[0, :actual_length]], -1)
        except Exception as e:
            print(f"LSTM debug - feats shape: {feats.shape}, actual_length: {actual_length}")
            raise e
        
        # 특성 융합 및 최종 예측
        h = torch.cat([x_attn, x_gcn], -1)  # (L, 4*hidden_dim)
        
        result = self.mlp(h).squeeze(-1)  # (L,)
        
        return result

    def focal_loss(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.focal_alpha * (1-pt)**self.focal_gamma * bce_loss
        return focal_loss.mean()

    def _common_step(self, batch):
        x, edge_attr, coords, y, _, _ = batch
        if x is None: return None, None, None
        
        # 배치 처리
        batch_size = x.shape[0]
        y_hat_list = []
        y_list = []
        
        for i in range(batch_size):
            mask = y[i] != -1
            if mask.sum() == 0: continue
            
            valid_len = mask.sum().item()
            x_i = x[i:i+1, :valid_len, :]  # (1, valid_len, 2573)
            edge_i = edge_attr[i:i+1, :valid_len, :valid_len, :] if edge_attr.dim() == 4 else None
            y_i = y[i, :valid_len]
            
            y_hat_i = self(x_i, edge_i)
            
            y_hat_list.append(y_hat_i)
            y_list.append(y_i)
        
        if not y_hat_list:
            return None, None, None
        
        y_hat_all = torch.cat(y_hat_list)
        y_all = torch.cat(y_list)
        
        # Logits으로 변환 (Sigmoid 출력을 다시 logits으로)
        y_hat_logits = torch.log(y_hat_all / (1 - y_hat_all + 1e-8))
        
        loss = self.focal_loss(y_hat_logits, y_all) if self.use_focal_loss else self.loss_fn(y_hat_logits, y_all.float())
        return loss, y_hat_all, y_all

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._common_step(batch)
        if loss is not None:
            self.log('train_loss', loss, batch_size=batch[0].size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self._common_step(batch)
        if loss is not None:
            batch_size = batch[0].size(0)
            self.log('val_loss', loss, batch_size=batch_size)
            f1_metric = BinaryF1Score().to(self.device)
            self.log('val_f1', f1_metric(y_hat, y.int()), batch_size=batch_size, prog_bar=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, edge_attr, coords, y, sequences, names = batch
        if x is None: return None
        
        batch_predictions = []
        for i in range(x.shape[0]):
            mask = y[i] != -1 if y is not None else torch.ones(x.shape[1], dtype=torch.bool)
            valid_len = mask.sum().item()
            
            x_i = x[i:i+1, :valid_len, :]
            edge_i = edge_attr[i:i+1, :valid_len, :valid_len, :] if edge_attr.dim() == 4 else None
            
            y_hat_i = self(x_i, edge_i)
            
            batch_predictions.append({
                'preds': y_hat_i,
                'labels': y[i, :valid_len] if y is not None else torch.zeros(valid_len),
                'sequence': sequences[i] if sequences else "",
                'name': names[i] if names else f"protein_{i}"
            })
        
        return batch_predictions
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        return optimizer

# =============== Nextflow 호환 Dataset ===============

class NextflowCompatibleDataset(Dataset):
    def __init__(self, samples, root_path):
        self.samples = [s for s in samples if getattr(s, 'is_processed', False)]
        self.root_path = root_path
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        try:
            # ESM2 임베딩 로드 (2560차원)
            feat_path = os.path.join(self.root_path, 'feat', f'{s.name}_esm2.ts')
            x_esm = torch.load(feat_path, weights_only=True).cpu()
            
            # DSSP 특성 (13차원)
            x_dssp = s.dssp.cpu()
            
            # 좌표 정보
            coord = s.coord.cpu()
            
            # 그래프 정보
            graph_path = os.path.join(self.root_path, 'graph', f'{s.name}.graph')
            graph = torch.load(graph_path, weights_only=True)
            edge = graph['edge']
            if edge.is_sparse:
                edge = edge.to_dense()
            
            # 라벨
            y = s.label.cpu()
            
            # 길이 맞춤
            min_len = min(x_esm.shape[0], x_dssp.shape[0], coord.shape[0], edge.shape[0], y.shape[0])
            x_esm = x_esm[:min_len, :]  # (L, 2560)
            x_dssp = x_dssp[:min_len, :]  # (L, 13)
            coord = coord[:min_len, :]
            edge = edge[:min_len, :min_len]
            y = y[:min_len]
            
            # Nextflow 호환: ESM2 + DSSP 결합
            x_final = torch.cat([x_esm, x_dssp], dim=1)  # (L, 2573)
            
            # 엣지 특성을 4차원으로 확장 (필요시)
            if edge.dim() == 2:
                edge_4d = torch.zeros(min_len, min_len, 51)  # 기본 엣지 특성 차원
                edge_4d[:, :, 0] = edge  # 인접 행렬을 첫 번째 채널에 저장
                edge = edge_4d
            
            return x_final, edge, coord, y, s.sequence[:min_len], s.name
            
        except Exception as e:
            print(f"Warning: Skipping sample {getattr(s, 'name', 'unknown')} due to error: {e}")
            return None

def nextflow_collate_wrapper(batch):
    """Nextflow 호환 collate 함수"""
    batch = [item for item in batch if item is not None]
    if not batch: 
        return None, None, None, None, None, None
    
    x_list, edge_list, coords_list, y_list, seq_list, name_list = zip(*batch)
    
    max_len = max([x.shape[0] for x in x_list])
    
    # 패딩
    x_padded = [F.pad(x, (0, 0, 0, max_len - x.shape[0])) for x in x_list]
    coords_padded = [F.pad(c, (0, 0, 0, max_len - c.shape[0])) for c in coords_list]
    y_padded = [F.pad(y, (0, max_len - y.shape[0]), value=-1) for y in y_list]
    
    # 엣지 패딩 (4차원)
    edge_padded = []
    for edge in edge_list:
        if edge.dim() == 3:  # (L, L, edge_dim)
            pad_size = max_len - edge.shape[0]
            edge_pad = F.pad(edge, (0, 0, 0, pad_size, 0, pad_size))
        else:  # 2차원인 경우
            pad_size = max_len - edge.shape[0]
            edge_pad = F.pad(edge, (0, pad_size, 0, pad_size))
            # 3차원으로 확장
            edge_pad = edge_pad.unsqueeze(-1).expand(-1, -1, 51)
        edge_padded.append(edge_pad)
    
    return (torch.stack(x_padded), torch.stack(edge_padded), torch.stack(coords_padded), 
            torch.stack(y_padded), seq_list, name_list)

# =============== 메인 함수 (기존과 동일하지만 새 모델 사용) ===============

def calculate_pos_weight(train_samples):
    total_pos, total_neg = 0, 0
    for sample in train_samples:
        if hasattr(sample, 'label') and hasattr(sample, 'is_processed') and sample.is_processed:
            labels = sample.label.cpu().numpy()
            total_pos += np.sum(labels == 1)
            total_neg += np.sum(labels == 0)
    pos_weight = total_neg / total_pos if total_pos > 0 else 1.0
    return pos_weight
# find_best_threshold_and_save 함수 위에 추가
def plot_confusion_matrix(labels, preds, threshold, output_path):
    """혼동행렬 생성 및 시각화"""
    import matplotlib
    matplotlib.use('Agg')  # GUI 없는 환경 대응
    
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating confusion matrix with {len(labels)} samples, threshold={threshold:.3f}")
    
    preds_binary = (preds > threshold).int()
    cm = confusion_matrix(labels.cpu().numpy(), preds_binary.cpu().numpy())
    
    print(f"Confusion matrix shape: {cm.shape}")
    print(f"Confusion matrix values:\n{cm}")
    
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    group_counts = [f"{value}" for value in cm.flatten()]
    group_percentages = [f"{value:.2%}" for value in cm_percent.flatten()]
    box_labels = [f"{v1}\n({v2})" for v1, v2 in zip(group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(2, 2)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=box_labels, fmt='', cmap='Blues',
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix (Threshold = {threshold:.3f})')
    
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {output_path}")
    except Exception as e:
        print(f"Error saving confusion matrix: {e}")
        
    plt.close()

def find_best_threshold_and_save(trainer, model, test_loader, ckpt_path, args, fold_num):
    print(f"\n--- [Fold {fold_num}] Generating predictions for test set ---")
    all_outputs = trainer.predict(model=model, dataloaders=test_loader, ckpt_path=ckpt_path)
    
    all_preds_scores, all_labels = [], []
    for batch_outputs in all_outputs:
        if batch_outputs is None: continue
        for output in batch_outputs:
            if output is None: continue
            mask = output['labels'] != -1
            all_preds_scores.append(output['preds'][mask])
            all_labels.append(output['labels'][mask])

    if not all_preds_scores:
        print("No valid predictions were made.")
        return {}

    all_preds_scores = torch.cat(all_preds_scores)
    all_labels = torch.cat(all_labels).int()

    print(f"\n--- [Fold {fold_num}] Finding Best Threshold ---")
    results = []
    f1_metric = BinaryF1Score().to(model.device)
    precision_metric = BinaryPrecision().to(model.device)
    recall_metric = BinaryRecall().to(model.device)

    for threshold in np.arange(0.01, 1.0, 0.01):
        preds_binary = (all_preds_scores > threshold).int()
        f1 = f1_metric(preds_binary, all_labels).item()
        precision = precision_metric(preds_binary, all_labels).item()
        recall = recall_metric(preds_binary, all_labels).item()
        results.append({'threshold': threshold, 'f1_score': f1, 'precision': precision, 'recall': recall})

    results_df = pd.DataFrame(results).sort_values(by='f1_score', ascending=False)
    best_result = results_df.iloc[0]
    
    # 추가 메트릭 계산
    best_threshold = best_result['threshold']
    preds_binary_final = (all_preds_scores > best_threshold).int()
    auroc_metric = BinaryAUROC().to(model.device)
    auroc = auroc_metric(all_preds_scores, all_labels).item()
    
    cm = confusion_matrix(all_labels.cpu().numpy(), preds_binary_final.cpu().numpy())
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    mcc_numerator = (tp * tn - fp * fn)
    mcc_denominator = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = mcc_numerator / mcc_denominator if mcc_denominator != 0 else 0.0
    
    fold_results = {
        'f1_score': best_result['f1_score'],
        'precision': best_result['precision'],
        'recall': best_result['recall'],
        'accuracy': accuracy,
        'auroc': auroc,
        'specificity': specificity,
        'mcc': mcc,
        'threshold': best_threshold
    }
    
    print(f"\n--- [Fold {fold_num}] Performance Summary ---")
    for key, value in fold_results.items():
        print(f"{key.capitalize():<12}: {value:.4f}")
    
    # 혼동행렬 생성 디버깅
    print(f"\nChecking confusion matrix generation for fold {fold_num}...")

    # 조건문 제거하고 모든 fold에서 생성
    output_dir = os.path.join(args.output, f'fold_{fold_num}')
    confusion_matrix_path = os.path.join(output_dir, f'confusion_matrix_fold_{fold_num}.png')

    try:
        plot_confusion_matrix(all_labels, all_preds_scores, best_threshold, confusion_matrix_path)
        
        # 파일 생성 확인
        if os.path.exists(confusion_matrix_path):
            file_size = os.path.getsize(confusion_matrix_path)
            print(f"✓ Confusion matrix created for fold {fold_num}! File size: {file_size} bytes")
        else:
            print(f"✗ Confusion matrix file not found for fold {fold_num}")
            
    except Exception as e:
        print(f"✗ Error creating confusion matrix for fold {fold_num}: {e}")

    print("Top 10 thresholds:")
    print(results_df[['threshold', 'f1_score', 'precision', 'recall']].head(10))
# =============== 앙상블 예측 함수 ===============

class EnsemblePredictor:
    """5-fold 모델 앙상블 예측기"""
    
    def __init__(self, fold_model_paths, device='cuda'):
        self.models = []
        self.device = device
        
        print("Loading ensemble models...")
        for i, path in enumerate(fold_model_paths):
            if os.path.exists(path):
                model = NextflowGraphBepi.load_from_checkpoint(path, map_location=device)
                model.eval()
                model.to(device)
                self.models.append(model)
                print(f"  Loaded fold {i} model: {path}")
            else:
                print(f"  Warning: Model not found: {path}")
        
        print(f"Ensemble ready with {len(self.models)} models")
    
    def predict_batch(self, batch):
        """배치에 대해 앙상블 예측 수행"""
        if not self.models:
            raise ValueError("No models loaded for ensemble")
        
        x, edge_attr, coords, y, sequences, names = batch
        batch_size = x.shape[0]
        
        # 각 모델의 예측 수집
        all_predictions = []
        
        for model in self.models:
            model_predictions = []
            
            with torch.no_grad():
                for i in range(batch_size):
                    mask = y[i] != -1 if y is not None else torch.ones(x.shape[1], dtype=torch.bool)
                    valid_len = mask.sum().item()
                    
                    x_i = x[i:i+1, :valid_len, :].to(self.device)
                    edge_i = edge_attr[i:i+1, :valid_len, :valid_len, :].to(self.device) if edge_attr.dim() == 4 else None
                    
                    pred_i = model(x_i, edge_i)  # 이미 Sigmoid 출력
                    model_predictions.append(pred_i.cpu())
            
            all_predictions.append(model_predictions)
        
        # 앙상블 예측 (평균)
        ensemble_predictions = []
        for i in range(batch_size):
            # 각 모델의 i번째 샘플 예측을 평균
            sample_preds = [all_predictions[model_idx][i] for model_idx in range(len(self.models))]
            ensemble_pred = torch.mean(torch.stack(sample_preds), dim=0)
            ensemble_predictions.append(ensemble_pred)
        
        # 결과 포맷팅
        results = []
        for i in range(batch_size):
            mask = y[i] != -1 if y is not None else torch.ones(x.shape[1], dtype=torch.bool)
            valid_len = mask.sum().item()
            
            results.append({
                'preds': ensemble_predictions[i],
                'labels': y[i, :valid_len] if y is not None else torch.zeros(valid_len),
                'sequence': sequences[i] if sequences else "",
                'name': names[i] if names else f"protein_{i}"
            })
        
        return results
    
    def predict_dataloader(self, dataloader):
        """전체 데이터로더에 대해 앙상블 예측"""
        all_results = []
        
        print("Performing ensemble predictions...")
        for batch in tqdm(dataloader, desc="Ensemble prediction"):
            if batch[0] is None:
                continue
            
            batch_results = self.predict_batch(batch)
            all_results.extend(batch_results)
        
        return all_results

def evaluate_ensemble(ensemble_results, threshold=None):
    """앙상블 결과 평가"""
    all_preds, all_labels = [], []
    
    for result in ensemble_results:
        if result is None:
            continue
        mask = result['labels'] != -1
        all_preds.append(result['preds'][mask])
        all_labels.append(result['labels'][mask])
    
    if not all_preds:
        return {}
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels).int()
    
    # 앙상블을 위한 최적 임계값 찾기
    if threshold is None:
        print("Finding optimal threshold for ensemble...")
        best_f1 = 0
        best_threshold = 0.5
        
        f1_metric = BinaryF1Score()
        for t in np.arange(0.01, 1.0, 0.01):
            preds_binary = (all_preds > t).int()
            f1 = f1_metric(preds_binary, all_labels).item()
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t
        
        threshold = best_threshold
        print(f"Optimal ensemble threshold: {threshold:.3f} (F1: {best_f1:.4f})")
    
    # 최적 임계값으로 메트릭 계산
    preds_binary = (all_preds > threshold).int()
    
    f1_metric = BinaryF1Score()
    precision_metric = BinaryPrecision()
    recall_metric = BinaryRecall()
    auroc_metric = BinaryAUROC()
    
    f1 = f1_metric(preds_binary, all_labels).item()
    precision = precision_metric(preds_binary, all_labels).item()
    recall = recall_metric(preds_binary, all_labels).item()
    auroc = auroc_metric(all_preds, all_labels).item()
    
    # Confusion matrix
    cm = confusion_matrix(all_labels.cpu().numpy(), preds_binary.cpu().numpy())
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # MCC
    mcc_numerator = (tp * tn - fp * fn)
    mcc_denominator = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = mcc_numerator / mcc_denominator if mcc_denominator != 0 else 0.0
    
    return {
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'auroc': auroc,
        'specificity': specificity,
        'mcc': mcc,
        'threshold': threshold,
        'num_samples': len(all_labels),
        'num_positive': all_labels.sum().item(),
        'num_negative': (all_labels == 0).sum().item()
    }
    #하이퍼 파라미터 값 조정
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./data/DEDUP90_FINETUNE')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('-o', '--output', type=str, default='./output_nextflow_compatible')
    parser.add_argument('--hidden_dim', type=int, default=512)  # Nextflow 기본값
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.3)  # Nextflow 기본값
    parser.add_argument('--focal_alpha', type=float, default=0.7)
    parser.add_argument('--focal_gamma', type=float, default=3.0)
    parser.add_argument('--k_folds', type=int, default=5)
    parser.add_argument('--ensemble_only', action='store_true', help='Skip training, only run ensemble evaluation')
    args = parser.parse_args()

    torch.set_float32_matmul_precision('medium')
    
    # 데이터 로드
    with open(os.path.join(args.root, 'train.pkl'), 'rb') as f: 
        train_samples = pk.load(f)
    with open(os.path.join(args.root, 'val.pkl'), 'rb') as f: 
        val_samples = pk.load(f)
    with open(os.path.join(args.root, 'test.pkl'), 'rb') as f: 
        test_samples = pk.load(f)
    
    all_train_val_samples = train_samples + val_samples
    test_dataset = NextflowCompatibleDataset(test_samples, args.root)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                           collate_fn=nextflow_collate_wrapper, num_workers=0)

    pos_weight = calculate_pos_weight(all_train_val_samples)
    print(f"Calculated pos_weight: {pos_weight:.3f}")
    
    # 앙상블만 실행하는 경우
    if args.ensemble_only:
        print("\n=== ENSEMBLE EVALUATION ONLY ===")
        
        # 기존 모델들 찾기
        fold_model_paths = []
        for fold in range(args.k_folds):
            model_path = f'checkpoints/fold_{fold}/best_model.ckpt'
            fold_model_paths.append(model_path)
        
        # 앙상블 예측기 생성
        device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
        ensemble = EnsemblePredictor(fold_model_paths, device)
        
        # 앙상블 예측 수행
        ensemble_results = ensemble.predict_dataloader(test_loader)
        
        # 앙상블 성능 평가
        ensemble_metrics = evaluate_ensemble(ensemble_results)
        
        print("\n=== ENSEMBLE RESULTS ===")
        for key, value in ensemble_metrics.items():
            print(f"{key.capitalize():<15}: {value:.4f}" if isinstance(value, float) else f"{key.capitalize():<15}: {value}")
        
        # 결과 저장
        os.makedirs(args.output, exist_ok=True)
        with open(os.path.join(args.output, 'ensemble_results.json'), 'w') as f:
            json.dump(ensemble_metrics, f, indent=2)
        
        print(f"\nEnsemble results saved to {args.output}/ensemble_results.json")
        return
    
    # 전체 K-Fold 훈련 + 앙상블 평가
    full_dataset = NextflowCompatibleDataset(all_train_val_samples, args.root)
    kfold = KFold(n_splits=args.k_folds, shuffle=True, random_state=42)
    all_fold_results = []
    fold_model_paths = []
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(full_dataset)):
        print(f"\n{'='*20} FOLD {fold} {'='*20}")

        train_sub_dataset = Subset(full_dataset, train_ids)
        val_sub_dataset = Subset(full_dataset, val_ids)

        train_loader = DataLoader(train_sub_dataset, batch_size=args.batch_size, shuffle=True, 
                                collate_fn=nextflow_collate_wrapper, num_workers=0)
        val_loader = DataLoader(val_sub_dataset, batch_size=args.batch_size, shuffle=False, 
                              collate_fn=nextflow_collate_wrapper, num_workers=0)
        
        # Nextflow 호환 모델 생성
        model = NextflowGraphBepi(
            feat_dim=2560,  # ESM2 차원
            hidden_dim=args.hidden_dim,
            exfeat_dim=13,  # DSSP 차원
            edge_dim=51,    # Nextflow 기본 엣지 차원
            dropout=args.dropout,
            lr=args.lr,
            pos_weight=pos_weight,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            threshold=0.1763  # GraphBepi 논문의 기본 임계값
        )
        
        checkpoint_callback = ModelCheckpoint(
            monitor='val_f1', mode='max', save_top_k=1,
            dirpath=f'checkpoints_nextflow/fold_{fold}', filename='best_model'
        )
        
        trainer = pl.Trainer(
            devices=[args.gpu] if torch.cuda.is_available() else 'auto', 
            accelerator='gpu' if torch.cuda.is_available() else 'auto',
            max_epochs=args.max_epochs,
            callbacks=[checkpoint_callback, EarlyStopping(monitor='val_f1', mode='max', patience=10, verbose=True)],
            precision="16-mixed",
            accumulate_grad_batches=2,
            logger=False,
            enable_progress_bar=True
        )
        
        print(f"Starting training for fold {fold}...")
        trainer.fit(model, train_loader, val_loader)
        
        best_model_path = trainer.checkpoint_callback.best_model_path
        fold_model_paths.append(best_model_path)
        
        fold_result = find_best_threshold_and_save(trainer, model, test_loader, best_model_path, args, fold)
        
        if fold_result:
            all_fold_results.append(fold_result)
    
    # K-Fold 개별 결과 요약
    if all_fold_results:
        print(f"\n{'='*20} K-FOLD INDIVIDUAL RESULTS {'='*20}")
        results_df = pd.DataFrame(all_fold_results)
        print(results_df.to_string())
        
        print("\n--- Individual Fold Statistics ---")
        print(f"Mean F1: {results_df['f1_score'].mean():.4f} ± {results_df['f1_score'].std():.4f}")
        print(f"Mean AUROC: {results_df['auroc'].mean():.4f} ± {results_df['auroc'].std():.4f}")
    
    # 앙상블 평가 수행
    print(f"\n{'='*20} ENSEMBLE EVALUATION {'='*20}")
    
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    ensemble = EnsemblePredictor(fold_model_paths, device)
    
    # 앙상블 예측
    ensemble_results = ensemble.predict_dataloader(test_loader)
    
    # 앙상블 성능 평가
    ensemble_metrics = evaluate_ensemble(ensemble_results)
    
    print("\n--- ENSEMBLE PERFORMANCE ---")
    for key, value in ensemble_metrics.items():
        if isinstance(value, float):
            print(f"{key.capitalize():<15}: {value:.4f}")
        else:
            print(f"{key.capitalize():<15}: {value}")
    
    # 개별 fold vs 앙상블 비교
    if all_fold_results:
        individual_mean_f1 = results_df['f1_score'].mean()
        ensemble_f1 = ensemble_metrics['f1_score']
        improvement = ensemble_f1 - individual_mean_f1
        
        print(f"\n--- ENSEMBLE vs INDIVIDUAL COMPARISON ---")
        print(f"Individual Average F1: {individual_mean_f1:.4f}")
        print(f"Ensemble F1:           {ensemble_f1:.4f}")
        print(f"Improvement:           {improvement:+.4f} ({improvement/individual_mean_f1*100:+.2f}%)")
    
    # 최종 결과 저장
        
        # 요약 통계 저장
        summary_stats = {
            'mean': results_df.mean().to_dict(),
            'std': results_df.std().to_dict(),
            'best_fold': results_df.loc[results_df['f1_score'].idxmax()].to_dict(),
            'model_config': {
                'feat_dim': 2560,
                'hidden_dim': args.hidden_dim,
                'exfeat_dim': 13,
                'edge_dim': 51,
                'dropout': args.dropout,
                'lr': args.lr,
                'focal_alpha': args.focal_alpha,
                'focal_gamma': args.focal_gamma,
                'batch_size': args.batch_size,
                'max_epochs': args.max_epochs
            }
        }
        
        with open(os.path.join(args.output, 'summary_stats.json'), 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        print(f"\nResults saved to {args.output}")
        print(f"Best F1 Score: {results_df['f1_score'].max():.4f}")
        print(f"Mean F1 Score: {results_df['f1_score'].mean():.4f} ± {results_df['f1_score'].std():.4f}")

if __name__ == "__main__":
    main()