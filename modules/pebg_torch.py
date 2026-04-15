import argparse
import ast
import math
import os
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse


class PNNFusion(nn.Module):
    """Product-based feature fusion used in original PEBG."""

    def __init__(self, embed_dim: int, hidden_dim: int, dropout_prob: float):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)

        # 3 embedding vectors + 3 pairwise inner products.
        self.fc1 = nn.Linear(self.embed_dim * 3 + 3, self.hidden_dim)
        self.drop = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(self.hidden_dim, 1)

    def forward(self, pro_embed: torch.Tensor, skill_embed: torch.Tensor, diff_feat_embed: torch.Tensor):
        # [B, 3, D]
        x = torch.stack([pro_embed, skill_embed, diff_feat_embed], dim=1)

        ip_01 = (x[:, 0, :] * x[:, 1, :]).sum(dim=-1, keepdim=True)
        ip_02 = (x[:, 0, :] * x[:, 2, :]).sum(dim=-1, keepdim=True)
        ip_12 = (x[:, 1, :] * x[:, 2, :]).sum(dim=-1, keepdim=True)

        linear_part = torch.cat([pro_embed, skill_embed, diff_feat_embed], dim=-1)
        product_part = torch.cat([ip_01, ip_02, ip_12], dim=-1)
        fusion_input = torch.cat([linear_part, product_part], dim=-1)

        hidden = F.relu(self.fc1(fusion_input))
        hidden = self.drop(hidden)
        pred = self.fc2(hidden).squeeze(-1)
        return hidden, pred


class PEBGPretrain(nn.Module):
    def __init__(self, pro_num: int, skill_num: int, diff_feat_dim: int, embed_dim: int, hidden_dim: int, dropout_prob: float):
        super().__init__()
        self.pro_embedding = nn.Embedding(pro_num, embed_dim)
        self.skill_embedding = nn.Embedding(skill_num, embed_dim)
        self.diff_embedding = nn.Embedding(diff_feat_dim, embed_dim)
        self.pnn = PNNFusion(embed_dim=embed_dim, hidden_dim=hidden_dim, dropout_prob=dropout_prob)

        nn.init.normal_(self.pro_embedding.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.skill_embedding.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.diff_embedding.weight, mean=0.0, std=0.1)

    def diff_feature_embed(self, diff_feat: torch.Tensor):
        # diff_feat: [B, diff_feat_dim] (float)
        return diff_feat @ self.diff_embedding.weight


def _to_device_tensor(arr: np.ndarray, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.as_tensor(arr, dtype=dtype, device=device)


def _build_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch PEBG pretraining")
    parser.add_argument("--data_dir", type=str, required=True, help="Absolute/relative data directory that contains PEBG assets")
    parser.add_argument("--con_sym", type=str, default="_", help="Composite skill separator symbol used when merging composed skills")
    parser.add_argument("--embed_dim", type=int, default=64, help="Node embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension in PNN")
    parser.add_argument("--keep_prob", type=float, default=0.5, help="Dropout keep probability (same meaning as TF code)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Training device")
    parser.add_argument("--save_every", type=str, default="50,100,200,500,1000,1500,2000", help="Comma-separated checkpoint epochs")
    parser.add_argument("--output_name", type=str, default="", help="Output embedding filename (.npz). Default: embedding_{epochs}_pt.npz")
    return parser.parse_args()


def _parse_save_every(spec: str) -> List[int]:
    if not spec:
        return []
    out = []
    for p in spec.split(","):
        p = p.strip()
        if not p:
            continue
        out.append(int(p))
    return out


def main():
    args = _build_cli()

    data_dir = os.path.normpath(args.data_dir)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data folder not found: {data_dir}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    con_sym = args.con_sym

    print(f"[PEBG-Torch] data_dir={data_dir}")
    print(f"[PEBG-Torch] device={device}")

    pro_skill_csr = sparse.load_npz(os.path.join(data_dir, "pro_skill_sparse.npz")).tocsr()
    skill_skill_csr = sparse.load_npz(os.path.join(data_dir, "skill_skill_sparse.npz")).tocsr()
    pro_pro_csr = sparse.load_npz(os.path.join(data_dir, "pro_pro_sparse.npz")).tocsr()

    pro_num, skill_num = pro_skill_csr.shape
    print(f"problem number {pro_num}, skill number {skill_num}")
    print(
        f"pro-skill edge {pro_skill_csr.nnz}, pro-pro edge {pro_pro_csr.nnz}, skill-skill edge {skill_skill_csr.nnz}"
    )

    pro_feat = np.load(os.path.join(data_dir, "pro_feat.npz"))["pro_feat"].astype(np.float32)
    diff_feat_dim = pro_feat.shape[1] - 1
    print(f"problem feature shape {pro_feat.shape}")

    diff_feat_all = pro_feat[:, :-1]
    aux_target_all = pro_feat[:, -1]

    skill_skill_targets = _to_device_tensor(skill_skill_csr.toarray().astype(np.float32), device=device)

    dropout_prob = 1.0 - float(args.keep_prob)
    model = PEBGPretrain(
        pro_num=pro_num,
        skill_num=skill_num,
        diff_feat_dim=diff_feat_dim,
        embed_dim=int(args.embed_dim),
        hidden_dim=int(args.hidden_dim),
        dropout_prob=dropout_prob,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    bce_logits = nn.BCEWithLogitsLoss()
    mse_loss = nn.MSELoss()

    save_every = set(_parse_save_every(args.save_every))
    ckpt_dir = os.path.join(data_dir, "pebg_model_torch")
    os.makedirs(ckpt_dir, exist_ok=True)

    train_steps = int(math.ceil(pro_num / float(args.batch_size)))

    print("finish building graph")

    for epoch in range(int(args.epochs)):
        model.train()
        total_loss = 0.0
        total_ps = 0.0
        total_pp = 0.0
        total_ss = 0.0
        total_mse = 0.0

        for step in range(train_steps):
            b = step * args.batch_size
            e = min((step + 1) * args.batch_size, pro_num)
            batch_idx_np = np.arange(b, e, dtype=np.int64)

            batch_pro = torch.as_tensor(batch_idx_np, dtype=torch.long, device=device)
            batch_diff_feat = _to_device_tensor(diff_feat_all[b:e], device=device)
            batch_aux_target = _to_device_tensor(aux_target_all[b:e], device=device)

            batch_pro_skill_targets = _to_device_tensor(pro_skill_csr[batch_idx_np].toarray().astype(np.float32), device=device)
            batch_pro_pro_targets = _to_device_tensor(pro_pro_csr[batch_idx_np].toarray().astype(np.float32), device=device)

            pro_embed = model.pro_embedding(batch_pro)
            skill_embed_matrix = model.skill_embedding.weight

            pro_skill_logits = pro_embed @ skill_embed_matrix.T
            pro_pro_logits = pro_embed @ model.pro_embedding.weight.T
            skill_skill_logits = skill_embed_matrix @ skill_embed_matrix.T

            denom = batch_pro_skill_targets.sum(dim=1, keepdim=True).clamp(min=1.0)
            skill_embed = (batch_pro_skill_targets @ skill_embed_matrix) / denom
            diff_feat_embed = model.diff_feature_embed(batch_diff_feat)
            _, aux_pred = model.pnn(pro_embed, skill_embed, diff_feat_embed)

            loss_ps = bce_logits(pro_skill_logits, batch_pro_skill_targets)
            loss_pp = bce_logits(pro_pro_logits, batch_pro_pro_targets)
            loss_ss = bce_logits(skill_skill_logits, skill_skill_targets)
            loss_aux = mse_loss(aux_pred, batch_aux_target)
            loss = loss_aux + loss_ps + loss_pp + loss_ss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_ps += float(loss_ps.item())
            total_pp += float(loss_pp.item())
            total_ss += float(loss_ss.item())
            total_mse += float(loss_aux.item())

        total_loss /= train_steps
        total_ps /= train_steps
        total_pp /= train_steps
        total_ss /= train_steps
        total_mse /= train_steps
        print(
            f"epoch {epoch + 1}, loss {total_loss:.4f} "
            f"(mse={total_mse:.4f}, ps={total_ps:.4f}, pp={total_pp:.4f}, ss={total_ss:.4f})"
        )

        if (epoch + 1) in save_every:
            ckpt_path = os.path.join(ckpt_dir, f"pebg_{epoch + 1}.pt")
            torch.save({"epoch": epoch + 1, "model_state_dict": model.state_dict()}, ckpt_path)

    print("finish training")

    model.eval()
    with torch.no_grad():
        pro_repre = model.pro_embedding.weight.detach().cpu().numpy()
        skill_repre = model.skill_embedding.weight.detach().cpu().numpy()

        pro_final_repre = np.zeros((pro_num, args.hidden_dim), dtype=np.float32)
        for step in range(train_steps):
            b = step * args.batch_size
            e = min((step + 1) * args.batch_size, pro_num)
            batch_idx_np = np.arange(b, e, dtype=np.int64)
            batch_pro = torch.as_tensor(batch_idx_np, dtype=torch.long, device=device)

            batch_diff_feat = _to_device_tensor(diff_feat_all[b:e], device=device)
            batch_pro_skill_targets = _to_device_tensor(pro_skill_csr[batch_idx_np].toarray().astype(np.float32), device=device)

            pro_embed = model.pro_embedding(batch_pro)
            skill_embed_matrix = model.skill_embedding.weight
            denom = batch_pro_skill_targets.sum(dim=1, keepdim=True).clamp(min=1.0)
            skill_embed = (batch_pro_skill_targets @ skill_embed_matrix) / denom
            diff_feat_embed = model.diff_feature_embed(batch_diff_feat)

            hidden, _ = model.pnn(pro_embed, skill_embed, diff_feat_embed)
            pro_final_repre[b:e] = hidden.detach().cpu().numpy()

    with open(os.path.join(data_dir, "skill_id_dict.txt"), "r", encoding="utf-8") as f:
        skill_id_dict = ast.literal_eval(f.read())

    join_skill_num = len(skill_id_dict)
    print(f"original skill number {skill_num}, joint skill number {join_skill_num}")

    skill_repre_new = np.zeros((join_skill_num, skill_repre.shape[1]), dtype=np.float32)
    skill_repre_new[:skill_num, :] = skill_repre
    for s in skill_id_dict.keys():
        s_str = str(s)
        if con_sym in s_str:
            tmp_skill_id = int(skill_id_dict[s])
            tmp_skills = [int(skill_id_dict[ele]) for ele in s_str.split(con_sym)]
            skill_repre_new[tmp_skill_id, :] = np.mean(skill_repre[tmp_skills], axis=0)

    output_name = args.output_name.strip() if args.output_name else f"embedding_{args.epochs}_pt.npz"
    output_path = os.path.join(data_dir, output_name)
    np.savez(
        output_path,
        pro_repre=pro_repre,
        skill_repre=skill_repre_new,
        pro_final_repre=pro_final_repre,
    )
    print(f"saved embeddings to: {output_path}")


if __name__ == "__main__":
    main()
