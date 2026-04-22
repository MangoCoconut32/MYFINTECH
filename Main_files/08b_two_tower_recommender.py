"""Two-tower neural recommender - learned alternative to the rule-based one.

Standard two-tower setup: one tower encodes the client features, another
holds a small embedding table for the products. Compatibility = dot
product between the two embeddings. Trained with BCE on the full label
matrix (essentially contrastive with in-batch negatives, since we only
have two products).
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_and_prepare_data

_script_dir = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.normpath(os.path.join(_script_dir, "..", "Dataset2_Needs.xls"))
OUT_DIR = os.path.normpath(os.path.join(_script_dir, "..", "Output", "08_recommender"))
os.makedirs(OUT_DIR, exist_ok=True)

EMBED_DIM = 16
EPOCHS = 80
BATCH = 256
LR = 1e-3
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


X_train, X_test, y_train_acc, y_test_acc = load_and_prepare_data(
    FILE_PATH, "AccumulationInvestment", use_engineered_features=True
)
full = pd.read_excel(FILE_PATH, sheet_name="Needs")
full.columns = full.columns.str.strip()
y_train_inc = full.loc[X_train.index, "IncomeInvestment"].values.astype(np.float32)
y_test_inc  = full.loc[X_test.index,  "IncomeInvestment"].values.astype(np.float32)

X_tr = torch.tensor(X_train.values.astype(np.float32))
X_te = torch.tensor(X_test.values.astype(np.float32))
Y_tr = torch.tensor(np.stack([y_train_acc.values, y_train_inc], axis=1).astype(np.float32))
Y_te = torch.tensor(np.stack([y_test_acc.values,  y_test_inc],  axis=1).astype(np.float32))

n_features = X_tr.shape[1]
n_products = 2
PRODUCT_NAMES = ["Accumulation", "Income"]


class ClientTower(nn.Module):
    def __init__(self, in_dim, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.GELU(),
            nn.Linear(32, embed_dim),
        )

    def forward(self, x):
        return self.net(x)


class TwoTower(nn.Module):
    def __init__(self, in_dim, n_products, embed_dim):
        super().__init__()
        self.client_tower = ClientTower(in_dim, embed_dim)
        self.product_embed = nn.Embedding(n_products, embed_dim)

    def forward(self, client_x):
        ce = self.client_tower(client_x)                            
        pe = self.product_embed.weight                              
        scores = ce @ pe.T                                          
        return scores


model = TwoTower(n_features, n_products, EMBED_DIM)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
loss_fn = nn.BCEWithLogitsLoss()


print(f"Training two-tower model ({n_features} feats, {n_products} products, D={EMBED_DIM})...")
n = X_tr.shape[0]
history = []
for epoch in range(EPOCHS):
    model.train()
    perm = torch.randperm(n)
    epoch_loss = 0.0
    for s in range(0, n, BATCH):
        idx = perm[s:s + BATCH]
        logits = model(X_tr[idx])
        loss = loss_fn(logits, Y_tr[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        epoch_loss += loss.item() * len(idx)
    epoch_loss /= n

    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            te_logits = model(X_te)
            te_probs = torch.sigmoid(te_logits).numpy()
        from sklearn.metrics import roc_auc_score
        auc_a = roc_auc_score(Y_te[:, 0].numpy(), te_probs[:, 0])
        auc_i = roc_auc_score(Y_te[:, 1].numpy(), te_probs[:, 1])
        history.append({"epoch": epoch + 1, "train_loss": epoch_loss,
                        "test_auc_acc": auc_a, "test_auc_inc": auc_i})
        print(f"  Epoch {epoch+1:3d} | loss={epoch_loss:.4f} | "
              f"AUC Acc={auc_a:.4f} | AUC Inc={auc_i:.4f}")


from sklearn.metrics import roc_auc_score
model.eval()
with torch.no_grad():
    te_logits = model(X_te).numpy()
    te_probs = 1 / (1 + np.exp(-te_logits))

scores_df = pd.DataFrame(te_probs, columns=PRODUCT_NAMES, index=X_test.index)
relevance = pd.DataFrame(Y_te.numpy(), columns=PRODUCT_NAMES, index=X_test.index)


def precision_at_k(scores, relevance, k):
    ranked = scores.rank(axis=1, ascending=False, method="first")
    top_k_mask = ranked <= k
    hits = (top_k_mask & (relevance == 1)).sum(axis=1)
    rel = relevance.sum(axis=1).clip(lower=1)
    return (hits / k).mean(), (hits / rel).mean()

rows = []
for k in (1, 2):
    p, r = precision_at_k(scores_df, relevance, k)
    rows.append({"K": k, "Precision@K": round(p, 4), "Recall@K": round(r, 4)})

auc_a = roc_auc_score(relevance["Accumulation"], scores_df["Accumulation"])
auc_i = roc_auc_score(relevance["Income"],       scores_df["Income"])

print("\n=== TWO-TOWER RESULTS ===")
print(f"AUC Accumulation: {auc_a:.4f}")
print(f"AUC Income:       {auc_i:.4f}")
for r in rows:
    print(f"K={r['K']}  P@K={r['Precision@K']}  R@K={r['Recall@K']}")

pd.DataFrame({"metric": ["AUC_Accumulation", "AUC_Income"],
              "value":  [round(auc_a, 4), round(auc_i, 4)]}
             ).to_csv(f"{OUT_DIR}/08b_two_tower_auc.csv", index=False)
pd.DataFrame(rows).to_csv(f"{OUT_DIR}/08b_two_tower_precision_at_k.csv", index=False)
pd.DataFrame(history).to_csv(f"{OUT_DIR}/08b_two_tower_history.csv", index=False)


prod_emb = model.product_embed.weight.detach().numpy()
pd.DataFrame(prod_emb, index=PRODUCT_NAMES).to_csv(f"{OUT_DIR}/08b_two_tower_product_embeddings.csv")

print(f"\nSaved: {OUT_DIR}/08b_two_tower_*.csv")
