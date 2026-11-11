import os
import random
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ===================== 0. 基本配置 =====================

RANDOM_SEED = 42
DIFFUSION_EPOCHS = 200          # 训练 Diffusion 的轮数
CLS_EPOCHS = 200                # 训练分类器的轮数
BATCH_SIZE_DIFF = 128
BATCH_SIZE_CLS = 64
LEARNING_RATE_DIFF = 1e-3
LEARNING_RATE_CLS = 1e-3
WEIGHT_DECAY_CLS = 1e-4

TEST_SIZE = 0.2                # train/test 划分比例
SYNTHETIC_RATIO = 2.0          # 生成样本数量 = ratio * 真实训练样本数

DIFFUSION_STEPS = 100          # 扩散时间步数 T（tabular 数据用 100 足够）
BETA_START = 1e-4
BETA_END = 2e-2

FEATURE_CSV_PATH = "merged_features_6x18_with_gender_and_name.csv"
DIFF_MODEL_PATH = "diffusion_gender_model.pt"
CLS_MODEL_PATH = "mlp_gender_classifier.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(RANDOM_SEED)


# ===================== 1. 数据读取与标准化 =====================

def load_feature_data(
    csv_path: str,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    读取特征文件，并拆分为特征矩阵 X 和标签 y。
    假设列中包含:
      - gender: 目标标签 (0/1)
      - candidate: 被试编号（非特征）
      - source: 原始文件路径（非特征）
      其余为数值特征。
    """
    df = pd.read_csv(csv_path)

    if "gender" not in df.columns:
        raise ValueError("CSV 中未找到 'gender' 列")

    y = df["gender"].values.astype(np.int64)

    drop_cols = ["gender", "candidate", "source"]
    drop_cols = [c for c in drop_cols if c in df.columns]
    X_df = df.drop(columns=drop_cols)

    X = X_df.values.astype(np.float32)

    print(f"[INFO] Data loaded from {csv_path}")
    print(f"[INFO] Samples: {len(df)}, Feature dim: {X.shape[1]}")
    print(f"[INFO] Label distribution (0,1): {np.bincount(y)}")

    return X, y, df


def standardize_train_test(
    X_train: np.ndarray, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    对特征做标准化 (z-score)，返回标准化后的 train / test 以及 mean/std（便于复现）
    """
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-6

    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std

    return X_train_norm, X_test_norm, mean, std


# ===================== 2. Dataset =====================

class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert X.shape[0] == len(y)
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y


# ===================== 3. Diffusion 相关模块 =====================

class SinusoidalTimeEmbedding(nn.Module):
    """
    标准的正弦时间步编码 (positional encoding style)
    t: (B,) int [0, T-1]
    输出: (B, dim)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        # shape: (half_dim,)
        freqs = torch.exp(
            torch.arange(half_dim, device=t.device) * (-np.log(10000.0) / (half_dim - 1))
        )
        # (B, half_dim)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            # 如果是奇数维，补一个 0
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class DenoiseMLP(nn.Module):
    """
    用于预测噪声 eps_theta(x_t, t, y) 的网络。
    输入：
      x_t: (B, D)
      t: (B,) 时间步
      y: (B,) 类别标签（gender: 0/1）
    输出：
      预测噪声: (B, D)
    """
    def __init__(self, input_dim: int, time_dim: int = 64, hidden_dim: int = 256, num_classes: int = 2):
        super().__init__()

        self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )

        self.label_emb = nn.Embedding(num_classes, time_dim)

        self.net = nn.Sequential(
            nn.Linear(input_dim + time_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        x: (B, D)
        t: (B,)
        y: (B,)
        """
        t_emb = self.time_embedding(t)      # (B, time_dim)
        t_emb = self.time_mlp(t_emb)        # (B, time_dim)

        y_emb = self.label_emb(y)           # (B, time_dim)

        h = torch.cat([x, t_emb, y_emb], dim=-1)  # (B, D + 2*time_dim)
        eps = self.net(h)                   # (B, D)
        return eps


class GaussianDiffusion:
    """
    标准 DDPM 的高斯扩散过程，用于 tabular 数据。
    """

    def __init__(self, timesteps: int, beta_start: float, beta_end: float, device: torch.device):
        self.device = device
        self.timesteps = timesteps

        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        self.betas = betas.to(device)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]], dim=0
        )

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # q(x_{t-1} | x_t, x_0) 的方差
        self.posterior_variance = (
            self.betas
            * (1.0 - self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )

    def extract(self, a: torch.Tensor, t: torch.Tensor, x_shape):
        """
        从时间数组 a 中抽取对应时间步 t 的值，并 reshape 成 x_shape 便于广播。
        a: (T,)
        t: (B,)
        返回: (B, 1, 1, ... ) 与 x_shape 兼容
        """
        out = a[t]  # (B,)
        while len(out.shape) < len(x_shape):
            out = out.unsqueeze(-1)
        return out

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        根据 q(x_t | x_0) 公式： sqrt(alpha_bar_t) * x0 + sqrt(1-alpha_bar_t) * noise
        """
        sqrt_alpha_bar = self.extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise

    @torch.no_grad()
    def p_sample(self, model: nn.Module, x_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        根据 p_theta(x_{t-1} | x_t) 采样一个时间步。
        """
        betas_t = self.extract(self.betas, t, x_t.shape)
        sqrt_one_minus_alpha_bar_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        sqrt_recip_alpha_t = self.extract(self.sqrt_recip_alphas, t, x_t.shape)

        # 预测噪声 eps_theta
        eps_theta = model(x_t, t, y)

        # 计算均值项
        model_mean = sqrt_recip_alpha_t * (x_t - betas_t / sqrt_one_minus_alpha_bar_t * eps_theta)

        # t == 0 时不再加噪声
        noise = torch.randn_like(x_t)
        posterior_var_t = self.extract(self.posterior_variance, t, x_t.shape)

        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))  # (B,1,..)
        sample = model_mean + nonzero_mask * torch.sqrt(posterior_var_t) * noise
        return sample

    @torch.no_grad()
    def p_sample_loop(self, model: nn.Module, shape, y: torch.Tensor) -> torch.Tensor:
        """
        从 x_T ~ N(0, I) 开始，逐步反推到 x_0
        shape: (B, D)
        y: (B,)
        """
        x_t = torch.randn(shape, device=self.device)
        batch_size = shape[0]

        for step in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), step, device=self.device, dtype=torch.long)
            x_t = self.p_sample(model, x_t, t, y)
        return x_t


# ===================== 4. 分类器 MLP =====================

class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ===================== 5. 训练与评估函数 =====================

def train_diffusion(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    dataloader: DataLoader,
    epochs: int,
    lr: float,
):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for x0, y in dataloader:
            x0 = x0.to(device)
            y = y.to(device)

            batch_size = x0.size(0)
            t = torch.randint(
                low=0,
                high=diffusion.timesteps,
                size=(batch_size,),
                device=device,
                dtype=torch.long,
            )

            noise = torch.randn_like(x0)
            x_t = diffusion.q_sample(x0, t, noise)

            eps_theta = model(x_t, t, y)
            loss = F.mse_loss(eps_theta, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_size
            total_samples += batch_size

        avg_loss = total_loss / total_samples
        print(f"[Diffusion][Epoch {epoch:03d}] Loss: {avg_loss:.6f}")


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_test_acc = -1.0
    best_state_dict = None

    for epoch in range(1, epochs + 1):
        # ---- 训练 ----
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            correct = (preds == y).sum().item()

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            total_correct += correct
            total_samples += batch_size

        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples

        # ---- 测试 ----
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)

                logits = model(x)
                loss = criterion(logits, y)

                preds = logits.argmax(dim=1)
                correct = (preds == y).sum().item()

                batch_size = y.size(0)
                total_loss += loss.item() * batch_size
                total_correct += correct
                total_samples += batch_size

        test_loss = total_loss / total_samples
        test_acc = total_correct / total_samples

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}

        print(
            f"[Classifier][Epoch {epoch:03d}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | "
            f"Validation Loss: {test_loss:.4f}, Validation Acc: {test_acc*100:.2f}%"
        )

    print(f"[Classifier] Best Validation Accuracy: {best_test_acc*100:.2f}%")

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)


def evaluate_classifier(model: nn.Module, X_test: np.ndarray, y_test: np.ndarray):
    model.eval()
    X_test_tensor = torch.from_numpy(X_test.astype(np.float32)).to(device)
    with torch.no_grad():
        logits = model(X_test_tensor)
        preds = logits.argmax(dim=1).cpu().numpy()

    print("[RESULT] Classification report on test set:")
    print(classification_report(y_test, preds, digits=4))
    print("[RESULT] Confusion matrix (rows: true, cols: pred):")
    print(confusion_matrix(y_test, preds))


# ===================== 6. 主流程 =====================

def main():
    # 1. 读取数据
    X, y, df = load_feature_data(FEATURE_CSV_PATH)
    if X.shape[1] != 108 or not set(np.unique(y)) == {0,1}:
        raise ValueError("Diffusion requires 108 features and both 0/1 gender labels")
    num_features = X.shape[1]

    # 2. train/test 划分
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_SEED,
    )

    # 3. 标准化
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train, random_state=RANDOM_SEED)
    from sklearn.impute import SimpleImputer
    import joblib
    imputer = SimpleImputer(strategy='median', keep_empty_features=True)
    X_train = imputer.fit_transform(np.where(np.isfinite(X_train), X_train, np.nan))
    X_test = imputer.transform(np.where(np.isfinite(X_test), X_test, np.nan))
    X_val = imputer.transform(np.where(np.isfinite(X_val), X_val, np.nan))
    joblib.dump(imputer, Path(CLS_MODEL_PATH).with_suffix('.imputer.joblib'))
    X_train_norm, X_test_norm, mean, std = standardize_train_test(X_train, X_test)
    X_val_norm = (X_val - mean) / std
    np.savez(str(Path(CLS_MODEL_PATH).with_suffix('.preprocessing.npz')), mean=mean, std=std)
    print("[INFO] Data normalized (z-score).")

    # 4. 构造 Dataset / DataLoader（真实训练集）
    train_dataset_real = TabularDataset(X_train_norm, y_train)
    train_loader_diff = DataLoader(
        train_dataset_real,
        batch_size=BATCH_SIZE_DIFF,
        shuffle=True,
        drop_last=False,
    )

    # 5. 定义 Diffusion model
    diffusion = GaussianDiffusion(
        timesteps=DIFFUSION_STEPS,
        beta_start=BETA_START,
        beta_end=BETA_END,
        device=device,
    )
    denoise_model = DenoiseMLP(
        input_dim=num_features,
        time_dim=64,
        hidden_dim=256,
        num_classes=2,
    ).to(device)

    # 6. 训练 Diffusion Model
    print("[INFO] Start training diffusion model ...")
    train_diffusion(
        model=denoise_model,
        diffusion=diffusion,
        dataloader=train_loader_diff,
        epochs=DIFFUSION_EPOCHS,
        lr=LEARNING_RATE_DIFF,
    )

    torch.save(denoise_model.state_dict(), DIFF_MODEL_PATH)
    print(f"[INFO] Diffusion model saved to: {DIFF_MODEL_PATH}")

    # 7. 根据训练好的 Diffusion Model 生成合成样本
    print("[INFO] Generating synthetic samples from diffusion model ...")
    n_train = X_train_norm.shape[0]
    n_synth = int(SYNTHETIC_RATIO * n_train)

    # 按训练集中标签比例生成各类样本数量
    unique, counts = np.unique(y_train, return_counts=True)
    label_counts = dict(zip(unique.tolist(), counts.tolist()))

    synth_X_list = []
    synth_y_list = []

    for label in unique:
        n_label_real = label_counts[label]
        n_label_synth = int(n_synth * n_label_real / n_train)

        if n_label_synth == 0:
            continue

        y_cond = torch.full(
            (n_label_synth,),
            int(label),
            device=device,
            dtype=torch.long,
        )

        synth_samples = diffusion.p_sample_loop(
            model=denoise_model,
            shape=(n_label_synth, num_features),
            y=y_cond,
        )  # (n_label_synth, D)

        synth_X_list.append(synth_samples.cpu().numpy())
        synth_y_list.append(np.full((n_label_synth,), label, dtype=np.int64))

    if len(synth_X_list) > 0:
        synth_X = np.concatenate(synth_X_list, axis=0)
        synth_y = np.concatenate(synth_y_list, axis=0)
    else:
        synth_X = np.empty((0, num_features), dtype=np.float32)
        synth_y = np.empty((0,), dtype=np.int64)

    print(f"[INFO] Generated synthetic samples: {synth_X.shape[0]}")

    # 8. 组合 “真实训练 + 合成样本” 作为分类器训练集
    X_train_cls = np.concatenate([X_train_norm, synth_X], axis=0)
    y_train_cls = np.concatenate([y_train, synth_y], axis=0)

    print(f"[INFO] Classifier train set size (real + synthetic): {X_train_cls.shape[0]}")

    train_dataset_cls = TabularDataset(X_train_cls, y_train_cls)
    test_dataset_cls = TabularDataset(X_val_norm, y_val)

    train_loader_cls = DataLoader(
        train_dataset_cls,
        batch_size=BATCH_SIZE_CLS,
        shuffle=True,
        drop_last=False,
    )
    test_loader_cls = DataLoader(
        test_dataset_cls,
        batch_size=BATCH_SIZE_CLS,
        shuffle=False,
        drop_last=False,
    )

    # 9. 定义并训练分类器
    cls_model = MLPClassifier(
        input_dim=num_features,
        hidden_dim=256,
        num_classes=2,
    )
    print("[INFO] Start training classifier on real + synthetic data ...")
    train_classifier(
        model=cls_model,
        train_loader=train_loader_cls,
        test_loader=test_loader_cls,
        epochs=CLS_EPOCHS,
        lr=LEARNING_RATE_CLS,
        weight_decay=WEIGHT_DECAY_CLS,
    )

    torch.save(cls_model.state_dict(), CLS_MODEL_PATH)
    print(f"[INFO] Classifier model saved to: {CLS_MODEL_PATH}")

    # 10. 最终在测试集上详细评估
    evaluate_classifier(cls_model, X_test_norm, y_test)


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    p = argparse.ArgumentParser(description="Conditional diffusion augmentation followed by MLP")
    p.add_argument("--csv", required=True)
    p.add_argument("--output", default="results/diffusion")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--steps", type=int, default=100)
    a = p.parse_args()
    if a.epochs < 1 or a.steps < 2: p.error("epochs >= 1 and steps >= 2 required")
    FEATURE_CSV_PATH = a.csv
    DIFFUSION_EPOCHS = CLS_EPOCHS = a.epochs
    DIFFUSION_STEPS = a.steps
    Path(a.output).mkdir(parents=True, exist_ok=True)
    DIFF_MODEL_PATH = str(Path(a.output) / "diffusion.pt")
    CLS_MODEL_PATH = str(Path(a.output) / "classifier.pt")
    torch.set_num_threads(2)
    main()
