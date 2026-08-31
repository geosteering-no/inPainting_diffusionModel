import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import os
import argparse
from tqdm import tqdm
from torchvision.utils import save_image
import time
import wandb


parser = argparse.ArgumentParser(description="Training diversity experiment")
parser.add_argument('--model', type=str, required=True, choices=['M1', 'M2', 'M3'],
                    help='Which model to train: M1 (R1), M2 (R1+R3), M3 (R1+R3+R4+R5)')
parser.add_argument('--seed', type=int, required=True,
                    help='Random seed (e.g. 42, 123, 456)')
parser.add_argument('--gpu', type=int, default=0,
                    help='GPU index to use (default: 0)')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of training epochs (default: 100)')
args = parser.parse_args()


SEED       = args.seed
MODEL_NAME = args.model
NUM_EPOCHS = args.epochs
BATCH_SIZE = 16
device     = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

print(f"Model:  {MODEL_NAME}")
print(f"Seed:   {SEED}")
print(f"Device: {device}")


torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

BASE_DATA = "/Home/siv36/hesal5042/Research/NORCE/inPainting_diffusionModel/Geology/sliced_data/Additional/XZ_numpy_patches"
BASE_RUNS = "/Home/siv36/hesal5042/Research/NORCE/inPainting_diffusionModel/Geology/training_runs"

FOLDER_R1 = os.path.join(BASE_DATA, "NOFAULT_MODEL_R1")
FOLDER_R3 = os.path.join(BASE_DATA, "NOFAULT_MODEL_R3")
FOLDER_R4 = os.path.join(BASE_DATA, "NOFAULT_MODEL_R4")
FOLDER_R5 = os.path.join(BASE_DATA, "NOFAULT_MODEL_R5")

if MODEL_NAME == 'M1':
    FOLDERS          = [FOLDER_R1]
    GEOMODEL_NAMES   = ["NOFAULT_MODEL_R1"]
elif MODEL_NAME == 'M2':
    FOLDERS          = [FOLDER_R1, FOLDER_R3]
    GEOMODEL_NAMES   = ["NOFAULT_MODEL_R1", "NOFAULT_MODEL_R3"]
elif MODEL_NAME == 'M3':
    FOLDERS          = [FOLDER_R1, FOLDER_R3, FOLDER_R4, FOLDER_R5]
    GEOMODEL_NAMES   = ["NOFAULT_MODEL_R1", "NOFAULT_MODEL_R3",
                        "NOFAULT_MODEL_R4", "NOFAULT_MODEL_R5"]

RUN_NAME        = f"{MODEL_NAME}_seed{SEED}"
RUN_FOLDER      = os.path.join(BASE_RUNS, RUN_NAME)
SAMPLE_FOLDER   = os.path.join(RUN_FOLDER, "samples")
CKPT_FOLDER     = os.path.join(RUN_FOLDER, "checkpoints")

os.makedirs(SAMPLE_FOLDER, exist_ok=True)
os.makedirs(CKPT_FOLDER,   exist_ok=True)


T          = 1000
beta_start = 1e-4
beta_end   = 0.02

betas                        = torch.linspace(beta_start, beta_end, T, device=device)
alphas                       = 1.0 - betas
alphas_cumprod               = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod          = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

class GeologyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1       = self._block(1,   64)
        self.enc2       = self._block(64,  128)
        self.enc3       = self._block(128, 256)
        self.enc4       = self._block(256, 512)
        self.bottleneck = self._block(512, 512)
        self.time_embed = nn.Sequential(
            nn.Linear(1, 256),
            nn.SiLU(),
            nn.Linear(256, 512),
        )
        self.dec1       = self._block(512 + 512, 256)
        self.dec2       = self._block(256 + 256, 128)
        self.dec3       = self._block(128 + 128, 64)
        self.dec4       = self._block(64  + 64,  64)
        self.final      = nn.Conv2d(64, 1, kernel_size=1)
        self.downsample = nn.MaxPool2d(2)
        self.upsample   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c,  out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, t):
        t_emb = self.time_embed(t.unsqueeze(-1).float())
        t_emb = t_emb.unsqueeze(-1).unsqueeze(-1)
        e1    = self.enc1(x)
        e2    = self.enc2(self.downsample(e1))
        e3    = self.enc3(self.downsample(e2))
        e4    = self.enc4(self.downsample(e3))
        bott  = self.bottleneck(self.downsample(e4))
        bott  = bott + t_emb.expand_as(bott)
        d1    = self.dec1(torch.cat([self.upsample(bott), e4], dim=1))
        d2    = self.dec2(torch.cat([self.upsample(d1),   e3], dim=1))
        d3    = self.dec3(torch.cat([self.upsample(d2),   e2], dim=1))
        d4    = self.dec4(torch.cat([self.upsample(d3),   e1], dim=1))
        return self.final(d4)


def forward_diffusion(x0, t):
    noise                       = torch.randn_like(x0)
    t                           = t.to(x0.device)
    sqrt_ac_t                   = sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
    sqrt_one_minus_ac_t         = sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
    return sqrt_ac_t * x0 + sqrt_one_minus_ac_t * noise, noise


class GeologyXZDataset(Dataset):
    def __init__(self, folder_paths):
        self.files = []

        for folder_path in folder_paths:
            if not os.path.isdir(folder_path):
                raise FileNotFoundError(f"Folder does not exist: {folder_path}")
            source_name  = os.path.basename(os.path.normpath(folder_path))
            folder_files = sorted(
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.endswith(".npy")
            )
            print(f"{source_name}: {len(folder_files)} patches")
            self.files.extend(folder_files)

        if not self.files:
            raise RuntimeError("No .npy files found.")

        print(f"Total patches: {len(self.files)}")

        all_min, all_max = [], []
        for fp in tqdm(self.files, desc="Computing global min/max"):
            arr = np.load(fp).astype(np.float32)
            all_min.append(arr.min())
            all_max.append(arr.max())

        self.global_min = float(np.min(all_min))
        self.global_max = float(np.max(all_max))
        self.eps        = 1e-8

        print(f"Global min: {self.global_min:.6f}  max: {self.global_max:.6f}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = np.load(self.files[idx]).astype(np.float32)
        img = (img - self.global_min) / (self.global_max - self.global_min + self.eps)
        img = img * 2.0 - 1.0
        return torch.from_numpy(img[np.newaxis]), 0


def generate_samples(model, epoch, global_min, global_max, num_samples=1, img_size=(64, 256)):
    model.eval()
    with torch.no_grad():
        x = torch.randn(num_samples, 1, *img_size, device=device)
        for timestep in reversed(range(T)):
            t_batch         = torch.full((num_samples,), timestep, device=device, dtype=torch.long)
            predicted_noise = model(x, t_batch)
            alpha_t         = alphas[timestep]
            alpha_cumprod_t = alphas_cumprod[timestep]
            noise           = torch.randn_like(x) if timestep > 0 else torch.zeros_like(x)
            x = (
                (1.0 / torch.sqrt(alpha_t))
                * (x - ((1.0 - alpha_t) / torch.sqrt(1.0 - alpha_cumprod_t)) * predicted_noise)
                + torch.sqrt(betas[timestep]) * noise
            )

        x_01   = (x + 1.0) / 2.0
        x_phys = x_01 * (global_max - global_min) + global_min

        np.save(os.path.join(SAMPLE_FOLDER, f"patch_epoch_{epoch}.npy"), x_phys.cpu().numpy())
        save_image(x_01.clamp(0.0, 1.0), os.path.join(SAMPLE_FOLDER, f"epoch_{epoch}.png"))

    print(f"Samples saved for epoch {epoch}")


def train():
    start = time.time()

    dataset      = GeologyXZDataset(FOLDERS)
    train_size   = int(0.8 * len(dataset))
    val_size     = len(dataset) - train_size
    generator    = torch.Generator().manual_seed(SEED)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")


    wandb.init(
        project="training-diversity-geology-ddpm",
        name=RUN_NAME,
        config={
            "model":            MODEL_NAME,
            "seed":             SEED,
            "epochs":           NUM_EPOCHS,
            "batch_size":       BATCH_SIZE,
            "learning_rate":    1e-4,
            "T":                T,
            "beta_start":       beta_start,
            "beta_end":         beta_end,
            "loss":             "MSE noise prediction",
            "training_geomodels": GEOMODEL_NAMES,
            "total_patches":    len(dataset),
            "global_min":       dataset.global_min,
            "global_max":       dataset.global_max,
        },
    )

    model     = GeologyUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    best_val_loss = float("inf")

    for epoch in range(NUM_EPOCHS):


        model.train()
        running_loss = 0.0
        for imgs, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            imgs = imgs.to(device)
            t    = torch.randint(0, T, (imgs.size(0),), device=device, dtype=torch.long)
            noisy_imgs, noise = forward_diffusion(imgs, t)
            pred_noise        = model(noisy_imgs, t)
            loss              = criterion(pred_noise, noise)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}  Train loss: {avg_train_loss:.6f}")

        #validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, _ in val_loader:
                imgs = imgs.to(device)
                t    = torch.randint(0, T, (imgs.size(0),), device=device, dtype=torch.long)
                noisy_imgs, noise = forward_diffusion(imgs, t)
                pred_noise        = model(noisy_imgs, t)
                val_loss         += criterion(pred_noise, noise).item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}  Val loss:   {avg_val_loss:.6f}")

        wandb.log({
            "epoch":           epoch + 1,
            "train_loss":      avg_train_loss,
            "validation_loss": avg_val_loss,
        })

        ckpt = {
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch":                epoch + 1,
            "global_min":           dataset.global_min,
            "global_max":           dataset.global_max,
            "training_geomodels":   GEOMODEL_NAMES,
            "batch_size":           BATCH_SIZE,
            "seed":                 SEED,
            "T":                    T,
            "beta_start":           beta_start,
            "beta_end":             beta_end,
        }

        if (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(CKPT_FOLDER, f"epoch_{epoch+1}.pth")
            torch.save(ckpt, ckpt_path)
            wandb.save(ckpt_path)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path     = os.path.join(CKPT_FOLDER, "best_checkpoint.pth")
            torch.save(ckpt, best_path)
            print(f"  New best val loss: {best_val_loss:.6f}")

        generate_samples(
            model      = model,
            epoch      = epoch + 1,
            global_min = dataset.global_min,
            global_max = dataset.global_max,
        )

    final_path = os.path.join(CKPT_FOLDER, "final_checkpoint.pth")
    torch.save({**ckpt, "best_validation_loss": best_val_loss}, final_path)
    wandb.save(final_path)
    wandb.finish()

    elapsed = time.time() - start
    print(f"\nFinished in {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    train()


    # python geomodels_train.py --model M3 --seed 789 --gpu 0 & python geomodels_train.py --model M3 --seed 321 --gpu 1 & python geomodels_train.py --model M3 --seed 654 --gpu 2 & python geomodels_train.py --model M3 --seed 987 --gpu 3 & python geomodels_train.py --model M3 --seed 213 --gpu 4 