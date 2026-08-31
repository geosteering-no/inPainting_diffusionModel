import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn

parser = argparse.ArgumentParser(description="RePaint inference for training diversity study")
parser.add_argument('--model',          type=str,   required=True, choices=['M1', 'M2', 'M3'],
                    help='Which model: M1, M2, M3')
parser.add_argument('--seed',           type=int,   required=True,
                    help='Training seed used (42, 123, 456)')
parser.add_argument('--gpu',            type=int,   default=0,
                    help='GPU index (default: 0)')
parser.add_argument('--n_realizations', type=int,   default=100,
                    help='Number of RePaint samples to generate (default: 100)')
parser.add_argument('--jump_length',    type=int,   default=5,
                    help='RePaint jump length (default: 5)')
parser.add_argument('--jump_n_sample',  type=int,   default=20,
                    help='RePaint jump n sample (default: 20)')
parser.add_argument('--mask_position',  type=int,   default=120,
                    help='Mask position in pixels (default: 120)')
parser.add_argument('--conditioning_indices', type=int, nargs='+', default=[100],
                    help='R2 patch indices to condition on (default: 100)')
args = parser.parse_args()


MODEL_NAME      = args.model
SEED            = args.seed
device          = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
n_realizations  = args.n_realizations
jump_length     = args.jump_length
jump_n_sample   = args.jump_n_sample
mask_position   = args.mask_position
conditioning_indices = args.conditioning_indices

print(f"Model:          {MODEL_NAME}")
print(f"Seed:           {SEED}")
print(f"Device:         {device}")
print(f"jump_length:    {jump_length}")
print(f"jump_n_sample:  {jump_n_sample}")
print(f"Conditioning:   {conditioning_indices}")


BASE_RUNS = "/Home/siv36/hesal5042/Research/NORCE/inPainting_diffusionModel/Geology/training_runs"
BASE_SAVE = "/Home/siv36/hesal5042/Research/NORCE/inPainting_diffusionModel/Geology/sliced_data/Additional/robustness"

RUN_NAME        = f"{MODEL_NAME}_seed{SEED}"
checkpoint_path = os.path.join(BASE_RUNS, RUN_NAME, "checkpoints", "best_checkpoint.pth")
save_root       = os.path.join(BASE_SAVE, RUN_NAME, f"jump{jump_n_sample}")

r2_folder = (
    "/Home/siv36/hesal5042/Research/NORCE/"
    "inPainting_diffusionModel/Geology/"
    "sliced_data/Additional/XZ_numpy_patches/"
    "NOFAULT_MODEL_R2"
)

os.makedirs(save_root, exist_ok=True)

print(f"Checkpoint:     {checkpoint_path}")
print(f"Save root:      {save_root}")



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
        self.down       = nn.MaxPool2d(2)
        self.up         = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

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
        e2    = self.enc2(self.down(e1))
        e3    = self.enc3(self.down(e2))
        e4    = self.enc4(self.down(e3))
        bott  = self.bottleneck(self.down(e4))
        bott  = bott + t_emb.expand_as(bott)
        d1    = self.dec1(torch.cat([self.up(bott), e4], dim=1))
        d2    = self.dec2(torch.cat([self.up(d1),   e3], dim=1))
        d3    = self.dec3(torch.cat([self.up(d2),   e2], dim=1))
        d4    = self.dec4(torch.cat([self.up(d3),   e1], dim=1))
        return self.final(d4)


model      = GeologyUNet().to(device)
checkpoint = torch.load(checkpoint_path, map_location=device)

# if not isinstance(checkpoint, dict):
#     raise ValueError("Expected structured checkpoint dict.")
# if "model_state_dict" not in checkpoint:
#     raise KeyError("Checkpoint missing model_state_dict.")
# if "global_min" not in checkpoint or "global_max" not in checkpoint:
#     raise KeyError("Checkpoint missing global_min/global_max.")

model.load_state_dict(checkpoint["model_state_dict"])
training_global_min = float(checkpoint["global_min"])
training_global_max = float(checkpoint["global_max"])
model.eval()

print(f"Loaded checkpoint: {checkpoint_path}")
print(f"Epoch:             {checkpoint.get('epoch', 'unknown')}")
print(f"Training geomodels: {checkpoint.get('training_geomodels', 'not recorded')}")
print(f"Global min/max:    {training_global_min:.6f} / {training_global_max:.6f}")

T          = 1000
beta_start = 1e-4
beta_end   = 0.02

betas                         = torch.linspace(beta_start, beta_end, T, device=device)
alphas                        = 1.0 - betas
alphas_cumprod                = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod           = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

class GeologyNPY(torch.utils.data.Dataset):
    def __init__(self, folder, global_min, global_max):
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"R2 folder not found: {folder}")
        self.paths = sorted([
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.endswith(".npy")
        ])
        if len(self.paths) == 0:
            raise RuntimeError(f"No .npy files found in {folder}")
        self.global_min = float(global_min)
        self.global_max = float(global_max)
        print(f"R2 test patches: {len(self.paths)}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        arr      = np.load(self.paths[idx]).astype(np.float32)
        arr_norm = (arr - self.global_min) / (self.global_max - self.global_min + 1e-8)
        arr_norm = arr_norm * 2.0 - 1.0
        return torch.from_numpy(arr_norm).unsqueeze(0)

dataset = GeologyNPY(r2_folder, training_global_min, training_global_max)

def get_repaint_schedule(T, jump_length=5, jump_n_sample=20):
    times = []
    t     = T - 1
    jumps = {}
    for j in range(0, T - jump_length, jump_length):
        jumps[j] = jump_n_sample - 1
    while t >= 1:
        t -= 1
        times.append(t)
        if jumps.get(t, 0) > 0:
            jumps[t] -= 1
            for _ in range(jump_length):
                t += 1
                times.append(t)
    times.append(-1)
    return times

def repaint(model, known_region, mask, T, jump_length, jump_n_sample):
    model.eval()
    current_device = known_region.device
    batch_size     = known_region.size(0)
    x_t            = torch.randn_like(known_region)
    times          = get_repaint_schedule(T=T, jump_length=jump_length,
                                          jump_n_sample=jump_n_sample)

    for i in tqdm(range(len(times) - 1), desc="RePaint sampling"):
        t      = times[i]
        t_next = times[i + 1]

        if t_next < t:
            t_tensor   = torch.full((batch_size,), t, device=current_device, dtype=torch.long)
            with torch.no_grad():
                pred_noise = model(x_t, t_tensor)

            alpha_t  = alphas[t]
            alpha_bar = alphas_cumprod[t]
            mean     = (1.0 / torch.sqrt(alpha_t)) * (
                x_t - ((1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar)) * pred_noise
            )

            if t_next >= 0:
                x_unknown = mean + torch.sqrt(betas[t]) * torch.randn_like(x_t)
            else:
                x_unknown = mean

            if t_next >= 0:
                x_known = (
                    sqrt_alphas_cumprod[t_next] * known_region
                    + sqrt_one_minus_alphas_cumprod[t_next] * torch.randn_like(known_region)
                )
            else:
                x_known = known_region

            x_t = mask * x_known + (1.0 - mask) * x_unknown

        else:
            x_t = (
                torch.sqrt(alphas[t_next]) * x_t
                + torch.sqrt(1.0 - alphas[t_next]) * torch.randn_like(x_t)
            )

    return x_t

def generate_ensemble(model, dataset, conditioning_idx, n_realizations,
                      mask_position, save_root):

    save_dir       = os.path.join(save_root, f"conditioning_{conditioning_idx}")
    individual_dir = os.path.join(save_dir, "individual_realizations")
    progress_path  = os.path.join(save_dir, "progress.txt")

    os.makedirs(individual_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"{MODEL_NAME} seed{SEED} | R2 conditioning index {conditioning_idx}")
    print(f"Saving to: {save_dir}")
    print("=" * 70)

    if conditioning_idx >= len(dataset):
        raise IndexError(f"Index {conditioning_idx} invalid for dataset size {len(dataset)}.")

    image        = dataset[conditioning_idx].unsqueeze(0).to(device)
    mask         = torch.ones_like(image)
    mask[:, :, :, mask_position:] = 0
    known_region = image * mask

    original_display = np.clip((image[0, 0].detach().cpu().numpy() + 1.0) / 2.0, 0.0, 1.0)
    truth_porosity   = original_display * (dataset.global_max - dataset.global_min) + dataset.global_min
    np.save(os.path.join(save_dir, "truth.npy"),                    truth_porosity)
    np.save(os.path.join(save_dir, "mask.npy"),                     mask.detach().cpu().numpy())
    np.save(os.path.join(save_dir, "known_region_normalized.npy"),  known_region.detach().cpu().numpy())

    height, width    = original_display.shape
    masked_display   = original_display.copy()
    masked_display[:, mask_position:] = 0


    for i in range(n_realizations):
        realization_path = os.path.join(individual_dir, f"realization_{i:04d}.npy")

        if os.path.exists(realization_path):
            print(f"Skipping realization {i+1}/{n_realizations} (already saved)")
            continue

        print(f"Generating realization {i+1}/{n_realizations}")

        output         = repaint(model=model, known_region=known_region, mask=mask,
                                 T=T, jump_length=jump_length, jump_n_sample=jump_n_sample)
        output_norm    = output.detach().cpu().squeeze().numpy()
        output_display = np.clip((output_norm + 1.0) / 2.0, 0.0, 1.0)
        output_porosity = output_display * (dataset.global_max - dataset.global_min) + dataset.global_min

        np.save(realization_path, output_porosity)

        with open(progress_path, "w") as f:
            f.write(str(i + 1))

        if i < 5:
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(original_display, cmap="gray", vmin=0, vmax=1)
            axes[0].set_title(f"Original R2 - {conditioning_idx}")
            axes[0].axis("off")

            axes[1].imshow(masked_display, cmap="gray", vmin=0, vmax=1)
            pink = np.zeros((height, width, 4))
            pink[:, mask_position:, :] = [1.0, 0.4, 0.7, 0.5]
            axes[1].imshow(pink)
            axes[1].set_title("Masked")
            axes[1].axis("off")

            axes[2].imshow(output_display, cmap="gray", vmin=0, vmax=1)
            axes[2].set_title(f"{MODEL_NAME} seed{SEED} RePaint")
            axes[2].axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"preview_{i:03d}.png"),
                        dpi=150, bbox_inches="tight")
            plt.close()

    realization_files = sorted([
        os.path.join(individual_dir, f)
        for f in os.listdir(individual_dir)
        if f.endswith(".npy")
    ])

    print(f"Saved realizations: {len(realization_files)}")

    if len(realization_files) != n_realizations:
        print(f"Incomplete: expected {n_realizations}, found {len(realization_files)}")
        return None

    ensemble = np.stack([np.load(p) for p in realization_files], axis=0)
    np.save(os.path.join(save_dir, "ensemble.npy"), ensemble)

    with open(progress_path, "w") as f:
        f.write(f"COMPLETE {n_realizations}")

    print(f"Final ensemble shape: {ensemble.shape}")
    print(f"Saved to: {os.path.join(save_dir, 'ensemble.npy')}")

    return ensemble


if __name__ == "__main__":
    for conditioning_index in conditioning_indices:
        generate_ensemble(
            model            = model,
            dataset          = dataset,
            conditioning_idx = conditioning_index,
            n_realizations   = n_realizations,
            mask_position    = mask_position,
            save_root        = save_root,
        )


# python geomodels_repaint.py --model M3 --seed 789 --gpu 0 & python geomodels_repaint.py --model M3 --seed 321 --gpu 1 & python geomodels_repaint.py --model M3 --seed 654 --gpu 2 & python geomodels_repaint.py --model M3 --seed 987 --gpu 3 & python geomodels_repaint.py --model M3 --seed 213 --gpu 4 