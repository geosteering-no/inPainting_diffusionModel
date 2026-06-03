import os
import random
import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn

# device
device = "cuda:1" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


#UNet model similar to training
class GeologyUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = self._block(1, 96)
        self.enc2 = self._block(96, 192)
        self.enc3 = self._block(192, 384)
        self.enc4 = self._block(384, 768)

        self.bottleneck = self._block(768, 768)

        self.time_embed = nn.Sequential(
            nn.Linear(1, 384),
            nn.SiLU(),
            nn.Linear(384, 768),
        )

        self.dec1 = self._block(768 + 768, 384)
        self.dec2 = self._block(384 + 384, 192)
        self.dec3 = self._block(192 + 192, 96)
        self.dec4 = self._block(96 + 96, 96)

        self.final = nn.Conv2d(96, 1, kernel_size=1)

        self.downsample = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, t):
        t_embed = self.time_embed(t.unsqueeze(-1).float())
        t_embed = t_embed.unsqueeze(-1).unsqueeze(-1)

        e1 = self.enc1(x)
        e2 = self.enc2(self.downsample(e1))
        e3 = self.enc3(self.downsample(e2))
        e4 = self.enc4(self.downsample(e3))

        bottleneck = self.bottleneck(self.downsample(e4))
        t_embed = t_embed.expand_as(bottleneck)
        bottleneck = bottleneck + t_embed

        d1 = self.dec1(torch.cat([self.upsample(bottleneck), e4], dim=1))
        d2 = self.dec2(torch.cat([self.upsample(d1), e3], dim=1))
        d3 = self.dec3(torch.cat([self.upsample(d2), e2], dim=1))
        d4 = self.dec4(torch.cat([self.upsample(d3), e1], dim=1))

        return self.final(d4)

model = GeologyUNet().to(device)

model.load_state_dict(
    torch.load(
        "/Home/siv36/hesal5042/Research/NORCE/hello/RePaint/guided_diffusion_mnist/guided_diffusion/Geology/Geology_Code/output/train_generated_patches/unet/model_100100.pth",
        map_location=device
    )
)

model.eval()


#DDPM hyperparameters
T = 1000
beta_start = 1e-4
beta_end = 0.02

betas = torch.linspace(beta_start, beta_end, T, device=device)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)



class GeologyNPY(torch.utils.data.Dataset):
    def __init__(self, folder):
        self.paths = sorted([
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.endswith(".npy")
        ])

        #compute global min/max once at init similar to the training
        all_min, all_max = [], []
        for p in tqdm(self.paths, desc="Computing global min/max"):
            arr = np.load(p).astype(np.float32)
            all_min.append(arr.min())
            all_max.append(arr.max())

        self.global_min = float(np.min(all_min))
        self.global_max = float(np.max(all_max))

        print(f"Global min: {self.global_min:.6f}")
        print(f"Global max: {self.global_max:.6f}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        arr = np.load(self.paths[idx]).astype(np.float32)

        #gglobal normalization to [-1, 1] similar to the training
        arr_norm = (arr - self.global_min) / (self.global_max - self.global_min + 1e-8)
        arr_norm = arr_norm * 2.0 - 1.0
        

        return torch.from_numpy(arr_norm).unsqueeze(0)


dataset = GeologyNPY(
    "/Home/siv36/hesal5042/Research/NORCE/inPainting_diffusionModel/Geology/sliced_data/XZ_numpy_patches"
)

print("Dataset size:", len(dataset))

def get_repaint_schedule(T, jump_length=5, jump_n_sample=20):
    """
    Creates RePaint-style forward/backward timestep schedule.

    Example idea:
    999 -> 998 -> 997 -> forward jump -> 998 -> 997 -> ...
    """

    times = []
    t = T - 1

    jumps = {}
    for j in range(0, T - jump_length, jump_length):
        jumps[j] = jump_n_sample - 1

    while t >= 1:
        t = t - 1
        times.append(t)

        if jumps.get(t, 0) > 0:
            jumps[t] -= 1

            for _ in range(jump_length):
                t = t + 1
                times.append(t)

    times.append(-1)
    return times

def repaint(model, x0, mask, T, jump_length, jump_n_sample):

    model.eval()
    device = x0.device
    B = x0.size(0)

    x_t = torch.randn_like(x0)

    times = get_repaint_schedule(
        T=T,
        jump_length=jump_length,
        jump_n_sample=jump_n_sample
    )

    for i in tqdm(range(len(times) - 1), desc="RePaint sampling"):

        t = times[i]
        t_next = times[i + 1]

        if t_next < t:
            # -------------------------
            # Reverse denoising step
            # x_t -> x_{t_next}
            # -------------------------

            t_tensor = torch.tensor([t] * B, device=device).long()

            with torch.no_grad():
                pred_noise = model(x_t, t_tensor)

            alpha_t = alphas[t]
            alpha_bar = alphas_cumprod[t]

            mean = (1 / torch.sqrt(alpha_t)) * (
                x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar)) * pred_noise
            )

            if t_next >= 0:
                noise = torch.randn_like(x_t)
                sigma = torch.sqrt(betas[t])
                x_unknown = mean + sigma * noise
            else:
                x_unknown = mean

            # Known region at timestep t_next
            if t_next >= 0:
                noise_k = torch.randn_like(x0)
                x_known = (
                    sqrt_alphas_cumprod[t_next] * x0 +
                    sqrt_one_minus_alphas_cumprod[t_next] * noise_k
                )
            else:
                x_known = x0

            x_t = mask * x_known + (1 - mask) * x_unknown

        else:
            # -------------------------
            # Forward re-noising jump
            # x_t -> x_{t_next}
            # -------------------------

            noise = torch.randn_like(x_t)

            x_t = (
                torch.sqrt(alphas[t_next]) * x_t +
                torch.sqrt(1 - alphas[t_next]) * noise
            )

    return x_t

def generate_ensemble_for_single_condition(
    model,
    dataset,
    n_realizations=1000,
    mask_position=120,
    save_dir="/Home/siv36/hesal5042/Research/NORCE/hello/RePaint/guided_diffusion_mnist/guided_diffusion/Geology/Geology_Code/output/train_generated_patches/unet/repaint_results",
    fixed_idx_path=None
):
    os.makedirs(save_dir, exist_ok=True)

    if fixed_idx_path is None:
        fixed_idx_path = os.path.join(save_dir, "fixed_conditioning_index.txt")

    # Fixed conditioning image
    # if os.path.exists(fixed_idx_path):
    #     with open(fixed_idx_path, "r") as f:
    #         idx = int(f.read().strip())
    #     print(f"Loaded fixed conditioning slice index: {idx}")
    # else:
    #     idx = random.randint(0, len(dataset) - 1)
    #     with open(fixed_idx_path, "w") as f:
    #         f.write(str(idx))
    #     print(f"Created fixed conditioning slice index: {idx}")
    idx = 100
    print(f"Using fixed conditioning slice index: {idx}")

    image = dataset[idx].unsqueeze(0).to(device)  # [-1,1], shape (1,1,64,256)

    mask = torch.ones_like(image)
    mask[:, :, :, mask_position:] = 0
    known_region = image * mask

    ensemble = []

    
    original_display = (image[0, 0].detach().cpu().numpy() + 1) / 2
    original_display = np.clip(original_display, 0, 1)

    masked_display = original_display.copy()
    masked_display[:, mask_position:] = 0

    H, W = original_display.shape

    for i in range(n_realizations):
        print(f"Generating realization {i+1}/{n_realizations}")

        output = repaint(
            model,
            known_region,
            mask,
            T,
            jump_length=5,
            jump_n_sample=5
        )
        #the process is to first convert [-1,1] to  [0,1] for display and plotting, then convert to physical porosity values for variogram analysis and ensemble saving. This way we keep the scale consistent for each use case.
        # output in [-1,1]
        output_norm = output.detach().cpu().squeeze().numpy()

        # convert to [0,1]
        output_display = (output_norm + 1) / 2
        output_display = np.clip(output_display, 0, 1)

        # convert to physical porosity for variograms
        output_porosity = (
            output_display * (dataset.global_max - dataset.global_min)
            + dataset.global_min
        )

        ensemble.append(output_porosity)#will be used for varigram analysis later, saved as ensemble.npy at the end

        #sampling generation
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(original_display, cmap="gray", vmin=0, vmax=1)
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(masked_display, cmap="gray", vmin=0, vmax=1)

        
        pink_mask = np.zeros((H, W, 4))
        pink_mask[:, mask_position:, :] = [1.0, 0.4, 0.7, 0.5]
        plt.imshow(pink_mask)

        plt.title("Masked")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(output_display, cmap="gray", vmin=0, vmax=1)
        plt.title("RePainted")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, f"realization_triplet_{i:03d}.png"),
            dpi=150,
            bbox_inches="tight"
        )
        plt.close()

    #saving ony one ensemble file for variograms
    ensemble = np.stack(ensemble, axis=0)  # (n_realizations, 64, 256)
    np.save(os.path.join(save_dir, "ensemble.npy"), ensemble)

    print("Finished generation.")
    print("Saved ensemble.npy:", ensemble.shape)
    print("Savedin:", save_dir)

    return ensemble



if __name__ == "__main__":

    ensemble = generate_ensemble_for_single_condition(
        model,
        dataset,
        n_realizations=100,
        mask_position=120,
        save_dir="/Home/siv36/hesal5042/Research/NORCE/hello/RePaint/guided_diffusion_mnist/guided_diffusion/Geology/Geology_Code/output/train_generated_patches/unet/wider"
    )