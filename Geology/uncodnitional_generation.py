import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
from torchvision.utils import save_image

device = "cuda:7" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

checkpoint_path = "/Home/siv36/hesal5042/Research/NORCE/hello/RePaint/guided_diffusion_mnist/guided_diffusion/Geology/Geology_Code/output/train_generated_patches/model_wandb100100.pth"

data_folder = "/Home/siv36/hesal5042/Research/NORCE/inPainting_diffusionModel/Geology/sliced_data/XZ_numpy_patches"

output_folder = "/Home/siv36/hesal5042/Research/NORCE/inPainting_diffusionModel/Geology/Geology_Code/output/unconditional_final"
os.makedirs(output_folder, exist_ok=True)

output_path = os.path.join(output_folder, "unconditional_1000.npy")

T = 1000
beta_start = 1e-4
beta_end = 0.02

betas = torch.linspace(beta_start, beta_end, T, device=device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)


class GeologyUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = self._block(1, 64)
        self.enc2 = self._block(64, 128)
        self.enc3 = self._block(128, 256)
        self.enc4 = self._block(256, 512)

        self.bottleneck = self._block(512, 512)

        self.time_embed = nn.Sequential(
            nn.Linear(1, 256),
            nn.SiLU(),
            nn.Linear(256, 512),
        )

        self.dec1 = self._block(512 + 512, 256)
        self.dec2 = self._block(256 + 256, 128)
        self.dec3 = self._block(128 + 128, 64)
        self.dec4 = self._block(64 + 64, 64)

        self.final = nn.Conv2d(64, 1, kernel_size=1)

        self.downsample = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=True
        )

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


def compute_global_min_max(folder_path):
    files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(".npy")
    ]

    files.sort()
    print(f"Found {len(files)} training images")

    all_min = []
    all_max = []

    for f in tqdm(files, desc="Computing global min/max"):
        arr = np.load(f).astype(np.float32)
        all_min.append(arr.min())
        all_max.append(arr.max())

    global_min = float(np.min(all_min))
    global_max = float(np.max(all_max))

    print("Global min:", global_min)
    print("Global max:", global_max)

    return global_min, global_max


@torch.no_grad()
def generate_unconditional_samples(
    model,
    num_samples=1000,
    img_size=(64, 256),
    batch_size=10,
    global_min=0.0,
    global_max=1.0,
):
    model.eval()

    all_samples = []

    num_batches = int(np.ceil(num_samples / batch_size))

    for b in range(num_batches):
        current_batch = min(batch_size, num_samples - b * batch_size)

        print(f"Generating batch {b+1}/{num_batches}, size={current_batch}")

        x = torch.randn(
            current_batch,
            1,
            img_size[0],
            img_size[1],
            device=device
        )

        for t in tqdm(reversed(range(T)), total=T, desc="Reverse diffusion"):
            t_batch = torch.full(
                (current_batch,),
                t,
                device=device,
                dtype=torch.long
            )

            predicted_noise = model(x, t_batch)

            alpha_t = alphas[t]
            alpha_cumprod_t = alphas_cumprod[t]
            beta_t = betas[t]

            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = (
                (1 / torch.sqrt(alpha_t))
                * (
                    x
                    - ((1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t))
                    * predicted_noise
                )
                + torch.sqrt(beta_t) * noise
            )

        # scale from [-1, 1] to [0, 1]
        x = (x + 1.0) / 2.0

        # scale back to physical porosity values
        x_phys = x * (global_max - global_min) + global_min

        all_samples.append(x_phys.cpu().numpy())

    ensemble = np.concatenate(all_samples, axis=0)

    print("Final unconditional ensemble shape:", ensemble.shape)

    return ensemble

if __name__ == "__main__":

    global_min, global_max = compute_global_min_max(data_folder)

    model = GeologyUNet().to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)

    print("Loaded checkpoint:", checkpoint_path)

    ensemble = generate_unconditional_samples(
        model=model,
        num_samples=1000,
        img_size=(64, 256),
        batch_size=10,
        global_min=global_min,
        global_max=global_max,
    )

    np.save(output_path, ensemble)
    print("Saved unconditional ensemble to:", output_path)

    preview = torch.from_numpy(ensemble[:8])


    preview_norm = (preview - preview.min()) / (preview.max() - preview.min() + 1e-8)

    # save_image(
    #     preview_norm,
    #     os.path.join(output_folder, "unconditional_preview.png"),
    #     nrow=4
    # )

