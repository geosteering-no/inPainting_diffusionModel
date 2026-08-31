import os
import math
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from tqdm import tqdm

device = "cuda:7" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

#Uncomment for M2 and M3
checkpoint_path = (
    "/Home/siv36/hesal5042/Research/NORCE/"
    "inPainting_diffusionModel/Geology/training_runs/"
    "M3_geomodels_R1_R3_R4_R5_seed42/checkpoints/best_checkpoint.pth"
)
output_folder = (
    "/Home/siv36/hesal5042/Research/NORCE/"
    "inPainting_diffusionModel/Geology/training_runs/"
    "M3_geomodels_R1_R3_R4_R5_seed42/unconditional"
)

#Below is for M1
# checkpoint_path = (
#     "/Home/siv36/hesal5042/Research/NORCE/inPainting_diffusionModel/Geology/training_runs/M3_geomodels_R1_R3_R4_R5_seed42/checkpoints/best_checkpoint.pth"
# )
# output_folder = (
#     "/Home/siv36/hesal5042/Research/NORCE/"
#     "inPainting_diffusionModel/Geology/training_runs/"
#     "M1_geomodeM3_geomodels_R1_R3_R4_R5_seed42ls_R1_seed42/unconditional"
# )

# training_data_folder = (
#     "/Home/siv36/hesal5042/Research/NORCE/inPainting_diffusionModel/Geology/sliced_data/Additional/XZ_numpy_patches/NOFAULT_MODEL_R1"
# )

os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, "unconditional_1000.npy")
preview_path = os.path.join(output_folder, "unconditional_preview.png")

NUM_SAMPLES = 1000
BATCH_SIZE = 16
IMAGE_SIZE = (64, 256)
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

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
        self.dec1 = self._block(1024, 256)
        self.dec2 = self._block(512, 128)
        self.dec3 = self._block(256, 64)
        self.dec4 = self._block(128, 64)
        self.final = nn.Conv2d(64, 1, kernel_size=1)
        self.downsample = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    @staticmethod
    def _block(in_channels, out_channels):
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
        bottleneck = bottleneck + t_embed.expand_as(bottleneck)
        d1 = self.dec1(torch.cat([self.upsample(bottleneck), e4], dim=1))
        d2 = self.dec2(torch.cat([self.upsample(d1), e3], dim=1))
        d3 = self.dec3(torch.cat([self.upsample(d2), e2], dim=1))
        d4 = self.dec4(torch.cat([self.upsample(d3), e1], dim=1))
        return self.final(d4)

#TODO: Uncomment this function if you want to load the model and normalization parameters from a checkpoint. for example for M2,M3 AND NOT M1
def load_model_and_normalization(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if "global_min" not in checkpoint or "global_max" not in checkpoint:
            raise KeyError("Checkpoint does not contain global_min/global_max.")
        global_min = float(checkpoint["global_min"])
        global_max = float(checkpoint["global_max"])
        print("Loaded structured checkpoint from epoch", checkpoint.get("epoch", "unknown"))
        print("Training geomodels:", checkpoint.get("training_geomodels", "not recorded"))
    else:
        raise ValueError(
            "Expected a structured checkpoint containing model_state_dict, global_min and global_max."
        )
    print("Checkpoint:", checkpoint_path)
    print("Saved training global min:", global_min)
    print("Saved training global max:", global_max)
    return model, global_min, global_max

#TODO: Uncomment this function if you want to load the model and normalization parameters from a checkpoint. for example for M1
# def load_model_and_normalization(
#     model,
#     checkpoint_path,
#     training_data_folder
# ):
#     checkpoint = torch.load(
#         checkpoint_path,
#         map_location=device
#     )

#     # Old M1 checkpoint = raw state_dict
#     model.load_state_dict(checkpoint)
#     model.eval()

#     print("Loaded M1 checkpoint:", checkpoint_path)

#     # Recompute the same normalization used during M1 training
#     files = [
#         os.path.join(training_data_folder, f)
#         for f in os.listdir(training_data_folder)
#         if f.endswith(".npy")
#     ]

#     files.sort()

#     all_min = []
#     all_max = []

#     for path in tqdm(
#         files,
#         desc="Computing M1 training min/max"
#     ):
#         arr = np.load(path).astype(np.float32)

#         all_min.append(arr.min())
#         all_max.append(arr.max())

#     global_min = float(np.min(all_min))
#     global_max = float(np.max(all_max))

#     print("M1 global min:", global_min)
#     print("M1 global max:", global_max)

#     return model, global_min, global_max

@torch.no_grad()
def generate_unconditional_samples(
    model,
    num_samples,
    img_size,
    batch_size,
    global_min,
    global_max,
):
    model.eval()
    all_samples = []
    num_batches = math.ceil(num_samples / batch_size)

    for batch_index in range(num_batches):
        current_batch = min(batch_size, num_samples - batch_index * batch_size)
        print(f"Generating batch {batch_index + 1}/{num_batches}, size={current_batch}")
        x = torch.randn(current_batch, 1, img_size[0], img_size[1], device=device)

        for timestep in tqdm(reversed(range(T)), total=T, desc="Reverse diffusion", leave=False):
            t_batch = torch.full((current_batch,), timestep, device=device, dtype=torch.long)
            predicted_noise = model(x, t_batch)
            alpha_t = alphas[timestep]
            alpha_cumprod_t = alphas_cumprod[timestep]
            beta_t = betas[timestep]
            noise = torch.randn_like(x) if timestep > 0 else torch.zeros_like(x)
            x = (
                (1.0 / torch.sqrt(alpha_t))
                * (
                    x
                    - ((1.0 - alpha_t) / torch.sqrt(1.0 - alpha_cumprod_t))
                    * predicted_noise
                )
                + torch.sqrt(beta_t) * noise
            )

        x_01 = (x + 1.0) / 2.0
        x_phys = x_01 * (global_max - global_min) + global_min
        all_samples.append(x_phys.cpu().numpy())

    ensemble = np.concatenate(all_samples, axis=0)
    print("Final unconditional ensemble shape:", ensemble.shape)
    print("Generated physical range:", float(ensemble.min()), float(ensemble.max()))
    return ensemble


if __name__ == "__main__":
    model = GeologyUNet().to(device)
    model, global_min, global_max = load_model_and_normalization(model, checkpoint_path)

    ensemble = generate_unconditional_samples(
        model=model,
        num_samples=NUM_SAMPLES,
        img_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        global_min=global_min,
        global_max=global_max,
    )

    np.save(output_path, ensemble)
    print("Saved unconditional ensemble to:", output_path)

    preview = torch.from_numpy(ensemble[:8])
    preview_norm = (preview - preview.min()) / (preview.max() - preview.min() + 1e-8)
    save_image(preview_norm, preview_path, nrow=4)
    print("Saved preview to:", preview_path)