import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
import random
import os
import torch
from tqdm import tqdm


device = "cuda:3" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

class HandwritingUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder (1 input channel - grayscale)
        self.enc1 = self._block(1, 64)
        self.enc2 = self._block(64, 128)
        self.enc3 = self._block(128, 256)
        self.enc4 = self._block(256, 512)

        # Bottleneck (with time conditioning)
        self.bottleneck = self._block(512, 512)
        self.time_embed = nn.Sequential(
            nn.Linear(1, 256),
            nn.SiLU(),
            nn.Linear(256, 512),
        )

        # Decoder
        self.dec1 = self._block(512 + 512, 256)  # skip from enc4
        self.dec2 = self._block(256 + 256, 128)  # skip from enc3
        self.dec3 = self._block(128 + 128, 64)   # skip from enc2
        self.dec4 = self._block(64 + 64, 64)     # skip from enc1

        # Final output layer
        self.final = nn.Conv2d(64, 1, kernel_size=1)

        # Sampling layers
        self.downsample = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

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
        # Time embedding
        t_embed = self.time_embed(t.unsqueeze(-1).float())
        t_embed = t_embed.unsqueeze(-1).unsqueeze(-1)

        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(self.downsample(e1))
        e3 = self.enc3(self.downsample(e2))
        e4 = self.enc4(self.downsample(e3))

        # Bottleneck
        bottleneck = self.bottleneck(self.downsample(e4))
        bottleneck = bottleneck + t_embed

        # Decoder path (with skip connections)
        d1 = self.dec1(torch.cat([self.upsample(bottleneck), e4], dim=1))
        d2 = self.dec2(torch.cat([self.upsample(d1), e3], dim=1))
        d3 = self.dec3(torch.cat([self.upsample(d2), e2], dim=1))
        d4 = self.dec4(torch.cat([self.upsample(d3), e1], dim=1))

        # Output
        return self.final(d4)




model = HandwritingUNet().to(device)
model.load_state_dict(torch.load("/Home/siv36/hesal5042/Research/NORCE/hello/RePaint/guided_diffusion_mnist/guided_diffusion/handwriting_outputsssss/model_epoch_100.pth"))
model.eval()

#same diffusion parameters from your training
T = 1000
beta_start = 1e-4
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, T, device=device)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod) 
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
print("=== FINALLY SOMTHEING WORKING REPAINT IMPLEMENTATION PHEWWW===")

class IAMDatasetHF(torch.utils.data.Dataset):
    def __init__(self, root_dir, img_size=(64, 128)):
        self.root_dir = root_dir
        self.img_size = img_size
        self.image_paths = []
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(subdir, file))
        
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        image = self.transform(image)
        return image, 0


iam_dataset = IAMDatasetHF(root_dir='iamHandwriting_dataset/words')

B = 4
indices = random.sample(range(len(iam_dataset)), B)
print(f"Selecting indices randomly: {indices}")

image = torch.stack([iam_dataset[i][0] for i in indices]).to(device)

#create mask
mask = torch.ones_like(image)
mask[:, :,:,50:] = 0  # Right half masked
known_region = image * mask


def repaint_handwriting(model, x0, mask, T, jump_length, jump_n_sample):

    model.eval()
    device = x0.device
    B = x0.size(0)

    x_t = torch.randn_like(x0)  # Start from pure noise

    for t in tqdm(range(T - 1, -1, -1), desc="RePaint Sampling"):

        for u in range(jump_n_sample if (t > 0 and t % jump_length == 0) else 1):
            t_tensor = torch.tensor([t] * B, device=device).long()


            with torch.no_grad():
                predicted_noise = model(x_t, t_tensor) #predict nise

            if t > 0:
                noise_known = torch.randn_like(x0)
            else:
                noise_known = torch.zeros_like(x0)
            
            x_t_minus_1_known = (
                sqrt_alphas_cumprod[t-1] * x0 + 
                sqrt_one_minus_alphas_cumprod[t-1] * noise_known
            ) if t > 0 else x0


            alpha_t = alphas[t]
            alpha_t_cumprod = alphas_cumprod[t]

            mean = (1 / torch.sqrt(alpha_t)) * (
                x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_t_cumprod)) * predicted_noise
            )
            

            if t > 0:
                noise = torch.randn_like(x_t)

                sigma_t = torch.sqrt(betas[t])  #or try sqrt((1-alpha_{t-1})/(1-alpha_t) * beta_t)
            else:
                noise = torch.zeros_like(x_t)
                sigma_t = 0
            
            x_t_minus_1_unknown = mean + sigma_t * noise




            x_t_minus_1 = mask * x_t_minus_1_known + (1 - mask) * x_t_minus_1_unknown


            if u < (jump_n_sample - 1) and t > 0:

                noise_resample = torch.randn_like(x_t_minus_1)
                x_t = (
                    torch.sqrt(alphas[t-1]) * x_t_minus_1 + 
                    torch.sqrt(1 - alphas[t-1]) * noise_resample
                )
            else:
                x_t = x_t_minus_1

    output = x_t
    # output = (x_t + 1) / 2
    output = torch.clamp(output, 0, 1)
    return output

output = repaint_handwriting(
    model, known_region, mask, T, jump_length=5, jump_n_sample=20

)

print("fdsff")
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(image[0,0].cpu(), cmap="gray", vmin=0, vmax=1)
plt.title("Original")


plt.subplot(1, 3, 2)
plt.imshow((image*mask)[0,0].cpu(), cmap="gray", vmin=0, vmax=1)
plt.title("Masked")


plt.subplot(1, 3, 3)
plt.imshow(output[0,0].detach().cpu(), cmap="gray", vmin=0, vmax=1)
plt.title("RePainted")
# plt.show()
plt.savefig('sine_wave10.png') 
