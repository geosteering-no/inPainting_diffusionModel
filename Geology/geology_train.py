import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from tqdm import tqdm
from torchvision.utils import save_image
from torch.utils.data import random_split
import time

start = time.time()

#device
device = "cuda:2" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

#hyperparameters for diffusion
T = 1000
beta_start = 1e-4
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, T, device=device)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)


#unet mdel
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
        t_embed = self.time_embed(t.unsqueeze(-1).float())
        t_embed = t_embed.unsqueeze(-1).unsqueeze(-1)

        e1 = self.enc1(x)
        e2 = self.enc2(self.downsample(e1))
        e3 = self.enc3(self.downsample(e2))
        e4 = self.enc4(self.downsample(e3))

        bottleneck = self.bottleneck(self.downsample(e4))
        # expand time embedding to match bottleneck shape
        t_embed = t_embed.expand_as(bottleneck)
        bottleneck = bottleneck + t_embed

        d1 = self.dec1(torch.cat([self.upsample(bottleneck), e4], dim=1))
        d2 = self.dec2(torch.cat([self.upsample(d1), e3], dim=1))
        d3 = self.dec3(torch.cat([self.upsample(d2), e2], dim=1))
        d4 = self.dec4(torch.cat([self.upsample(d3), e1], dim=1))

        return self.final(d4)


#forward diffusion procesd
def forward_diffusion(x0, t):
    noise = torch.randn_like(x0)
    t = t.to(x0.device)
    sqrt_alpha_cumprod_t = sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
    sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
    return sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise, noise


#geology xz pacthes dataset
class GeologyXZDataset(Dataset):
    def __init__(self, folder_path):
        self.files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.endswith(".npy")
        ]
        self.files.sort()

        print(f"Found {len(self.files)} images")

        # --------------------------------------------------
        # Compute GLOBAL min/max once (diffusion-safe)
        # --------------------------------------------------
        all_min = []
        all_max = []

        for f in tqdm(self.files, desc="Computing global min/max"):
            arr = np.load(f).astype(np.float32)
            all_min.append(arr.min())
            all_max.append(arr.max())

        self.global_min = float(np.min(all_min))
        self.global_max = float(np.max(all_max))

        print(f"Global min: {self.global_min:.6f}")
        print(f"Global max: {self.global_max:.6f}")

        # small epsilon to avoid division by zero
        self.eps = 1e-8

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = np.load(self.files[idx]).astype(np.float32)

        # --------------------------------------------------
        # GLOBAL normalization → [-1, 1]
        # --------------------------------------------------
        img = (img - self.global_min) / (self.global_max - self.global_min + self.eps)
        img = img * 2.0 - 1.0

        # add channel dimension
        img = np.expand_dims(img, axis=0)

        return torch.from_numpy(img), 0




#sample generation
def generate_samples(model, epoch, num_samples=1, img_size=(64, 256)):
    model.eval()
    with torch.no_grad():
        x = torch.randn(num_samples, 1, img_size[0], img_size[1]).to(device)
        for t in reversed(range(T)):
            t_batch = torch.tensor([t] * num_samples, device=device).long()
            predicted_noise = model(x, t_batch)
            alpha_t = alphas[t].to(device)
            alpha_cumprod_t = alphas_cumprod[t].to(device)
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise) + torch.sqrt(betas[t]) * noise
        # x = (x.clamp(-1, 1) + 1) / 2
        x = (x+ 1) / 2 # scale back to [0,1] now in ~[0,1], can slightly exceed 0 or 1
        x_phys = x * (dataset.global_max - dataset.global_min) + dataset.global_min # scale back to physical values or porosity values
        # save PHYSICAL data
        np.save(f"/Home/siv36/hesal5042/Research/NORCE/inPainting_diffusionModel/Geology/Geology_Code/output/train_generated_patches/1000/patch_epoch_{epoch}.npy",
        x_phys.cpu().numpy())
        save_image(x, f'/Home/siv36/hesal5042/Research/NORCE/inPainting_diffusionModel/Geology/Geology_Code/output/train_geology_output/1000/epoch_{epoch}.png')
    print(f"Samples saved for epoch {epoch}")


#training
# dataset = GeologyXZDataset("/Home/siv36/hesal5042/Research/NORCE/hello/RePaint/guided_diffusion_mnist/guided_diffusion/Geology/XZ_numpy_patches")
dataset = GeologyXZDataset("/Home/siv36/hesal5042/Research/NORCE/inPainting_diffusionModel/Geology/sliced_data/XZ_numpy_patches")
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

print("Total dataset:", len(dataset))
print("Train dataset:", len(train_dataset))
print("Batch size:", train_loader.batch_size)
print("Train batches per epoch:", len(train_loader))
def train_geology_ddpm():


    model = GeologyUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    os.makedirs('/Home/siv36/hesal5042/Research/NORCE/inPainting_diffusionModel/Geology/Geology_Code/output/100', exist_ok=True)
    num_epochs = 100

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for imgs, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            imgs = imgs.to(device)
            batch_size = imgs.size(0)
            t = torch.randint(0, T, (batch_size,), device=device).long()
            noisy_imgs, noise = forward_diffusion(imgs, t)
            predicted_noise = model(noisy_imgs, t)

            loss = criterion(predicted_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")


        # test set evaluation
       
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for imgs, _ in test_loader:
                imgs = imgs.to(device)
                batch_size = imgs.size(0)
                t = torch.randint(0, T, (batch_size,), device=device).long()
                noisy_imgs, noise = forward_diffusion(imgs, t)
                predicted_noise = model(noisy_imgs, t)

                loss = criterion(predicted_noise, noise)
                test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader)
        print(f"Test Loss: {avg_test_loss:.4f}")

        # model cgeckpoint saves every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f'/Home/siv36/hesal5042/Research/NORCE/hello/RePaint/guided_diffusion_mnist/guided_diffusion/Geology/Geology_Code/output/train_generated_patches/model_100{epoch+1}.pth')
            
        #sample images generated every epock
        generate_samples(model, epoch+1, num_samples=1)
     



if __name__ == "__main__":
    train_geology_ddpm()
