import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time


# ==========================================
# Residual Block
# ==========================================

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        return x + self.conv2(F.relu(self.conv1(x)))


# ==========================================
# BasicVSR Lite CPU
# ==========================================

class BasicVSRLite(nn.Module):
    def __init__(self, num_feat=64, num_blocks=10):
        super().__init__()

        print("🧠 Inizializzazione BasicVSR-Lite")

        self.feat_extract = nn.Conv2d(3, num_feat, 3, 1, 1)

        self.backward_blocks = nn.Sequential(
            *[ResidualBlock(num_feat) for _ in range(num_blocks)]
        )

        self.forward_blocks = nn.Sequential(
            *[ResidualBlock(num_feat) for _ in range(num_blocks)]
        )

        self.fusion = nn.Conv2d(num_feat * 2, num_feat, 1)

        self.up1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
        self.up2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)

        self.final = nn.Conv2d(num_feat, 3, 3, 1, 1)

        print("✅ Modello creato con:")
        print(f"   - Feature channels: {num_feat}")
        print(f"   - Residual blocks: {num_blocks}")
        print("   - Upsampling: x4")


    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        print(f"\n📦 Input tensor shape: {x.shape}")
        print(f"   Batch: {B}, Frames: {T}, Size: {H}x{W}")

        feats = []
        print("\n🔍 Estrazione feature per frame...")
        for t in range(T):
            feat = self.feat_extract(x[:, t])
            feats.append(feat)
            print(f"   Frame {t+1}/{T} → Feature shape: {feat.shape}")

        # BACKWARD
        print("\n⬅ Backward propagation...")
        feat_b = torch.zeros_like(feats[0])
        backward_feats = []

        for t in reversed(range(T)):
            feat_b = self.backward_blocks(feat_b + feats[t])
            backward_feats.insert(0, feat_b)
            print(f"   Backward step frame {t}")

        # FORWARD
        print("\n➡ Forward propagation + Upsampling...")
        feat_f = torch.zeros_like(feats[0])
        outputs = []

        for t in range(T):
            feat_f = self.forward_blocks(feat_f + feats[t])

            # Upsample ×4 con bicubic (pesi casuali ignorati)
            base = F.interpolate(
                x[:, t], scale_factor=4, mode="bicubic", align_corners=False
            )

            outputs.append(base)  # ← MODIFICA QUI

            print(f"   Upscaled frame {t+1}/{T}")

        print("\n🎯 Forward completo.\n")
        return torch.stack(outputs, dim=1)


# ==========================================
# VIDEO INFERENCE
# ==========================================

def process_video(input_path, output_path, model, device="cpu"):
    print("📂 Apertura video:", input_path)

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frames = []
    frame_count = 0

    print("📥 Lettura frame...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        frame_count += 1

    cap.release()

    print(f"✅ Totale frame letti: {frame_count}")
    print("⏳ Preparazione tensori...")

    frames = np.array(frames).astype(np.float32) / 255.0
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2).unsqueeze(0)
    frames = frames.to(device)

    print("🚀 Avvio inferenza...\n")
    start = time.time()

    with torch.no_grad():
        output = model(frames)

    end = time.time()
    print(f"⏱ Tempo totale inferenza: {end - start:.2f} secondi")

    print("💾 Conversione e salvataggio video...")

    output = output.squeeze(0).permute(0, 2, 3, 1).cpu().numpy()
    print("Valore minimo:", output.min())
    print("Valore massimo:", output.max())

    if output.max() <= 1.5:
        print("Output in range [0,1] → moltiplico per 255")
        output = output * 255.0
    else:
        print("Output già in range [0,255]")

    output = np.clip(output, 0, 255).astype(np.uint8)
    h, w, _ = output[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for i, frame in enumerate(output):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
        print(f"   Frame salvato {i+1}/{len(output)}")

    out.release()
    print("✅ Video salvato in:", output_path)


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    device = "cpu"
    model = BasicVSRLite()
    model.eval()

    input_video = "Videos/Videos_Down/7.mp4"
    output_video = "BasicVSR_Lite/7_basicVSR_Lite.mp4"

    os.makedirs("BasicVSR_Lite", exist_ok=True)

    process_video(input_video, output_video, model, device)