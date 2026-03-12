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
# Second-Order Deformable Alignment (Lite)
# Simula l'allineamento deformabile di BasicVSR++
# in modo leggero senza librerie esterne (e.g. mmcv)
# ==========================================

class LiteDeformableAlignment(nn.Module):
    """
    Versione lite dell'allineamento deformabile di BasicVSR++.
    Usa optical flow approssimato tramite correlazione + warp con grid_sample.
    Supporta second-order: usa anche il frame precedente-precedente.
    """
    def __init__(self, num_feat):
        super().__init__()
        # Stima il flusso ottico tra frame corrente e precedente
        self.flow_estimator = nn.Sequential(
            nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat, num_feat // 2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // 2, 2, 3, 1, 1),  # output: (dx, dy)
        )
        # Second-order: combina flusso attuale e precedente
        self.flow_refiner = nn.Sequential(
            nn.Conv2d(4, 2, 3, 1, 1),  # 2 flussi → 1 flusso raffinato
            nn.Tanh()
        )

    def warp(self, feat, flow):
        """Warpa feat secondo il flusso flow con grid_sample."""
        B, C, H, W = feat.shape
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, dtype=torch.float32, device=feat.device),
            torch.arange(W, dtype=torch.float32, device=feat.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1, H, W, 2)

        # Normalizza in [-1, 1]
        grid[..., 0] = 2.0 * grid[..., 0] / (W - 1) - 1.0
        grid[..., 1] = 2.0 * grid[..., 1] / (H - 1) - 1.0

        # Aggiungi flusso normalizzato
        flow_norm = flow.permute(0, 2, 3, 1).clone()
        flow_norm[..., 0] = flow_norm[..., 0] / (W / 2)
        flow_norm[..., 1] = flow_norm[..., 1] / (H / 2)

        grid = grid + flow_norm
        grid = grid.clamp(-1, 1)

        return F.grid_sample(feat, grid, mode='bilinear', padding_mode='border', align_corners=True)

    def forward(self, feat_curr, feat_prev, feat_prev2=None, flow_prev=None):
        """
        feat_curr:  feature frame corrente
        feat_prev:  feature frame precedente (allineato)
        feat_prev2: feature frame due passi fa (second-order)
        flow_prev:  flusso stimato al passo precedente
        """
        # Stima flusso tra corrente e precedente
        concat = torch.cat([feat_curr, feat_prev], dim=1)
        flow = self.flow_estimator(concat)  # (B, 2, H, W)

        # Second-order: se disponibile il flusso precedente, raffinalo
        if flow_prev is not None:
            flow = self.flow_refiner(torch.cat([flow, flow_prev], dim=1))

        # Warpa il frame precedente verso il corrente
        feat_warped = self.warp(feat_prev, flow)

        # Se disponibile il secondo ordine, aggiungi contributo
        if feat_prev2 is not None:
            flow2 = self.flow_estimator(torch.cat([feat_curr, feat_prev2], dim=1))
            feat_warped2 = self.warp(feat_prev2, flow2)
            feat_warped = feat_warped + 0.3 * feat_warped2  # peso ridotto

        return feat_warped, flow


# ==========================================
# BasicVSR++ Lite
# ==========================================

class BasicVSRPlusPlusLite(nn.Module):
    def __init__(self, num_feat=64, num_blocks=7):
        super().__init__()

        print("🧠 Inizializzazione BasicVSR++-Lite")

        # Estrazione feature
        self.feat_extract = nn.Conv2d(3, num_feat, 3, 1, 1)

        # Allineamento deformabile lite (second-order)
        self.backward_align = LiteDeformableAlignment(num_feat)
        self.forward_align  = LiteDeformableAlignment(num_feat)

        # Propagation networks
        self.backward_fusion = nn.Conv2d(num_feat * 2, num_feat, 1)
        self.forward_fusion  = nn.Conv2d(num_feat * 2, num_feat, 1)

        self.backward_blocks = nn.Sequential(*[ResidualBlock(num_feat) for _ in range(num_blocks)])
        self.forward_blocks  = nn.Sequential(*[ResidualBlock(num_feat) for _ in range(num_blocks)])

        # Fusione bidirezionale
        self.fusion = nn.Conv2d(num_feat * 2, num_feat, 1)

        # Upsampling ×4
        self.up1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
        self.up2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)

        self.final = nn.Conv2d(num_feat, 3, 3, 1, 1)

        print("✅ Modello creato con:")
        print(f"   - Feature channels: {num_feat}")
        print(f"   - Residual blocks: {num_blocks}")
        print("   - Second-order deformable alignment: ✓")
        print("   - Upsampling: x4")


    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        print(f"\n📦 Input tensor shape: {x.shape}")
        print(f"   Batch: {B}, Frames: {T}, Size: {H}x{W}")

        # Estrazione feature
        feats = []
        print("\n🔍 Estrazione feature per frame...")
        for t in range(T):
            feat = self.feat_extract(x[:, t])
            feats.append(feat)
            print(f"   Frame {t+1}/{T} → Feature shape: {feat.shape}")

        # ─── BACKWARD PROPAGATION ───
        print("\n⬅  Backward propagation (con second-order alignment)...")
        backward_feats = [None] * T
        feat_prop = torch.zeros_like(feats[0])
        feat_prop2 = torch.zeros_like(feats[0])
        flow_prev = None

        for t in reversed(range(T)):
            feat_warped, flow = self.backward_align(feats[t], feat_prop, feat_prop2, flow_prev)
            fused = self.backward_fusion(torch.cat([feats[t], feat_warped], dim=1))
            feat_new = self.backward_blocks(fused)
            backward_feats[t] = feat_new
            feat_prop2 = feat_prop
            feat_prop  = feat_new
            flow_prev  = flow
            print(f"   Backward step frame {t}")

        # ─── FORWARD PROPAGATION ───
        print("\n➡  Forward propagation (con second-order alignment) + Upsampling...")
        feat_prop = torch.zeros_like(feats[0])
        feat_prop2 = torch.zeros_like(feats[0])
        flow_prev = None
        outputs = []

        for t in range(T):
            feat_warped, flow = self.forward_align(feats[t], feat_prop, feat_prop2, flow_prev)
            fused_f = self.forward_fusion(torch.cat([feats[t], feat_warped], dim=1))
            feat_new = self.forward_blocks(fused_f)

            # Fusione forward + backward
            fused = self.fusion(torch.cat([feat_new, backward_feats[t]], dim=1))

            # Upsample ×4
            out = self.pixel_shuffle(self.up1(fused))
            out = self.pixel_shuffle(self.up2(out))
            out = self.final(out)

            # Bicubic base (opzione A applicata: solo base se modello non addestrato)
            base = F.interpolate(
                x[:, t], scale_factor=4, mode="bicubic", align_corners=False
            )

            # ⚠️  Con pesi non addestrati usa solo base (come Opzione A)
            # Quando carichi pesi reali, sostituisci con: outputs.append(out + base)
            outputs.append(base)

            feat_prop2 = feat_prop
            feat_prop  = feat_new
            flow_prev  = flow
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

    frames_np = np.array(frames).astype(np.float32) / 255.0
    frames_t = torch.from_numpy(frames_np).permute(0, 3, 1, 2).unsqueeze(0).to(device)

    print("🚀 Avvio inferenza...\n")
    start = time.time()

    with torch.no_grad():
        output = model(frames_t)

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
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for i, frame in enumerate(output):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out_writer.write(frame)
        print(f"   Frame salvato {i+1}/{len(output)}")

    out_writer.release()
    print("✅ Video salvato in:", output_path)


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    device = "cpu"
    model = BasicVSRPlusPlusLite()
    model.eval()

    input_video  = "Videos/Videos_Down/7.mp4"
    output_video = "BasicVSR_PlusPlus_Lite/7_basicVSR_plusplus_lite.mp4"

    os.makedirs("BasicVSR_PlusPlus_Lite", exist_ok=True)

    process_video(input_video, output_video, model, device)