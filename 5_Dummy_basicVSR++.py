import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import time

# -------------------------
# Blocchi base
# -------------------------
class ResidualBlockNoBN(nn.Module):
    def __init__(self, num_feat=64):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        )
    def forward(self, x):
        return x + self.main(x)

class TemporalRefine(nn.Module):
    def __init__(self, num_feat):
        super().__init__()
        self.conv = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)

    def forward(self, curr, prev):
        if prev is None:
            return curr
        return self.conv(torch.cat([curr, prev], dim=1))

# -------------------------
# VSR migliorato
# -------------------------
class BasicVSRLikeEnhanced(nn.Module):
    def __init__(self, num_feat=64, num_block=7):
        super().__init__()
        self.num_feat = num_feat

        self.conv_first = nn.Conv2d(3, num_feat, 3, 1, 1)
        self.backward_resblocks = nn.ModuleList([ResidualBlockNoBN(num_feat) for _ in range(num_block)])
        self.forward_resblocks = nn.ModuleList([ResidualBlockNoBN(num_feat) for _ in range(num_block)])
        self.fusion = nn.Conv2d(num_feat * 2, num_feat, 1, 1, 0)
        self.temporal_refine = TemporalRefine(num_feat)

        self.upsample1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
        self.upsample2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_last = nn.Conv2d(num_feat, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, lrs):
        b, t, c, h, w = lrs.size()
        feats = [self.lrelu(self.conv_first(lrs[:, i])) for i in range(t)]

        print(f"🧠 Estrazione feature completata su {t} frame")

        out_prop = feats[0].new_zeros(b, self.num_feat, h, w)
        backward_outputs = []
        for i in range(t - 1, -1, -1):
            out_prop = feats[i] + out_prop
            for block in self.backward_resblocks:
                out_prop = block(out_prop)
            backward_outputs.append(out_prop)
        backward_outputs = backward_outputs[::-1]

        print("🔁 Propagazione backward completata")

        out_prop = feats[0].new_zeros(b, self.num_feat, h, w)
        prev_fused = None
        final_outputs = []

        for i in range(t):
            out_prop = feats[i] + out_prop
            for block in self.forward_resblocks:
                out_prop = block(out_prop)

            fused = self.fusion(torch.cat([out_prop, backward_outputs[i]], dim=1))
            fused = self.temporal_refine(fused, prev_fused)
            prev_fused = fused.detach()

            out = self.lrelu(self.pixel_shuffle(self.upsample1(fused)))
            out = self.lrelu(self.pixel_shuffle(self.upsample2(out)))
            out = self.conv_last(out)

            final_outputs.append(out + nn.functional.interpolate(lrs[:, i], scale_factor=4, mode='bilinear', align_corners=False))

        print("🎨 Upscaling e fusione completati")
        return torch.stack(final_outputs, dim=1)

# -------------------------
# Inference su video con log
# -------------------------
def main():
    print("🚀 Avvio pipeline VSR...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Device: {device}")

    model = BasicVSRLikeEnhanced(num_feat=64).to(device)
    model.eval()
    print("✅ Modello caricato e in modalità eval")

    input_dir = 'Videos_Down/realistic'
    output_dir = 'VSRPP'
    os.makedirs(output_dir, exist_ok=True)

    WINDOW = 15
    OVERLAP = 5

    for video in os.listdir(input_dir):
        if not video.endswith(('.mp4', '.avi', '.mov')):
            continue

        print(f"\n🎥 Apro video: {video}")
        cap = cv2.VideoCapture(os.path.join(input_dir, video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"📐 Risoluzione: {w}x{h} | FPS: {fps:.2f} | Frame totali: {total_frames}")

        writer = cv2.VideoWriter(
            os.path.join(output_dir, f"HQ_{video}"),
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (w * 4, h * 4)
        )

        frames = []
        carry = []
        processed = 0
        t_start = time.time()

        with tqdm(total=total_frames, desc="📈 Avanzamento") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                frames.append(torch.from_numpy(img.transpose(2, 0, 1)))

                if len(frames) == WINDOW:
                    print(f"\n⚙️ Elaboro chunk: frame {processed} → {processed + WINDOW}")
                    input_t = torch.stack(frames).unsqueeze(0).to(device)

                    with torch.no_grad():
                        output = model(input_t)

                    start = 0 if not carry else OVERLAP
                    for i in range(start, output.size(1)):
                        res = output[0, i].cpu().numpy().transpose(1, 2, 0)

                        # protezione numerica
                        res = np.nan_to_num(res, nan=0.0, posinf=1.0, neginf=0.0)

                        # clamp serio
                        res = np.clip(res, 0.0, 1.0)

                        # bilanciamento con input LR upscalato (stabilizza i colori)
                        lr = input_t[0, i].cpu().numpy().transpose(1, 2, 0)
                        lr_up = cv2.resize(lr, (res.shape[1], res.shape[0]), interpolation=cv2.INTER_CUBIC)
                        res = 0.8 * res + 0.2 * lr_up

                        res = (res * 255.0).astype(np.uint8)
                        writer.write(cv2.cvtColor(res, cv2.COLOR_RGB2BGR))

                        pbar.update(1)
                        processed += 1

                    carry = frames[-OVERLAP:]
                    frames = frames[-OVERLAP:]

        cap.release()
        writer.release()
        elapsed = time.time() - t_start
        print(f"✅ Video completato in {elapsed:.1f}s → {video}")

    print("\n🎉 Tutti i video sono stati processati.")

if __name__ == "__main__":
    main()
