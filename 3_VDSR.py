import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import types, sys

# -------------------------
# Classi dummy per il caricamento del checkpoint legacy
# -------------------------
class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.conv(x))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.residual_layer = self.make_layer(Conv_ReLU_Block, 18)
        self.input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dump_patches = False
    
    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = self.relu(self.input(x))
        out = self.residual_layer(out)
        out = self.output(out)
        out = torch.add(out, residual)
        return out

# -------------------------
# Modello VDSR (tua implementazione pulita)
# -------------------------
class VDSR(nn.Module):
    def __init__(self):
        super(VDSR, self).__init__()
        self.input = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.residual_layer = self.make_layer(64, 18)
        self.output = nn.Conv2d(64, 3, kernel_size=3, padding=1, bias=False)

    def make_layer(self, channels, num_layers):
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x 
        out = self.relu(self.input(x))
        out = self.residual_layer(out)
        out = self.output(out)
        return torch.add(out, residual)

# -------------------------
# Applicazione VDSR su video
# -------------------------
def apply_vdsr_video(input_folder, output_folder, model, device, scale=4):
    if not os.path.exists(input_folder):
        print(f"❌ Cartella input non trovata: {input_folder}")
        return

    os.makedirs(output_folder, exist_ok=True)
        
    videos = [f for f in os.listdir(input_folder) 
              if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    if not videos:
        print(f"⚠️ Nessun video trovato in {input_folder}")
        return

    for video_name in videos:
        input_path = os.path.join(input_folder, video_name)
        output_path = os.path.join(output_folder, f"VDSR_{video_name}")
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"❌ Errore nell'aprire: {video_name}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) * scale
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) * scale
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n🎥 VDSR pretrainato su: {video_name}")

        with torch.no_grad():
            with tqdm(total=total_frames, desc=f"Rendering VDSR", unit="frame") as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    img_res = cv2.resize(frame, (w, h), interpolation=cv2.INTER_CUBIC)
                    
                    img_in = img_res.astype(np.float32) / 255.0
                    img_in = torch.from_numpy(img_in.transpose(2, 0, 1)).unsqueeze(0).to(device)
                    
                    prediction = model(img_in)
                    
                    output = prediction.squeeze(0).permute(1, 2, 0)
                    output = (output * 255.0).clamp(0, 255).cpu().numpy().astype(np.uint8)
                    
                    out.write(output)
                    pbar.update(1)
                
        cap.release()
        out.release()
        print(f"✅ Video salvato: {output_path}")

# -------------------------
# MAIN
# -------------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint_path = "model_epoch_50.pth"

    # Creiamo un modulo finto chiamato "vdsr" con le classi necessarie
    fake_module = types.ModuleType("vdsr")
    fake_module.Net = Net
    fake_module.Conv_ReLU_Block = Conv_ReLU_Block
    sys.modules["vdsr"] = fake_module

    # Carichiamo il checkpoint legacy
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Estraiamo i pesi
    if hasattr(checkpoint, "state_dict"):
        state_dict = checkpoint.state_dict()
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint  # Potrebbe essere direttamente il state_dict

    # Creiamo il modello pulito e carichiamo i pesi
    model = VDSR().to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    print(f"✅ Modello VDSR caricato da {checkpoint_path} su: {device}")

    apply_vdsr_video(
        input_folder='Videos_Down',
        output_folder='Videos_Restored_VDSR',
        model=model,
        device=device,
        scale=4
    )

if __name__ == "__main__":
    main()