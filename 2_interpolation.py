import os
import cv2
from tqdm import tqdm

def run_baseline_interpolation(input_folder, output_folder, method=cv2.INTER_CUBIC, suffix="Bicubic", scale=4):
    if not os.path.exists(input_folder):
        print(f"❌ Errore: La cartella di input '{input_folder}' non esiste.")
        return

    os.makedirs(output_folder, exist_ok=True)
    
    video_files = [f for f in os.listdir(input_folder) 
                   if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    if not video_files:
        print(f"⚠️ Nessun video trovato in {input_folder}")
        return

    for video_name in video_files:
        input_path = os.path.join(input_folder, video_name)
        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            print(f"❌ Errore nell'aprire il video: {video_name}")
            continue
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) * scale
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) * scale
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        output_path = os.path.join(
            output_folder, 
            f"{os.path.splitext(video_name)[0]}_{suffix}.mp4"
        )

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        print(f"\n🎞️ --- Elaborazione {suffix} per: {video_name} ---")
        
        with tqdm(total=total_frames, unit='frame', desc=f"Rendering {suffix}") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                rescaled_frame = cv2.resize(frame, (w, h), interpolation=method)
                out.write(rescaled_frame)
                pbar.update(1)
        
        cap.release()
        out.release()
        print(f"✅ Salvataggio completato: {output_path}")

def main():
    run_baseline_interpolation(
        input_folder='Videos_Down',
        output_folder='Videos_Restored_Bicubic',
        method=cv2.INTER_CUBIC,
        suffix="Bicubic",
        scale=4
    )

    run_baseline_interpolation(
        input_folder='Videos_Down',
        output_folder='Videos_Restored_Lanczos',
        method=cv2.INTER_LANCZOS4,
        suffix="Lanczos",
        scale=4
    )

if __name__ == "__main__":
    main()
