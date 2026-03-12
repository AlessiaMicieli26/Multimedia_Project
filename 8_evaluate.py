import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# -------------------------
# Metriche PSNR / SSIM su un video
# -------------------------
def evaluate_restoration(video_path_orig, video_path_restored):
    cap_orig = cv2.VideoCapture(video_path_orig)
    cap_rest = cv2.VideoCapture(video_path_restored)

    if not cap_orig.isOpened() or not cap_rest.isOpened():
        print(f"❌ Errore apertura video:\n- {video_path_orig}\n- {video_path_restored}")
        return 0, 0

    psnr_values, ssim_values = [], []
    
    while True:
        ret_o, frame_o = cap_orig.read()
        ret_r, frame_r = cap_rest.read()
        if not ret_o or not ret_r:
            break

        if frame_o.shape != frame_r.shape:
            frame_r = cv2.resize(frame_r, (frame_o.shape[1], frame_o.shape[0]))

        gray_o = cv2.cvtColor(frame_o, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

        psnr_values.append(psnr(gray_o, gray_r, data_range=255))
        ssim_values.append(ssim(gray_o, gray_r, data_range=255))

    cap_orig.release()
    cap_rest.release()

    if not psnr_values:
        return 0, 0

    return float(np.mean(psnr_values)), float(np.mean(ssim_values))


# -------------------------
# Valutazione per cartelle di metodi
# -------------------------
def evaluate_restoration_folder(original_dir, methods_config):
    all_results = []

    original_videos = [
        f for f in os.listdir(original_dir)
        if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    ]

    if not original_videos:
        print(f"⚠️ Nessun video trovato in {original_dir}")
        return None

    for method_name, restored_dir in methods_config.items():
        if not os.path.exists(restored_dir):
            print(f"⚠️ Cartella non trovata: {restored_dir}")
            continue

        method_psnr, method_ssim = [], []

        for video_name in tqdm(original_videos, desc=f"Calcolo {method_name}"):

            original_path = os.path.join(original_dir, video_name)
            restored_files = os.listdir(restored_dir)

            base_name = os.path.splitext(video_name)[0]

            match = [
                f for f in restored_files
                if base_name in f and f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
            ]

            if match:
                restored_path = os.path.join(restored_dir, match[0])
                p, s = evaluate_restoration(original_path, restored_path)

                if p > 0:
                    method_psnr.append(p)
                    method_ssim.append(s)

        if method_psnr:
            all_results.append({
                "Metodo": method_name,
                "PSNR Medio": np.mean(method_psnr),
                "SSIM Medio": np.mean(method_ssim)
            })

    return pd.DataFrame(all_results)


# -------------------------
# Plot risultati
# -------------------------
def plot_results(df_results, out_dir="plots"):

    if df_results is None or df_results.empty:
        print("⚠️ Nessun risultato da plottare.")
        return

    os.makedirs(out_dir, exist_ok=True)

    df_results = df_results.sort_values(by="PSNR Medio", ascending=False)

    # --- PSNR ---
    plt.figure(figsize=(10, 5))
    sns.barplot(x="Metodo", y="PSNR Medio", data=df_results)
    plt.title("Confronto Qualità Video (PSNR)")
    plt.ylabel("PSNR Medio (dB)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    psnr_path = os.path.join(out_dir, "psnr_comparison.png")
    plt.savefig(psnr_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"💾 Grafico PSNR salvato in: {psnr_path}")

    # --- SSIM ---
    df_ssim = df_results.sort_values(by="SSIM Medio", ascending=False)

    plt.figure(figsize=(10, 5))
    sns.barplot(x="Metodo", y="SSIM Medio", data=df_ssim)
    plt.title("Confronto Fedeltà Strutturale (SSIM)")
    plt.ylabel("SSIM Medio")
    plt.ylim(df_ssim["SSIM Medio"].min() - 0.02, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    ssim_path = os.path.join(out_dir, "ssim_comparison.png")
    plt.savefig(ssim_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"💾 Grafico SSIM salvato in: {ssim_path}")


# -------------------------
# Salvataggio frame confronto
# -------------------------
def save_comparison_frame(original_video, methods_config, frame_idx=50, out_path="comparison_grid.jpg"):

    frames = []

    cap_orig = cv2.VideoCapture(original_video)
    cap_orig.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap_orig.read()
    cap_orig.release()

    if not ret:
        print("❌ Impossibile leggere il frame originale.")
        return

    cv2.putText(frame, "Originale", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    frames.append(frame)

    h, w, _ = frame.shape

    for name, folder in methods_config.items():

        if not os.path.exists(folder):
            continue

        restored_files = [
            f for f in os.listdir(folder)
            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
        ]

        if not restored_files:
            continue

        cap = cv2.VideoCapture(os.path.join(folder, restored_files[0]))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if ret:
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h))

            cv2.putText(frame, name, (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            frames.append(frame)

    if not frames:
        print("❌ Nessun frame disponibile.")
        return

    cols = 3
    rows = int(np.ceil(len(frames) / cols))

    grid = []

    for r in range(rows):
        row_frames = frames[r*cols:(r+1)*cols]
        if len(row_frames) < cols:
            row_frames += [np.zeros((h, w, 3), dtype=np.uint8)] * (cols - len(row_frames))
        grid.append(np.hstack(row_frames))

    comparison = np.vstack(grid)
    cv2.imwrite(out_path, comparison)
    print(f"✅ Griglia di confronto salvata in: {out_path}")


# -------------------------
# MAIN
# -------------------------
def main():

    methods_folders = {
        "Bicubic": "Videos/Videos_Restored/Restore_Bicubic",
        "Lanczos": "Videos/Videos_Restored/Restore_Lanzcoz",  
        "VDSR": "Videos/Videos_Restored/Restore_VDSR",
        "Dummy_BasicVSR": "Videos/Videos_Restored/Videos_Restored_basicvsr_v2",
        "Lite_BasicVSR": "Videos/Videos_Restored/BasicVSR_Lite",
        "Dummy_BasicVSR++": "Videos/Videos_Restored/VSRPP_degradation_v2",
        "Lite_BasicVSR++": "Videos/Videos_Restored/BasicVSR_PlusPlus_Lite"
    }

    # 👇 FIX IMPORTANTE
    original_folder = "Videos/Videos_Original"

    df_results = evaluate_restoration_folder(original_folder, methods_folders)

    if df_results is not None and not df_results.empty:

        print("\n📊 RISULTATI:\n", df_results)
        df_results.to_csv("benchmark_results.csv", index=False)
        print("💾 Risultati salvati in benchmark_results.csv")

        plot_results(df_results)

        # # confronto visivo sul primo video trovato
        # original_videos = [
        #     f for f in os.listdir(original_folder)
        #     if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
        # ]

        # if original_videos:
        #     first_video = original_videos[0]

        #     save_comparison_frame(
        #         original_video=os.path.join(original_folder, first_video),
        #         methods_config=methods_folders,
        #         frame_idx=50,
        #         out_path="comparison_grid.jpg"
        #     )

    else:
        print("⚠️ Nessun risultato prodotto. Controlla i percorsi.")


if __name__ == "__main__":
    main()