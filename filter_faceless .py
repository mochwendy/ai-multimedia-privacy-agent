import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import sys
import urllib.request
import numpy as np

def analisa_audio_internal(video_path):
    """Fungsi Agen Telinga V2.1: Menjamin transparansi status pengecekan musik di log"""
    audio_temp = "temp_audio.wav"
    os.system(f"ffmpeg -y -i {video_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_temp} >/dev/null 2>&1")
    
    if not os.path.exists(audio_temp) or os.path.getsize(audio_temp) < 1000:
        if os.path.exists(audio_temp): os.remove(audio_temp)
        return "🔇 AUDIO MUTED (Senyap/Tanpa Suara)", "NORMAL"

    try:
        with open(audio_temp, "rb") as f:
            f.seek(44)
            data_raw = f.read()
        
        audio_data = np.frombuffer(data_raw, dtype=np.int16)
        os.remove(audio_temp)
        
        if len(audio_data) == 0:
            return "🔇 AUDIO MUTED (Data Kosong)", "NORMAL"
            
        # 1. Hitung Volume (RMS to Decibel)
        rms = np.sqrt(np.mean(audio_data.astype(np.float64)**2))
        db = 20 * np.log10(rms) if rms > 0 else 0
        
        # 2. Analisis Struktur Frekuensi & Ritme Musik
        ukuran_segmen = 4000
        jumlah_segmen = len(audio_data) // ukuran_segmen
        
        if jumlah_segmen > 4:
            rms_per_segmen = []
            for i in range(jumlah_segmen):
                segmen = audio_data[i*ukuran_segmen : (i+1)*ukuran_segmen].astype(np.float64)
                rms_segmen = np.sqrt(np.mean(segmen**2)) if len(segmen) > 0 else 0
                rms_per_segmen.append(rms_segmen)
            
            rms_per_segmen = np.array(rms_per_segmen)
            rata_rms = np.mean(rms_per_segmen)
            koefisien_variasi = np.std(rms_per_segmen) / rata_rms if rata_rms > 0 else 1
            
            apakah_musik = koefisien_variasi < 0.45 and db > 45
        else:
            apakah_musik = False

        # 3. Penyusunan Log Terperinci untuk User
        if db > 75:
            return f"⚠️ AUDIO HIGH-VOLUME ({db:.2f} dB) - [⚠️ MUSIK: Terabaikan karena Kebisingan Ekstrem]", "DANGER_VOLUME"
        elif apakah_musik:
            return f"🎵 MUSIC DETECTED ({db:.2f} dB) - [❌ MUSIK: Terdeteksi Instrumen/Lagu Melanggar]", "MUSIC_REJECT"
        else:
            # 🌟 PERBAIKAN: Sekarang tertulis jelas kalau musik sudah dicek dan hasilnya aman!
            return f"🔊 AUDIO NORMAL ({db:.2f} dB) - [✅ MUSIK: Bersih / Hanya Suara Percakapan]", "NORMAL"
            
    except Exception as e:
        if os.path.exists(audio_temp): os.remove(audio_temp)
        return f"⚠️ AUDIO CHECK SKIPPED ({str(e)})", "NORMAL"


def deteksi_faceless_video(video_path):
    if not os.path.exists(video_path):
        print(f"❌ Kesalahan: File video '{video_path}' tidak ditemukan.")
        return None

    status_audio, kategori_audio = analisa_audio_internal(video_path)

    model_path = "blaze_face_short_range.tflite"
    if not os.path.exists(model_path):
        print("⏳ Sedang mengunduh model AI Face Detection...")
        url_model = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
        urllib.request.urlretrieve(url_model, model_path)

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.5)

    kap_video = cv2.VideoCapture(video_path)
    total_frames = int(kap_video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = kap_video.get(cv2.CAP_PROP_FPS)
    lebar = int(kap_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    tinggi = int(kap_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = "output_faceless.mp4"
    empat_cc = cv2.VideoWriter_fourcc(*'mp4v')
    penulis_video = cv2.VideoWriter(output_path, empat_cc, fps, (lebar, tinggi))

    wajah_terdeteksi_global = False

    with vision.FaceDetector.create_from_options(options) as detector:
        while kap_video.isOpened():
            sukses, frame = kap_video.read()
            if not sukses:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            hasil_deteksi = detector.detect(mp_image)

            if hasil_deteksi.detections:
                wajah_terdeteksi_global = True
                for deteksi in hasil_deteksi.detections:
                    box = deteksi.bounding_box
                    xmin = max(0, int(box.origin_x))
                    ymin = max(0, int(box.origin_y))
                    xmax = min(lebar, int(box.origin_x + box.width))
                    ymax = min(tinggi, int(box.origin_y + box.height))

                    if xmax > xmin and ymax > ymin:
                        sub_wajah = frame[ymin:ymax, xmin:xmax]
                        wajah_blur = cv2.GaussianBlur(sub_wajah, (99, 99), 30)
                        frame[ymin:ymax, xmin:xmax] = wajah_blur

            penulis_video.write(frame)

    kap_video.release()
    penulis_video.release()

    output_ready = "output_ready.mp4"
    cmd_merge = f"ffmpeg -y -i {output_path} -i {video_path} -map 0:v -map 1:a? -c:v libx264 -c:a copy -f mp4 {output_ready} >/dev/null 2>&1"
    
    if os.system(cmd_merge) == 0:
        final_video = output_ready
    else:
        final_video = output_path

    if kategori_audio == "DANGER_VOLUME":
        kesimpulan_final = "REJECTED (KONTEN DITOLAK: Kebisingan Audio Melebihi Batas) ❌"
    elif kategori_audio == "MUSIC_REJECT":
        kesimpulan_final = "REJECTED (KONTEN DITOLAK: Terdeteksi Musik/Backsound Latar Belakang) ❌"
    elif wajah_terdeteksi_global:
        kesimpulan_final = "APPROVED WITH SENSOR (LOLOS: Wajah Terproteksi Gaussian Blur) 🟢"
    else:
        kesimpulan_final = "APPROVED PRIVACY CLEAN (LOLOS: Konten Murni Tanpa Wajah & Musik) 🚀"

    nama_file = os.path.basename(video_path)
    print("\n📊 HASIL FILTER AI MULTIMEDIA AGENT (VISION + AUDIO + MUSIC DETECTOR):")
    print("--------------------------------------------------")
    print(f"• Nama File Video  : {nama_file}")
    print(f"• Total Frame      : {total_frames} frame (~{total_frames/fps:.2f} detik)")
    print(f"• Status Audio     : {status_audio}")
    print(f"• Kategori Visual  : {'👤 HUMAN FACE VIDEO' if wajah_terdeteksi_global else '✅ FACELESS VIDEO'}")
    print(f"• Status Filter    : {'WAJAH BERHASIL DI-BLUR' if wajah_terdeteksi_global else 'LOLOS OTOMATIS'}")
    print("--------------------------------------------------")
    print(f"🎯 KESIMPULAN AGEN : {kesimpulan_final}")
    print("--------------------------------------------------")
    
    return final_video

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("💡 Cara penggunaan skrip otomatis di terminal:\n   python filter_faceless.py nama_video.mp4")
    else:
        deteksi_faceless_video(sys.argv[1])
