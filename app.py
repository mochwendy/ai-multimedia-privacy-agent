import os
import sys
import io
import traceback
import contextlib
import gradio as gr
from filter_faceless import deteksi_faceless_video 

os.environ["GRADIO_MCP_SERVER"] = "True"

def wrapper_deteksi_wajah(video_path):
    if not video_path:
        return None, "⚠️ Mohon masukkan nama file video terlebih dahulu!"
        
    if not os.path.exists(video_path):
        return None, f"❌ File '{video_path}' tidak ditemukan di server Hugging Face."
        
    f = io.StringIO()
    try:
        with contextlib.redirect_stdout(f):
            # hasil akan berisi path file "output_ready.mp4" atau "output_faceless.mp4"
            hasil = deteksi_faceless_video(video_path)
        
        log_tertangkap = f.getvalue()
        
        # Kirim file video hasil sensor dan log analitik ke UI Gradio
        if hasil and os.path.exists(hasil):
            return hasil, log_tertangkap
        else:
            return None, log_tertangkap + "\n\n⚠️ Gagal memuat file hasil render video."
        
    except Exception as e:
        error_msg = traceback.format_exc()
        return None, f"💥 Terjadi Eror pada Modul Wajah:\n\n{error_msg}"

with gr.Blocks(title="AI Faceless Video Filter") as web_filter:
    gr.Markdown("# 🎬 AI Faceless Video Filter & Moderator")
    gr.Markdown("Aplikasi pemindai dan penutup wajah otonom berbasis MediaPipe & MCP.")
    
    with gr.Row():
        with gr.Column():
            video_input = gr.Textbox(label="Nama File Video", placeholder="video_saya.mp4")
            btn = gr.Button("Filter Wajah Video dengan AI", variant="primary")
        with gr.Column():
            output_video = gr.Video(label="Hasil Video Tanpa Wajah (Faceless)")
            status_text = gr.Textbox(label="Log Status / Pesan Eror AI Backend", interactive=False, lines=15)
            
    btn.click(fn=wrapper_deteksi_wajah, inputs=video_input, outputs=[output_video, status_text])

if os.environ.get("SPACE_ID") or len(sys.argv) < 2:
    web_filter.launch(share=False)
else:
    print("💡 Cara penggunaan skrip otomatis di terminal:\n   python filter_faceless.py nama_video.mp4")
