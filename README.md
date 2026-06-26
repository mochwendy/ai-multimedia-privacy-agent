# 📌 Autonomous Multimedia Content Moderator: Privacy-Preserving AI Agent

**Developed by:** Moch Wendy  
**Architecture Stack:** Python, MediaPipe (Tasks API v2), OpenCV, FFmpeg, NumPy, Gradio UI, MCP Architecture  

---

## 🚀 Project Overview

This repository houses the production-ready source code for an **Autonomous Multimodal AI Content Moderator Agent**. Built specifically for high-throughput compliance systems, the system operates dual native pipelines—simultaneously parsing **high-frequency visual streams** for PII redacting (Privacy Preservation) and **discrete acoustic waveforms** to filter background copyrighted tracks or high-volume audio distortion.# 📌 Autonomous Multimedia Content Moderator: Privacy-Preserving AI Agent

**Developed by:** Moch Wendy  
**Architecture Stack:** Python, MediaPipe (Tasks API v2), OpenCV, FFmpeg, NumPy, Gradio UI, MCP Architecture  

---

## 🚀 Project Overview

This repository houses the production-ready source code for an **Autonomous Multimodal AI Content Moderator Agent**. Built specifically for high-throughput compliance systems, the system operates dual native pipelines—simultaneously parsing **high-frequency visual streams** for PII redacting (Privacy Preservation) and **discrete acoustic waveforms** to filter background copyrighted tracks or high-volume audio distortion.

### 🎯 Core Capabilities:
*   **Privacy-Preserving Vision Engine:** Ingests compressed H.264 video arrays, feeds normalized frames to an embedded ultra-lightweight TFLite face topology graph, extracts absolute bounding boxes, and performs inline frame mutation via an adaptive Gaussian Blur kernel topology.
*   **Zero-Dependency DSP Audio Engine:** Bypasses typical heavy native packages (e.g., *Librosa*, *PyAudioAnalysis*) in favor of high-speed vector mechanics using *NumPy* and *FFmpeg*, delivering scalable execution suited for tight Kubernetes/Docker node resource limits.
*   **Deterministic Decision Matrix:** Concludes state execution dynamically, flagging content directly into `APPROVED_CLEAN`, `APPROVED_WITH_SENSOR`, or `REJECTED` conditions based on multimodal sensory logs.

---

## 🏗️ Multimodal Pipeline Architecture

| Pipeline Stage | Technology Applied | Engineering Detail / Vector Process |
| :--- | :--- | :--- |
| **Audio-Video Demuxing** | FFmpeg Wrapper | Extracts 16kHz mono-channel signed 16-bit PCM WAV byte-streams via native sub-process pipes, bypassing container-level OS codec boundaries. |
| **Acoustic Structural Filter** | NumPy Math Array | Segments raw arrays into 0.25s intervals ($4000 \text{ samples}$). Calculates running RMS to Decibel ($dB$) conversion and parses the **Coefficient of Variation (CV)**. Music tracks with stable periodic rhythm vectors are dynamically filtered out when $CV < 0.45$ and $dB > 45$. |
| **Geometric Inference** | MediaPipe Tasks Vision | Implements the latest `blaze_face_short_range.tflite` graph executor for localized, lightning-fast edge inference at $\ge 0.5$ confidence floors. |
| **Privacy Mask Mutation** | OpenCV Matrix Mapping | Isolates spatial ROIs (Region of Interest) and transforms high-frequency facial matrices using an intensive $99 \times 99$ Gaussian convolution matrix. |
| **Acoustic Remuxing & Sync** | FFmpeg H.264 Encoder | Encodes sanitized image sequences with H.264 native baseline levels, safely copying original clean verbal streams back without desynchronization. |

---

## 📊 Performance Analytics & Console Output Log

The agent reports real-time metadata tracking directly via hijacked `stdout` terminal hooks streamed directly into the frontend interface:

```text
📊 HASIL FILTER AI MULTIMEDIA AGENT (VISION + AUDIO + MUSIC DETECTOR):
------------------------------------------------------------------
• Nama File Video  : compliance_payload_test.mp4
• Total Frame      : 209 frame (~10.38 detik)
• Status Audio     : 🔊 AUDIO NORMAL (56.35 dB) - [✅ MUSIK: Bersih / Hanya Percakapan]
• Kategori Visual  : 👤 HUMAN FACE VIDEO
• Status Filter    : WAJAH BERHASIL DI-BLUR
------------------------------------------------------------------
🎯 KESIMPULAN AGEN : APPROVED WITH SENSOR (LOLOS: Wajah Terproteksi Gaussian Blur) 🟢
------------------------------------------------------------------
