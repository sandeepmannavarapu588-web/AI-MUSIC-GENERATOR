"""
Flask Backend Application for AI Music Generator Using Deep Learning (LSTM)
Provides REST APIs for dataset collection, preprocessing, LSTM training,
AI music generation, MIDI conversion, and audio playback.
"""

import os
import glob
import json
import time
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS

from dataset_manager import DatasetManager, GENRES
from preprocessing import MidiPreprocessor
from model import ModelManager
from generator import MusicGenerator
from audio_converter import AudioConverter

app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GENERATED_DIR = os.path.join(BASE_DIR, "generated_music")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")

# Initialize backend service managers
dataset_mgr = DatasetManager(BASE_DIR)
preprocessor = MidiPreprocessor(BASE_DIR)
model_mgr = ModelManager(BASE_DIR)
generator = MusicGenerator(BASE_DIR)
audio_converter = AudioConverter(BASE_DIR)


# ==========================================
# Frontend Routes
# ==========================================
@app.route("/")
def index():
    if os.path.exists(os.path.join(BASE_DIR, "index.html")):
        return send_from_directory(BASE_DIR, "index.html")
    return send_from_directory(app.static_folder, "index.html")


@app.route("/css/<path:filename>")
def serve_css(filename):
    css_dir = os.path.join(BASE_DIR, "css") if os.path.exists(os.path.join(BASE_DIR, "css")) else os.path.join(app.static_folder, "css")
    return send_from_directory(css_dir, filename)


@app.route("/js/<path:filename>")
def serve_js(filename):
    js_dir = os.path.join(BASE_DIR, "js") if os.path.exists(os.path.join(BASE_DIR, "js")) else os.path.join(app.static_folder, "js")
    return send_from_directory(js_dir, filename)


# ==========================================
# Module 1 & System Status APIs
# ==========================================
@app.route("/api/status", methods=["GET"])
def get_system_status():
    """Return overarching system metrics for Dashboard."""
    datasets_info = dataset_mgr.list_datasets()
    generated_midis = glob.glob(os.path.join(GENERATED_DIR, "*.mid"))
    generated_wavs = glob.glob(os.path.join(GENERATED_DIR, "*.wav"))
    
    # Check trained models count
    model_files = glob.glob(os.path.join(MODELS_DIR, "lstm_model_*.npz")) + glob.glob(os.path.join(MODELS_DIR, "lstm_model_*.keras"))
    
    training_status = model_mgr.get_status()

    return jsonify({
        "status": "online",
        "total_midi_files": datasets_info["total_files"],
        "total_generated_music": len(generated_midis),
        "total_wav_audio": len(generated_wavs),
        "trained_models_count": len(model_files),
        "training": training_status,
        "engine": training_status["engine_type"],
        "has_fluidsynth": audio_converter.has_fluidsynth
    })


@app.route("/api/datasets", methods=["GET"])
def get_datasets():
    """List all available datasets and file details."""
    data = dataset_mgr.list_datasets()
    return jsonify(data)


@app.route("/api/upload", methods=["POST"])
def upload_midi():
    """Upload MIDI file with validation."""
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file part in request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No selected file"}), 400

    genre = request.form.get("genre", "piano")
    success, res = dataset_mgr.save_uploaded_file(file, genre=genre)

    if not success:
        return jsonify({"success": False, "error": res}), 400

    return jsonify({
        "success": True,
        "message": f"Successfully uploaded {res['filename']} to genre '{res['genre']}'",
        "data": res
    })


# ==========================================
# Module 2: Data Preprocessing APIs
# ==========================================
@app.route("/api/preprocess", methods=["POST"])
def run_preprocessing():
    """Trigger data preprocessing for a genre or all genres."""
    data = request.get_json() or {}
    genre = data.get("genre", "piano")
    seq_length = int(data.get("seq_length", 16))

    try:
        stats = preprocessor.process_genre(genre=genre, seq_length=seq_length)
        return jsonify({
            "success": True,
            "message": f"Preprocessed {stats['files_processed_count']} files for genre '{genre}'.",
            "stats": stats
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/preprocessing/status", methods=["GET"])
def get_preprocessing_status():
    """Get preprocessing status for a genre."""
    genre = request.args.get("genre", "piano")
    exists, stats = preprocessor.get_preprocessing_status(genre=genre)
    return jsonify({
        "genre": genre,
        "preprocessed": exists,
        "stats": stats
    })


# ==========================================
# Module 3: LSTM Model Development APIs
# ==========================================
@app.route("/api/train", methods=["POST"])
def train_model():
    """Start training the LSTM model."""
    data = request.get_json() or {}
    genre = data.get("genre", "piano")
    epochs = int(data.get("epochs", 25))
    batch_size = int(data.get("batch_size", 8))
    learning_rate = float(data.get("learning_rate", 0.005))
    seq_length = int(data.get("seq_length", 16))

    success, msg = model_mgr.start_training(
        genre=genre,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seq_length=seq_length
    )

    if not success:
        return jsonify({"success": False, "error": msg}), 400

    return jsonify({"success": True, "message": msg})


@app.route("/api/train/status", methods=["GET"])
def get_train_status():
    """Get real-time training progress, current loss, and loss curve."""
    return jsonify(model_mgr.get_status())


@app.route("/api/train/stop", methods=["POST"])
def stop_train():
    """Stop ongoing training."""
    success, msg = model_mgr.stop_training()
    return jsonify({"success": success, "message": msg})


# ==========================================
# Module 4 & 5: AI Music Generation & MIDI
# ==========================================
@app.route("/api/generate", methods=["POST"])
def generate_music():
    """Generate new music using trained LSTM model."""
    data = request.get_json() or {}
    genre = data.get("genre", "piano")
    length = int(data.get("length", 60))
    temperature = float(data.get("temperature", 0.8))
    tempo = int(data.get("tempo", 110))

    try:
        # Step 1: AI sequence generation using LSTM
        generated_tokens = generator.generate_sequence(
            model_manager=model_mgr,
            genre=genre,
            length=length,
            temperature=temperature
        )

        # Step 2: Convert tokens to standard MIDI file
        midi_meta = generator.tokens_to_midi(
            tokens=generated_tokens,
            genre=genre,
            tempo_bpm=tempo
        )

        # Step 3: Convert MIDI to WAV audio
        midi_filepath = midi_meta["midi_path"]
        audio_meta = audio_converter.midi_to_wav(midi_filepath)

        # Combine response
        combined_meta = {**midi_meta, **audio_meta}
        combined_meta["temperature"] = temperature
        combined_meta["requested_length"] = length

        # Update JSON record
        meta_filepath = os.path.join(GENERATED_DIR, f"{os.path.splitext(midi_meta['filename'])[0]}.json")
        with open(meta_filepath, "w", encoding="utf-8") as f:
            json.dump(combined_meta, f, indent=2)

        return jsonify({
            "success": True,
            "message": f"Successfully generated {midi_meta['notes_count']} notes in {genre} style!",
            "track": combined_meta
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ==========================================
# Module 6: Audio, History & Downloads
# ==========================================
@app.route("/api/history", methods=["GET"])
def get_history():
    """List all generated music with playback and download info."""
    json_files = glob.glob(os.path.join(GENERATED_DIR, "*.json"))
    history = []

    for jf in sorted(json_files, key=os.path.getmtime, reverse=True):
        try:
            with open(jf, "r", encoding="utf-8") as f:
                item = json.load(f)
                
                # Verify files exist on disk
                midi_name = item.get("filename")
                wav_name = item.get("wav_filename", f"{os.path.splitext(midi_name)[0]}.wav")
                
                midi_exists = os.path.exists(os.path.join(GENERATED_DIR, midi_name))
                wav_exists = os.path.exists(os.path.join(GENERATED_DIR, wav_name))
                
                item["has_midi"] = midi_exists
                item["has_wav"] = wav_exists
                item["wav_filename"] = wav_name
                history.append(item)
        except Exception:
            continue

    return jsonify({"history": history, "total": len(history)})


@app.route("/api/download/midi/<filename>", methods=["GET"])
def download_midi(filename):
    """Download generated MIDI file."""
    filepath = os.path.join(GENERATED_DIR, filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404
    return send_file(filepath, as_attachment=True, download_name=filename, mimetype="audio/midi")


@app.route("/api/download/wav/<filename>", methods=["GET"])
def download_wav(filename):
    """Download generated WAV audio file."""
    filepath = os.path.join(GENERATED_DIR, filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404
    return send_file(filepath, as_attachment=True, download_name=filename, mimetype="audio/wav")


@app.route("/api/stream/<filename>", methods=["GET"])
def stream_audio(filename):
    """Stream WAV audio for HTML5 audio player."""
    filepath = os.path.join(GENERATED_DIR, filename)
    if not os.path.exists(filepath):
        # If WAV does not exist but MIDI does, auto-synthesize
        midi_filename = f"{os.path.splitext(filename)[0]}.mid"
        midi_path = os.path.join(GENERATED_DIR, midi_filename)
        if os.path.exists(midi_path):
            try:
                audio_converter.midi_to_wav(midi_path, filepath)
            except Exception:
                return jsonify({"error": "Failed to synthesize audio"}), 500
        else:
            return jsonify({"error": "Audio file not found"}), 404

    return send_file(filepath, mimetype="audio/wav")


@app.route("/api/history/<filename>", methods=["DELETE"])
def delete_generated_track(filename):
    """Delete a generated track (MIDI, WAV, and metadata)."""
    base = os.path.splitext(filename)[0]
    for ext in [".mid", ".wav", ".json"]:
        target = os.path.join(GENERATED_DIR, f"{base}{ext}")
        if os.path.exists(target):
            try:
                os.remove(target)
            except Exception:
                pass
    return jsonify({"success": True, "message": f"Deleted track {base}"})


if __name__ == "__main__":
    import socket
    import sys

    # Support custom port via command-line arg or PORT env var; default to 5001 to avoid conflict with faq-chatbot (port 5000)
    port = int(os.environ.get("PORT", 5001))
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])

    # Check if selected port is in use; if so, pick next available port
    def is_port_in_use(p):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('127.0.0.1', p)) == 0

    if is_port_in_use(port):
        for fallback_port in [5001, 5002, 5050, 8080]:
            if not is_port_in_use(fallback_port):
                port = fallback_port
                break

    print("=" * 65)
    print("  🎵  AI Music Generator Using Deep Learning (LSTM)")
    print("=" * 65)
    print(f"  Deep Learning Engine : {model_mgr.engine_type}")
    print(f"  Base Directory       : {BASE_DIR}")
    print(f"  FluidSynth Available : {audio_converter.has_fluidsynth}")
    print("-" * 65)
    print(f"  👉 STUDIO URL: http://127.0.0.1:{port}")
    print(f"  👉 (or http://localhost:{port})")
    print("=" * 65)
    app.run(host="0.0.0.0", port=port, debug=False)
