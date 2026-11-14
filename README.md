# AI Music Generator Using Deep Learning (LSTM)

An intelligent neural composition web application that learns musical patterns from MIDI datasets and generates new polyphonic melodies and chord sequences using Deep Learning (LSTM Recurrent Neural Networks).

![Dark Cyber Studio UI](frontend/css/style.css)

---

## 🎵 Key Features

### 1. Multi-Genre MIDI Dataset Collection
- Pre-populated starter datasets across 4 genres:
  - **Classical:** Beethoven Ode to Joy theme, Bach Counterpoint Inventions, Mozart Sonatas.
  - **Jazz:** ii-V-I 7th chords, Autumn Leaves harmonic cycles, Blues scales.
  - **Piano:** Romantic Nocturnes, A minor Waltzes, Satie Impressionist progressions.
  - **Instrumental:** Celtic guitar themes, Flute pastorals, Violin strings.
- Interactive drag-and-drop file upload with automated validation.
- Metadata extraction: Note count, duration, tempo (BPM), instrument program IDs.

### 2. Polyphonic Music Preprocessing
- Powered by `music21` and `pretty_midi`.
- Extracts individual notes, chords (dot-separated pitch notation), and quantized durations.
- Converts musical events into sequential integer tokens.
- Sliding window sequence generation with configurable window length (8 to 64 notes).
- Normalized input sequence preparation for Deep Learning.

### 3. Dual LSTM Deep Learning Architecture
- **Keras / TensorFlow Sequential Model Support:** Dual LSTM layers with Dropout and Dense softmax projection.
- **Native Vectorized NumPy LSTM Engine:** Includes full Forget, Input, Cell candidate, and Output gates, Backpropagation Through Time (BPTT), Adam optimizer with gradient clipping, and Cross-Entropy loss computation.
- Real-time training progress updates (epoch, loss, elapsed time) via background threads.
- Dynamic loss curve visualization using Chart.js.

### 4. Creative AI Music Generation
- Temperature-controlled Softmax sampling ($T \in [0.1, 1.5]$) to balance harmony and artistic variation.
- Custom length selection (30 to 200 notes).
- Configurable tempo (60 to 160 BPM).
- Realistic human velocity simulation ($82 - 104$ MIDI velocity).

### 5. MIDI & WAV Audio Synthesis
- Real-time conversion of generated note sequences to standard `.mid` files.
- WAV audio generation via FluidSynth or built-in acoustic harmonic synthesis.
- Studio audio player with Play, Pause, Stop, Seek, Volume, and Mute controls.
- Web Audio API real-time spectrum waveform visualizer.
- Interactive Virtual Piano Roll keyboard highlighting active notes.
- Complete download history with one-click direct MIDI and WAV downloads.

---

## 🛠️ Technology Stack

- **Backend:** Python 3.10+, Flask 3.x, Flask-CORS
- **Deep Learning:** LSTM / RNN (TensorFlow / Keras + Native NumPy LSTM Engine)
- **Music Processing:** `music21`, `pretty_midi`, `mido`, `scipy`
- **Frontend:** HTML5, CSS3, JavaScript (ES6+), Bootstrap 5, Chart.js, Font Awesome
- **Audio Rendering:** FluidSynth SoundFont synthesis / PrettyMIDI acoustic synthesis

---

## 📁 Project Folder Structure

```text
AI-Music-Generator/
├── backend/
│   ├── app.py                # Flask REST API & Web Server
│   ├── dataset_manager.py    # MIDI dataset scanning, validation, & seed generator
│   ├── preprocessing.py      # music21 parsing, tokenization & sequence windowing
│   ├── model.py              # LSTM Neural Network (TF + Native NumPy BPTT)
│   ├── generator.py          # Temperature-based AI note sampling & MIDI export
│   └── audio_converter.py    # MIDI-to-WAV audio synthesis (FluidSynth & Scipy)
├── frontend/
│   ├── index.html            # Studio Single Page Application (8 views)
│   ├── css/
│   │   └── style.css         # Dark theme, glassmorphic cards & animations
│   └── js/
│       └── app.js            # Frontend controller, Chart.js, Web Audio visualizer
├── datasets/
│   ├── classical/            # Classical MIDI datasets
│   ├── jazz/                 # Jazz MIDI datasets
│   ├── piano/                # Piano MIDI datasets
│   └── instrumental/         # Instrumental MIDI datasets
├── models/                   # Saved model weights (.npz, .keras) & vocabulary caches
├── generated_music/          # Output generated .mid, .wav, and .json metadata
├── uploads/                  # User-uploaded MIDI files
├── requirements.txt          # Python dependencies
└── README.md                 # Documentation
```

---

## 🚀 Installation & Running

### 1. Clone or Navigate to Directory
```bash
cd AI-Music-Generator
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Start the Web Application
You can run it in any of the following ways:

**Option A (Recommended - 1-Click Python Launcher):**
```bash
python run_app.py
```
*(This automatically picks an open port like `5001` so it does not conflict with other apps like FAQ Chatbot on `5000`, and opens your default browser!)*

**Option B (Windows 1-Click Batch File):**
Double click `start_studio.bat` inside the `AI-Music-Generator` folder.

**Option C (Direct Backend App):**
```bash
python backend/app.py
```

### 4. Open in Browser
If your browser does not open automatically, navigate to:
```
http://localhost:5001
```
*(Or `http://127.0.0.1:5001`)*

---

## 🎹 REST API Reference

| Endpoint | Method | Description |
|---|---|---|
| `GET /api/status` | GET | System status, counts, engine, and current training state |
| `GET /api/datasets` | GET | List available MIDI datasets grouped by genre |
| `POST /api/upload` | POST | Upload and validate a new MIDI file (`multipart/form-data`) |
| `POST /api/preprocess` | POST | Run note tokenization for selected genre |
| `GET /api/preprocessing/status` | GET | Check preprocessing cache for a genre |
| `POST /api/train` | POST | Start LSTM model training in background thread |
| `GET /api/train/status` | GET | Poll real-time training progress and loss history |
| `POST /api/train/stop` | POST | Stop active training run |
| `POST /api/generate` | POST | Generate new music sequence, export MIDI & WAV |
| `GET /api/history` | GET | List all generated music tracks |
| `GET /api/download/midi/<filename>` | GET | Download generated MIDI file |
| `GET /api/download/wav/<filename>` | GET | Download synthesized WAV audio |
| `GET /api/stream/<filename>` | GET | Stream audio for in-browser HTML5 player |
| `DELETE /api/history/<filename>` | DELETE | Remove generated track files from storage |

---

## 🎼 The 8 Studio Dashboard Views

1. **Home Dashboard:** Quick stats, hero banner, live loss curve, and recent generated compositions.
2. **Upload MIDI Dataset:** Drag & drop MIDI uploader with automated note verification and dataset explorer.
3. **Data Preprocessing:** Extract pitch tokens, examine unique vocabulary size, sequence windowing.
4. **Train AI Model:** Configure epochs, batch size, learning rate, and observe live loss graphs.
5. **Generate Music:** Choose genre, note sequence length, tempo, and temperature creativity slider.
6. **Music Player:** Sleek dark music studio player with real-time waveform visualizer and virtual piano roll.
7. **Download History:** Audio library with inline playback and instant MIDI / WAV downloads.
8. **About Project:** Architecture details, LSTM gate explanations, and mathematical formulation.
