/**
 * AI Music Generator Studio - Full Client Application Controller
 * Handles SPA navigation, REST API communication, live training chart polling,
 * music generation, Web Audio API visualization, and history management.
 */

// Global Application State
const AppState = {
  currentView: 'dashboard',
  selectedGenre: 'piano',
  isTraining: false,
  trainPollInterval: null,
  lossChart: null,
  dashLossChart: null,
  prepChart: null,
  
  // Audio Player State
  currentTrack: null,
  trackList: [],
  currentTrackIndex: -1,
  audioElement: null,
  audioContext: null,
  analyser: null,
  audioSource: null,
  isVisualizerRunning: false,
  animationFrameId: null
};

// Initialize Application on DOM Ready
document.addEventListener('DOMContentLoaded', () => {
  initNavigation();
  initAudioPlayer();
  initCharts();
  initVirtualPiano();
  initEventListeners();
  
  // Initial Data Load
  loadSystemStatus();
  loadDatasets();
  loadHistory();
});

/* ===================================================================
   1. SPA Navigation & Views
   =================================================================== */
function initNavigation() {
  const navItems = document.querySelectorAll('.sidebar-nav .nav-item');
  navItems.forEach(item => {
    item.addEventListener('click', (e) => {
      e.preventDefault();
      const viewId = item.getAttribute('data-view');
      switchView(viewId);
    });
  });

  // Mobile sidebar toggle
  const toggleBtn = document.getElementById('sidebarToggle');
  const sidebar = document.getElementById('appSidebar');
  if (toggleBtn && sidebar) {
    toggleBtn.addEventListener('click', () => {
      sidebar.classList.toggle('open');
    });
  }

  // Quick generate button
  document.getElementById('btnQuickGenerate').addEventListener('click', () => {
    switchView('generate');
  });

  // Topbar player click opens player view
  document.getElementById('topbarPlayer').addEventListener('click', () => {
    switchView('player');
  });
}

function switchView(viewId) {
  AppState.currentView = viewId;

  // Update nav item active states
  document.querySelectorAll('.sidebar-nav .nav-item').forEach(item => {
    if (item.getAttribute('data-view') === viewId) {
      item.classList.add('active');
    } else {
      item.classList.remove('active');
    }
  });

  // Update page-view visibility
  document.querySelectorAll('.page-view').forEach(view => {
    view.classList.remove('active');
  });
  const targetView = document.getElementById(`view-${viewId}`);
  if (targetView) {
    targetView.classList.add('active');
  }

  // Close mobile sidebar if open
  const sidebar = document.getElementById('appSidebar');
  if (sidebar && sidebar.classList.contains('open')) {
    sidebar.classList.remove('open');
  }

  // Update top bar title
  const titleMap = {
    'dashboard': '<i class="fa-solid fa-chart-line text-info"></i> Home Dashboard',
    'upload': '<i class="fa-solid fa-cloud-arrow-up text-primary"></i> Upload MIDI Dataset',
    'preprocess': '<i class="fa-solid fa-sliders text-warning"></i> Data Preprocessing',
    'train': '<i class="fa-solid fa-brain text-purple"></i> Train AI Model (LSTM)',
    'generate': '<i class="fa-solid fa-wand-magic-sparkles text-info"></i> Generate Music',
    'player': '<i class="fa-solid fa-circle-play text-success"></i> Music Studio Player',
    'history': '<i class="fa-solid fa-clock-rotate-left text-primary"></i> Download History',
    'about': '<i class="fa-solid fa-graduation-cap text-cyan"></i> About AI Music Generator'
  };
  document.getElementById('currentViewTitle').innerHTML = titleMap[viewId] || 'AI Music Studio';

  // Trigger view-specific data refresh
  if (viewId === 'dashboard') {
    loadSystemStatus();
    loadHistory();
  } else if (viewId === 'upload') {
    loadDatasets();
  } else if (viewId === 'history') {
    loadHistory();
  } else if (viewId === 'preprocess') {
    checkPrepStatus();
  }
}

/* ===================================================================
   2. Charts Initialization (Chart.js)
   =================================================================== */
function initCharts() {
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 400 },
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: '#161828',
        borderColor: '#7928ca',
        borderWidth: 1,
        titleColor: '#00dfd8',
        bodyColor: '#fff',
        displayColors: false
      }
    },
    scales: {
      x: {
        grid: { color: 'rgba(255, 255, 255, 0.05)' },
        ticks: { color: '#64748b' }
      },
      y: {
        grid: { color: 'rgba(255, 255, 255, 0.05)' },
        ticks: { color: '#64748b' }
      }
    }
  };

  // 1. Dashboard Loss Chart
  const dashCtx = document.getElementById('dashLossChart');
  if (dashCtx) {
    AppState.dashLossChart = new Chart(dashCtx.getContext('2d'), {
      type: 'line',
      data: {
        labels: [1, 2, 3, 4, 5],
        datasets: [{
          label: 'Cross-Entropy Loss',
          data: [4.12, 3.84, 3.52, 3.31, 3.10],
          borderColor: '#00dfd8',
          backgroundColor: 'rgba(0, 223, 216, 0.1)',
          fill: true,
          tension: 0.35,
          pointRadius: 4,
          pointBackgroundColor: '#00dfd8'
        }]
      },
      options: chartOptions
    });
  }

  // 2. Training View Loss Chart
  const trainCtx = document.getElementById('trainLossChart');
  if (trainCtx) {
    AppState.lossChart = new Chart(trainCtx.getContext('2d'), {
      type: 'line',
      data: {
        labels: [],
        datasets: [{
          label: 'Training Loss',
          data: [],
          borderColor: '#7928ca',
          backgroundColor: 'rgba(121, 40, 202, 0.15)',
          fill: true,
          tension: 0.35,
          pointRadius: 4,
          pointBackgroundColor: '#a855f7'
        }]
      },
      options: chartOptions
    });
  }

  // 3. Preprocessing Top Tokens Bar Chart
  const prepCtx = document.getElementById('prepTokensChart');
  if (prepCtx) {
    AppState.prepChart = new Chart(prepCtx.getContext('2d'), {
      type: 'bar',
      data: {
        labels: ['C4', 'E4', 'G4', 'C5', 'D4', 'B4', 'A4', 'F4'],
        datasets: [{
          label: 'Frequency',
          data: [12, 10, 8, 8, 6, 5, 5, 4],
          backgroundColor: 'rgba(121, 40, 202, 0.6)',
          borderColor: '#7928ca',
          borderWidth: 1,
          borderRadius: 4
        }]
      },
      options: {
        ...chartOptions,
        scales: {
          x: { grid: { display: false }, ticks: { color: '#94a3b8' } },
          y: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#94a3b8' } }
        }
      }
    });
  }
}

/* ===================================================================
   3. System Status & Metrics
   =================================================================== */
async function loadSystemStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();

    document.getElementById('statTotalMidis').textContent = data.total_midi_files || 0;
    document.getElementById('statTrainedModels').textContent = data.trained_models_count || 0;
    document.getElementById('statGeneratedCount').textContent = data.total_generated_music || 0;
    
    const engineLabel = data.engine || 'LSTM Active';
    document.getElementById('engineBadge').textContent = `Engine: ${engineLabel}`;
    
    if (data.training && data.training.is_training) {
      document.getElementById('statModelStatus').textContent = 'Training...';
      document.getElementById('statModelStatus').style.color = '#ffab00';
    } else {
      document.getElementById('statModelStatus').textContent = 'Ready';
      document.getElementById('statModelStatus').style.color = '#00e676';
    }

    // Sync dashboard loss graph if history exists
    if (data.training && data.training.loss_history && data.training.loss_history.length > 0) {
      const history = data.training.loss_history;
      const labels = history.map(h => `Ep ${h.epoch}`);
      const losses = history.map(h => h.loss);
      if (AppState.dashLossChart) {
        AppState.dashLossChart.data.labels = labels;
        AppState.dashLossChart.data.datasets[0].data = losses;
        AppState.dashLossChart.update();
      }
    }
  } catch (err) {
    console.error('Failed to load status:', err);
  }
}

/* ===================================================================
   4. Module 1: Dataset Management & Upload
   =================================================================== */
async function loadDatasets() {
  const tbody = document.getElementById('datasetsTableBody');
  if (!tbody) return;

  try {
    const res = await fetch('/api/datasets');
    const data = await res.json();

    tbody.innerHTML = '';
    let hasFiles = false;

    for (const [genre, gData] of Object.entries(data.genres)) {
      if (gData.files && gData.files.length > 0) {
        hasFiles = true;
        gData.files.forEach(file => {
          const row = document.createElement('tr');
          row.innerHTML = `
            <td>
              <div class="d-flex align-items-center gap-2">
                <i class="fa-solid fa-file-audio text-secondary"></i>
                <span class="fw-medium">${file.filename}</span>
              </div>
            </td>
            <td><span class="badge-studio badge-${genre}">${genre}</span></td>
            <td>${file.notes_count}</td>
            <td>${file.duration_sec}s</td>
            <td>${file.tempo} BPM</td>
            <td>${file.size_kb} KB</td>
          `;
          tbody.appendChild(row);
        });
      }
    }

    if (!hasFiles) {
      tbody.innerHTML = `<tr><td colspan="6" class="text-center text-muted">No dataset files found.</td></tr>`;
    }
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="6" class="text-center text-danger">Error loading datasets: ${err.message}</td></tr>`;
  }
}

/* ===================================================================
   5. Module 2: Preprocessing
   =================================================================== */
async function checkPrepStatus() {
  const genre = document.getElementById('prepGenreSelect').value;
  try {
    const res = await fetch(`/api/preprocessing/status?genre=${genre}`);
    const data = await res.json();
    if (data.preprocessed && data.stats) {
      updatePrepUI(data.stats);
    }
  } catch (err) {
    console.log('No cached preprocessing:', err);
  }
}

function updatePrepUI(stats) {
  document.getElementById('prepTotalTokens').textContent = stats.total_tokens || 0;
  document.getElementById('prepUniqueTokens').textContent = stats.unique_tokens || 0;
  document.getElementById('prepTotalSequences').textContent = stats.total_sequences || 0;
  document.getElementById('prepStatsBadge').textContent = `Processed (${stats.genre})`;

  // Update bar chart
  if (stats.top_tokens && stats.top_tokens.length > 0 && AppState.prepChart) {
    AppState.prepChart.data.labels = stats.top_tokens.map(t => t.token.split('_')[1] || t.token);
    AppState.prepChart.data.datasets[0].data = stats.top_tokens.map(t => t.count);
    AppState.prepChart.update();

    // Update tokens list preview
    const tokenList = document.getElementById('prepTokensList');
    tokenList.innerHTML = stats.top_tokens.map(t => `
      <span class="badge bg-dark border border-secondary text-info p-2 font-monospace">
        ${t.token} <span class="text-muted">(${t.count})</span>
      </span>
    `).join('');
  }
}

/* ===================================================================
   6. Module 3: LSTM Model Training
   =================================================================== */
function startTrainingPoll() {
  if (AppState.trainPollInterval) clearInterval(AppState.trainPollInterval);

  AppState.trainPollInterval = setInterval(async () => {
    try {
      const res = await fetch('/api/train/status');
      const data = await res.json();

      AppState.isTraining = data.is_training;

      // Update UI elements
      document.getElementById('trainStatusText').textContent = data.status_message || 'Training...';
      document.getElementById('trainCurrentLoss').textContent = data.current_loss || '0.00';
      document.getElementById('trainEpochCounter').textContent = `Epoch ${data.current_epoch} / ${data.total_epochs}`;

      const percent = data.total_epochs > 0 ? Math.round((data.current_epoch / data.total_epochs) * 100) : 0;
      document.getElementById('trainProgressBar').style.width = `${percent}%`;
      document.getElementById('trainProgressPercent').textContent = `${percent}%`;

      // Update Chart.js curve
      if (data.loss_history && data.loss_history.length > 0 && AppState.lossChart) {
        AppState.lossChart.data.labels = data.loss_history.map(h => `Ep ${h.epoch}`);
        AppState.lossChart.data.datasets[0].data = data.loss_history.map(h => h.loss);
        AppState.lossChart.update();

        // Also update dashboard chart
        if (AppState.dashLossChart) {
          AppState.dashLossChart.data.labels = data.loss_history.map(h => `Ep ${h.epoch}`);
          AppState.dashLossChart.data.datasets[0].data = data.loss_history.map(h => h.loss);
          AppState.dashLossChart.update();
        }
      }

      if (!data.is_training) {
        clearInterval(AppState.trainPollInterval);
        AppState.trainPollInterval = null;
        document.getElementById('btnStartTrain').style.display = 'block';
        document.getElementById('btnStopTrain').style.display = 'none';
        showToast(data.status_message, 'success');
        loadSystemStatus();
      }
    } catch (err) {
      console.error('Error polling train status:', err);
    }
  }, 900);
}

/* ===================================================================
   7. Module 4 & 5: AI Music Generation
   =================================================================== */
async function generateMusicAction() {
  const genre = AppState.selectedGenre || 'piano';
  const length = parseInt(document.getElementById('genLengthRange').value, 10);
  const temperature = parseFloat(document.getElementById('genTempRange').value);
  const tempo = parseInt(document.getElementById('genTempoRange').value, 10);

  const btn = document.getElementById('btnGenerateMusic');
  const loader = document.getElementById('genLoadingState');
  const resultCard = document.getElementById('genResultCard');

  btn.disabled = true;
  loader.style.display = 'block';
  resultCard.style.display = 'none';

  try {
    const res = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ genre, length, temperature, tempo })
    });
    const data = await res.json();

    if (!res.ok || !data.success) {
      throw new Error(data.error || 'Generation failed');
    }

    const track = data.track;
    showToast(`Composed ${track.notes_count} notes successfully!`, 'success');

    // Populate result card
    document.getElementById('resTrackTitle').textContent = track.filename;
    document.getElementById('resGenreBadge').textContent = track.genre;
    document.getElementById('resGenreBadge').className = `badge-studio badge-${track.genre}`;
    document.getElementById('resNotesCount').textContent = `${track.notes_count} notes`;
    document.getElementById('resDuration').textContent = `${track.duration_sec}s`;
    document.getElementById('resFilesize').textContent = `${track.filesize_kb} KB`;

    // Download links
    document.getElementById('resBtnDownloadMidi').href = `/api/download/midi/${track.filename}`;
    document.getElementById('resBtnDownloadWav').href = `/api/download/wav/${track.wav_filename}`;

    // Note preview tokens
    if (track.tokens && track.tokens.length > 0) {
      document.getElementById('resTokensList').innerHTML = track.tokens.map(tok => `
        <span class="badge bg-secondary p-1 font-monospace" style="font-size: 0.75rem;">${tok}</span>
      `).join('');
    }

    resultCard.style.display = 'block';

    // Direct Play button on result card
    document.getElementById('resBtnPlayNow').onclick = () => {
      loadTrackIntoPlayer(track);
      switchView('player');
      playAudio();
    };

    // Refresh history and stats in background
    loadHistory();
    loadSystemStatus();
  } catch (err) {
    showToast(`Generation error: ${err.message}`, 'danger');
  } finally {
    btn.disabled = false;
    loader.style.display = 'none';
  }
}

/* ===================================================================
   8. Module 6: Audio Player & Waveform Visualizer
   =================================================================== */
function initAudioPlayer() {
  AppState.audioElement = document.getElementById('mainAudioElement');
  const audio = AppState.audioElement;

  const playBtn = document.getElementById('btnPlayPause');
  const stopBtn = document.getElementById('btnStopAudio');
  const seekRange = document.getElementById('playerSeekRange');
  const volRange = document.getElementById('volumeRange');
  const muteBtn = document.getElementById('btnMuteToggle');

  playBtn.addEventListener('click', togglePlayPause);
  stopBtn.addEventListener('click', stopAudio);

  // Time & Progress Updates
  audio.addEventListener('timeupdate', () => {
    if (!isNaN(audio.duration) && audio.duration > 0) {
      const progress = (audio.currentTime / audio.duration) * 100;
      seekRange.value = progress;
      document.getElementById('playerCurrentTime').textContent = formatTime(audio.currentTime);
      document.getElementById('playerTotalTime').textContent = formatTime(audio.duration);

      // Trigger Virtual Piano roll animation based on time
      triggerPianoRollNote();
    }
  });

  audio.addEventListener('loadedmetadata', () => {
    document.getElementById('playerTotalTime').textContent = formatTime(audio.duration);
    document.getElementById('playerCurrentTime').textContent = '0:00';
  });

  audio.addEventListener('ended', () => {
    stopAudio();
    // Auto-advance to next track if available
    playNextTrack();
  });

  // Seek Control
  seekRange.addEventListener('input', () => {
    if (!isNaN(audio.duration) && audio.duration > 0) {
      audio.currentTime = (seekRange.value / 100) * audio.duration;
    }
  });

  // Volume Control
  volRange.addEventListener('input', () => {
    audio.volume = volRange.value;
    updateVolumeIcon(audio.volume);
  });

  muteBtn.addEventListener('click', () => {
    audio.muted = !audio.muted;
    if (audio.muted) {
      document.getElementById('volumeIcon').className = 'fa-solid fa-volume-xmark';
    } else {
      updateVolumeIcon(audio.volume);
    }
  });

  // Next / Prev buttons
  document.getElementById('btnNextTrack').addEventListener('click', playNextTrack);
  document.getElementById('btnPrevTrack').addEventListener('click', playPrevTrack);

  // Download buttons on Player
  document.getElementById('btnPlayerDlMidi').addEventListener('click', () => {
    if (AppState.currentTrack) {
      window.location.href = `/api/download/midi/${AppState.currentTrack.filename}`;
    }
  });
  document.getElementById('btnPlayerDlWav').addEventListener('click', () => {
    if (AppState.currentTrack) {
      window.location.href = `/api/download/wav/${AppState.currentTrack.wav_filename}`;
    }
  });
}

function updateVolumeIcon(vol) {
  const icon = document.getElementById('volumeIcon');
  if (vol <= 0.05) icon.className = 'fa-solid fa-volume-off';
  else if (vol < 0.5) icon.className = 'fa-solid fa-volume-low';
  else icon.className = 'fa-solid fa-volume-high';
}

function loadTrackIntoPlayer(track) {
  AppState.currentTrack = track;
  const audio = AppState.audioElement;

  audio.src = `/api/stream/${track.wav_filename}`;
  audio.load();

  // Update Player UI
  document.getElementById('playerTrackTitle').textContent = track.filename;
  document.getElementById('playerGenreBadge').textContent = track.genre;
  document.getElementById('playerGenreBadge').className = `badge-studio badge-${track.genre} mb-2`;
  document.getElementById('playerTrackMeta').textContent = `${track.notes_count} notes • Duration: ${track.duration_sec}s • Created: ${track.created_at || 'Just now'}`;

  // Topbar player sync
  document.getElementById('topbarTrackName').textContent = track.filename;

  // Genre art icon
  const artIcon = document.getElementById('playerArtIcon');
  const icons = { 'piano': '🎹', 'classical': '🎻', 'jazz': '🎷', 'instrumental': '🎸' };
  artIcon.textContent = icons[track.genre] || '🎵';
}

function togglePlayPause() {
  const audio = AppState.audioElement;
  if (!AppState.currentTrack && AppState.trackList.length > 0) {
    loadTrackIntoPlayer(AppState.trackList[0]);
  }

  if (audio.paused) {
    playAudio();
  } else {
    pauseAudio();
  }
}

function playAudio() {
  const audio = AppState.audioElement;
  if (!audio.src) return;

  setupWebAudioContext();

  audio.play().then(() => {
    document.getElementById('playPauseIcon').className = 'fa-solid fa-pause';
    document.getElementById('topbarEqualizer').classList.add('active');
    startWaveformVisualizer();
  }).catch(err => {
    console.error('Audio play error:', err);
    showToast('Click track to start playback', 'info');
  });
}

function pauseAudio() {
  const audio = AppState.audioElement;
  audio.pause();
  document.getElementById('playPauseIcon').className = 'fa-solid fa-play';
  document.getElementById('topbarEqualizer').classList.remove('active');
}

function stopAudio() {
  const audio = AppState.audioElement;
  audio.pause();
  audio.currentTime = 0;
  document.getElementById('playerSeekRange').value = 0;
  document.getElementById('playerCurrentTime').textContent = '0:00';
  document.getElementById('playPauseIcon').className = 'fa-solid fa-play';
  document.getElementById('topbarEqualizer').classList.remove('active');
}

function playNextTrack() {
  if (AppState.trackList.length === 0) return;
  AppState.currentTrackIndex = (AppState.currentTrackIndex + 1) % AppState.trackList.length;
  loadTrackIntoPlayer(AppState.trackList[AppState.currentTrackIndex]);
  playAudio();
}

function playPrevTrack() {
  if (AppState.trackList.length === 0) return;
  AppState.currentTrackIndex = (AppState.currentTrackIndex - 1 + AppState.trackList.length) % AppState.trackList.length;
  loadTrackIntoPlayer(AppState.trackList[AppState.currentTrackIndex]);
  playAudio();
}

/* ===================================================================
   9. Web Audio API Spectrum Waveform Visualizer
   =================================================================== */
function setupWebAudioContext() {
  if (AppState.audioContext) return;

  try {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    AppState.audioContext = new AudioCtx();
    AppState.analyser = AppState.audioContext.createAnalyser();
    AppState.analyser.fftSize = 128;

    AppState.audioSource = AppState.audioContext.createMediaElementSource(AppState.audioElement);
    AppState.audioSource.connect(AppState.analyser);
    AppState.analyser.connect(AppState.audioContext.destination);
  } catch (e) {
    console.warn('Web Audio API not supported or blocked:', e);
  }
}

function startWaveformVisualizer() {
  if (AppState.isVisualizerRunning) return;
  AppState.isVisualizerRunning = true;

  const canvas = document.getElementById('audioVisualizerCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  function renderFrame() {
    if (!AppState.isVisualizerRunning) return;
    AppState.animationFrameId = requestAnimationFrame(renderFrame);

    canvas.width = canvas.parentElement.clientWidth;
    canvas.height = canvas.parentElement.clientHeight;

    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    if (AppState.analyser && !AppState.audioElement.paused) {
      const bufferLength = AppState.analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);
      AppState.analyser.getByteFrequencyData(dataArray);

      const barWidth = (width / bufferLength) * 2.2;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        const barHeight = (dataArray[i] / 255) * height * 0.9;

        // Gradient for neon audio bars
        const gradient = ctx.createLinearGradient(0, height, 0, 0);
        gradient.addColorStop(0, '#7928ca');
        gradient.addColorStop(0.5, '#0070f3');
        gradient.addColorStop(1, '#00dfd8');

        ctx.fillStyle = gradient;
        ctx.fillRect(x, height - barHeight, barWidth - 2, barHeight);

        x += barWidth;
      }
    } else {
      // Gentle idle wave animation
      const time = Date.now() * 0.003;
      ctx.beginPath();
      ctx.moveTo(0, height / 2);
      for (let x = 0; x < width; x += 5) {
        const y = height / 2 + Math.sin(x * 0.02 + time) * 6;
        ctx.lineTo(x, y);
      }
      ctx.strokeStyle = 'rgba(121, 40, 202, 0.4)';
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }

  renderFrame();
}

/* ===================================================================
   10. Virtual Piano Roll Visualizer
   =================================================================== */
const PIANO_NOTES = [
  { note: 'C3', black: false }, { note: 'C#3', black: true },
  { note: 'D3', black: false }, { note: 'D#3', black: true },
  { note: 'E3', black: false },
  { note: 'F3', black: false }, { note: 'F#3', black: true },
  { note: 'G3', black: false }, { note: 'G#3', black: true },
  { note: 'A3', black: false }, { note: 'A#3', black: true },
  { note: 'B3', black: false },
  { note: 'C4', black: false }, { note: 'C#4', black: true },
  { note: 'D4', black: false }, { note: 'D#4', black: true },
  { note: 'E4', black: false },
  { note: 'F4', black: false }, { note: 'F#4', black: true },
  { note: 'G4', black: false }, { note: 'G#4', black: true },
  { note: 'A4', black: false }, { note: 'A#4', black: true },
  { note: 'B4', black: false },
  { note: 'C5', black: false }
];

function initVirtualPiano() {
  const container = document.getElementById('virtualPianoKeys');
  if (!container) return;

  container.innerHTML = PIANO_NOTES.map(item => `
    <div class="piano-key ${item.black ? 'black' : 'white'}" data-note="${item.note}" title="${item.note}"></div>
  `).join('');
}

let lastActiveKey = null;
function triggerPianoRollNote() {
  if (AppState.audioElement.paused) return;

  // Pick a simulated note from the track's vocabulary/melodic flow
  const keys = document.querySelectorAll('.piano-key');
  if (keys.length === 0) return;

  if (lastActiveKey) {
    lastActiveKey.classList.remove('active');
  }

  // Random rhythmic activation
  if (Math.random() > 0.4) {
    const randomIdx = Math.floor(Math.random() * keys.length);
    const key = keys[randomIdx];
    key.classList.add('active');
    lastActiveKey = key;
    document.getElementById('activeNoteDisplay').textContent = `Active Note: ${key.getAttribute('data-note')}`;
  }
}

/* ===================================================================
   11. History & Downloads Management
   =================================================================== */
async function loadHistory() {
  const tbody = document.getElementById('historyTableBody');
  const dashContainer = document.getElementById('dashRecentTracks');

  try {
    const res = await fetch('/api/history');
    const data = await res.json();
    AppState.trackList = data.history || [];

    // Populate History Table View
    if (tbody) {
      if (AppState.trackList.length === 0) {
        tbody.innerHTML = `<tr><td colspan="8" class="text-center text-muted">No music generated yet.</td></tr>`;
      } else {
        tbody.innerHTML = AppState.trackList.map((track, idx) => `
          <tr>
            <td>
              <div class="d-flex align-items-center gap-2">
                <i class="fa-solid fa-compact-disc text-info"></i>
                <span class="fw-bold">${track.filename}</span>
              </div>
            </td>
            <td><span class="badge-studio badge-${track.genre}">${track.genre}</span></td>
            <td>${track.notes_count}</td>
            <td>${track.duration_sec}s</td>
            <td>${track.created_at || 'Recent'}</td>
            <td>
              <button class="btn-studio btn-primary-gradient btn-sm-studio" onclick="playTrackFromHistory(${idx})">
                <i class="fa-solid fa-play"></i> Play
              </button>
            </td>
            <td>
              <div class="d-flex gap-2">
                <a class="btn-studio btn-outline-studio btn-sm-studio" href="/api/download/midi/${track.filename}" download title="Download MIDI">
                  <i class="fa-solid fa-file-arrow-down"></i> MIDI
                </a>
                <a class="btn-studio btn-outline-studio btn-sm-studio" href="/api/download/wav/${track.wav_filename}" download title="Download WAV">
                  <i class="fa-solid fa-headphones"></i> WAV
                </a>
              </div>
            </td>
            <td>
              <button class="btn-studio btn-danger-studio btn-sm-studio" onclick="deleteTrack('${track.filename}')" title="Delete Track">
                <i class="fa-solid fa-trash-can"></i>
              </button>
            </td>
          </tr>
        `).join('');
      }
    }

    // Populate Dashboard Recent Tracks
    if (dashContainer) {
      if (AppState.trackList.length === 0) {
        dashContainer.innerHTML = `
          <div class="text-center text-muted py-5">
            <i class="fa-solid fa-music fa-2x mb-2 text-secondary"></i>
            <p>No tracks generated yet. Click "Quick Generate" to compose!</p>
          </div>
        `;
      } else {
        dashContainer.innerHTML = AppState.trackList.slice(0, 4).map((track, idx) => `
          <div class="p-3 bg-dark rounded border border-secondary d-flex justify-content-between align-items-center">
            <div>
              <div class="fw-bold">${track.filename}</div>
              <div class="text-muted small">
                <span class="badge-studio badge-${track.genre}">${track.genre}</span> • ${track.notes_count} notes • ${track.duration_sec}s
              </div>
            </div>
            <div class="d-flex gap-2">
              <button class="btn-studio btn-primary-gradient btn-sm-studio" onclick="playTrackFromHistory(${idx})">
                <i class="fa-solid fa-play"></i>
              </button>
              <a class="btn-studio btn-outline-studio btn-sm-studio" href="/api/download/midi/${track.filename}" download>
                <i class="fa-solid fa-download"></i>
              </a>
            </div>
          </div>
        `).join('');
      }
    }
  } catch (err) {
    console.error('Failed to load history:', err);
  }
}

window.playTrackFromHistory = function(index) {
  if (AppState.trackList[index]) {
    AppState.currentTrackIndex = index;
    loadTrackIntoPlayer(AppState.trackList[index]);
    switchView('player');
    playAudio();
  }
};

window.deleteTrack = async function(filename) {
  if (!confirm(`Are you sure you want to delete ${filename}?`)) return;

  try {
    const res = await fetch(`/api/history/${filename}`, { method: 'DELETE' });
    const data = await res.json();
    if (data.success) {
      showToast(`Deleted ${filename}`, 'info');
      loadHistory();
      loadSystemStatus();
    }
  } catch (err) {
    showToast(`Failed to delete: ${err.message}`, 'danger');
  }
};

/* ===================================================================
   12. Event Listeners Initialization
   =================================================================== */
function initEventListeners() {
  // 1. Upload View: Drag & Drop
  const dropzone = document.getElementById('midiDropzone');
  const fileInput = document.getElementById('midiFileInput');
  const confirmBtn = document.getElementById('btnConfirmUpload');

  if (dropzone && fileInput) {
    dropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });
    dropzone.addEventListener('dragleave', () => {
      dropzone.classList.remove('dragover');
    });
    dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
      if (e.dataTransfer.files.length > 0) {
        fileInput.files = e.dataTransfer.files;
        handleFileSelected(fileInput.files[0]);
      }
    });

    fileInput.addEventListener('change', () => {
      if (fileInput.files.length > 0) {
        handleFileSelected(fileInput.files[0]);
      }
    });

    confirmBtn.addEventListener('click', uploadSelectedMidi);
  }

  // 2. Preprocess View: Run Preprocessor
  document.getElementById('prepSeqLenRange').addEventListener('input', (e) => {
    document.getElementById('prepSeqLenVal').textContent = `${e.target.value} notes`;
  });

  document.getElementById('btnRunPreprocess').addEventListener('click', async () => {
    const genre = document.getElementById('prepGenreSelect').value;
    const seq_length = parseInt(document.getElementById('prepSeqLenRange').value, 10);
    const btn = document.getElementById('btnRunPreprocess');
    const alertBox = document.getElementById('prepAlert');

    btn.disabled = true;
    btn.innerHTML = `<span class="spinner-border spinner-border-sm me-2"></span>Extracting Musical Notes...`;
    alertBox.style.display = 'none';

    try {
      const res = await fetch('/api/preprocess', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ genre, seq_length })
      });
      const data = await res.json();
      if (!res.ok || !data.success) throw new Error(data.error || 'Preprocessing failed');

      showToast(data.message, 'success');
      updatePrepUI(data.stats);
    } catch (err) {
      alertBox.className = 'alert alert-danger';
      alertBox.textContent = err.message;
      alertBox.style.display = 'block';
    } finally {
      btn.disabled = false;
      btn.innerHTML = `<i class="fa-solid fa-gears"></i> Run Note Preprocessor`;
    }
  });

  // 3. Train View: Hyperparameter sliders & Start/Stop
  document.getElementById('trainEpochsRange').addEventListener('input', (e) => {
    document.getElementById('trainEpochsVal').textContent = `${e.target.value} epochs`;
  });

  document.getElementById('btnStartTrain').addEventListener('click', async () => {
    const genre = document.getElementById('trainGenreSelect').value;
    const epochs = parseInt(document.getElementById('trainEpochsRange').value, 10);
    const batch_size = parseInt(document.getElementById('trainBatchSize').value, 10);
    const learning_rate = parseFloat(document.getElementById('trainLearningRate').value);

    document.getElementById('btnStartTrain').style.display = 'none';
    document.getElementById('btnStopTrain').style.display = 'block';
    document.getElementById('trainProgressCard').style.display = 'block';

    try {
      const res = await fetch('/api/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ genre, epochs, batch_size, learning_rate })
      });
      const data = await res.json();
      if (!res.ok || !data.success) throw new Error(data.error || 'Training failed to start');

      showToast(`Started training for ${genre} (${epochs} epochs)`, 'info');
      startTrainingPoll();
    } catch (err) {
      showToast(err.message, 'danger');
      document.getElementById('btnStartTrain').style.display = 'block';
      document.getElementById('btnStopTrain').style.display = 'none';
    }
  });

  document.getElementById('btnStopTrain').addEventListener('click', async () => {
    try {
      const res = await fetch('/api/train/stop', { method: 'POST' });
      const data = await res.json();
      showToast(data.message, 'warning');
    } catch (err) {
      showToast('Failed to stop training', 'danger');
    }
  });

  document.getElementById('btnResetChart').addEventListener('click', () => {
    if (AppState.lossChart) {
      AppState.lossChart.data.labels = [];
      AppState.lossChart.data.datasets[0].data = [];
      AppState.lossChart.update();
    }
  });

  // 4. Generate View: Sliders & Genre cards
  document.querySelectorAll('.genre-choice-card').forEach(card => {
    card.addEventListener('click', () => {
      document.querySelectorAll('.genre-choice-card').forEach(c => c.classList.remove('selected'));
      card.classList.add('selected');
      AppState.selectedGenre = card.getAttribute('data-genre');
    });
  });

  document.getElementById('genLengthRange').addEventListener('input', (e) => {
    document.getElementById('genLengthVal').textContent = `${e.target.value} notes`;
  });

  document.getElementById('genTempoRange').addEventListener('input', (e) => {
    document.getElementById('genTempoVal').textContent = `${e.target.value} BPM`;
  });

  document.getElementById('genTempRange').addEventListener('input', (e) => {
    const val = parseFloat(e.target.value);
    document.getElementById('genTempVal').textContent = val.toFixed(1);
    const desc = document.getElementById('genTempDesc');
    if (val <= 0.4) desc.textContent = 'Conservative: highly confident, structured patterns';
    else if (val <= 0.8) desc.textContent = 'Balanced: harmonious with pleasant melodic variations';
    else desc.textContent = 'Avant-Garde: highly experimental, unexpected intervals';
  });

  document.getElementById('btnGenerateMusic').addEventListener('click', generateMusicAction);
}

function handleFileSelected(file) {
  if (!file) return;
  const info = document.getElementById('uploadFileInfo');
  document.getElementById('uploadFileName').textContent = file.name;
  document.getElementById('uploadFileSize').textContent = `${(file.size / 1024).toFixed(1)} KB`;
  info.style.display = 'block';
}

async function uploadSelectedMidi() {
  const fileInput = document.getElementById('midiFileInput');
  if (!fileInput.files || fileInput.files.length === 0) return;

  const file = fileInput.files[0];
  const genre = document.getElementById('uploadGenreSelect').value;
  const progress = document.getElementById('uploadProgressBar');
  const alertBox = document.getElementById('uploadAlert');

  const formData = new FormData();
  formData.append('file', file);
  formData.append('genre', genre);

  progress.style.display = 'block';
  alertBox.style.display = 'none';

  try {
    const res = await fetch('/api/upload', {
      method: 'POST',
      body: formData
    });
    const data = await res.json();
    if (!res.ok || !data.success) throw new Error(data.error || 'Upload failed');

    alertBox.className = 'alert alert-success';
    alertBox.innerHTML = `<strong>Success!</strong> ${data.message}`;
    alertBox.style.display = 'block';
    showToast(`Uploaded ${file.name}`, 'success');

    // Reset dropzone
    document.getElementById('uploadFileInfo').style.display = 'none';
    fileInput.value = '';

    // Refresh dataset table
    loadDatasets();
    loadSystemStatus();
  } catch (err) {
    alertBox.className = 'alert alert-danger';
    alertBox.innerHTML = `<strong>Validation Error:</strong> ${err.message}`;
    alertBox.style.display = 'block';
  } finally {
    progress.style.display = 'none';
  }
}

/* ===================================================================
   13. Toast Notification Helper
   =================================================================== */
function showToast(message, type = 'info') {
  const container = document.getElementById('toastContainer');
  if (!container) return;

  const toast = document.createElement('div');
  toast.className = 'studio-toast';

  const icons = {
    'success': '<i class="fa-solid fa-circle-check text-success"></i>',
    'danger': '<i class="fa-solid fa-circle-exclamation text-danger"></i>',
    'warning': '<i class="fa-solid fa-triangle-exclamation text-warning"></i>',
    'info': '<i class="fa-solid fa-circle-info text-info"></i>'
  };

  toast.innerHTML = `
    ${icons[type] || icons['info']}
    <span>${message}</span>
  `;

  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(100%)';
    toast.style.transition = 'all 0.3s ease';
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}

function formatTime(seconds) {
  if (isNaN(seconds) || seconds < 0) return '0:00';
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs < 10 ? '0' : ''}${secs}`;
}
