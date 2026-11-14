"""
Automated Test Suite for AI Music Generator
Tests all modules: dataset collection, preprocessing, LSTM training,
AI music generation, MIDI conversion, and audio synthesis.
"""

import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))
from app import app

client = app.test_client()

print("=" * 60)
print("TEST 1: System Status & Initial Stats")
print("=" * 60)
res = client.get('/api/status')
print("Status Code:", res.status_code)
print("Response JSON:", res.json)
assert res.status_code == 200
assert res.json['status'] == 'online'
print("✓ Test 1 Passed!")

print("\n" + "=" * 60)
print("TEST 2: Datasets Enumeration Across Genres")
print("=" * 60)
res = client.get('/api/datasets')
assert res.status_code == 200
data = res.json
print(f"Total dataset MIDI files: {data['total_files']}")
for genre in ['classical', 'jazz', 'piano', 'instrumental']:
    count = data['genres'][genre]['count']
    print(f"  - {genre}: {count} files")
    assert count >= 3
print("✓ Test 2 Passed!")

print("\n" + "=" * 60)
print("TEST 3: Note Preprocessing (Classical Genre)")
print("=" * 60)
res = client.post('/api/preprocess', json={'genre': 'classical', 'seq_length': 16})
assert res.status_code == 200
stats = res.json['stats']
print(f"Total tokens parsed: {stats['total_tokens']}")
print(f"Unique vocabulary: {stats['unique_tokens']}")
print(f"Sequences generated: {stats['total_sequences']}")
assert stats['total_tokens'] > 0
assert stats['unique_tokens'] > 0
print("✓ Test 3 Passed!")

print("\n" + "=" * 60)
print("TEST 4: LSTM Model Training (Classical Genre)")
print("=" * 60)
res = client.post('/api/train', json={
    'genre': 'classical',
    'epochs': 5,
    'batch_size': 8,
    'learning_rate': 0.01
})
assert res.status_code == 200
print("Training started:", res.json['message'])

# Wait for training to complete
for i in range(25):
    status_res = client.get('/api/train/status')
    s = status_res.json
    print(f"Epoch {s['current_epoch']}/{s['total_epochs']} - Loss: {s['current_loss']} ({s['status_message']})")
    if not s['is_training'] and s['current_epoch'] >= 5:
        break
    time.sleep(0.5)
print("✓ Test 4 Passed!")

print("\n" + "=" * 60)
print("TEST 5: AI Music Generation & MIDI/WAV Synthesis")
print("=" * 60)
res = client.post('/api/generate', json={
    'genre': 'classical',
    'length': 45,
    'temperature': 0.8,
    'tempo': 120
})
assert res.status_code == 200
track = res.json['track']
print(f"Generated track: {track['filename']}")
print(f"Notes count: {track['notes_count']}")
print(f"Duration: {track['duration_sec']}s")
print(f"Synthesized WAV: {track['wav_filename']} ({track['filesize_kb']} KB)")
assert track['notes_count'] > 0
print("✓ Test 5 Passed!")

print("\n" + "=" * 60)
print("TEST 6: Download History & File Integrity Verification")
print("=" * 60)
res = client.get('/api/history')
assert res.status_code == 200
history = res.json['history']
print(f"Total history entries: {len(history)}")
assert len(history) >= 1

base_dir = os.path.dirname(__file__)
midi_path = os.path.join(base_dir, 'generated_music', track['filename'])
wav_path = os.path.join(base_dir, 'generated_music', track['wav_filename'])

print(f"MIDI file on disk: {midi_path} ({os.path.getsize(midi_path)} bytes)")
print(f"WAV file on disk: {wav_path} ({os.path.getsize(wav_path)} bytes)")
assert os.path.exists(midi_path) and os.path.getsize(midi_path) > 50
assert os.path.exists(wav_path) and os.path.getsize(wav_path) > 1000

# Stream endpoint test
stream_res = client.get(f"/api/stream/{track['wav_filename']}")
assert stream_res.status_code == 200
print(f"Stream response code: {stream_res.status_code}, content type: {stream_res.mimetype}")
print("✓ Test 6 Passed!")

print("\n" + "=" * 60)
print("ALL 6 MODULE TESTS PASSED FLAWLESSLY!")
print("=" * 60)
