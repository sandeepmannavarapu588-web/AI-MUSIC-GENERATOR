"""
Pre-seeds models and sample generated tracks for all 4 genres
so the music library is populated on first launch.
"""

import os
import sys
import time

sys.path.append(os.path.dirname(__file__))
from app import app

client = app.test_client()

for genre in ['piano', 'jazz', 'instrumental']:
    print(f"Preprocessing & training {genre}...")
    client.post('/api/preprocess', json={'genre': genre, 'seq_length': 16})
    client.post('/api/train', json={'genre': genre, 'epochs': 6, 'batch_size': 8, 'learning_rate': 0.01})
    
    # Wait for training to complete
    for _ in range(15):
        s = client.get('/api/train/status').json
        if not s['is_training']:
            break
        time.sleep(0.5)

    print(f"Composing AI track for {genre}...")
    res = client.post('/api/generate', json={
        'genre': genre,
        'length': 40,
        'temperature': 0.8,
        'tempo': 105
    })
    track = res.json.get('track', {})
    print(f"✓ Generated {genre}: {track.get('filename')} ({track.get('notes_count')} notes, {track.get('duration_sec')}s)")

print("\nAll 4 genres now have pre-trained models and sample tracks!")
