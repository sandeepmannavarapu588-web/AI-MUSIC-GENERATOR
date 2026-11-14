"""
AI Music Generator Module
Generates new music using the trained LSTM model with temperature-based sampling,
and converts generated sequences into standard MIDI files.
"""

import os
import time
import json
import random
import numpy as np
import music21 as m21

class MusicGenerator:
    def __init__(self, base_dir=None):
        if base_dir is None:
            self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        else:
            self.base_dir = os.path.abspath(base_dir)

        self.generated_dir = os.path.join(self.base_dir, 'generated_music')
        self.models_dir = os.path.join(self.base_dir, 'models')
        os.makedirs(self.generated_dir, exist_ok=True)

    def sample_with_temperature(self, probabilities, temperature=1.0):
        """
        Temperature-based categorical sampling.
        Higher temperature = more diverse/creative.
        Lower temperature = more confident/conservative.
        """
        temperature = max(0.1, min(2.0, float(temperature)))
        probs = np.asarray(probabilities).astype('float64')
        probs = np.clip(probs, 1e-12, 1.0)

        # Log-odds scaled by temperature
        log_probs = np.log(probs) / temperature
        exp_probs = np.exp(log_probs - np.max(log_probs))
        probs_adjusted = exp_probs / np.sum(exp_probs)

        # Ensure valid probability distribution sums to 1.0
        probs_adjusted = probs_adjusted / np.sum(probs_adjusted)

        return int(np.random.choice(len(probs_adjusted), p=probs_adjusted))

    def generate_sequence(self, model_manager, genre='piano', length=60, temperature=0.8):
        """
        Generates a sequence of musical tokens using the trained LSTM model.
        """
        inference_ctx = model_manager.load_model_for_inference(genre=genre)
        predict_fn = inference_ctx['predict_fn']
        vocab_size = inference_ctx['vocab_size']
        seq_length = inference_ctx['seq_length']
        int_to_note = inference_ctx['int_to_note']
        network_input = inference_ctx['network_input']

        # Pick a random starting seed pattern from the dataset
        start_idx = random.randint(0, len(network_input) - 1)
        pattern = list(network_input[start_idx])

        prediction_output = []

        for note_index in range(length):
            # Normalize pattern for model input
            pattern_norm = np.array(pattern, dtype=np.float32) / float(vocab_size)
            probs = predict_fn(pattern_norm)

            next_idx = self.sample_with_temperature(probs, temperature=temperature)
            next_token = int_to_note[next_idx]
            prediction_output.append(next_token)

            pattern.append(next_idx)
            pattern = pattern[1:]

        return prediction_output

    def tokens_to_midi(self, tokens, genre='piano', output_filename=None, tempo_bpm=110):
        """
        Converts generated token sequence into a valid MIDI file with rich expressive dynamics.
        """
        midi_stream = m21.stream.Stream()
        midi_stream.append(m21.tempo.MetronomeMark(number=tempo_bpm))
        midi_stream.append(m21.meter.TimeSignature('4/4'))

        # Set genre instrument
        if genre == 'classical':
            midi_stream.append(m21.instrument.Piano())
        elif genre == 'jazz':
            midi_stream.append(m21.instrument.Saxophone())
        elif genre == 'instrumental':
            midi_stream.append(m21.instrument.AcousticGuitar())
        else:  # piano
            midi_stream.append(m21.instrument.Piano())

        total_quarter_length = 0.0
        parsed_notes_count = 0

        for token in tokens:
            try:
                parts = token.split('_')
                if len(parts) >= 3:
                    token_type = parts[0]
                    content = parts[1]
                    dur = float(parts[2])
                else:
                    token_type = 'N'
                    content = 'C4'
                    dur = 1.0

                # Natural human-like velocity variation (80 to 105)
                vel = random.randint(82, 104)

                if token_type == 'N':
                    # Single note
                    new_note = m21.note.Note(content, quarterLength=dur)
                    new_note.volume.velocity = vel
                    new_note.storedInstrument = midi_stream[0]
                    midi_stream.append(new_note)
                    total_quarter_length += dur
                    parsed_notes_count += 1

                elif token_type == 'C':
                    # Chord
                    pitches_in_chord = [int(p) for p in content.split('.') if p.isdigit()]
                    if pitches_in_chord:
                        chord_notes = [m21.note.Note(midi=p) for p in pitches_in_chord]
                        new_chord = m21.chord.Chord(chord_notes, quarterLength=dur)
                        new_chord.volume.velocity = vel
                        midi_stream.append(new_chord)
                        total_quarter_length += dur
                        parsed_notes_count += len(pitches_in_chord)

                elif token_type == 'R':
                    # Rest
                    new_rest = m21.note.Rest(quarterLength=dur)
                    midi_stream.append(new_rest)
                    total_quarter_length += dur
            except Exception as e:
                # Graceful skip for any ill-formed token
                continue

        if parsed_notes_count == 0:
            # Fallback safe chord progression
            fallback_chord = m21.chord.Chord(['C4', 'E4', 'G4', 'C5'], quarterLength=4.0)
            midi_stream.append(fallback_chord)

        # Generate unique filename if not provided
        timestamp = int(time.time())
        if not output_filename:
            output_filename = f"generated_{genre}_{timestamp}.mid"

        midi_filepath = os.path.join(self.generated_dir, output_filename)
        midi_stream.write('midi', fp=midi_filepath)

        duration_sec = round((total_quarter_length / (tempo_bpm / 60.0)), 2)
        filesize_kb = round(os.path.getsize(midi_filepath) / 1024, 2)

        meta = {
            'filename': output_filename,
            'midi_path': midi_filepath,
            'genre': genre,
            'notes_count': parsed_notes_count,
            'duration_sec': duration_sec,
            'tempo': tempo_bpm,
            'filesize_kb': filesize_kb,
            'created_at': time.strftime("%Y-%m-%d %H:%M:%S"),
            'tokens': tokens[:20]  # sample of first tokens
        }

        # Save metadata JSON
        meta_filepath = os.path.join(self.generated_dir, f"{os.path.splitext(output_filename)[0]}.json")
        with open(meta_filepath, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)

        return meta
