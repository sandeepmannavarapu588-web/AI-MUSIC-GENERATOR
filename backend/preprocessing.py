"""
Data Preprocessing Module for AI Music Generator
Parses MIDI files using music21, extracts notes/chords/durations,
creates vocabulary mappings and normalized training sequences for LSTM.
"""

import os
import glob
import pickle
import json
from collections import Counter
import numpy as np
import music21 as m21

class MidiPreprocessor:
    def __init__(self, base_dir=None):
        if base_dir is None:
            self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        else:
            self.base_dir = os.path.abspath(base_dir)

        self.datasets_dir = os.path.join(self.base_dir, 'datasets')
        self.models_dir = os.path.join(self.base_dir, 'models')
        os.makedirs(self.models_dir, exist_ok=True)

    def extract_notes_from_midi(self, filepath):
        """
        Parses a MIDI file using music21 and extracts a sequence of tokens.
        Each token represents a Note or Chord along with duration.
        Format: 'N_C4_1.0' or 'C_60.64.67_2.0'
        """
        notes_and_chords = []
        try:
            midi = m21.converter.parse(filepath)
            parts = m21.instrument.partitionByInstrument(midi)
            notes_to_parse = parts.parts[0].recurse() if parts else midi.flatten().notes

            for element in notes_to_parse:
                # Quantize duration to nearest 0.25 or 0.5 to keep vocabulary clean & musical
                dur = round(float(element.duration.quarterLength) * 2) / 2
                dur = max(0.25, min(4.0, dur))

                if isinstance(element, m21.note.Note):
                    # Single note
                    token = f"N_{str(element.pitch)}_{dur}"
                    notes_and_chords.append(token)
                elif isinstance(element, m21.chord.Chord):
                    # Multi-pitch chord
                    pitches = ".".join(str(p.midi) for p in sorted(element.pitches))
                    token = f"C_{pitches}_{dur}"
                    notes_and_chords.append(token)
                elif isinstance(element, m21.note.Rest):
                    if dur >= 0.5:
                        notes_and_chords.append(f"R_Rest_{dur}")
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")

        return notes_and_chords

    def process_genre(self, genre='piano', seq_length=16):
        """
        Preprocesses all MIDI files for a given genre (or all genres),
        builds vocabulary, sliding-window sequences, and saves cached artifacts.
        """
        all_tokens = []
        files_processed = []

        if genre == 'all':
            search_folders = [
                os.path.join(self.datasets_dir, g)
                for g in ['classical', 'jazz', 'piano', 'instrumental']
            ] + [os.path.join(self.base_dir, 'uploads')]
        else:
            search_folders = [os.path.join(self.datasets_dir, genre)]

        for folder in search_folders:
            if not os.path.exists(folder):
                continue
            for f in glob.glob(os.path.join(folder, '*.mid*')):
                tokens = self.extract_notes_from_midi(f)
                if len(tokens) > 5:
                    all_tokens.extend(tokens)
                    files_processed.append(os.path.basename(f))

        if not all_tokens:
            raise ValueError(f"No note events found in genre '{genre}'. Please add MIDI files first.")

        # Build vocabulary
        token_counter = Counter(all_tokens)
        pitchnames = sorted(list(set(all_tokens)))
        vocab_size = len(pitchnames)

        note_to_int = {note: number for number, note in enumerate(pitchnames)}
        int_to_note = {number: note for number, note in enumerate(pitchnames)}

        # Prepare sequence windows
        network_input = []
        network_output = []

        # If data is shorter than seq_length, adapt seq_length
        actual_seq_length = min(seq_length, max(4, len(all_tokens) - 1))

        for i in range(0, len(all_tokens) - actual_seq_length):
            seq_in = all_tokens[i:i + actual_seq_length]
            seq_out = all_tokens[i + actual_seq_length]
            network_input.append([note_to_int[char] for char in seq_in])
            network_output.append(note_to_int[seq_out])

        n_patterns = len(network_input)
        if n_patterns == 0:
            raise ValueError("Dataset too small for the specified sequence length.")

        # Normalize input for neural network [0, 1]
        X = np.reshape(network_input, (n_patterns, actual_seq_length, 1))
        X_normalized = X / float(vocab_size)
        y = np.array(network_output, dtype=np.int32)

        # Cache preprocessed data
        cache_path = os.path.join(self.models_dir, f"preprocessed_{genre}.pkl")
        stats_path = os.path.join(self.models_dir, f"preprocessed_{genre}_stats.json")

        data_payload = {
            'genre': genre,
            'vocab_size': vocab_size,
            'seq_length': actual_seq_length,
            'note_to_int': note_to_int,
            'int_to_note': int_to_note,
            'pitchnames': pitchnames,
            'network_input': network_input,
            'X': X_normalized,
            'y': y,
            'raw_tokens_count': len(all_tokens),
            'patterns_count': n_patterns,
            'files_processed': files_processed
        }

        with open(cache_path, 'wb') as f:
            pickle.dump(data_payload, f)

        # Save readable summary stats for API and UI
        top_tokens = [{'token': k, 'count': v} for k, v in token_counter.most_common(12)]
        stats_summary = {
            'genre': genre,
            'files_processed_count': len(files_processed),
            'files': files_processed,
            'total_tokens': len(all_tokens),
            'unique_tokens': vocab_size,
            'sequence_length': actual_seq_length,
            'total_sequences': n_patterns,
            'top_tokens': top_tokens
        }

        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats_summary, f, indent=2)

        return stats_summary

    def get_preprocessing_status(self, genre='piano'):
        """Check if preprocessed data exists and return its summary stats."""
        stats_path = os.path.join(self.models_dir, f"preprocessed_{genre}_stats.json")
        if os.path.exists(stats_path):
            with open(stats_path, 'r', encoding='utf-8') as f:
                return True, json.load(f)
        return False, None

    def load_preprocessed_data(self, genre='piano'):
        """Load the preprocessed tensor data and mappings for training."""
        cache_path = os.path.join(self.models_dir, f"preprocessed_{genre}.pkl")
        if not os.path.exists(cache_path):
            # Auto-process if not already done
            self.process_genre(genre=genre)
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
