"""
Dataset Manager for AI Music Generator
Handles scanning, validating, uploading, and providing starter datasets across genres.
"""

import os
import glob
import math
import music21 as m21
import pretty_midi

GENRES = ['classical', 'jazz', 'piano', 'instrumental']

class DatasetManager:
    def __init__(self, base_dir=None):
        if base_dir is None:
            # AI-Music-Generator root directory
            self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        else:
            self.base_dir = os.path.abspath(base_dir)

        self.datasets_dir = os.path.join(self.base_dir, 'datasets')
        self.uploads_dir = os.path.join(self.base_dir, 'uploads')

        # Ensure directories exist
        for genre in GENRES:
            os.makedirs(os.path.join(self.datasets_dir, genre), exist_ok=True)
        os.makedirs(self.uploads_dir, exist_ok=True)

        # Seed starter datasets if empty
        self.ensure_starter_datasets()

    def ensure_starter_datasets(self):
        """Populate starter datasets with musically coherent MIDI pieces if directories are empty."""
        for genre in GENRES:
            genre_dir = os.path.join(self.datasets_dir, genre)
            existing_midis = glob.glob(os.path.join(genre_dir, '*.mid')) + glob.glob(os.path.join(genre_dir, '*.midi'))
            if len(existing_midis) < 2:
                self._generate_starter_tracks(genre, genre_dir)

    def _generate_starter_tracks(self, genre, target_dir):
        """Generate high-quality musical seed MIDI tracks for training and exploration."""
        if genre == 'classical':
            # Beethoven Ode to Joy theme & Bach & Mozart motifs
            self._create_classical_piece_1(os.path.join(target_dir, 'classical_ode_theme.mid'))
            self._create_classical_piece_2(os.path.join(target_dir, 'classical_bach_invention.mid'))
            self._create_classical_piece_3(os.path.join(target_dir, 'classical_mozart_melody.mid'))
        elif genre == 'jazz':
            # Jazz ii-V-I progressions, swing rhythms, 7th chords
            self._create_jazz_piece_1(os.path.join(target_dir, 'jazz_autumn_breeze.mid'))
            self._create_jazz_piece_2(os.path.join(target_dir, 'jazz_blues_swing.mid'))
            self._create_jazz_piece_3(os.path.join(target_dir, 'jazz_midnight_groove.mid'))
        elif genre == 'piano':
            # Romantic arpeggios, Chopin-inspired nocturne, Satie style chords
            self._create_piano_piece_1(os.path.join(target_dir, 'piano_nocturne_dreams.mid'))
            self._create_piano_piece_2(os.path.join(target_dir, 'piano_waltz_in_am.mid'))
            self._create_piano_piece_3(os.path.join(target_dir, 'piano_tranquil_stream.mid'))
        elif genre == 'instrumental':
            # Strings / acoustic guitar / orchestral melodic sequences
            self._create_instrumental_piece_1(os.path.join(target_dir, 'instrumental_celtic_journey.mid'))
            self._create_instrumental_piece_2(os.path.join(target_dir, 'instrumental_pastoral_morning.mid'))
            self._create_instrumental_piece_3(os.path.join(target_dir, 'instrumental_acoustic_horizon.mid'))

    def _create_classical_piece_1(self, filename):
        s = m21.stream.Stream()
        s.append(m21.tempo.MetronomeMark(number=132))
        s.append(m21.meter.TimeSignature('4/4'))
        s.append(m21.instrument.Piano())

        # Ode to Joy motif with chords
        melody = [
            ('E4', 1.0), ('E4', 1.0), ('F4', 1.0), ('G4', 1.0),
            ('G4', 1.0), ('F4', 1.0), ('E4', 1.0), ('D4', 1.0),
            ('C4', 1.0), ('C4', 1.0), ('D4', 1.0), ('E4', 1.0),
            ('E4', 1.5), ('D4', 0.5), ('D4', 2.0),
            ('E4', 1.0), ('E4', 1.0), ('F4', 1.0), ('G4', 1.0),
            ('G4', 1.0), ('F4', 1.0), ('E4', 1.0), ('D4', 1.0),
            ('C4', 1.0), ('C4', 1.0), ('D4', 1.0), ('E4', 1.0),
            ('D4', 1.5), ('C4', 0.5), ('C4', 2.0),
            ('D4', 1.0), ('D4', 1.0), ('E4', 1.0), ('C4', 1.0),
            ('D4', 1.0), ('E4', 0.5), ('F4', 0.5), ('E4', 1.0), ('C4', 1.0),
            ('D4', 1.0), ('E4', 0.5), ('F4', 0.5), ('E4', 1.0), ('D4', 1.0),
            ('C4', 1.0), ('D4', 1.0), ('G3', 2.0)
        ]
        for note_name, dur in melody:
            n = m21.note.Note(note_name, quarterLength=dur)
            s.append(n)
        s.write('midi', fp=filename)

    def _create_classical_piece_2(self, filename):
        s = m21.stream.Stream()
        s.append(m21.tempo.MetronomeMark(number=108))
        s.append(m21.instrument.Piano())
        # Bach style counterpoint motif
        notes = [
            ('C4', 0.5), ('D4', 0.5), ('E4', 0.5), ('F4', 0.5), ('G4', 1.0), ('C5', 1.0),
            ('B4', 0.5), ('A4', 0.5), ('G4', 0.5), ('F4', 0.5), ('E4', 1.0), ('G4', 1.0),
            ('F4', 0.5), ('E4', 0.5), ('D4', 0.5), ('C4', 0.5), ('B3', 1.0), ('D4', 1.0),
            ('C4', 2.0), ('E4', 0.5), ('G4', 0.5), ('C5', 1.0), ('E5', 1.0),
            ('D5', 0.5), ('C5', 0.5), ('B4', 0.5), ('A4', 0.5), ('G4', 2.0),
            ('A4', 0.5), ('B4', 0.5), ('C5', 1.0), ('F4', 0.5), ('G4', 0.5), ('A4', 1.0),
            ('D4', 0.5), ('E4', 0.5), ('F4', 1.0), ('G3', 1.0), ('B3', 1.0), ('C4', 2.0)
        ]
        for n_name, dur in notes:
            s.append(m21.note.Note(n_name, quarterLength=dur))
        s.write('midi', fp=filename)

    def _create_classical_piece_3(self, filename):
        s = m21.stream.Stream()
        s.append(m21.tempo.MetronomeMark(number=120))
        s.append(m21.instrument.Piano())
        # Mozart Sonata theme motif
        seq = [
            ('C5', 2.0), ('E5', 1.0), ('G5', 1.0), ('B4', 1.5), ('C5', 0.5), ('D5', 1.0), ('C5', 1.0),
            ('A4', 1.0), ('D5', 1.0), ('G4', 2.0), ('F4', 1.0), ('E4', 1.0), ('D4', 2.0),
            ('C4', 2.0), ('E4', 1.0), ('G4', 1.0), ('C5', 1.5), ('D5', 0.5), ('E5', 1.0), ('F5', 1.0),
            ('G5', 2.0), ('E5', 1.0), ('C5', 1.0), ('D5', 2.0), ('C5', 2.0)
        ]
        for n_name, dur in seq:
            s.append(m21.note.Note(n_name, quarterLength=dur))
        s.write('midi', fp=filename)

    def _create_jazz_piece_1(self, filename):
        s = m21.stream.Stream()
        s.append(m21.tempo.MetronomeMark(number=110))
        s.append(m21.instrument.Piano())
        # Autumn Breeze jazz harmony with 7th chords and melody
        events = [
            m21.chord.Chord(['D4', 'F4', 'A4', 'C5'], quarterLength=2.0), # Dm7
            m21.note.Note('E5', quarterLength=1.0),
            m21.note.Note('D5', quarterLength=1.0),
            m21.chord.Chord(['G3', 'B3', 'D4', 'F4'], quarterLength=2.0), # G7
            m21.note.Note('C5', quarterLength=1.0),
            m21.note.Note('B4', quarterLength=1.0),
            m21.chord.Chord(['C4', 'E4', 'G4', 'B4'], quarterLength=3.0), # Cmaj7
            m21.note.Note('A4', quarterLength=1.0),
            m21.chord.Chord(['F3', 'A3', 'C4', 'E4'], quarterLength=2.0), # Fmaj7
            m21.note.Note('G4', quarterLength=1.0),
            m21.note.Note('F4', quarterLength=1.0),
            m21.chord.Chord(['B3', 'D4', 'F4', 'A4'], quarterLength=2.0), # Bm7b5
            m21.chord.Chord(['E3', 'G#3', 'B3', 'D4'], quarterLength=2.0), # E7
            m21.chord.Chord(['A3', 'C4', 'E4', 'G4'], quarterLength=4.0)  # Am7
        ]
        for e in events:
            s.append(e)
        s.write('midi', fp=filename)

    def _create_jazz_piece_2(self, filename):
        s = m21.stream.Stream()
        s.append(m21.tempo.MetronomeMark(number=124))
        s.append(m21.instrument.Saxophone())
        # Blues scale swing
        notes = [
            ('C4', 0.75), ('Eb4', 0.75), ('F4', 0.5), ('F#4', 0.5), ('G4', 1.0), ('Bb4', 0.5),
            ('C5', 1.5), ('Bb4', 0.5), ('G4', 1.0), ('F#4', 0.5), ('F4', 0.5), ('Eb4', 1.0),
            ('C4', 2.0), ('Eb4', 0.5), ('F4', 0.5), ('G4', 1.0), ('Bb4', 1.0), ('C5', 2.0),
            ('Eb5', 0.75), ('D5', 0.25), ('C5', 1.0), ('G4', 1.0), ('F4', 1.0), ('C4', 2.0)
        ]
        for n_name, dur in notes:
            s.append(m21.note.Note(n_name, quarterLength=dur))
        s.write('midi', fp=filename)

    def _create_jazz_piece_3(self, filename):
        s = m21.stream.Stream()
        s.append(m21.tempo.MetronomeMark(number=96))
        s.append(m21.instrument.Piano())
        events = [
            m21.chord.Chord(['F3', 'A3', 'C4', 'Eb4'], quarterLength=2.0), # F7
            m21.note.Note('A4', quarterLength=1.0),
            m21.note.Note('C5', quarterLength=1.0),
            m21.chord.Chord(['Bb3', 'D4', 'F4', 'Ab4'], quarterLength=2.0), # Bb7
            m21.note.Note('D5', quarterLength=1.0),
            m21.note.Note('F5', quarterLength=1.0),
            m21.chord.Chord(['G3', 'B3', 'D4', 'F4'], quarterLength=2.0), # G7
            m21.chord.Chord(['C3', 'E3', 'G3', 'Bb3'], quarterLength=2.0), # C7
            m21.chord.Chord(['F3', 'A3', 'C4', 'E4'], quarterLength=4.0)
        ]
        for e in events:
            s.append(e)
        s.write('midi', fp=filename)

    def _create_piano_piece_1(self, filename):
        s = m21.stream.Stream()
        s.append(m21.tempo.MetronomeMark(number=80))
        s.append(m21.instrument.Piano())
        # Romantic Nocturne arpeggiated piano flow
        notes = [
            ('C4', 0.5), ('E4', 0.5), ('G4', 0.5), ('B4', 0.5), ('C5', 2.0),
            ('B4', 0.5), ('G4', 0.5), ('E4', 0.5), ('D4', 0.5), ('E4', 2.0),
            ('A3', 0.5), ('C4', 0.5), ('E4', 0.5), ('A4', 0.5), ('B4', 1.5), ('A4', 0.5),
            ('F4', 1.0), ('D4', 1.0), ('G3', 0.5), ('B3', 0.5), ('D4', 0.5), ('F4', 0.5),
            ('E4', 2.0), ('C4', 2.0), ('G4', 1.0), ('E5', 2.0), ('D5', 1.0), ('C5', 3.0)
        ]
        for n_name, dur in notes:
            s.append(m21.note.Note(n_name, quarterLength=dur))
        s.write('midi', fp=filename)

    def _create_piano_piece_2(self, filename):
        s = m21.stream.Stream()
        s.append(m21.tempo.MetronomeMark(number=138))
        s.append(m21.meter.TimeSignature('3/4'))
        s.append(m21.instrument.Piano())
        # Waltz in A minor
        events = [
            m21.note.Note('E5', quarterLength=2.0),
            m21.note.Note('D#5', quarterLength=1.0),
            m21.note.Note('E5', quarterLength=2.0),
            m21.note.Note('B4', quarterLength=1.0),
            m21.note.Note('D5', quarterLength=2.0),
            m21.note.Note('C5', quarterLength=1.0),
            m21.note.Note('A4', quarterLength=3.0),
            m21.chord.Chord(['A3', 'C4', 'E4'], quarterLength=2.0),
            m21.note.Note('C5', quarterLength=1.0),
            m21.note.Note('E5', quarterLength=2.0),
            m21.note.Note('A5', quarterLength=1.0),
            m21.note.Note('G#5', quarterLength=3.0),
            m21.chord.Chord(['E3', 'G#3', 'B3'], quarterLength=3.0)
        ]
        for e in events:
            s.append(e)
        s.write('midi', fp=filename)

    def _create_piano_piece_3(self, filename):
        s = m21.stream.Stream()
        s.append(m21.tempo.MetronomeMark(number=72))
        s.append(m21.instrument.Piano())
        # Tranquil Satie-esque meditation
        events = [
            m21.chord.Chord(['G3', 'B3', 'D4', 'F#4'], quarterLength=3.0),
            m21.note.Note('B4', quarterLength=1.0),
            m21.note.Note('D5', quarterLength=2.0),
            m21.chord.Chord(['D3', 'F#3', 'A3', 'C#4'], quarterLength=3.0),
            m21.note.Note('A4', quarterLength=1.0),
            m21.note.Note('F#4', quarterLength=2.0),
            m21.chord.Chord(['G3', 'B3', 'D4'], quarterLength=3.0),
            m21.chord.Chord(['D3', 'F#3', 'A3'], quarterLength=3.0)
        ]
        for e in events:
            s.append(e)
        s.write('midi', fp=filename)

    def _create_instrumental_piece_1(self, filename):
        s = m21.stream.Stream()
        s.append(m21.tempo.MetronomeMark(number=115))
        s.append(m21.instrument.AcousticGuitar())
        # Celtic melodic folk theme
        notes = [
            ('D4', 1.0), ('E4', 0.5), ('F#4', 0.5), ('A4', 1.0), ('B4', 1.0),
            ('A4', 0.5), ('F#4', 0.5), ('E4', 1.0), ('D4', 1.5), ('E4', 0.5),
            ('F#4', 1.0), ('A4', 1.0), ('B4', 1.5), ('C#5', 0.5), ('D5', 2.0),
            ('C#5', 1.0), ('B4', 1.0), ('A4', 1.0), ('F#4', 1.0), ('E4', 2.0),
            ('D4', 1.0), ('F#4', 1.0), ('A4', 1.0), ('D5', 2.0)
        ]
        for n_name, dur in notes:
            s.append(m21.note.Note(n_name, quarterLength=dur))
        s.write('midi', fp=filename)

    def _create_instrumental_piece_2(self, filename):
        s = m21.stream.Stream()
        s.append(m21.tempo.MetronomeMark(number=90))
        s.append(m21.instrument.Flute())
        notes = [
            ('G4', 1.0), ('A4', 1.0), ('B4', 2.0), ('D5', 1.0), ('C5', 1.0), ('B4', 1.0), ('A4', 1.0),
            ('G4', 2.0), ('E4', 1.0), ('D4', 1.0), ('G4', 3.0),
            ('B4', 1.0), ('C5', 1.0), ('D5', 2.0), ('E5', 1.0), ('D5', 1.0), ('B4', 2.0),
            ('A4', 1.0), ('G4', 1.0), ('A4', 2.0), ('G4', 3.0)
        ]
        for n_name, dur in notes:
            s.append(m21.note.Note(n_name, quarterLength=dur))
        s.write('midi', fp=filename)

    def _create_instrumental_piece_3(self, filename):
        s = m21.stream.Stream()
        s.append(m21.tempo.MetronomeMark(number=100))
        s.append(m21.instrument.Violin())
        notes = [
            ('A4', 1.0), ('C5', 1.0), ('E5', 2.0), ('D5', 1.0), ('C5', 1.0), ('B4', 2.0),
            ('G4', 1.0), ('B4', 1.0), ('D5', 2.0), ('C5', 1.0), ('B4', 1.0), ('A4', 3.0),
            ('E4', 1.0), ('A4', 1.0), ('C5', 1.0), ('E5', 2.0), ('F5', 1.0), ('E5', 1.0),
            ('D5', 1.0), ('B4', 1.0), ('A4', 3.0)
        ]
        for n_name, dur in notes:
            s.append(m21.note.Note(n_name, quarterLength=dur))
        s.write('midi', fp=filename)

    def list_datasets(self):
        """Return information about all available datasets and uploaded files."""
        result = {}
        total_files = 0
        total_size = 0

        for genre in GENRES + ['uploads']:
            folder = self.uploads_dir if genre == 'uploads' else os.path.join(self.datasets_dir, genre)
            pattern = os.path.join(folder, '*.*')
            files = [f for f in glob.glob(pattern) if f.lower().endswith(('.mid', '.midi'))]
            
            file_details = []
            for filepath in files:
                stat = os.stat(filepath)
                size_kb = round(stat.st_size / 1024, 2)
                total_size += stat.st_size
                meta = self.get_midi_metadata(filepath)
                file_details.append({
                    'filename': os.path.basename(filepath),
                    'path': filepath,
                    'size_kb': size_kb,
                    'notes_count': meta.get('notes_count', 0),
                    'duration_sec': meta.get('duration_sec', 0.0),
                    'tempo': meta.get('tempo', 120),
                    'instruments': meta.get('instruments', [])
                })
            
            result[genre] = {
                'count': len(files),
                'files': file_details
            }
            total_files += len(files)

        return {
            'genres': result,
            'total_files': total_files,
            'total_size_mb': round(total_size / (1024 * 1024), 2)
        }

    def validate_midi_file(self, filepath):
        """Validate if a file is a valid MIDI file with readable note data."""
        try:
            pm = pretty_midi.PrettyMIDI(filepath)
            notes_count = sum(len(inst.notes) for inst in pm.instruments)
            if notes_count == 0:
                score = m21.converter.parse(filepath)
                elements = len(score.flatten().notes)
                if elements == 0:
                    return False, "File contains 0 musical notes."
                return True, f"Valid MIDI with {elements} notes."
            return True, f"Valid MIDI with {notes_count} notes and {len(pm.instruments)} instrument(s)."
        except Exception as e:
            return False, f"Invalid MIDI file: {str(e)}"

    def get_midi_metadata(self, filepath):
        """Extract rich musical metadata from a MIDI file."""
        try:
            pm = pretty_midi.PrettyMIDI(filepath)
            total_notes = sum(len(inst.notes) for inst in pm.instruments)
            instruments = [inst.name if inst.name else f"Program {inst.program}" for inst in pm.instruments]
            tempo = round(pm.estimate_tempo(), 1) if pm.get_tempo_changes()[1].size > 0 else 120.0
            duration = round(pm.get_end_time(), 2)

            return {
                'notes_count': total_notes,
                'duration_sec': duration,
                'tempo': tempo,
                'instruments': instruments[:3]
            }
        except Exception:
            return {
                'notes_count': 0,
                'duration_sec': 0.0,
                'tempo': 120,
                'instruments': ['Piano']
            }

    def save_uploaded_file(self, file_storage, genre='piano'):
        """Save an uploaded MIDI file into uploads/ and appropriate genre folder."""
        filename = file_storage.filename
        if not filename.lower().endswith(('.mid', '.midi')):
            return False, "File must be a .mid or .midi file."

        safe_name = os.path.basename(filename).replace(' ', '_')
        upload_path = os.path.join(self.uploads_dir, safe_name)
        file_storage.save(upload_path)

        is_valid, msg = self.validate_midi_file(upload_path)
        if not is_valid:
            try:
                os.remove(upload_path)
            except Exception:
                pass
            return False, f"Validation failed: {msg}"

        target_genre = genre if genre in GENRES else 'piano'
        genre_path = os.path.join(self.datasets_dir, target_genre, safe_name)
        
        with open(upload_path, 'rb') as src, open(genre_path, 'wb') as dst:
            dst.write(src.read())

        metadata = self.get_midi_metadata(upload_path)
        return True, {
            'filename': safe_name,
            'genre': target_genre,
            'message': msg,
            'metadata': metadata
        }
