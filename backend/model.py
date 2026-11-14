"""
LSTM Neural Network Architecture & Model Manager for AI Music Generation
Supports TensorFlow/Keras if installed, with a high-performance,
fully vectorized native NumPy LSTM Neural Network fallback with BPTT and Adam.
"""

import os
import time
import json
import threading
import numpy as np

# Try importing TensorFlow / Keras
HAS_TF = False
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.callbacks import Callback
    HAS_TF = True
except Exception:
    HAS_TF = False


class NativeLSTM:
    """
    Vectorized LSTM Neural Network with Adam Optimizer and Backpropagation Through Time (BPTT).
    Guarantees genuine Deep Learning training and note prediction on any Python platform.
    """
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=50, lr=0.005):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lr = lr

        concat_dim = input_dim + hidden_dim

        # Xavier / Glorot initialization for LSTM gates
        scale = np.sqrt(2.0 / (concat_dim + hidden_dim))
        self.Wf = np.random.randn(hidden_dim, concat_dim) * scale
        self.Wi = np.random.randn(hidden_dim, concat_dim) * scale
        self.Wc = np.random.randn(hidden_dim, concat_dim) * scale
        self.Wo = np.random.randn(hidden_dim, concat_dim) * scale

        self.bf = np.ones((hidden_dim, 1)) * 0.5  # Forget gate positive bias trick
        self.bi = np.zeros((hidden_dim, 1))
        self.bc = np.zeros((hidden_dim, 1))
        self.bo = np.zeros((hidden_dim, 1))

        # Output projection layer
        out_scale = np.sqrt(2.0 / (hidden_dim + output_dim))
        self.Wy = np.random.randn(output_dim, hidden_dim) * out_scale
        self.by = np.zeros((output_dim, 1))

        # Adam optimizer state variables
        self.m = {k: np.zeros_like(v) for k, v in self.get_params().items()}
        self.v = {k: np.zeros_like(v) for k, v in self.get_params().items()}
        self.t = 0

    def get_params(self):
        return {
            'Wf': self.Wf, 'Wi': self.Wi, 'Wc': self.Wc, 'Wo': self.Wo,
            'bf': self.bf, 'bi': self.bi, 'bc': self.bc, 'bo': self.bo,
            'Wy': self.Wy, 'by': self.by
        }

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return e_x / (np.sum(e_x, axis=0, keepdims=True) + 1e-12)

    def forward(self, x_seq):
        """
        Forward pass for a sequence: x_seq has shape (T, 1)
        Returns:
            probs: output probabilities (output_dim, 1)
            cache: intermediate activations for BPTT
        """
        T = len(x_seq)
        h = np.zeros((self.hidden_dim, 1))
        c = np.zeros((self.hidden_dim, 1))

        caches = []

        for t in range(T):
            xt = x_seq[t].reshape(-1, 1)
            concat = np.vstack((h, xt))

            f = self.sigmoid(np.dot(self.Wf, concat) + self.bf)
            i = self.sigmoid(np.dot(self.Wi, concat) + self.bi)
            c_cand = np.tanh(np.dot(self.Wc, concat) + self.bc)
            c_next = f * c + i * c_cand
            o = self.sigmoid(np.dot(self.Wo, concat) + self.bo)
            h_next = o * np.tanh(c_next)

            caches.append((concat, h, c, f, i, c_cand, c_next, o, h_next))
            h = h_next
            c = c_next

        # Final step prediction
        logits = np.dot(self.Wy, h) + self.by
        probs = self.softmax(logits)

        return probs, (caches, h)

    def backward(self, x_seq, target_idx, probs, cache):
        caches, last_h = cache
        T = len(x_seq)

        grads = {k: np.zeros_like(v) for k, v in self.get_params().items()}

        # Cross-entropy derivative with respect to logits
        dlogits = probs.copy()
        dlogits[target_idx] -= 1.0

        grads['Wy'] += np.dot(dlogits, last_h.T)
        grads['by'] += dlogits

        dh_next = np.dot(self.Wy.T, dlogits)
        dc_next = np.zeros((self.hidden_dim, 1))

        # Backward through time (BPTT)
        for t in reversed(range(T)):
            concat, h_prev, c_prev, f, i, c_cand, c_curr, o, h_curr = caches[t]

            tanh_c = np.tanh(c_curr)
            do = dh_next * tanh_c
            do_raw = do * o * (1.0 - o)

            dc = dc_next + dh_next * o * (1.0 - tanh_c ** 2)
            dc_cand = dc * i
            dc_cand_raw = dc_cand * (1.0 - c_cand ** 2)

            di = dc * c_cand
            di_raw = di * i * (1.0 - i)

            df = dc * c_prev
            df_raw = df * f * (1.0 - f)

            grads['Wf'] += np.dot(df_raw, concat.T)
            grads['bf'] += df_raw

            grads['Wi'] += np.dot(di_raw, concat.T)
            grads['bi'] += di_raw

            grads['Wc'] += np.dot(dc_cand_raw, concat.T)
            grads['bc'] += dc_cand_raw

            grads['Wo'] += np.dot(do_raw, concat.T)
            grads['bo'] += do_raw

            dconcat = (
                np.dot(self.Wf.T, df_raw) +
                np.dot(self.Wi.T, di_raw) +
                np.dot(self.Wc.T, dc_cand_raw) +
                np.dot(self.Wo.T, do_raw)
            )

            dh_next = dconcat[:self.hidden_dim, :]
            dc_next = dc * f

        return grads

    def apply_gradients(self, grads, beta1=0.9, beta2=0.999, eps=1e-8):
        """Adam optimizer step with gradient clipping."""
        self.t += 1
        params = self.get_params()

        for k in params:
            # Gradient clipping to prevent exploding gradients
            g = np.clip(grads[k], -5.0, 5.0)

            self.m[k] = beta1 * self.m[k] + (1.0 - beta1) * g
            self.v[k] = beta2 * self.v[k] + (1.0 - beta2) * (g ** 2)

            m_hat = self.m[k] / (1.0 - beta1 ** self.t)
            v_hat = self.v[k] / (1.0 - beta2 ** self.t)

            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

    def train_epoch(self, X, y, batch_size=16):
        """Train 1 epoch across sequences, returns average loss."""
        indices = np.arange(len(X))
        np.random.shuffle(indices)

        total_loss = 0.0
        n_samples = len(X)

        for start in range(0, n_samples, batch_size):
            batch_idx = indices[start:start + batch_size]
            accum_grads = {k: np.zeros_like(v) for k, v in self.get_params().items()}
            batch_loss = 0.0

            for idx in batch_idx:
                x_seq = X[idx]
                target = y[idx]

                probs, cache = self.forward(x_seq)
                loss = -np.log(probs[target, 0] + 1e-12)
                batch_loss += loss

                grads = self.backward(x_seq, target, probs, cache)
                for k in accum_grads:
                    accum_grads[k] += grads[k] / len(batch_idx)

            self.apply_gradients(accum_grads)
            total_loss += batch_loss

        return float(total_loss / n_samples)

    def predict(self, x_seq):
        """Predict next note probability distribution."""
        probs, _ = self.forward(x_seq)
        return probs.flatten()

    def save(self, filepath):
        """Save model parameters to disk."""
        params = self.get_params()
        params['meta'] = {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'lr': self.lr,
            't': self.t
        }
        np.savez_compressed(filepath, **params)

    def load(self, filepath):
        """Load model parameters from disk."""
        data = np.load(filepath, allow_pickle=True)
        self.Wf = data['Wf']
        self.Wi = data['Wi']
        self.Wc = data['Wc']
        self.Wo = data['Wo']
        self.bf = data['bf']
        self.bi = data['bi']
        self.bc = data['bc']
        self.bo = data['bo']
        self.Wy = data['Wy']
        self.by = data['by']
        if 'meta' in data:
            meta = data['meta'].item()
            self.input_dim = meta.get('input_dim', self.input_dim)
            self.hidden_dim = meta.get('hidden_dim', self.hidden_dim)
            self.output_dim = meta.get('output_dim', self.output_dim)
            self.lr = meta.get('lr', self.lr)


class ModelManager:
    def __init__(self, base_dir=None):
        if base_dir is None:
            self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        else:
            self.base_dir = os.path.abspath(base_dir)

        self.models_dir = os.path.join(self.base_dir, 'models')
        os.makedirs(self.models_dir, exist_ok=True)

        self.is_training = False
        self.stop_requested = False
        self.current_genre = None
        self.current_epoch = 0
        self.total_epochs = 0
        self.current_loss = 0.0
        self.loss_history = []
        self.training_thread = None
        self.status_message = "Idle"
        self.engine_type = "TensorFlow / Keras" if HAS_TF else "Vectorized Native LSTM"

    def get_status(self):
        return {
            'is_training': self.is_training,
            'genre': self.current_genre,
            'current_epoch': self.current_epoch,
            'total_epochs': self.total_epochs,
            'current_loss': round(self.current_loss, 4),
            'loss_history': self.loss_history,
            'status_message': self.status_message,
            'engine_type': self.engine_type,
            'has_tf': HAS_TF
        }

    def stop_training(self):
        if self.is_training:
            self.stop_requested = True
            self.status_message = "Stopping training..."
            return True, "Training stop requested."
        return False, "No active training to stop."

    def start_training(self, genre='piano', epochs=30, batch_size=8, learning_rate=0.005, seq_length=16):
        if self.is_training:
            return False, "Training is already in progress."

        self.is_training = True
        self.stop_requested = False
        self.current_genre = genre
        self.current_epoch = 0
        self.total_epochs = epochs
        self.current_loss = 0.0
        self.loss_history = []
        self.status_message = f"Initializing training for {genre}..."

        self.training_thread = threading.Thread(
            target=self._run_training_worker,
            args=(genre, epochs, batch_size, learning_rate, seq_length),
            daemon=True
        )
        self.training_thread.start()
        return True, f"Training started for {genre} ({epochs} epochs)."

    def _run_training_worker(self, genre, epochs, batch_size, learning_rate, seq_length):
        from preprocessing import MidiPreprocessor
        preprocessor = MidiPreprocessor(self.base_dir)

        try:
            self.status_message = f"Loading preprocessed sequences for {genre}..."
            data = preprocessor.load_preprocessed_data(genre)
            X = data['X']
            y = data['y']
            vocab_size = data['vocab_size']

            if HAS_TF:
                self._train_with_tf(X, y, vocab_size, genre, epochs, batch_size, learning_rate)
            else:
                self._train_with_native(X, y, vocab_size, genre, epochs, batch_size, learning_rate)

            self.status_message = f"Training completed successfully! Model saved for {genre}."
        except Exception as e:
            self.status_message = f"Training error: {str(e)}"
            print(f"Training exception: {e}")
        finally:
            self.is_training = False
            self.stop_requested = False

    def _train_with_native(self, X, y, vocab_size, genre, epochs, batch_size, learning_rate):
        model = NativeLSTM(input_dim=1, hidden_dim=64, output_dim=vocab_size, lr=learning_rate)
        
        # Load existing weights if present to allow continuous learning
        model_path = os.path.join(self.models_dir, f"lstm_model_{genre}.npz")
        if os.path.exists(model_path):
            try:
                model.load(model_path)
            except Exception:
                pass

        for epoch in range(1, epochs + 1):
            if self.stop_requested:
                self.status_message = "Training halted by user."
                break

            loss = model.train_epoch(X, y, batch_size=batch_size)
            self.current_epoch = epoch
            self.current_loss = loss
            self.loss_history.append({'epoch': epoch, 'loss': round(loss, 4)})
            self.status_message = f"Epoch {epoch}/{epochs} - Loss: {loss:.4f}"
            time.sleep(0.01)  # Yield CPU lightly

        # Save model
        model.save(model_path)
        # Save training metadata
        meta_path = os.path.join(self.models_dir, f"lstm_meta_{genre}.json")
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump({
                'genre': genre,
                'epochs_completed': self.current_epoch,
                'final_loss': self.current_loss,
                'loss_history': self.loss_history,
                'vocab_size': vocab_size,
                'engine': 'NativeLSTM',
                'updated_at': time.strftime("%Y-%m-%d %H:%M:%S")
            }, f, indent=2)

    def _train_with_tf(self, X, y, vocab_size, genre, epochs, batch_size, learning_rate):
        model = Sequential([
            Input(shape=(X.shape[1], X.shape[2])),
            LSTM(128, return_sequences=True),
            Dropout(0.2),
            LSTM(128),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(vocab_size, activation='softmax')
        ])
        
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=opt)

        class ProgressCallback(Callback):
            def __init__(self, manager):
                super().__init__()
                self.mgr = manager

            def on_epoch_end(self, epoch, logs=None):
                if self.mgr.stop_requested:
                    self.model.stop_training = True
                loss = logs.get('loss', 0.0)
                self.mgr.current_epoch = epoch + 1
                self.mgr.current_loss = float(loss)
                self.mgr.loss_history.append({'epoch': epoch + 1, 'loss': round(float(loss), 4)})
                self.mgr.status_message = f"Epoch {epoch + 1}/{self.mgr.total_epochs} - Loss: {loss:.4f}"

        model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[ProgressCallback(self)],
            verbose=0
        )

        model_path = os.path.join(self.models_dir, f"lstm_model_{genre}.keras")
        model.save(model_path)

        meta_path = os.path.join(self.models_dir, f"lstm_meta_{genre}.json")
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump({
                'genre': genre,
                'epochs_completed': self.current_epoch,
                'final_loss': self.current_loss,
                'loss_history': self.loss_history,
                'vocab_size': vocab_size,
                'engine': 'TensorFlow / Keras',
                'updated_at': time.strftime("%Y-%m-%d %H:%M:%S")
            }, f, indent=2)

    def load_model_for_inference(self, genre='piano'):
        """Load trained model weights and preprocessed vocabulary for generation."""
        from preprocessing import MidiPreprocessor
        preprocessor = MidiPreprocessor(self.base_dir)
        data = preprocessor.load_preprocessed_data(genre)

        vocab_size = data['vocab_size']
        seq_length = data['seq_length']
        int_to_note = data['int_to_note']
        note_to_int = data['note_to_int']
        network_input = data['network_input']

        # Check for trained model
        npz_path = os.path.join(self.models_dir, f"lstm_model_{genre}.npz")
        keras_path = os.path.join(self.models_dir, f"lstm_model_{genre}.keras")

        if HAS_TF and os.path.exists(keras_path):
            model = tf.keras.models.load_model(keras_path)
            predict_fn = lambda seq: model.predict(seq.reshape(1, seq_length, 1), verbose=0)[0]
        elif os.path.exists(npz_path):
            native_model = NativeLSTM(input_dim=1, hidden_dim=64, output_dim=vocab_size)
            native_model.load(npz_path)
            predict_fn = lambda seq: native_model.predict(seq.reshape(-1, 1))
        else:
            # If no trained model yet, initialize and quick-train 10 epochs automatically
            native_model = NativeLSTM(input_dim=1, hidden_dim=64, output_dim=vocab_size, lr=0.01)
            for _ in range(12):
                native_model.train_epoch(data['X'], data['y'], batch_size=8)
            native_model.save(npz_path)
            predict_fn = lambda seq: native_model.predict(seq.reshape(-1, 1))

        return {
            'predict_fn': predict_fn,
            'vocab_size': vocab_size,
            'seq_length': seq_length,
            'int_to_note': int_to_note,
            'note_to_int': note_to_int,
            'network_input': network_input
        }
