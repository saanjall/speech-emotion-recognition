"""
models.py - the three Keras models compared in this project.

1. MLP          : handcrafted 1-D feature vector  -> Dense layers            (baseline)
2. CNN          : 2-D log-mel "image"             -> 4 conv blocks -> GAP
3. CNN + BiLSTM : 2-D log-mel -> conv blocks (shrink frequency, KEEP time)
                  -> sequence of frame vectors -> Bidirectional LSTM
                  -> attention pooling (learns WHICH frames matter)        (main model)
"""
import keras                      # Keras 3 (installed with TensorFlow >= 2.16)
from keras import layers, ops     # keras.ops works on any backend (TensorFlow or PyTorch)

import config


# --------------------------------------------------------------------------
# Custom layer: additive (Bahdanau-style) attention pooling over time
# --------------------------------------------------------------------------
@keras.saving.register_keras_serializable(package="ser")
class AttentionPooling(layers.Layer):
    """Turns a sequence (batch, T, D) into one vector (batch, D).

        score_t = v^T tanh(W h_t + b)       one scalar per time step
        alpha   = softmax(score)            weights that sum to 1 over time
        context = sum_t alpha_t * h_t       weighted average of the LSTM outputs

    Emotional cues are often concentrated in a few frames (a shout, a laugh),
    so a learned weighted average works better than plain averaging.
    """

    def __init__(self, units=128, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        dim = int(input_shape[-1])
        self.W = self.add_weight(name="W", shape=(dim, self.units), initializer="glorot_uniform")
        self.b = self.add_weight(name="b", shape=(self.units,), initializer="zeros")
        self.v = self.add_weight(name="v", shape=(self.units, 1), initializer="glorot_uniform")
        super().build(input_shape)

    def call(self, h):
        score = ops.matmul(ops.tanh(ops.matmul(h, self.W) + self.b), self.v)   # (B, T, 1)
        alpha = ops.softmax(score, axis=1)                                     # (B, T, 1)
        return ops.sum(alpha * h, axis=1)                                      # (B, D)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units})
        return cfg


# --------------------------------------------------------------------------
# 1. MLP baseline
# --------------------------------------------------------------------------
def build_mlp(input_dim, n_classes=config.NUM_CLASSES):
    inp = keras.Input(shape=(input_dim,), name="features")
    x = inp
    for units, drop in [(512, 0.4), (256, 0.4), (128, 0.3)]:
        x = layers.Dense(units)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(drop)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inp, out, name="mlp")


# --------------------------------------------------------------------------
# Shared conv block: (Conv2D -> BN -> ReLU) x 2 -> MaxPool -> Dropout
# --------------------------------------------------------------------------
def conv_block(x, filters, pool, dropout):
    for _ in range(2):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=pool)(x)
    return layers.Dropout(dropout)(x)


# --------------------------------------------------------------------------
# 2. CNN on log-mel spectrograms
# --------------------------------------------------------------------------
def build_cnn(input_shape=(config.N_MELS, config.N_FRAMES, 1), n_classes=config.NUM_CLASSES):
    inp = keras.Input(shape=input_shape, name="logmel")
    x = inp
    for filters in [32, 64, 128, 256]:
        x = conv_block(x, filters, pool=(2, 2), dropout=0.25)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inp, out, name="cnn")


# --------------------------------------------------------------------------
# 3. CNN + BiLSTM + attention (main model)
# --------------------------------------------------------------------------
def build_cnn_bilstm(input_shape=(config.N_MELS, config.N_FRAMES, 1), n_classes=config.NUM_CLASSES):
    inp = keras.Input(shape=input_shape, name="logmel")            # (freq=128, time=130, 1)
    x = inp
    # Pool ONLY along frequency (pool_size=(k, 1)) so the time axis stays intact for the LSTM.
    for filters, pool in [(32, (2, 1)), (64, (2, 1)), (128, (2, 1)), (128, (4, 1))]:
        x = conv_block(x, filters, pool=pool, dropout=0.2)          # -> (4, 130, 128)
    # (freq, time, ch) -> (time, freq, ch) -> (time, freq*ch): one feature vector per frame
    x = layers.Permute((2, 1, 3))(x)
    x = layers.Reshape((input_shape[1], -1))(x)                     # (130, 512)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)   # (130, 256)
    x = layers.Dropout(0.3)(x)
    x = AttentionPooling(128, name="attention")(x)                  # (256,)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inp, out, name="cnn_bilstm")


def build_model(name, input_shape):
    """Factory used by train.py.  input_shape excludes the batch dimension."""
    if name == "mlp":
        model = build_mlp(input_shape[0])
    elif name == "cnn":
        model = build_cnn(input_shape)
    elif name == "cnn_bilstm":
        model = build_cnn_bilstm(input_shape)
    else:
        raise ValueError(f"Unknown model: {name}")
    model.compile(optimizer=keras.optimizers.Adam(config.LEARNING_RATE),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def load_keras_model(path):
    """Load a saved .keras model (the custom AttentionPooling layer is registered above)."""
    return keras.models.load_model(path, custom_objects={"AttentionPooling": AttentionPooling}, compile=False)
