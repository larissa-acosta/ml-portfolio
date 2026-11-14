import pickle

import gradio as gr
import matplotlib
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
import tensorflow as tf
from tensorflow.keras.models import load_model

try:
    import spaces  # available on Hugging Face Spaces (ZeroGPU hardware requires a @spaces.GPU function)
    gpu = spaces.GPU
except ImportError:  # running locally
    def gpu(fn):
        return fn

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "larissaac/steel-energy-lstm"
SEQ_LENGTH = 60
FEATURES = [
    "Usage_kWh",
    "Lagging_Current_Reactive.Power_kVarh",
    "Leading_Current_Reactive_Power_kVarh",
    "Lagging_Current_Power_Factor",
    "Leading_Current_Power_Factor",
]



# Models saved with newer Keras versions include a `quantization_config` key that older
# Keras releases reject; drop it when deserializing Dense layers.
_orig_dense_from_config = tf.keras.layers.Dense.from_config.__func__


def _compat_dense_from_config(cls, config):
    config = {k: v for k, v in config.items() if k != "quantization_config"}
    return _orig_dense_from_config(cls, config)


tf.keras.layers.Dense.from_config = classmethod(_compat_dense_from_config)

model = load_model(hf_hub_download(REPO, "lstm_model.keras"), compile=False)
with open(hf_hub_download(REPO, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)


@gpu
def _zerogpu_placeholder():
    """ZeroGPU Spaces require at least one @spaces.GPU function. The model is TensorFlow (CPU),
    so the real analysis below runs in the main process and does not use the GPU worker."""
    return 0


def analyze(file, sigma, whole_series):
    df = pd.read_csv(file if isinstance(file, str) else file.name)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise gr.Error(f"Missing columns: {missing}")
    data = df[FEATURES].dropna().values
    if len(data) <= SEQ_LENGTH + 1:
        raise gr.Error(f"Need more than {SEQ_LENGTH + 1} rows.")

    # Same protocol as the paper: chronological 80/20 split, evaluation on the final 20%
    # (the previous SEQ_LENGTH rows are only used as context for the first window).
    if not whole_series:
        n_train = int(len(data) * 0.8)
        data = data[n_train - SEQ_LENGTH:]

    scaled = scaler.transform(data)
    X = np.array([scaled[i:i + SEQ_LENGTH] for i in range(len(scaled) - SEQ_LENGTH)])
    y = data[SEQ_LENGTH:, 0]

    pred_scaled = model.predict(X, batch_size=1024, verbose=0).flatten()
    dummy = np.zeros((len(pred_scaled), len(FEATURES)))
    dummy[:, 0] = pred_scaled
    pred = scaler.inverse_transform(dummy)[:, 0]

    err = np.abs(y - pred)
    mae = err.mean()
    threshold = mae + sigma * err.std()
    flagged = np.where(err > threshold)[0]

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(y, label="Actual", alpha=0.7)
    ax.plot(pred, label="Forecast", alpha=0.7)
    ax.scatter(flagged, y[flagged], color="red", s=18, label=f"Flagged ({len(flagged)})", zorder=5)
    ax.set_xlabel("Time step (15 min)")
    ax.set_ylabel("Usage (kWh)")
    ax.legend()
    fig.tight_layout()

    table = pd.DataFrame({
        "step": flagged,
        "actual_kWh": y[flagged].round(2),
        "forecast_kWh": pred[flagged].round(2),
        "error_kWh": err[flagged].round(2),
    }).sort_values("error_kWh", ascending=False).head(50)

    summary = (f"MAE: {mae:.2f} kWh | threshold (MAE + {sigma}σ): {threshold:.2f} kWh | "
               f"flagged: {len(flagged)} of {len(err)} periods ({100 * len(flagged) / len(err):.2f}%)")
    return summary, fig, table


demo = gr.Interface(
    fn=analyze,
    inputs=[
        gr.File(label="CSV (Steel_industry_data format)"),
        gr.Slider(0.5, 3.0, value=1.5, step=0.1, label="Sigma multiplier for the bottleneck threshold"),
        gr.Checkbox(value=False, label="Analyze the whole file (default: only the final 20%, the test period used in the paper)"),
    ],
    outputs=[gr.Textbox(label="Summary"), gr.Plot(label="Forecast vs actual"), gr.Dataframe(label="Top flagged periods")],
    title="Steel plant energy bottleneck detection (LSTM)",
    description="Forecasts next-step consumption from the previous 15 hours; large forecast errors are flagged as potential bottlenecks. Academic project (PUCPR).",
)

if __name__ == "__main__":
    demo.launch()
