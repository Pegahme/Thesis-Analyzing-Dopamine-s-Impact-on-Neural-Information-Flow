import numpy as np
from pathlib import Path
import json
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from kneed import KneeLocator
from mne import create_info, EpochsArray
from mne.preprocessing import ICA
import joblib

# === CONFIG ===
base = Path("/lustre/majlepy2/myproject/50_epochs_tc3d")       # Input dir with localized timecourses
out_dir = Path("/lustre/majlepy2/myproject/50_epochs_idtxl")   # Output dir for PCA+ICA results
out_dir.mkdir(parents=True, exist_ok=True)
sfreq = 20.0  # Sampling frequency (matches preprocessing)

# === Collect all input arrays ===
# Each file: (nodes × samples × epochs)
paths = sorted(base.glob("*_68localized_50epochs.npy"))

arrays = []
meta = []
total_so_far = 0

for p in paths:
    arr = np.load(p)
    n_nodes, n_samples, n_epochs = arr.shape

    # Flatten epochs into long 2D array: (samples*epochs, nodes)
    arr_flat = arr.reshape(n_nodes, -1).T
    arrays.append(arr_flat)

    # Track file-specific metadata for later reshaping
    meta.append({
        "file": str(p),
        "n_epochs": n_epochs,
        "n_samples": n_samples,
        "start": total_so_far,
        "stop": total_so_far + arr_flat.shape[0]
    })

    total_so_far += arr_flat.shape[0]
    print(f"Included: {p.name}: {arr.shape} → {arr_flat.shape}")

# === Stack across subjects ===
data = np.vstack(arrays)  # (samples*epochs*subjects, nodes)
print("Data shape before PCA:", data.shape)

# === Standardize features ===
scaler = StandardScaler()
X_z = scaler.fit_transform(data)
print("After z-scoring:", X_z.shape)

# === PCA (first: determine optimal #components) ===
pca_full = PCA()
pca_full.fit(X_z)
explained_var = pca_full.explained_variance_ratio_
cum_explained = np.cumsum(explained_var)

# Use "elbow" method (or fallback to 75% variance explained)
x = np.arange(1, len(cum_explained) + 1)
knee = KneeLocator(x, cum_explained, curve='concave', direction='increasing')
elbow_point = knee.knee if knee.knee else np.argmax(cum_explained >= 0.75) + 1
print(f"Elbow at {elbow_point} components ({cum_explained[elbow_point - 1]:.2%} variance explained)")

# === PCA with selected number of components ===
pca = PCA(n_components=elbow_point)
X_pca = pca.fit_transform(X_z)
print(f"PCA output shape: {X_pca.shape}")

# === ICA on PCA output ===
n_components = X_pca.shape[1]
ch_names = [f"PC{i+1}" for i in range(n_components)]
info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")

# Wrap into MNE EpochsArray (shape: samples × components × 1 “epoch”)
epochs_data = X_pca[:, :, np.newaxis]
epochs = EpochsArray(epochs_data, info, verbose=False)

# Fit ICA (FastICA)
ica = ICA(n_components=n_components, method="fastica",
          random_state=42, max_iter="auto", verbose=True)
ica.fit(epochs)

# Extract ICA source time series (samples × components)
sources = ica.get_sources(epochs).get_data().squeeze(-1)

# === Save explained variance (PCA diagnostics) ===
np.save(out_dir / "group_pca_explained_variance_ratio.npy", pca.explained_variance_ratio_)
np.save(out_dir / "group_pca_cum_explained.npy", cum_explained)

# === Save ICA outputs per subject/session ===
brain_ts = sources

for m in meta:
    start, stop = m["start"], m["stop"]
    n_epochs = m["n_epochs"]
    n_samples = m["n_samples"]

    # Slice subject-specific data and reshape back into (components × samples × epochs)
    this_Y = brain_ts[start:stop, :].T.reshape(n_components, n_samples, n_epochs)

    # Save with updated filename marker
    stem = Path(m["file"]).stem.replace("_68localized_50epochs", "_pca_ica_23_130_50")
    fname = f"{stem}.npy"

    np.save(out_dir / fname, this_Y)
    print(f"Saved {fname}: {this_Y.shape}")

# === Save group-level outputs (for reuse) ===
np.save(out_dir / "group_ica_brain_timeseries_flat.npy", brain_ts)
np.save(out_dir / "group_ica_mixing_matrix.npy", ica.mixing_matrix_)
np.save(out_dir / "group_ica_sources_flat.npy", sources)

# Save fitted models for later reproducibility
joblib.dump(ica, out_dir / "group_ica_model.joblib")
joblib.dump(pca, out_dir / "group_pca_model.joblib")
joblib.dump(scaler, out_dir / "group_scaler.joblib")

# === Save metadata for reshaping back later ===
with open(out_dir / "flattening_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"\nDone. All outputs saved in: {out_dir}")
