# File: preprocessing_functions.py (modified for log file instead of Excel)

import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import mne
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr
from datetime import datetime
import plotly.graph_objects as go
import shutil
import pandas as pd

############################################
# Logging Setup
############################################
def setup_logging(log_dir, subject_id):
    os.makedirs(log_dir, exist_ok=True)
    log_filename = f"log_{subject_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_path = os.path.join(log_dir, log_filename)
    with open(log_path, 'w') as f:
        f.write(f"Log started for subject: {subject_id}\n")
    return log_path

############################################
def log_dropped_channels(log_path, subject_id, raw, required_channels):
    channels_to_drop = [ch for ch in required_channels if ch in raw.info['ch_names']]
    dropped_channels_str = ', '.join(channels_to_drop) if channels_to_drop else "Required channels missing"

    if channels_to_drop:
        raw.drop_channels(channels_to_drop)

    with open(log_path, 'a') as f:
        f.write(f"{subject_id} - Dropped Channels: {dropped_channels_str}\n")

    return raw

############################################
def log_crop_data(log_path, subject_id, tmin, tmax, duration):
    with open(log_path, 'a') as f:
        f.write(f"{subject_id} - Crop Data:\n")
        f.write(f"    tmin: {tmin if tmin is not None else 'Not defined'}\n")
        f.write(f"    tmax: {tmax if tmax is not None else 'Not defined'}\n")
        f.write(f"    duration: {duration}\n")

############################################
def log_bad_channels(log_path, subject_id, raw_prep):
    bad_channels = raw_prep.info.get("bads", [])
    bad_channels_str = ', '.join(bad_channels) if bad_channels else "None"

    with open(log_path, 'a') as f:
        f.write(f"{subject_id} - Bad Channels: {bad_channels_str}\n")

    return raw_prep

############################################
def log_ica_parameters(log_path, subject_id, n_components, l_freq, h_freq, random_state):
    with open(log_path, 'a') as f:
        f.write(f"{subject_id} - ICA Parameters:\n")
        f.write(f"    n_components: {n_components}\n")
        f.write(f"    l_freq: {l_freq}, h_freq: {h_freq if h_freq is not None else 'Not defined'}\n")
        f.write(f"    random_state: {random_state}\n")

############################################
def log_excluded_ica_sources(log_path, subject_id, ica_exclude):
    exclude_str = ', '.join(map(str, ica_exclude)) if ica_exclude else "None"
    with open(log_path, 'a') as f:
        f.write(f"{subject_id} - Excluded ICA Sources: {exclude_str}\n")

############################################
def regress_out_accelerometer_artifacts(raw_filtered, accel_ch_names=['X', 'Y', 'Z'], l_freq=4, h_freq=6, alpha=1.0):
    raw_reg = raw_filtered.copy()
    eeg_picks = mne.pick_types(raw_filtered.info, eeg=True, exclude='bads')
    accel_picks = mne.pick_channels(raw_filtered.info['ch_names'], accel_ch_names)

    eeg_data = raw_filtered.get_data(picks=eeg_picks)
    accel_data = raw_filtered.get_data(picks=accel_picks)

    accel_data_filtered = mne.filter.filter_data(accel_data, sfreq=raw_filtered.info['sfreq'], l_freq=l_freq, h_freq=h_freq, verbose=False)
    accel_predictors = np.column_stack([accel_data_filtered.T, np.ones(accel_data_filtered.shape[1])])

    eeg_cleaned = np.zeros_like(eeg_data)
    model = Ridge(alpha=alpha)

    for ch_idx in range(eeg_data.shape[0]):
        y = eeg_data[ch_idx, :]
        model.fit(accel_predictors, y)
        y_pred = model.predict(accel_predictors)
        eeg_cleaned[ch_idx, :] = y - y_pred

    raw_reg._data[eeg_picks, :] = eeg_cleaned
    return raw_reg, accel_data_filtered

############################################
def log_eeg_regression_metrics(raw_before, raw_reg, subject_id, save_dir, filename=None):
    eeg_picks = mne.pick_types(raw_before.info, eeg=True, exclude='bads')
    ch_names = np.array(raw_before.ch_names)[eeg_picks]
    data_before = raw_before.get_data(picks=eeg_picks)
    data_after = raw_reg.get_data(picks=eeg_picks)

    rms_diffs, var_reductions, correlations, r_squared = [], [], [], []

    for i in range(len(eeg_picks)):
        x = data_before[i]
        y = data_after[i]
        rms = np.sqrt(np.mean((x - y) ** 2))
        var_diff = np.var(x) - np.var(y)
        corr, _ = pearsonr(x, y)
        r2 = 1 - np.var(y - x) / np.var(x)

        rms_diffs.append(rms)
        var_reductions.append(var_diff)
        correlations.append(corr)
        r_squared.append(r2)

    metrics_df = pd.DataFrame({
        'subject': subject_id,
        'channel': ch_names,
        'rms_difference': rms_diffs,
        'variance_reduction': var_reductions,
        'correlation': correlations,
        'r_squared': r_squared
    })

    if filename is None:
        filename = f'regression_metrics_{subject_id}.csv'
    filepath = os.path.join(save_dir, filename)

    metrics_df.to_csv(filepath, sep=';', index=False)
    print(f"Regression metrics saved to: {filepath}")
    return metrics_df

############################################
def save_preprocessed_data(preproc_data, save_dir, subject_id, session_id=None, notebook_name=None, log_path=None):

    os.makedirs(save_dir, exist_ok=True)

    # Construct filename
    filename = f"{subject_id}_{session_id}_preprocessed_SimonTask_epo.fif" if session_id else f"{subject_id}_preprocessed_SimonTask_epo.fif"
    fif_path = os.path.join(save_dir, filename)

    # Save EEG data
    preproc_data.save(fif_path, overwrite=True)
    print(f"Preprocessed data saved: {fif_path}")

    # Backup notebook
    notebook_backup_path = None
    if notebook_name:
        notebook_source = os.path.join(os.getcwd(), notebook_name)
        notebook_backup_path = os.path.join(save_dir, notebook_name)

        if os.path.exists(notebook_source):
            shutil.copy(notebook_source, notebook_backup_path)
            print(f"Notebook backed up to: {notebook_backup_path}")
        else:
            print(f"Warning: Notebook '{notebook_name}' not found in {os.getcwd()}.")

    # === Log both paths ===
    if log_path:
        with open(log_path, 'a') as f:
            f.write(f"Final preprocessed .fif saved: {fif_path}\n")
            if notebook_backup_path:
                f.write(f"Notebook backup saved: {notebook_backup_path}\n")

    return fif_path, notebook_backup_path
############################################
def plot_var(raw, save_path, title=None):
    eeg_data = raw.get_data(picks='eeg', reject_by_annotation='NaN')
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # Variance over time
    ax[0].plot(raw.times, np.nanvar(eeg_data, axis=0), color='b')
    ax[0].set_title("EEG Variance Over Time")
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Variance")

    # Variance distribution
    ax[1].hist(np.nanvar(eeg_data, axis=1), bins=24, color='r', histtype='step', linewidth=1.5)
    ax[1].set_title("EEG Channel Variance Distribution")
    ax[1].set_xlabel("Variance")
    ax[1].set_ylabel("Number of Channels")

    # Optional overall title
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Variance plot saved at {save_path}")
    return fig, ax

############################################

"""
def plot_var(raw, save_path):
    eeg_data = raw.get_data(picks='eeg', reject_by_annotation='NaN')
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(raw.times, np.nanvar(eeg_data, axis=0), color='b')
    ax[0].set_title("EEG Variance Over Time")
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Variance")
    ax[1].hist(np.nanvar(eeg_data, axis=1), bins=24, color='r', histtype='step', linewidth=1.5)
    ax[1].set_title("EEG Channel Variance Distribution")
    ax[1].set_xlabel("Variance")
    ax[1].set_ylabel("Number of Channels")
    plt.tight_layout()
    plt.show()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Variance plot saved at {save_path}")
    return fig, ax
"""
############################################

def plot_var_epochs(epochs_ar, save_dir, subject_id, session_id=None, log_path=None):
    # Prepare data
    eeg_data = epochs_ar.get_data(picks='eeg')  # (n_epochs, n_ch, n_times)
    data_2d = eeg_data.transpose(1, 0, 2).reshape(eeg_data.shape[1], -1)  # (n_ch, n_epochs*n_times)

    # Make figure
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # Variance over time (blue)
    time_var = np.var(data_2d, axis=0)
    ax[0].plot(
        np.linspace(0, epochs_ar.tmax * eeg_data.shape[0], time_var.size),
        time_var,
        color='blue'
    )
    ax[0].set_title("EEG Variance Over Time")
    ax[0].set_xlabel("Time (pseudo-concatenated)")
    ax[0].set_ylabel("Variance")

    # Variance per channel (red)
    chan_var = np.var(data_2d, axis=1)
    ax[1].hist(chan_var, bins=24, histtype='step', linewidth=1.5, color='red')
    ax[1].set_title("EEG Channel Variance Distribution")
    ax[1].set_xlabel("Variance")
    ax[1].set_ylabel("Number of Channels")

    fig.tight_layout()

    # Build output path
    parts = [subject_id]
    if session_id:
        parts.append(session_id)
    parts.append("epochs_variance.png")
    out = Path(save_dir) / "_".join(parts)
    out.parent.mkdir(parents=True, exist_ok=True)

    # No-clobber increment
    final = out
    i = 1
    while final.exists():
        final = out.with_name(out.stem + f"_{i}.png")
        i += 1

    fig.savefig(final, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[QC] Epoch variance plot saved: {final}")

    if log_path:
        with open(log_path, 'a') as f:
            f.write(f"Epoch variance plot saved: {final}\n")

    return final

"""

def plot_var_epochs(epochs_ar, save_dir, subject_id, session_id=None, log_path=None):
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    eeg_data = epochs_ar.get_data(picks='eeg')
    data_2d = eeg_data.transpose(1, 0, 2).reshape(eeg_data.shape[1], -1)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    time_var = np.var(data_2d, axis=0)
    ax[0].plot(np.linspace(0, epochs_ar.tmax * eeg_data.shape[0], time_var.size), time_var, color='b')
    ax[0].set_title("EEG Variance Over Time")
    ax[0].set_xlabel("Time (pseudo-concatenated)")
    ax[0].set_ylabel("Variance")

    chan_var = np.var(data_2d, axis=1)
    ax[1].hist(chan_var, bins=24, color='r', histtype='step', linewidth=1.5)
    ax[1].set_title("EEG Channel Variance Distribution")
    ax[1].set_xlabel("Variance")
    ax[1].set_ylabel("Number of Channels")

    plt.tight_layout()
    plt.show()

    filename_parts = [subject_id]
    if session_id:
        filename_parts.append(session_id)
    filename_parts.append("ICA_components_plot.png")
    filename = "_".join(filename_parts)
    save_path = os.path.join(save_dir, filename)

    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Variance plot saved at {save_path}")

    # === Log to file if log_path is given ===
    if log_path:
        with open(log_path, 'a') as f:
            f.write(f"Epoch variance plot saved: {save_path}\n")

    return fig, ax
"""
############################################
def ica_and_plot_sources(raw_prep, save_dir, subject_id, session_id, n_components, l_freq, h_freq, random_state):
    ica_raw = raw_prep.copy()
    os.makedirs(save_dir, exist_ok=True)
    ica_filtered = ica_raw.filter(l_freq=l_freq, h_freq=h_freq)
    ica = mne.preprocessing.ICA(n_components=n_components, method="fastica", random_state=random_state, max_iter="auto")
    ica.fit(ica_filtered)
    if 'VEOG' in raw_prep.ch_names:
        eog_inds, scores = ica.find_bads_eog(ica_filtered, ch_name='VEOG')
        print(f"VEOG identified ICA components: {eog_inds}")
        if isinstance(eog_inds, (list, np.ndarray)) and len(eog_inds) > 0:
            ica.plot_scores(scores)
            ica.plot_components(picks=eog_inds)
        else:
            print("No blink-related components identified using VEOG.")
    else:
        print("VEOG channel not found in the data.")
    ica_sources = ica.get_sources(ica_filtered).get_data()
    time_points = ica_filtered.times
    fig = go.Figure([
        go.Scatter(x=time_points, y=ica_sources[i, :] + i * 10, mode='lines', name=f'ICA {i + 1}')
        for i in range(ica_sources.shape[0])
    ])
    fig.update_layout(
        title=f'ICA Components - {subject_id} - {session_id}',
        xaxis_title='Time (s)',
        yaxis_title='Amplitude'
    )
    fig_path = os.path.join(save_dir, f"{subject_id}_{session_id}_ICA_sources.html")
    fig.write_html(fig_path)
    print(f"ICA interactive plot saved to: {fig_path}")
    ica.plot_sources(ica_filtered.copy().pick_types(eeg=True), show=True)
    return ica, ica_raw
