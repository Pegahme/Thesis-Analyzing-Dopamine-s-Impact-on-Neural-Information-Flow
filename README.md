# Thesis – Analyzing Dopamine’s Impact on Neural Information Flow

This repository accompanies the Master’s thesis and contains the full analysis workflow and results.  
It mirrors the thesis chapters: **Chapter 3 (Methods)** and **Chapter 4 (Results)**, plus complementary analyses.  

The project investigates how dopamine influences neural information flow in Parkinson’s disease using EEG, source estimation, dimensionality reduction, and effective connectivity inference.

---

## Repository Structure

### 📂 Chapter_3_Methods
Analysis pipelines corresponding to the methods section.

- **1. Preprocessing**  
  Scripts and notebooks for EEG preprocessing: filtering, referencing, ICA artifact correction, and trial rejection/interpolation.

- **2. Source_Estimation**  
  Code and resources for source reconstruction and extracting regional time series.  

- **3. Subject and Segment Selection**  
  Notebooks for selecting participants, sessions, and harmonizing epoch counts across subjects.

- **4. PCA+ICA**  
  Scripts and notebooks for dimensionality reduction (group-level PCA, ICA).

- **5. Inference and aggregation (gauss / ksg)**  
  Pipelines for effective connectivity inference:  
  - *Gaussian estimator* (linear)  
  - *KSG estimator* (nonlinear, nearest-neighbor based)  
  Each folder contains inference scripts and aggregated adjacency matrices.

---

### 📂 Chapter_4_Results
Jupyter notebooks reproducing the main thesis results.

- **Global analyses** (`1. Global_gauss.ipynb`, `1. Global_ksg.ipynb`)  
- **Reproducibility & similarity** (`2. ...`)  
- **Temporal lag structure** (`3. ...`)  
- **Group comparisons**: Healthy vs. PD, ON vs. OFF medication  
- **Estimator comparison**: KSG vs. Gaussian

---

### 📂 Results_Complementary
Supporting analyses and additional figures.

- **Global-level**, **ROI-level**, **Edgewise** → complementary analyses at different resolutions.  
- **Comparison/** → supporting plots and .csv files
- **figs/** → supporting plots (e.g., TE lag distributions, group contrasts).  
- Additional files:  
  - `Anatomical spatial maps.ipynb` – anatomical visualization of components  
  - `IC_anatomy_labels_top3.csv` – top anatomical labels for independent components

---

### 📄 subject_session_metadata.csv
Metadata file mapping subjects to:  
- Group (Parkinson’s disease vs. control)  
- Medication status (ON vs. OFF)  
- Session identifiers  

Used for reproducibility of group-level analyses.

---

## Requirements
The analysis was performed in a Python environment with the following main libraries:

- Python 
- [MNE-Python](https://mne.tools)  
- [PyPREP](https://github.com/sappelhoff/pyprep)  
- [AutoReject](https://autoreject.github.io)  
- [IDTxl](https://github.com/pwollstadt/IDTxl)  
- SLURM (for batch/cluster job submission)

---

## Notes
- This repository is intended as a **research record** for examiners.  
- Scripts and notebooks correspond directly to the thesis chapters.  
- Large raw datasets are not included here but are publicly available via OpenNeuro.  
