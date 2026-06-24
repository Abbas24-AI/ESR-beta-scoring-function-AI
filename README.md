# ERβ-Score

**An interpretable machine-learning scoring function and web server for Estrogen Receptor β (ERβ)–guided drug discovery in triple-negative breast cancer (TNBC).**

ERβ-Score predicts the probability that a compound is an active ERβ modulator directly from its SMILES string, reports an **applicability-domain (AD)** confidence flag, evaluates **Lipinski drug-likeness**, and runs **on-demand structure-based docking** against the ERβ co-crystal structure (PDB: **7XWQ**) — all from a single Streamlit web app.

> Companion code for the manuscript *“ERβ-Score: An Interpretable Machine Learning–Based Scoring Function and Web Server for Estrogen Receptor β–Guided Drug Discovery in Triple-Negative Breast Cancer.”* See [Citation](#citation).

---

## Features

- **Activity prediction** — a Gradient Boosting Classifier trained on 988 curated ChEMBL ERβ compounds (CHEMBL242), using 39 interpretable 2D/3D RDKit descriptors. Returns a calibrated `P(active)` with an Active/Inactive badge.
- **Applicability domain** — k-nearest-neighbor (k = 5) Tanimoto similarity on ECFP4 (2048-bit) fingerprints, with an in-/out-of-domain flag so unreliable extrapolations are surfaced rather than hidden.
- **Drug-likeness** — Lipinski Rule-of-Five summary (MW, LogP, HBD, HBA, TPSA, rotatable bonds).
- **3D visualization** — interactive protein–ligand viewer rendered in-browser via 3Dmol.js (no plugin).
- **Structure-based docking** — one-click Smina docking into the ERβ pocket, with the docked pose rendered in the same viewer and the full scoring log.

---

## Quick start

```bash
# 1. Clone
git clone https://github.com/<your-org>/ESR_beta_New.git
cd ESR_beta_New

# 2. Create the environment (conda recommended — see INSTALL.md for details)
conda create -n erbeta python=3.10 -y
conda activate erbeta
pip install -r requirements.txt

# 3. (Optional, for the Docking tab) install the docking toolchain
conda install -c conda-forge smina openbabel -y

# 4. Launch
streamlit run app.py
```

The app opens at `http://localhost:8501`. Paste a SMILES string (e.g. `CC(=O)Oc1ccccc1C(=O)O`) and explore the **Prediction**, **3D Viewer**, and **Docking** tabs.

> Full setup, troubleshooting, and deployment instructions are in **[INSTALL.md](INSTALL.md)**.

---

## Repository structure

```
ESR_beta_New/
├── app.py                       # Streamlit web application (entry point)
├── requirements.txt             # Python dependencies (must be at repo root for Streamlit Cloud)
├── packages.txt                 # system libs for RDKit on Streamlit Cloud (libxrender1, etc.)
├── saved_model/                 # Serialized model artifacts (loaded by app.py)
│   ├── best_model.pkl           #   trained GradientBoosting classifier
│   ├── best_scaler.pkl          #   fitted StandardScaler
│   ├── feature_columns.json     #   39 descriptor names (ordered)
│   ├── feature_medians.json     #   training medians for imputation
│   ├── ad_config.json           #   AD method, k, threshold, fingerprint config
│   └── train_fps_ecfp4.npy      #   training ECFP4 fingerprint matrix (for AD)
├── 7XWQ.pdb                     # ERβ co-crystal structure (full complex)
├── docking_results/             # prepared receptor + docking outputs
│   └── receptor_protein_only.pdb
├── ESRB-notebook.ipynb          # end-to-end pipeline (curation → model → VS → docking)
├── ESRB_ML_ready.csv            # curated modeling dataset (SMILES + activity labels)
├── ESRB_train.csv / ESRB_test.csv   # scaffold-disjoint split
├── virtual_screening_results*.csv   # prospective screening outputs
├── fig_*.png                    # manuscript figures
└── README.md / INSTALL.md
```

> ⚠️ **Large files:** several screening libraries in this project (e.g. `TCM-R5 filtered.sdf` ≈ 70 MB, `IBscreen.csv` ≈ 30 MB, `virtual_screening_combined.csv` ≈ 55 MB) exceed GitHub’s 100 MB hard limit / 50 MB warning. Do **not** commit them directly — archive them on **Zenodo** and/or track them with **Git LFS**. A ready-to-use [`.gitignore`](.gitignore) excludes them by default. See INSTALL.md → *Handling large data files*.

---

## Model & method summary

| Component | Setting |
|---|---|
| Algorithm | Gradient Boosting Classifier (selected by mean CV PR-AUC) |
| Descriptors | 39 RDKit 2D physicochemical + Kier–Hall + EState/PEOE VSA + 3D shape (PMI, NPR, spherocity) |
| Data | 988 ERβ compounds from ChEMBL (CHEMBL242); active pIC₅₀ ≥ 7.5, inactive pIC₅₀ < 5.5 |
| Split | Scaffold-disjoint (Bemis–Murcko), 80/20 |
| Applicability domain | kNN-Tanimoto (k = 5), ECFP4 2048-bit, threshold ≈ 0.242 (5th percentile of training scores) |
| Conformers | ETKDGv3 + MMFF94 (random seed 42) |
| Docking | Smina, exhaustiveness 16, 9 modes; grid centered at (−3.244, −7.848, −39.600), box 22.5³ Å |

Reported scaffold-disjoint hold-out performance: **PR-AUC ≈ 0.905, ROC-AUC ≈ 0.864, MCC ≈ 0.578**. In-domain compounds substantially outperform out-of-domain compounds — by design, the AD flag identifies the latter. See the manuscript for full metrics, Y-randomization, and AD analysis.

---

## Inputs & outputs

- **Input:** a single SMILES string (entered in the sidebar).
- **Prediction tab:** activity probability gauge, Active/Inactive badge, AD score vs. threshold, Lipinski table.
- **3D Viewer tab:** MMFF94-optimized ligand conformer in the ERβ pocket (7XWQ).
- **Docking tab:** Smina best binding affinity (kcal/mol), docked pose, and scoring log. Grid parameters are editable in the sidebar.

---

## Reproducibility

A global random seed (`42`) is enforced across stochastic steps. The complete training/validation/screening pipeline is in `ESRB-notebook.ipynb`. Model artifacts in `saved_model/` are the exact objects loaded at runtime by `app.py`.

---

## Citation

If you use ERβ-Score, please cite:

> Khan A., Zahid M.A., Kouidri W., *et al.* **ERβ-Score: An Interpretable Machine Learning–Based Scoring Function and Web Server for Estrogen Receptor β–Guided Drug Discovery in Triple-Negative Breast Cancer.** *International Journal of Molecular Sciences*, 2025. *(in revision)*

A `CITATION.cff` and the archived dataset DOI will be added on acceptance.

---

## Disclaimer

ERβ-Score is a research tool for **prioritization and triage** within a multi-stage computational workflow. Predictions are computational, are bounded by the applicability domain, and are **not** a substitute for experimental validation. It is not intended for clinical or diagnostic use.

## License

Released under the MIT License (add a `LICENSE` file before publishing). Third-party tools (RDKit, scikit-learn, Smina, Open Babel, 3Dmol.js) retain their own licenses.
