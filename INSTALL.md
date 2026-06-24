# Installation & Usage Guide — ERβ-Score

This guide walks through setting up and running the ERβ-Score Streamlit app (`streamlit run app.py`), including the optional docking toolchain and public deployment.

---

## 1. Prerequisites

- **Python 3.10+**
- **conda** (Miniconda or Anaconda) — strongly recommended, because RDKit, Smina, and Open Babel install most reliably from `conda-forge`.
- ~2 GB free disk for the environment.
- OS: tested on macOS and Linux. On Windows, use WSL2 for the docking tools (Smina/Open Babel are easiest under Linux).

---

## 2. Set up the environment

```bash
# clone
git clone https://github.com/<your-org>/ESR_beta_New.git
cd ESR_beta_New

# create & activate
conda create -n erbeta python=3.10 -y
conda activate erbeta

# core Python dependencies
pip install -r requirements.txt
```

`requirements.txt` pins the core stack:

```
streamlit>=1.35.0
rdkit>=2023.9.0
scikit-learn>=1.3.0
joblib>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
```

> If `pip install rdkit` fails on your platform, install it from conda instead:
> `conda install -c conda-forge rdkit`

---

## 3. Install the docking toolchain (optional — only for the Docking tab)

The **Prediction**, **3D Viewer**, Lipinski, and AD features work without anything extra. The **Docking** tab additionally needs **Smina** and **Open Babel** on your `PATH`:

```bash
conda install -c conda-forge smina openbabel -y
```

Verify:

```bash
smina --version
obabel -V
```

`app.py` locates `smina` via `shutil.which` and a few common conda prefixes. If docking reports *“smina not found,”* make sure the `erbeta` environment is active when you launch Streamlit.

---

## 4. Run the app

```bash
conda activate erbeta
streamlit run app.py
```

Open the URL Streamlit prints (default `http://localhost:8501`).

**Workflow inside the app:**
1. Enter a SMILES string in the sidebar (e.g. estradiol `C[C@]12CC[C@H]3[C@@H](CC[C@H]4=CC(=O)CC[C@H]34)[C@@H]1CC[C@@H]2O`).
2. **Prediction** tab → activity probability, Active/Inactive badge, AD flag, Lipinski table.
3. **3D Viewer** tab → ligand conformer in the ERβ (7XWQ) pocket.
4. **Docking** tab → adjust the grid if needed, click **Run Smina Docking**, read the affinity + pose.

A full SMILES → docked-complex run typically takes ~1–3 minutes on standard hardware.

---

## 5. Handling large data files (important for GitHub)

Several files in this project exceed GitHub limits (100 MB hard / 50 MB warning):

| File | Approx. size |
|---|---|
| `TCM-R5 filtered.sdf` | 70 MB |
| `virtual_screening_combined.csv` | 55 MB |
| `virtual_screening_results_ibscreen.csv` | 44 MB |
| `IBscreen.csv` | 31 MB |

**Do not commit these directly.** Two recommended options:

**Option A — Zenodo (recommended for archival + DOI):**
Upload the raw screening libraries and full result CSVs to Zenodo, get a DOI, and link it from the README / Data Availability statement. Keep only the small, essential files (`saved_model/`, `app.py`, curated `ESRB_ML_ready.csv`, split files) in Git.

**Option B — Git LFS:**
```bash
git lfs install
git lfs track "*.sdf" "virtual_screening_*.csv" "IBscreen.csv"
git add .gitattributes
```

A ready-made [`.gitignore`](.gitignore) already excludes the large libraries, OS cruft, and Python caches. Review it before your first commit.

---

## 6. First push to GitHub

```bash
git init
git add README.md INSTALL.md .gitignore app.py requirements.txt saved_model 7XWQ.pdb docking_results/receptor_protein_only.pdb
git commit -m "ERβ-Score: app, model artifacts, and docs"
git branch -M main
git remote add origin https://github.com/<your-org>/ESR_beta_New.git
git push -u origin main
```

Add the curated dataset and notebook as suits your data-release plan (small ones in Git; large ones on Zenodo/LFS).

---

## 7. Deploy a public web server

Reviewers flagged that the abstract’s `http://localhost:8501/` link is not publicly accessible. Deploy the app and replace that link with the live URL.

**Streamlit Community Cloud (free, simplest):**
1. Push the repo to GitHub with these files **at the repository root** (next to `app.py`):
   `app.py`, `requirements.txt`, `packages.txt`, `saved_model/`, `7XWQ.pdb`, `docking_results/receptor_protein_only.pdb`.
2. Go to <https://share.streamlit.io>, connect the repo, set the main file to `app.py`.
3. **Before clicking Deploy, open “Advanced settings” and set Python version to 3.12** (or 3.11). Do **not** use the newest default (3.13/3.14) — RDKit and scikit-learn do not yet publish wheels for it, so the build will fail.
4. Deploy. You get a persistent URL like `https://<app-name>.streamlit.app`.

> **The two failure modes we already hit, and how these files fix them:**
>
> - *`ModuleNotFoundError: No module named 'matplotlib'` (or rdkit/sklearn).* This means Streamlit installed **only** its own dependencies because `requirements.txt` was not found at the repo root. Confirm the file is committed and visible at
>   `https://github.com/<you>/<repo>/blob/main/requirements.txt`, then **Reboot app**. On a correct build you will see rdkit, scikit-learn, matplotlib, and joblib installing (far more than ~42 packages).
> - *RDKit import errors about `libXrender.so.1` / `libXext`.* The included `packages.txt` installs the required system libraries (`libxrender1`, `libxext6`, `libsm6`).

> ⚠️ **Docking on the cloud:** the Docking tab needs **Smina**, which is not installable via apt, so it cannot run on Streamlit Community Cloud. The app is written to detect this and simply disable docking (showing “smina not found”) while Prediction, Applicability Domain, Lipinski, and the 3D Viewer all work normally. If you need hosted docking, deploy on **HuggingFace Spaces (Docker)** and `conda install -c conda-forge smina openbabel` in the Dockerfile.

Once deployed, update the manuscript abstract and Section 2.10 with the working URL.

---

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: rdkit` | `conda install -c conda-forge rdkit` inside the active env |
| Docking tab: *“smina not found”* | `conda install -c conda-forge smina openbabel`; relaunch from the active env |
| 3D viewer blank | needs internet (3Dmol.js is loaded from `https://3dmol.org` CDN) |
| Slow first prediction | RDKit 3D embedding (ETKDGv3 + MMFF94) runs once per molecule; cached thereafter |
| Streamlit cache stale after model swap | press **C** → *Clear cache*, or restart the app |

---

## Support

Open an issue on the GitHub repository, or contact the corresponding authors listed in the manuscript.
