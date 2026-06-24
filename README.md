# ---- OS / editor cruft ----
.DS_Store
Thumbs.db
*~
.idea/
.vscode/
.claude/

# ---- Python ----
__pycache__/
*.py[cod]
*.egg-info/
.ipynb_checkpoints/
.venv/
venv/
env/
.streamlit/secrets.toml

# ---- Large screening libraries / raw data (archive on Zenodo or use Git LFS) ----
TCM-R5 filtered.sdf
IBscreen.csv
cleaned_NPAtlas.csv
virtual_screening_combined.csv
virtual_screening_results.csv
virtual_screening_results_*.csv
ESRB_ecfp4.csv
ESRB_ecfp6.csv
ESRB_rdkitfp.csv
ESRB_maccs.csv
ESRB_descriptors_full.csv
esrbetsa_raw.csv

# ---- Generated docking outputs (keep prepared receptor; ignore bulky poses) ----
docking_results/Docking_complexes/
docking_results/ligands/
docking_results/smina/
docking_results/gnina/
*.pdbqt

# ---- Temp / build ----
*.log
*.tmp
~$*           # Word lock files
