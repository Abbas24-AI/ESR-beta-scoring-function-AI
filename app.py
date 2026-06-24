"""
ESRβ Activity Predictor — Streamlit App
GradientBoosting model · kNN-Tanimoto AD · Lipinski · Smina docking
"""
import os, re, shutil, subprocess, tempfile, json, warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from io import StringIO

warnings.filterwarnings("ignore")

# ── RDKit ────────────────────────────────────────────────────────────────────
from rdkit import Chem
from rdkit.Chem import (Descriptors, rdMolDescriptors, AllChem,
                        DataStructs, Draw, rdFingerprintGenerator)
from rdkit.Chem.rdMolDescriptors import (
    CalcTPSA, CalcNumHBD, CalcNumHBA, CalcNumRotatableBonds,
    CalcNumAromaticRings, CalcFractionCSP3, CalcPMI1, CalcPMI2,
    CalcNPR1, CalcSpherocityIndex,
)
from rdkit.Chem import RingInfo
import joblib

# ── Paths ────────────────────────────────────────────────────────────────────
BASE        = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(BASE, "saved_model")
RECEPTOR    = os.path.join(BASE, "docking_results", "receptor_protein_only.pdb")
PDB_COMPLEX = os.path.join(BASE, "7XWQ.pdb")

# ── Grid defaults (from 7XWQ.pdb HETATM; BOX_PAD=8.0) ──────────────────────
DEFAULT_GRID = dict(
    center_x=-3.244, center_y=-7.848, center_z=-39.600,
    size_x=22.5,     size_y=22.5,     size_z=22.5,
)
EXHAUSTIVENESS = 16
N_MODES        = 9
CPU_DOCK       = 4
BOX_PAD        = 8.0

# ─────────────────────────────────────────────────────────────────────────────
# Load saved artefacts  (called AFTER set_page_config, cached across reruns)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    model   = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
    scaler  = joblib.load(os.path.join(MODEL_DIR, "best_scaler.pkl"))
    with open(os.path.join(MODEL_DIR, "feature_columns.json")) as f:
        feats = json.load(f)
    with open(os.path.join(MODEL_DIR, "feature_medians.json")) as f:
        medians = json.load(f)
    with open(os.path.join(MODEL_DIR, "ad_config.json")) as f:
        ad_cfg = json.load(f)
    train_fps = np.load(os.path.join(MODEL_DIR, "train_fps_ecfp4.npy"))
    return model, scaler, feats, medians, ad_cfg, train_fps

# ─────────────────────────────────────────────────────────────────────────────
# Molecular feature computation  (mirrors notebook _calc_2d / _calc_3d exactly)
# ─────────────────────────────────────────────────────────────────────────────
def _embed3d(mol):
    """Return mol with 3-D conformer, or None on failure."""
    try:
        mh = Chem.AddHs(mol)
        ps = AllChem.ETKDGv3()
        ps.randomSeed = 42
        if AllChem.EmbedMolecule(mh, ps) == -1:
            ps.useRandomCoords = True
            if AllChem.EmbedMolecule(mh, ps) == -1:
                return None
        AllChem.MMFFOptimizeMolecule(mh)
        return mh
    except Exception:
        return None

def compute_features(mol, medians):
    """Compute all 39 descriptors; use median for any that fail."""
    # ── 2-D descriptors ──────────────────────────────────────────────────────
    def _safe(fn, *args, key=None):
        try:
            v = fn(*args)
            return v if (v is not None and not (isinstance(v, float) and np.isnan(v))) else medians.get(key, 0.0)
        except Exception:
            return medians.get(key, 0.0)

    fd = {
        "MolWt":             _safe(Descriptors.MolWt,          mol,  key="MolWt"),
        "LogP":              _safe(Descriptors.MolLogP,         mol,  key="LogP"),
        "TPSA":              _safe(CalcTPSA,                    mol,  key="TPSA"),
        "NumHDonors":        _safe(CalcNumHBD,                  mol,  key="NumHDonors"),
        "NumHAcceptors":     _safe(CalcNumHBA,                  mol,  key="NumHAcceptors"),
        "NumRotatableBonds": _safe(CalcNumRotatableBonds,        mol,  key="NumRotatableBonds"),
        "NumAromaticRings":  _safe(CalcNumAromaticRings,         mol,  key="NumAromaticRings"),
        "RingCount":         _safe(rdMolDescriptors.CalcNumRings,mol,  key="RingCount"),
        "FractionCSP3":      _safe(CalcFractionCSP3,            mol,  key="FractionCSP3"),
        "BalabanJ":          _safe(Descriptors.BalabanJ,         mol,  key="BalabanJ"),
    }
    for i in range(1, 12):
        k = f"EState_VSA{i}"
        fd[k] = _safe(getattr(Descriptors, k), mol, key=k)
    for i in range(1, 15):
        k = f"PEOE_VSA{i}"
        fd[k] = _safe(getattr(Descriptors, k), mol, key=k)

    # ── 3-D descriptors ──────────────────────────────────────────────────────
    mh = _embed3d(mol)
    if mh is not None:
        fd["PMI1"]            = _safe(CalcPMI1,            mh, key="PMI1")
        fd["PMI2"]            = _safe(CalcPMI2,            mh, key="PMI2")
        fd["NPR1"]            = _safe(CalcNPR1,            mh, key="NPR1")
        fd["SpherocityIndex"] = _safe(CalcSpherocityIndex, mh, key="SpherocityIndex")
    else:
        # embedding failed — fall back to training-set medians
        for k in ("PMI1", "PMI2", "NPR1", "SpherocityIndex"):
            fd[k] = medians.get(k, 0.0)

    return fd

def lipinski(mol):
    return {
        "MW":         round(Descriptors.MolWt(mol), 2),
        "LogP":       round(Descriptors.MolLogP(mol), 2),
        "HBD":        CalcNumHBD(mol),
        "HBA":        CalcNumHBA(mol),
        "RotBonds":   CalcNumRotatableBonds(mol),
        "TPSA":       round(CalcTPSA(mol), 2),
        "RingCount":  mol.GetRingInfo().NumRings(),
        "AromaticRings": CalcNumAromaticRings(mol),
    }

def ro5_pass(lp):
    return (lp["MW"] <= 500 and lp["LogP"] <= 5 and
            lp["HBD"] <= 5  and lp["HBA"] <= 10)

def ecfp4(mol):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    arr = np.zeros(2048, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(gen.GetFingerprint(mol), arr)
    return arr

def predict(mol, model, scaler, feature_cols, medians):
    feat_dict = compute_features(mol, medians)
    row = [feat_dict.get(c, medians.get(c, 0.0)) for c in feature_cols]
    X   = np.array(row, dtype=np.float64).reshape(1, -1)
    X_s = scaler.transform(X)
    prob = float(model.predict_proba(X_s)[0, 1])
    return prob, feat_dict

def domain_applicability(mol, train_fps, ad_cfg):
    fp     = ecfp4(mol).astype(np.float32)
    train  = train_fps.astype(np.float32)
    # Tanimoto: (A∩B) / (A+B−A∩B)  via dot product on bit vectors
    inter  = train @ fp
    denom  = train.sum(1) + fp.sum() - inter
    sim    = np.where(denom > 0, inter / denom, 0.0)
    k      = ad_cfg["knn_k"]
    top_k  = np.partition(sim, -k)[-k:]
    score  = float(top_k.mean())
    thr    = ad_cfg["ad_threshold"]
    return score, thr, score >= thr

# ─────────────────────────────────────────────────────────────────────────────
# 3-D conformer → PDB string (for viewer)
# ─────────────────────────────────────────────────────────────────────────────
def mol_to_pdb_str(mol):
    m3 = _embed3d(mol)
    tmp = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False)
    from rdkit.Chem import PDBWriter
    w = PDBWriter(tmp.name); w.write(m3); w.close()
    txt = open(tmp.name).read()
    os.unlink(tmp.name)
    return txt

# ─────────────────────────────────────────────────────────────────────────────
# Py3Dmol viewer (HTML component)
# ─────────────────────────────────────────────────────────────────────────────
def viewer_html(receptor_pdb_path, ligand_pdb_str, height=480):
    receptor_txt = open(receptor_pdb_path).read().replace("`", "'").replace("\\", "\\\\").replace("\n", "\\n")
    ligand_txt   = ligand_pdb_str.replace("`", "'").replace("\\", "\\\\").replace("\n", "\\n")
    html = f"""
    <script src="https://3dmol.org/build/3Dmol-min.js"></script>
    <div id="viewer" style="width:100%;height:{height}px;position:relative;border:1px solid #ccc;border-radius:8px;background:#f0f4f8;"></div>
    <script>
      (function() {{
        var viewer = $3Dmol.createViewer('viewer', {{backgroundColor: '#f0f4f8'}});
        viewer.addModel(`{receptor_txt}`, 'pdb');
        viewer.setStyle({{model:0}}, {{cartoon:{{color:'#5e9ea0', opacity:0.85}}}});
        viewer.addModel(`{ligand_txt}`, 'pdb');
        viewer.setStyle({{model:1}}, {{stick:{{colorscheme:'Jmol', radius:0.25}}}});
        viewer.addSurface($3Dmol.SurfaceType.VDW, {{
          opacity: 0.12, color: '#88bbff'
        }}, {{model:0}});
        viewer.zoomTo({{model:1}});
        viewer.render();
      }})();
    </script>
    """
    return html

# ─────────────────────────────────────────────────────────────────────────────
# Tool finder
# ─────────────────────────────────────────────────────────────────────────────
def find_tool(name):
    hit = shutil.which(name)
    if hit and os.path.isfile(hit) and os.access(hit, os.X_OK) and os.path.getsize(hit) > 1000:
        return hit
    for pfx in [os.path.expanduser("~/miniconda3"), os.path.expanduser("~/anaconda3"),
                "/opt/homebrew/Caskroom/miniconda/base", "/opt/conda"]:
        p = os.path.join(pfx, "bin", name)
        if os.path.isfile(p) and os.access(p, os.X_OK) and os.path.getsize(p) > 1000:
            return p
    return None

# ─────────────────────────────────────────────────────────────────────────────
# Docking
# ─────────────────────────────────────────────────────────────────────────────
def run_docking(smiles, name, grid):
    smina = find_tool("smina")
    obabel = find_tool("obabel")
    if not smina:
        return None, "smina not found. Install: conda install -c conda-forge smina"

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "Invalid SMILES"

    tmpdir = tempfile.mkdtemp()
    try:
        safe_name = re.sub(r"[^A-Za-z0-9_-]", "_", name)[:30]
        sdf_path  = os.path.join(tmpdir, f"{safe_name}.sdf")
        out_sdf   = os.path.join(tmpdir, f"{safe_name}_docked.sdf")
        log_path  = os.path.join(tmpdir, f"{safe_name}.log")

        # 3-D conformer
        m3 = _embed3d(mol)
        from rdkit.Chem import SDWriter
        w = SDWriter(sdf_path); w.write(m3); w.close()

        # Convert receptor to pdbqt if obabel available
        rec = RECEPTOR
        if obabel:
            pdbqt = os.path.join(tmpdir, "receptor.pdbqt")
            subprocess.run([obabel, RECEPTOR, "-O", pdbqt, "-xr"],
                           capture_output=True)
            if os.path.exists(pdbqt) and os.path.getsize(pdbqt) > 100:
                rec = pdbqt

        cmd = [
            smina,
            "--receptor",       rec,
            "--ligand",         sdf_path,
            "--out",            out_sdf,
            "--log",            log_path,
            "--center_x",       f"{grid['center_x']:.3f}",
            "--center_y",       f"{grid['center_y']:.3f}",
            "--center_z",       f"{grid['center_z']:.3f}",
            "--size_x",         f"{grid['size_x']:.1f}",
            "--size_y",         f"{grid['size_y']:.1f}",
            "--size_z",         f"{grid['size_z']:.1f}",
            "--exhaustiveness", str(EXHAUSTIVENESS),
            "--num_modes",      str(N_MODES),
            "--cpu",            str(CPU_DOCK),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        # Parse score
        score = None
        if os.path.exists(out_sdf):
            from rdkit.Chem import SDMolSupplier
            for m in SDMolSupplier(out_sdf, removeHs=False):
                if m is not None and m.HasProp("minimizedAffinity"):
                    score = float(m.GetProp("minimizedAffinity"))
                    break
        if score is None and os.path.exists(log_path):
            for line in open(log_path):
                mt = re.search(r"^\s*1\s+([-\d.]+)", line)
                if mt:
                    score = float(mt.group(1)); break

        # Build complex PDB
        complex_pdb = None
        if os.path.exists(out_sdf):
            lig_pdb = os.path.join(tmpdir, f"{safe_name}_lig.pdb")
            ok = False
            if obabel:
                r2 = subprocess.run([obabel, out_sdf, "-O", lig_pdb, "-f", "1", "-l", "1"],
                                    capture_output=True)
                ok = os.path.exists(lig_pdb) and os.path.getsize(lig_pdb) > 50
            if not ok:
                try:
                    from rdkit.Chem import PDBWriter as _PW
                    for m in SDMolSupplier(out_sdf, removeHs=False):
                        if m:
                            pw = _PW(lig_pdb); pw.write(m); pw.close()
                            ok = True; break
                except Exception:
                    pass
            if ok:
                prot_txt = open(RECEPTOR).read()
                lig_txt  = open(lig_pdb).read()
                complex_pdb = prot_txt + "\nREMARK  Docked ligand\n" + lig_txt

        log_txt = open(log_path).read() if os.path.exists(log_path) else res.stderr
        return {"score": score, "complex_pdb": complex_pdb, "log": log_txt}, None
    except subprocess.TimeoutExpired:
        return None, "Docking timed out (>5 min)"
    except Exception as e:
        return None, str(e)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────
def prob_gauge(prob):
    fig, ax = plt.subplots(figsize=(4, 2.2), subplot_kw=dict(aspect="equal"))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    theta = np.linspace(np.pi, 0, 300)
    ax.plot(np.cos(theta), np.sin(theta), lw=14, color="#e0e0e0", solid_capstyle="round")
    fill = np.linspace(np.pi, np.pi - prob * np.pi, 300)
    color = "#e74c3c" if prob < 0.4 else "#f39c12" if prob < 0.6 else "#2ecc71"
    ax.plot(np.cos(fill), np.sin(fill), lw=14, color=color, solid_capstyle="round")
    ax.text(0, -0.15, f"{prob*100:.1f}%", ha="center", va="center",
            fontsize=22, fontweight="bold", color=color)
    ax.text(0, -0.5, "Activity Probability", ha="center", va="center",
            fontsize=9, color="#555555")
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-0.7, 1.2)
    ax.axis("off")
    return fig

def lipinski_bar(lp):
    props  = ["MW\n(≤500)", "LogP\n(≤5)", "HBD\n(≤5)", "HBA\n(≤10)", "TPSA\n(≤140)", "RotBonds\n(≤10)"]
    vals   = [lp["MW"], lp["LogP"], lp["HBD"], lp["HBA"], lp["TPSA"], lp["RotBonds"]]
    limits = [500, 5, 5, 10, 140, 10]
    colors = ["#2ecc71" if v <= lim else "#e74c3c" for v, lim in zip(vals, limits)]

    fig, ax = plt.subplots(figsize=(6, 2.8))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8f9fb")
    bars = ax.bar(props, vals, color=colors, edgecolor="none", width=0.55)
    for bar, lim, val in zip(bars, limits, vals):
        ax.axhline(lim, xmin=bar.get_x()/(len(props)-0.5), color="#00000030", lw=0.8, ls="--")
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.02,
                f"{val}", ha="center", va="bottom", fontsize=8, color="#222")
    ax.set_ylabel("Value", color="#555", fontsize=9)
    ax.tick_params(colors="#555", labelsize=8)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.set_title("Lipinski / Drug-like Properties", color="#222", fontsize=10, pad=6)
    fig.tight_layout()
    return fig

def ad_dial(score, threshold):
    fig, ax = plt.subplots(figsize=(3.2, 1.8))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ax.barh([0], [1], color="#e0e0e0", height=0.4)
    color = "#2ecc71" if score >= threshold else "#e74c3c"
    ax.barh([0], [min(score, 1.0)], color=color, height=0.4)
    ax.axvline(threshold, color="#e67e22", lw=1.5, ls="--")
    ax.text(threshold, 0.28, f"Thr={threshold:.3f}", color="#e67e22",
            fontsize=7, ha="center")
    ax.text(0.5, -0.38, f"AD Score: {score:.3f}", ha="center", va="top",
            fontsize=10, color=color, fontweight="bold",
            transform=ax.transAxes)
    ax.set_xlim(0, 1); ax.set_yticks([]); ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.tick_params(colors="#555", labelsize=7)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.set_title("Applicability Domain (kNN-Tanimoto)", color="#222", fontsize=9, pad=4)
    fig.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# Page config  ← must be FIRST Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ESRβ Activity Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load model artefacts AFTER set_page_config (cache_resource requires it) ──
MODEL, SCALER, FEATURE_COLS, MEDIANS, AD_CFG, TRAIN_FPS = load_model()

st.markdown("""
<style>
  body, .stApp { background:#ffffff; color:#1a1a1a; }
  section[data-testid="stSidebar"] { background:#f5f7fa; }
  .metric-card {
    background:#f0f4f8; border:1px solid #d0d7e0; border-radius:10px;
    padding:16px 20px; margin:4px 0;
  }
  .tag-active   { background:#e8f8ee; color:#1a7a3a; border:1px solid #2ecc71;
                  border-radius:6px; padding:4px 12px; font-weight:700; font-size:15px; }
  .tag-inactive { background:#fdecea; color:#b71c1c; border:1px solid #e74c3c;
                  border-radius:6px; padding:4px 12px; font-weight:700; font-size:15px; }
  .tag-domain   { background:#e3f0fb; color:#1565c0; border:1px solid #3498db;
                  border-radius:6px; padding:4px 12px; font-size:13px; }
  .tag-outside  { background:#fff3e0; color:#e65100; border:1px solid #e67e22;
                  border-radius:6px; padding:4px 12px; font-size:13px; }
  hr { border-color:#d0d7e0; }
  .cite-box {
    font-size:11.5px; color:#555; font-style:italic;
    border-left:3px solid #aaa; padding:4px 10px; margin:6px 0 2px 0;
  }
  .dev-box {
    font-size:13px; color:#333; margin:2px 0 8px 0;
  }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ESRβ Predictor")
    st.caption("Estrogen Receptor Beta · GradBoost · kNN-Tanimoto AD")
    st.markdown("---")

    smiles_input = st.text_area(
        "SMILES",
        value="",
        height=100,
        placeholder="Paste SMILES here…",
        help="Single SMILES string for the query molecule",
    )
    predict_btn = st.button("▶  Predict", use_container_width=True, type="primary")
    st.markdown("---")

    st.markdown("**Model info**")
    st.caption("Type: GradientBoostingClassifier")
    st.caption("Features: 39 (2D + 3D descriptors)")
    st.caption("Test PR-AUC: 0.891 · ROC-AUC: 0.888")
    st.markdown("---")

    with st.expander("⚙️  Docking Grid (Smina)"):
        cx = st.number_input("center_x", value=DEFAULT_GRID["center_x"], format="%.3f")
        cy = st.number_input("center_y", value=DEFAULT_GRID["center_y"], format="%.3f")
        cz = st.number_input("center_z", value=DEFAULT_GRID["center_z"], format="%.3f")
        sx = st.number_input("size_x (Å)", value=DEFAULT_GRID["size_x"], format="%.1f", min_value=10.0)
        sy = st.number_input("size_y (Å)", value=DEFAULT_GRID["size_y"], format="%.1f", min_value=10.0)
        sz = st.number_input("size_z (Å)", value=DEFAULT_GRID["size_z"], format="%.1f", min_value=10.0)
        grid_cfg = dict(center_x=cx, center_y=cy, center_z=cz,
                        size_x=sx,   size_y=sy,   size_z=sz)

    dock_btn = st.button("⚗️  Run Smina Docking", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
header_img = os.path.join(BASE, "Header.png")
if os.path.exists(header_img):
    st.image(header_img, use_container_width=True)

st.markdown(
    '<div class="cite-box">'
    'Please cite: <b>ESRβ-Score: An Interpretable Machine Learning Scoring Function '
    'for Estrogen Receptor β–Driven Precision Drug Discovery in Triple-Negative Breast Cancer</b>'
    '</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="dev-box">'
    'Developed by: <b>Dr. Abbas Khan &amp; Dr. Abdelali Agouni</b> '
    '(College of Pharmacy, Qatar University, Doha, Qatar)'
    '</div>',
    unsafe_allow_html=True,
)
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Main output
# ─────────────────────────────────────────────────────────────────────────────
if predict_btn or dock_btn:
    smi = smiles_input.strip()
    if not smi:
        st.error("Enter a SMILES string first.")
        st.stop()

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        st.error("Invalid SMILES — RDKit could not parse it.")
        st.stop()

    # ── Run prediction ───────────────────────────────────────────────────────
    with st.spinner("Computing descriptors & predicting…"):
        try:
            prob, feat_dict = predict(mol, MODEL, SCALER, FEATURE_COLS, MEDIANS)
            lp              = lipinski(mol)
            ad_score, ad_thr, in_domain = domain_applicability(mol, TRAIN_FPS, AD_CFG)
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()

    tab1, tab2, tab3 = st.tabs(["📊 Prediction", "🔬 3D Viewer", "⚗️ Docking"])

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 1 — Prediction
    # ═════════════════════════════════════════════════════════════════════════
    with tab1:
        col_img, col_res = st.columns([1, 2])

        with col_img:
            st.markdown("#### Molecule")
            img = Draw.MolToImage(mol, size=(300, 250))
            st.image(img, use_container_width=True)
            st.caption(f"`{smi[:80]}{'…' if len(smi)>80 else ''}`")

        with col_res:
            # Activity badge
            label = "ACTIVE" if prob >= 0.5 else "INACTIVE"
            tag_cls = "tag-active" if prob >= 0.5 else "tag-inactive"
            dom_cls = "tag-domain" if in_domain else "tag-outside"
            dom_lbl = "In-Domain" if in_domain else "Out-of-Domain"

            st.markdown(
                f'<span class="{tag_cls}">{label}</span>&nbsp;&nbsp;'
                f'<span class="{dom_cls}">{dom_lbl}</span>',
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)

            # Gauge + AD side by side
            g1, g2 = st.columns(2)
            with g1:
                fig_g = prob_gauge(prob)
                st.pyplot(fig_g, use_container_width=True)
                plt.close(fig_g)
            with g2:
                fig_ad = ad_dial(ad_score, ad_thr)
                st.pyplot(fig_ad, use_container_width=True)
                plt.close(fig_ad)

        # ── Lipinski ─────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### Lipinski / Drug-likeness")

        ro5 = ro5_pass(lp)
        ro5_badge = "✅ Passes Lipinski Rule-of-5" if ro5 else "⚠️ Violates Lipinski Rule-of-5"
        st.info(ro5_badge)

        lip_cols = st.columns(4)
        props_fmt = [
            ("MW", lp["MW"], "≤ 500 Da"),
            ("LogP", lp["LogP"], "≤ 5"),
            ("HB Donors", lp["HBD"], "≤ 5"),
            ("HB Acceptors", lp["HBA"], "≤ 10"),
            ("TPSA", lp["TPSA"], "≤ 140 Å²"),
            ("Rot. Bonds", lp["RotBonds"], "≤ 10"),
            ("Rings", lp["RingCount"], "–"),
            ("Arom. Rings", lp["AromaticRings"], "–"),
        ]
        for i, (name, val, rule) in enumerate(props_fmt):
            with st.columns(8)[i % 8] if i < 4 else lip_cols[i % 4]:
                pass

        # Use a table instead for cleaner layout
        lp_df = pd.DataFrame(props_fmt, columns=["Property", "Value", "Threshold"])
        lp_df["Status"] = lp_df.apply(
            lambda r: ("✅" if (
                (r.Property == "MW"          and r.Value <= 500) or
                (r.Property == "LogP"        and r.Value <= 5) or
                (r.Property == "HB Donors"   and r.Value <= 5) or
                (r.Property == "HB Acceptors"and r.Value <= 10) or
                (r.Property == "TPSA"        and r.Value <= 140) or
                (r.Property == "Rot. Bonds"  and r.Value <= 10) or
                r.Property in ("Rings", "Arom. Rings")
            ) else "❌"), axis=1
        )
        st.dataframe(lp_df, use_container_width=True, hide_index=True)

        fig_lip = lipinski_bar(lp)
        st.pyplot(fig_lip, use_container_width=True)
        plt.close(fig_lip)

        # ── Feature table ────────────────────────────────────────────────────
        with st.expander("🔍 All 39 Computed Descriptors"):
            fd_df = pd.DataFrame(
                [(k, round(v, 4)) for k, v in feat_dict.items()],
                columns=["Descriptor", "Value"],
            )
            st.dataframe(fd_df, use_container_width=True, hide_index=True)

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 2 — 3-D Viewer
    # ═════════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("#### Protein–Ligand 3D View")
        st.caption(
            "Receptor: ESRβ (PDB: 7XWQ) · "
            "Ligand: query molecule (ETKDGv3 + MMFF94 3-D conformer)"
        )

        if not os.path.exists(RECEPTOR):
            st.warning(f"Receptor PDB not found at `{RECEPTOR}`. Run docking first to generate it, or check path.")
        else:
            with st.spinner("Generating 3-D conformer & rendering…"):
                try:
                    lig_pdb = mol_to_pdb_str(mol)
                    html = viewer_html(RECEPTOR, lig_pdb, height=500)
                    st.components.v1.html(html, height=510, scrolling=False)
                    st.markdown("""
**Controls:**
- 🖱️ Left-drag → rotate  &nbsp;|&nbsp; Scroll → zoom  &nbsp;|&nbsp; Right-drag → translate
- Protein shown as **cartoon** (teal) with transparent surface
- Ligand shown as **sticks** (element colours)
""")
                except Exception as e:
                    st.error(f"3D viewer error: {e}")

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 3 — Docking
    # ═════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("#### Smina Docking")

        smina_avail = find_tool("smina") is not None
        if not smina_avail:
            st.error("**smina** not found in PATH or common conda prefixes.")
            st.code("conda install -c conda-forge smina", language="bash")
            st.stop()

        st.markdown(
            f"**Grid centre:** ({grid_cfg['center_x']:.3f}, {grid_cfg['center_y']:.3f}, "
            f"{grid_cfg['center_z']:.3f}) Å  \n"
            f"**Box size:** {grid_cfg['size_x']:.1f} × {grid_cfg['size_y']:.1f} × "
            f"{grid_cfg['size_z']:.1f} Å  \n"
            f"Exhaustiveness: {EXHAUSTIVENESS} · Modes: {N_MODES}"
        )

        if dock_btn:
            with st.spinner("Running Smina docking… this may take 1–3 minutes."):
                result, err = run_docking(smi, "query_mol", grid_cfg)

            if err:
                st.error(f"Docking failed: {err}")
            else:
                score = result.get("score")
                if score is not None:
                    col_s, col_i = st.columns([1, 2])
                    with col_s:
                        color = "#2ecc71" if score < -7 else "#f39c12" if score < -5 else "#e74c3c"
                        st.markdown(
                            f"""<div class="metric-card" style="text-align:center">
                            <div style="font-size:13px;color:#aaa">Best Affinity</div>
                            <div style="font-size:36px;font-weight:700;color:{color}">{score:.2f}</div>
                            <div style="font-size:13px;color:#aaa">kcal/mol</div>
                            </div>""",
                            unsafe_allow_html=True,
                        )
                        st.markdown("""
<small style='color:#888'>
**Interpretation:**
< -7 kcal/mol → strong binding
-5 to -7 → moderate
> -5 → weak
</small>""", unsafe_allow_html=True)
                    with col_i:
                        st.markdown("**Smina Log**")
                        st.text(result.get("log", "")[:2000])
                else:
                    st.warning("Docking ran but could not parse affinity score.")
                    st.text(result.get("log", "")[:2000])

                # 3-D docked complex viewer
                if result.get("complex_pdb"):
                    st.markdown("---")
                    st.markdown("#### Docked Complex")
                    lig_lines = []
                    in_lig    = False
                    for ln in result["complex_pdb"].split("\n"):
                        if "Docked ligand" in ln:
                            in_lig = True
                        if in_lig:
                            lig_lines.append(ln)
                    lig_pdb_str = "\n".join(lig_lines)

                    # Write protein portion
                    prot_lines = [ln for ln in result["complex_pdb"].split("\n")
                                  if ln.startswith(("ATOM", "TER", "END"))]
                    prot_pdb_tmp = tempfile.NamedTemporaryFile(
                        suffix=".pdb", delete=False, mode="w")
                    prot_pdb_tmp.write("\n".join(prot_lines))
                    prot_pdb_tmp.close()

                    try:
                        html2 = viewer_html(prot_pdb_tmp.name, lig_pdb_str, height=500)
                        st.components.v1.html(html2, height=510, scrolling=False)
                        st.caption("Docked pose from Smina (best mode)")
                    finally:
                        os.unlink(prot_pdb_tmp.name)
        else:
            st.info("Adjust the grid in the sidebar if needed, then click **Run Smina Docking**.")

else:
    # ── Landing page ─────────────────────────────────────────────────────────
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
**📊 Activity Prediction**
GradientBoosting model trained on 988 ESRβ compounds.
Outputs probability + Active/Inactive call.
""")
    with col2:
        st.markdown("""
**🎯 Applicability Domain**
kNN-Tanimoto (k=5, ECFP4 2048-bit).
Flags if query is structurally outside training set.
""")
    with col3:
        st.markdown("""
**⚗️ Smina Docking**
Auto-grid from co-crystal 7XWQ.
Runs smina and renders docked pose in 3-D.
""")
    st.markdown("---")
    st.markdown("**Enter a SMILES in the sidebar and click ▶ Predict to begin.**")
