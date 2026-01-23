import pickle
import os
import requests
import streamlit as st
import numpy as np
import matplotlib
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
import rdkit.Chem.Descriptors as Descriptors

# Set backend to avoid GUI crashes
matplotlib.use('Agg')

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="ToxPred AI: Enterprise Edition", 
    page_icon="🧬", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional look
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }
    .stMetric {
        background-color: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. MODEL LOADING ---
@st.cache_resource
def load_models():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    paths = {
        'sol': os.path.join(current_dir, 'solubility_model.pkl'),
        'tox': os.path.join(current_dir, 'toxicity_model.pkl'),
        'bbb': os.path.join(current_dir, 'bbb_model.pkl')
    }
    models = {}
    for name, path in paths.items():
        try:
            with open(path, 'rb') as f:
                models[name] = pickle.load(f)
        except:
            models[name] = None
    return models

models = load_models()

# --- 3. ANALYSIS LOGIC ---
def analyze_compound(compound_name):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{compound_name}/property/IsomericSMILES,CanonicalSMILES,ConnectivitySMILES,SMILES/JSON"
    result = {"Compound": compound_name}
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            props = data['PropertyTable']['Properties'][0]
            
            # Find Best SMILES
            smiles = next((props[k] for k in ['IsomericSMILES', 'CanonicalSMILES', 'SMILES'] if k in props), None)
            result["SMILES"] = smiles
            result["CID"] = props.get('CID', None)
            
            if smiles and (mol := Chem.MolFromSmiles(smiles)):
                # --- A. CALCULATED PROPERTIES (RDKit) ---
                result["MolWeight"] = Descriptors.MolWt(mol)
                result["LogP"] = Descriptors.MolLogP(mol)
                result["HBDonors"] = Descriptors.NumHDonors(mol)
                result["HBAcceptors"] = Descriptors.NumHAcceptors(mol)
                result["RotBonds"] = Descriptors.NumRotatableBonds(mol)
                result["TPSA"] = Descriptors.TPSA(mol) # New Metric!
                result["Status"] = "OK"

                # --- B. AI PREDICTIONS (Random Forest) ---
                # 1. Generate Morgan Fingerprint (2048 bits)
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                fp_array = np.array(fp).reshape(1, -1)

                # 2. Run Models
                # Solubility
                if models['sol']:
                    pred = models['sol'].predict(fp_array)[0]
                    result["LogS"] = pred
                    result["Sol_Class"] = "High" if pred > -2 else "Medium" if pred > -4 else "Low"
                
                # Toxicity
                if models['tox']:
                    prob = models['tox'].predict_proba(fp_array)[0][1]
                    result["Tox_Prob"] = prob
                    result["Tox_Class"] = "Safe" if prob > 0.6 else "Toxic"

                # BBB (Brain)
                if models['bbb']:
                    prob_bbb = models['bbb'].predict_proba(fp_array)[0][1]
                    result["BBB_Prob"] = prob_bbb
                    result["BBB_Class"] = "Permeable" if prob_bbb > 0.5 else "Impermeable"

            else: result["Status"] = "Invalid SMILES structure"
        else: result["Status"] = "Compound not found in PubChem"
    except Exception as e: result["Status"] = f"Connection Error: {e}"
    
    return result

# --- 4. SIDEBAR INFO ---
with st.sidebar:
    st.header("📘 Methodology & Models")
    
    st.markdown("### 🧠 The AI Engine")
    st.info("""
    **Algorithm:** Random Forest Ensemble (Scikit-Learn).
    **Features:** Morgan Fingerprints (ECFP4), radius 2, 2048 bits.
    """)
    
    st.markdown("### 📊 Model Performance")
    st.write("**1. Solubility (Regression)**")
    st.caption("Training Data: ESOL (Delaney)")
    st.progress(0.87, text="R² Score: 0.87")
    
    st.write("**2. Toxicity (Classification)**")
    st.caption("Training Data: ClinTox (FDA)")
    st.progress(0.76, text="Accuracy: 76%")
    
    st.write("**3. Blood-Brain Barrier**")
    st.caption("Training Data: BBBP")
    st.progress(0.85, text="ROC-AUC: 0.85")
    
    st.markdown("---")
    st.caption("Built with Python, RDKit & Streamlit.")

# --- 5. MAIN UI ---
st.title("🧬 ToxPred AI: In-Silico Screening")
st.markdown("""
**Accelerate Drug Discovery.** Predict physicochemical properties, toxicity risks, and brain penetration using advanced Machine Learning.
""")

col_search, col_btn = st.columns([3, 1])
with col_search:
    compound_name = st.text_input("Enter Chemical Name (e.g., Aspirin, Dopamine, Dieldrin)", "Aspirin")
with col_btn:
    st.write("")
    st.write("")
    run_btn = st.button("🚀 Analyze Molecule", type="primary", use_container_width=True)

if run_btn:
    with st.spinner(f"Querying PubChem & Computing Fingerprints for {compound_name}..."):
        res = analyze_compound(compound_name)
        
        if res["Status"] == "OK":
            # --- RESULTS LAYOUT ---
            
            # TOP SECTION: Structure & Key Links
            c_left, c_right = st.columns([1, 2])
            
            with c_left:
                st.subheader("Structure")
                mol = Chem.MolFromSmiles(res["SMILES"])
                st.image(Draw.MolToImage(mol, size=(350, 350)), caption="2D Chemical Structure")
                if res.get("CID"):
                    st.link_button("View on PubChem ↗", f"https://pubchem.ncbi.nlm.nih.gov/compound/{res['CID']}", use_container_width=True)

            with c_right:
                st.subheader("🤖 AI-Driven ADMET Profile")
                
                # METRICS ROW 1: The Predictions
                m1, m2, m3 = st.columns(3)
                
                with m1:
                    st.metric("Solubility (LogS)", f"{res.get('LogS',0):.2f}", res.get('Sol_Class',''))
                    st.caption("Log Molar Solubility. > -2 is good.")

                with m2:
                    tox_c = res.get('Tox_Class', '')
                    tox_p = res.get('Tox_Prob', 0)
                    st.metric("Clinical Toxicity", tox_c, f"{tox_p:.0%} Safe")
                    st.caption("Probability of FDA Approval vs Failure.")

                with m3:
                    bbb_c = res.get('BBB_Class', '')
                    bbb_p = res.get('BBB_Prob', 0)
                    st.metric("Brain Barrier (BBB)", bbb_c, f"{bbb_p:.0%} Prob.")
                    st.caption("Can it cross into the brain?")

                st.markdown("---")

                # METRICS ROW 2: Drug Likeness (Lipinski)
                st.subheader("💊 Drug-Likeness (Lipinski Rules)")
                
                # Check Rules
                fails = 0
                r_mw = res['MolWeight'] < 500
                r_logp = res['LogP'] < 5
                r_hbd = res['HBDonors'] < 5
                r_hba = res['HBAcceptors'] < 10
                if not r_mw: fails += 1
                if not r_logp: fails += 1
                if not r_hbd: fails += 1
                if not r_hba: fails += 1

                l1, l2, l3, l4 = st.columns(4)
                l1.metric("MW", f"{res['MolWeight']:.0f}", "Pass (<500)" if r_mw else "Fail")
                l2.metric("LogP", f"{res['LogP']:.2f}", "Pass (<5)" if r_logp else "Fail")
                l3.metric("H-Donors", res['HBDonors'], "Pass (<5)" if r_hbd else "Fail")
                l4.metric("H-Accept", res['HBAcceptors'], "Pass (<10)" if r_hba else "Fail")

                if fails == 0: st.success("✨ **Excellent Candidate:** Follows all Lipinski rules for oral drugs.")
                elif fails == 1: st.warning("⚠️ **Warning:** Violates 1 Lipinski rule.")
                else: st.error(f"❌ **Poor Candidate:** Violates {fails} Lipinski rules (likely poor absorption).")

            # --- DEEP DIVE SECTION ---
            st.markdown("---")
            with st.expander("🔍 Deep Dive: Parameter Explanations & Importance", expanded=False):
                st.markdown("""
                ### Why do these numbers matter?
                
                #### 1. LogP (Lipophilicity)
                * **What is it?** A measure of how "greasy" a molecule is.
                * **Why it matters:** * **Too Low (< 0):** The drug stays in water/blood and cannot pass through cell membranes.
                    * **Too High (> 5):** The drug gets stuck in fat tissue or becomes toxic.
                    * **Ideal Range:** 1 to 5 for oral drugs.

                #### 2. TPSA (Topological Polar Surface Area)
                * **Value:** **{:.1f} Å²**
                * **What is it?** The surface area of the molecule that is polar (oxygen/nitrogen atoms).
                * **Why it matters:** * **< 140 Å²:** Needed for good cell membrane permeability.
                    * **< 90 Å²:** Usually required to cross the Blood-Brain Barrier (BBB).
                
                #### 3. LogS (Solubility)
                * **What is it?** The logarithm of solubility in mol/L.
                * **Why it matters:** If a drug doesn't dissolve, it passes right through the body.
                    * **> -2:** Highly Soluble.
                    * **< -6:** Poorly Soluble (Requires special formulation).

                #### 4. Blood-Brain Barrier (BBB)
                * **Importance:** * **CNS Drugs (Depression, Alzheimer's):** MUST cross the BBB.
                    * **Non-CNS Drugs (Heart, Stomach):** Should NOT cross the BBB (to avoid side effects like drowsiness).
                """.format(res['TPSA']))

        else:
            st.error(f"Analysis Failed: {res['Status']}")