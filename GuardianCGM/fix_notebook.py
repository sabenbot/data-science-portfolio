import json

# Read the notebook
with open('01_Signal_Processing_and_EDA.ipynb', 'r') as f:
    nb = json.load(f)

# Find and replace the imports cell
found_imports = False
for cell in nb['cells']:
    source_text = ''.join(cell.get('source', []))
    if cell['cell_type'] == 'code' and 'import plotly.graph_objects as go' in source_text:
        cell['source'] = [
            'import pandas as pd\n',
            'import numpy as np\n',
            'import matplotlib.pyplot as plt\n',
            'from scipy.signal import savgol_filter\n',
            '\n',
            'def generate_patient_data(days=3):\n',
            '    np.random.seed(42)\n',
            '    # 5-minute intervals (standard for CGM sensors)\n',
            '    periods = (days * 24 * 60) // 5\n',
            '    time = pd.date_range("2026-01-01", periods=periods, freq="5min")\n',
            '    \n',
            '    # Simulate physiological glucose oscillations\n',
            '    t = np.linspace(0, 2 * np.pi * days, periods)\n',
            '    base = 120 + 40 * np.sin(t) + 15 * np.sin(4 * t)\n',
            '    \n',
            '    # Add sensor noise (simulating electrochemical variance)\n',
            '    noise = np.random.normal(0, 4.5, periods)\n',
            '    return pd.DataFrame({"timestamp": time, "glucose_raw": base + noise})\n',
            '\n',
            'df = generate_patient_data()'
        ]
        cell['outputs'] = []
        cell['execution_count'] = None
        found_imports = True
        print('Updated imports cell')
        break

if not found_imports:
    print('WARNING: Imports cell not found!')

# Find and replace the velocity plot cell
found_velocity = False
for cell in nb['cells']:
    source_text = ''.join(cell.get('source', []))
    if cell['cell_type'] == 'code' and 'fig_vel = go.Figure()' in source_text:
        cell['source'] = [
            'fig_vel, ax = plt.subplots(figsize=(14, 5))\n',
            '\n',
            '# Plot Velocity\n',
            'ax.plot(df_biomarkers["timestamp"][:288], df_biomarkers["velocity"][:288], \n',
            '        color="orange", linewidth=2, label="Glucose Velocity")\n',
            '\n',
            '# Add a "Danger Zone" threshold (e.g., falling faster than 2mg/dL per min)\n',
            'ax.axhline(y=-2.0, linestyle="--", color="red", linewidth=2, label="Rapid Drop Risk")\n',
            'ax.axhline(y=0, color="black", alpha=0.5)\n',
            '\n',
            'ax.set_title("Digital Biomarker: Rate of Change (Velocity)", fontsize=14, fontweight="bold")\n',
            'ax.set_ylabel("mg/dL / min", fontsize=12)\n',
            'ax.set_xlabel("Time", fontsize=12)\n',
            'ax.legend()\n',
            'ax.grid(True, alpha=0.3)\n',
            'plt.tight_layout()\n',
            'plt.show()'
        ]
        # Clear outputs for this cell
        cell['outputs'] = []
        cell['execution_count'] = None
        found_velocity = True
        print('Updated velocity plot cell')
        break

if not found_velocity:
    print('WARNING: Velocity plot cell not found!')

# Write back
with open('01_Signal_Processing_and_EDA.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print('Notebook updated successfully')
