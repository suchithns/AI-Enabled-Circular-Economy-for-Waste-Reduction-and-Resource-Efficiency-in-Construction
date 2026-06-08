"""
Circular Economy Score Predictor
==================================
A machine learning app built with Streamlit and scikit-learn.
Predicts 5 sustainability outputs from 17 construction project inputs.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Circular Economy ML Predictor",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 800;
        color: #1a6b3a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #e8f5e9, #f1f8e9);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border-left: 5px solid #2e7d32;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1b5e20;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #555;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .grade-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1rem;
        margin-top: 0.5rem;
    }
    .section-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1a6b3a;
        border-bottom: 2px solid #a5d6a7;
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background: #e3f2fd;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        border-left: 4px solid #1976d2;
        margin: 0.5rem 0 1rem 0;
        font-size: 0.92rem;
        color: #1a237e;
    }
    .highlight-box {
        background: #fff8e1;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        border-left: 4px solid #f9a825;
        margin: 0.5rem 0;
        font-size: 0.92rem;
    }
</style>
""", unsafe_allow_html=True)


# ─── Load & Prepare Data ──────────────────────────────────────────────────────
@st.cache_data
def load_data():
    """Load and encode the training dataset."""
    data = {
        'PROJECT TYPE': ['Residential','Commercial','Smart City','Hospital','Infrastructure',
                         'Educational Campus','Airport','Green Building','Metro/Railway','Renovation',
                         'IT Park','Bridge Construction','Warehouse','Stadium','Affordable Housing',
                         'Industrial Plant','Coastal Infrastructure','Highway Project','Urban Redevelopment',
                         'Data Center','Shopping Mall','Tunnel Construction','Railway Station',
                         'Smart Township','Sports Complex','Government Building','Hotel & Resort',
                         'Manufacturing Unit','Solar Power Plant','Water Treatment Plant'],
        'AREA_m2':      [120,500,2200,900,1800,750,3200,450,2600,180,1500,2800,950,4000,600,3500,2700,5000,1300,2100,1700,2400,3100,2800,1900,1100,1450,3600,4200,2500],
        'FLOORS':       [2,5,10,6,1,4,3,3,2,1,8,1,2,5,4,4,2,1,6,5,6,3,1,9,4,5,4,3,1,2],
        'CONCRETE_PCT': [58,72,80,68,84,60,82,52,78,40,74,86,66,79,57,88,76,90,64,70,73,77,89,71,69,65,63,87,55,67],
        'STEEL_PCT':    [20,15,18,16,9,18,20,14,17,10,19,22,16,24,15,26,18,8,18,21,19,18,20,17,18,17,16,24,12,15],
        'WOOD_PCT':     [12,5,8,10,2,14,4,22,6,28,6,2,12,5,18,3,5,1,14,4,7,5,1,12,8,10,18,2,4,6],
        'GLASS_PCT':    [10,20,28,24,6,18,30,16,12,8,26,4,10,32,12,10,14,2,20,28,25,18,3,24,22,20,26,8,10,12],
        'RECYCLED_PCT': [18,10,35,22,5,26,12,42,15,38,20,8,24,18,36,7,16,4,34,18,20,14,6,38,24,28,30,8,46,32],
        'DEMO_WASTE_PCT':[15,8,32,20,2,24,10,35,12,42,18,5,20,12,30,5,14,2,28,15,16,12,4,34,20,24,25,6,40,28],
        'PREFAB_PCT':   [25,5,40,18,2,32,10,48,8,20,35,3,30,22,42,6,12,1,36,24,28,15,2,44,26,32,35,5,52,38],
        'MODULAR_LEVEL':['Medium','Low','High','Medium','Low','High','Medium','High','Medium','Medium',
                         'High','Low','Medium','Medium','High','Low','Medium','Low','High','Medium',
                         'Medium','Medium','Low','High','Medium','Medium','High','Low','High','High'],
        'CONSTRUCTION_METHOD':['Prefab','Conventional','Modular','Prefab','Conventional','Modular',
                                'Conventional','Modular','Conventional','Prefab','Modular','Conventional',
                                'Prefab','Conventional','Prefab','Conventional','Conventional','Conventional',
                                'Modular','Prefab','Prefab','Conventional','Conventional','Modular',
                                'Prefab','Prefab','Modular','Conventional','Modular','Prefab'],
        'WASTE_SEGREGATION':['Yes','No','Yes','Yes','No','Yes','Yes','Yes','Yes','Yes','Yes','No',
                              'Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','No','Yes',
                              'Yes','Yes','Yes','No','Yes','Yes'],
        'RENEW_ENERGY_PCT':[22,8,45,28,4,38,18,55,14,30,40,3,32,25,48,6,20,2,42,35,34,16,2,50,30,36,42,5,65,48],
        'WATER_REUSE_PCT': [30,12,42,25,5,40,15,48,18,35,36,4,30,20,45,8,22,3,40,30,32,20,4,45,28,34,40,6,55,60],
        'ENERGY_KWH':    [2200,7800,12500,6400,14800,5400,18200,3200,13200,1800,9200,16500,6100,21000,4200,24000,15400,28500,7600,19500,9800,14600,22600,11800,8900,6900,7600,24800,9800,8400],
        'MATERIAL_WASTAGE_PCT':[5,14,4,6,17,5,12,3,11,4,5,16,6,13,4,18,10,20,5,8,7,11,19,4,7,5,5,17,3,5],
        # Outputs
        'RECYCLABLE_WASTE_PCT': [84,46,95,72,38,89,58,97,42,86,80,30,69,63,90,36,40,99,83,77,74,34,100,85,71,76,88,28,100,92],
        'CARBON_EMISSION_SCORE':[42,89,28,61,96,36,79,22,91,40,52,98,66,73,34,94,92,18,47,58,62,97,15,44,64,57,39,99,12,31],
        'WASTE_GEN_KG':  [4800,9200,2100,6400,12500,3500,7800,1800,10400,2600,4300,14800,5600,6100,2900,13200,11800,1500,3200,4700,5200,13900,1200,3000,5900,4500,2500,14200,1000,2200],
        'MAT_RECOVERY_PCT':[81,40,93,70,35,88,55,96,38,84,78,26,68,65,89,32,37,97,82,75,73,30,99,83,69,74,87,24,100,91],
        'CIRCULAR_ECONOMY_SCORE':[86,39,94,74,34,91,57,98,41,89,82,25,71,67,92,30,35,99,85,79,76,28,100,87,87,72,78,90,22,100]
    }
    return pd.DataFrame(data)


@st.cache_resource
def train_models(df):
    """Train one Random Forest model per output target."""
    input_cols = ['PROJECT TYPE','AREA_m2','FLOORS','CONCRETE_PCT','STEEL_PCT','WOOD_PCT',
                  'GLASS_PCT','RECYCLED_PCT','DEMO_WASTE_PCT','PREFAB_PCT',
                  'MODULAR_LEVEL','CONSTRUCTION_METHOD','WASTE_SEGREGATION',
                  'RENEW_ENERGY_PCT','WATER_REUSE_PCT','ENERGY_KWH','MATERIAL_WASTAGE_PCT']
    output_cols = ['RECYCLABLE_WASTE_PCT','CARBON_EMISSION_SCORE','WASTE_GEN_KG',
                   'MAT_RECOVERY_PCT','CIRCULAR_ECONOMY_SCORE']
    cat_cols = ['PROJECT TYPE','MODULAR_LEVEL','CONSTRUCTION_METHOD','WASTE_SEGREGATION']

    df_enc = df.copy()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    X = df_enc[input_cols]
    models = {}
    loo_results = {}

    for out in output_cols:
        y = df_enc[out]
        model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        model.fit(X, y)
        models[out] = model

        # LOO cross-validation for honest accuracy estimate
        loo = LeaveOneOut()
        preds, actuals = [], []
        for train_idx, test_idx in loo.split(X):
            m = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            m.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds.append(m.predict(X.iloc[test_idx])[0])
            actuals.append(y.iloc[test_idx].values[0])

        loo_results[out] = {
            'mae': mean_absolute_error(actuals, preds),
            'preds': preds,
            'actuals': actuals
        }

    return models, encoders, loo_results, input_cols, output_cols, df_enc, X


# ─── Helper: encode single prediction ────────────────────────────────────────
def encode_input(row_dict, encoders, input_cols):
    row = row_dict.copy()
    for col, le in encoders.items():
        if col in row:
            if row[col] in le.classes_:
                row[col] = le.transform([row[col]])[0]
            else:
                row[col] = 0
    return pd.DataFrame([row])[input_cols]


# ─── Score Grade ──────────────────────────────────────────────────────────────
def get_grade(score):
    if score >= 90:   return "🌟 Excellent", "#1b5e20", "#e8f5e9"
    elif score >= 75: return "✅ Good",      "#2e7d32", "#c8e6c9"
    elif score >= 55: return "⚠️ Moderate",  "#f57f17", "#fff9c4"
    elif score >= 35: return "🔶 Below Avg", "#e65100", "#ffe0b2"
    else:             return "❌ Poor",      "#b71c1c", "#ffebee"


# ─── Main App ─────────────────────────────────────────────────────────────────
def main():
    df = load_data()
    models, encoders, loo_results, input_cols, output_cols, df_enc, X_train = train_models(df)

    # ── Sidebar Navigation ────────────────────────────────────────────────────
    st.sidebar.image("https://img.icons8.com/color/96/recycle-sign.png", width=80)
    st.sidebar.markdown("## Navigation")
    page = st.sidebar.radio("Go to", [
        "🔮 Predict",
        "📊 Model Performance",
        "🧠 How the ML Works",
        "📈 Data Explorer",
        "📋 About the Project"
    ])

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Model:** Random Forest (200 trees)")
    st.sidebar.markdown("**Training samples:** 30 projects")
    st.sidebar.markdown("**Input features:** 17")
    st.sidebar.markdown("**Predicted outputs:** 5")

    # ════════════════════════════════════════════════════════════════════════
    #  PAGE 1: PREDICT
    # ════════════════════════════════════════════════════════════════════════
    if page == "🔮 Predict":
        st.markdown('<p class="main-header">AI-Enabled Circular Economy for Waste Reduction and Resource Efficiency in Construction</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Enter construction project parameters to predict sustainability outcomes using a trained Random Forest model</p>', unsafe_allow_html=True)

        st.markdown('<div class="info-box">💡 Fill in the 17 input parameters on the left. The ML model will predict 5 sustainability outputs including the final Circular Economy Score.</div>', unsafe_allow_html=True)

        col_form, col_results = st.columns([1.1, 0.9])

        with col_form:
            # ── Project Info ──────────────────────────────────────────────
            st.markdown('<p class="section-title">🏗️ Project Information</p>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                project_type = st.selectbox("Project Type", sorted(df['PROJECT TYPE'].unique()))
                area = st.number_input("Area (m²)", min_value=50, max_value=10000, value=500, step=50)
                floors = st.slider("Number of Floors", 1, 20, 3)
            with c2:
                modular_level = st.selectbox("Modular Level", ["High", "Medium", "Low"])
                construction_method = st.selectbox("Construction Method", ["Prefab", "Modular", "Conventional"])
                waste_segregation = st.selectbox("Waste Segregation", ["Yes", "No"])

            # ── Material Composition ──────────────────────────────────────
            st.markdown('<p class="section-title">🧱 Material Composition (%)</p>', unsafe_allow_html=True)
            c3, c4 = st.columns(2)
            with c3:
                concrete = st.slider("Concrete %", 30, 95, 60)
                steel    = st.slider("Steel %",    5,  30, 18)
                wood     = st.slider("Wood %",     0,  35, 10)
                glass    = st.slider("Glass %",    0,  35, 15)
            with c4:
                recycled  = st.slider("Recycled Material %",       0, 60, 20)
                demo_waste= st.slider("Demolition Waste Reused %", 0, 50, 15)
                prefab    = st.slider("Prefabrication %",          0, 60, 25)

            # ── Energy & Environment ──────────────────────────────────────
            st.markdown('<p class="section-title">🌱 Energy & Environment</p>', unsafe_allow_html=True)
            c5, c6 = st.columns(2)
            with c5:
                renew_energy = st.slider("Renewable Energy Usage %", 0,  80, 25)
                water_reuse  = st.slider("Water Reuse %",            0,  80, 25)
            with c6:
                energy_cons   = st.number_input("Energy Consumption (kWh)", min_value=500, max_value=50000, value=5000, step=100)
                mat_wastage   = st.slider("Material Wastage %", 1, 25, 6)

            predict_btn = st.button("🔮 Predict All Outputs", type="primary", use_container_width=True)

        with col_results:
            if predict_btn:
                row_dict = {
                    'PROJECT TYPE': project_type,
                    'AREA_m2': area, 'FLOORS': floors,
                    'CONCRETE_PCT': concrete, 'STEEL_PCT': steel,
                    'WOOD_PCT': wood, 'GLASS_PCT': glass,
                    'RECYCLED_PCT': recycled, 'DEMO_WASTE_PCT': demo_waste,
                    'PREFAB_PCT': prefab, 'MODULAR_LEVEL': modular_level,
                    'CONSTRUCTION_METHOD': construction_method,
                    'WASTE_SEGREGATION': waste_segregation,
                    'RENEW_ENERGY_PCT': renew_energy, 'WATER_REUSE_PCT': water_reuse,
                    'ENERGY_KWH': energy_cons, 'MATERIAL_WASTAGE_PCT': mat_wastage
                }
                X_pred = encode_input(row_dict, encoders, input_cols)

                predictions = {out: models[out].predict(X_pred)[0] for out in output_cols}

                st.markdown('<p class="section-title">📊 Predicted Outputs</p>', unsafe_allow_html=True)

                # ── Circular Economy Score (hero metric) ──────────────────
                score = predictions['CIRCULAR_ECONOMY_SCORE']
                grade_label, grade_color, grade_bg = get_grade(score)

                st.markdown(f"""
                <div class="metric-card" style="border-left-color:{grade_color}; background: linear-gradient(135deg, {grade_bg}, #ffffff);">
                    <div class="metric-label">🏆 Circular Economy Score</div>
                    <div class="metric-value" style="color:{grade_color}">{score:.0f} / 100</div>
                    <span class="grade-badge" style="background:{grade_bg}; color:{grade_color}; border:2px solid {grade_color}">{grade_label}</span>
                </div>""", unsafe_allow_html=True)

                # ── Progress bar ──────────────────────────────────────────
                fig_gauge, ax_gauge = plt.subplots(figsize=(5, 0.6))
                ax_gauge.barh([0], [100], color='#e0e0e0', height=0.5)
                bar_color = '#2e7d32' if score >= 75 else '#f9a825' if score >= 50 else '#c62828'
                ax_gauge.barh([0], [score], color=bar_color, height=0.5)
                ax_gauge.text(score + 1, 0, f'{score:.0f}', va='center', fontweight='bold', color=bar_color)
                ax_gauge.set_xlim(0, 110)
                ax_gauge.axis('off')
                fig_gauge.patch.set_alpha(0)
                st.pyplot(fig_gauge, use_container_width=True)
                plt.close()

                st.markdown("---")
                st.markdown('<p class="section-title">📋 Other Predicted Outputs</p>', unsafe_allow_html=True)

                output_meta = {
                    'RECYCLABLE_WASTE_PCT':  ("♻️ Recyclable Waste",       "%",   "Higher = Better"),
                    'CARBON_EMISSION_SCORE': ("🌫️ Carbon Emission Score",  "",    "Lower = Better"),
                    'WASTE_GEN_KG':          ("🗑️ Waste Generated",         " kg", "Lower = Better"),
                    'MAT_RECOVERY_PCT':      ("🔄 Material Recovery Rate", "%",   "Higher = Better"),
                }
                for out, (label, unit, hint) in output_meta.items():
                    val = predictions[out]
                    mae = loo_results[out]['mae']
                    st.metric(
                        label=f"{label} ({hint})",
                        value=f"{val:.0f}{unit}",
                        delta=f"±{mae:.1f}{unit} model error"
                    )

                # ── Radar chart ───────────────────────────────────────────
                st.markdown("---")
                st.markdown('<p class="section-title">🕸️ Sustainability Profile</p>', unsafe_allow_html=True)

                categories = ['Recyclable\nWaste', 'Carbon\n(inv)', 'Waste Gen\n(inv)', 'Mat Recovery', 'CE Score']
                raw = [
                    predictions['RECYCLABLE_WASTE_PCT'],
                    100 - predictions['CARBON_EMISSION_SCORE'],
                    100 - (predictions['WASTE_GEN_KG'] / 148),
                    predictions['MAT_RECOVERY_PCT'],
                    predictions['CIRCULAR_ECONOMY_SCORE']
                ]
                values = [max(0, min(100, v)) for v in raw]
                values_norm = [v / 100 for v in values]

                angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
                values_norm += values_norm[:1]
                angles += angles[:1]

                fig_r, ax_r = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
                ax_r.fill(angles, values_norm, alpha=0.25, color='#2e7d32')
                ax_r.plot(angles, values_norm, 'o-', linewidth=2, color='#2e7d32')
                ax_r.set_xticks(angles[:-1])
                ax_r.set_xticklabels(categories, fontsize=8)
                ax_r.set_ylim(0, 1)
                ax_r.set_yticks([0.25, 0.5, 0.75, 1.0])
                ax_r.set_yticklabels(['25', '50', '75', '100'], fontsize=7, color='grey')
                ax_r.grid(color='grey', alpha=0.3)
                ax_r.set_title("Sustainability Profile\n(higher = better)", size=10, pad=15)
                st.pyplot(fig_r, use_container_width=True)
                plt.close()
            else:
                st.markdown("""
                <div style="background:#f5f5f5; border-radius:12px; padding:3rem; text-align:center; color:#999; margin-top:2rem;">
                    <div style="font-size:4rem">🔮</div>
                    <div style="font-size:1.1rem; margin-top:1rem; font-weight:600">Fill in the parameters and click Predict</div>
                    <div style="font-size:0.9rem; margin-top:0.5rem">5 sustainability outputs will appear here</div>
                </div>""", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    #  PAGE 2: MODEL PERFORMANCE
    # ════════════════════════════════════════════════════════════════════════
    elif page == "📊 Model Performance":
        st.markdown('<p class="main-header">📊 Model Performance</p>', unsafe_allow_html=True)
        st.markdown("How accurate is the model? We use **Leave-One-Out Cross-Validation** — the fairest method for small datasets.")

        st.markdown('<div class="info-box">🔬 <b>Leave-One-Out CV:</b> Train on 29 samples, predict the 1 left out. Repeat 30 times. This gives a realistic estimate of how the model performs on <i>unseen</i> data.</div>', unsafe_allow_html=True)

        output_labels = {
            'RECYCLABLE_WASTE_PCT':  "Recyclable Waste %",
            'CARBON_EMISSION_SCORE': "Carbon Emission Score",
            'WASTE_GEN_KG':          "Waste Generated (kg)",
            'MAT_RECOVERY_PCT':      "Material Recovery Rate %",
            'CIRCULAR_ECONOMY_SCORE':"Circular Economy Score"
        }

        # ── Summary table ─────────────────────────────────────────────────
        st.markdown('<p class="section-title">📋 Accuracy Summary</p>', unsafe_allow_html=True)
        perf_rows = []
        for out in output_cols:
            res = loo_results[out]
            r2 = r2_score(res['actuals'], res['preds'])
            perf_rows.append({
                "Output": output_labels[out],
                "Mean Abs Error (MAE)": f"{res['mae']:.1f}",
                "R² Score": f"{max(0, r2):.3f}",
                "Data Range": f"{min(res['actuals'])} – {max(res['actuals'])}"
            })
        st.dataframe(pd.DataFrame(perf_rows), use_container_width=True, hide_index=True)

        # ── Actual vs Predicted plots ──────────────────────────────────────
        st.markdown('<p class="section-title">📈 Actual vs Predicted (LOO Cross-Validation)</p>', unsafe_allow_html=True)

        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        axes = axes.flatten()
        fig.suptitle("Actual vs Predicted — Leave-One-Out Cross-Validation", fontsize=14, fontweight='bold')

        for i, out in enumerate(output_cols):
            res = loo_results[out]
            ax = axes[i]
            ax.scatter(res['actuals'], res['preds'], color='#2e7d32', alpha=0.7, s=60, edgecolors='white', linewidth=0.5)
            mn = min(min(res['actuals']), min(res['preds']))
            mx = max(max(res['actuals']), max(res['preds']))
            ax.plot([mn, mx], [mn, mx], 'r--', alpha=0.6, label='Perfect prediction')
            ax.set_xlabel("Actual", fontsize=9)
            ax.set_ylabel("Predicted", fontsize=9)
            ax.set_title(output_labels[out], fontsize=10, fontweight='bold')
            ax.legend(fontsize=7)
            r2 = r2_score(res['actuals'], res['preds'])
            ax.text(0.05, 0.92, f"MAE={res['mae']:.1f} | R²={max(0,r2):.2f}",
                    transform=ax.transAxes, fontsize=8, color='#444',
                    bbox=dict(boxstyle='round', facecolor='#e8f5e9', alpha=0.8))
            ax.grid(True, alpha=0.3)

        axes[-1].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # ── Feature Importance ─────────────────────────────────────────────
        st.markdown('<p class="section-title">🔑 Feature Importance for Circular Economy Score</p>', unsafe_allow_html=True)
        st.markdown("Which input features matter most for predicting the Circular Economy Score?")

        fi = models['CIRCULAR_ECONOMY_SCORE'].feature_importances_
        feature_names_display = [
            'Project Type','Area (m²)','Floors','Concrete %','Steel %','Wood %',
            'Glass %','Recycled Material %','Demolition Waste Reused %','Prefab %',
            'Modular Level','Construction Method','Waste Segregation',
            'Renewable Energy %','Water Reuse %','Energy (kWh)','Material Wastage %'
        ]
        fi_df = pd.DataFrame({'Feature': feature_names_display, 'Importance': fi})
        fi_df = fi_df.sort_values('Importance', ascending=True)

        fig_fi, ax_fi = plt.subplots(figsize=(8, 7))
        colors = ['#1b5e20' if v > 0.07 else '#4caf50' if v > 0.04 else '#a5d6a7' for v in fi_df['Importance']]
        bars = ax_fi.barh(fi_df['Feature'], fi_df['Importance'], color=colors)
        ax_fi.set_xlabel('Importance Score', fontsize=10)
        ax_fi.set_title('Feature Importance — Random Forest\n(Circular Economy Score)', fontweight='bold')
        for bar, val in zip(bars, fi_df['Importance']):
            ax_fi.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                      f'{val:.3f}', va='center', fontsize=8)
        ax_fi.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_fi, use_container_width=True)
        plt.close()

    # ════════════════════════════════════════════════════════════════════════
    #  PAGE 3: HOW THE ML WORKS
    # ════════════════════════════════════════════════════════════════════════
    elif page == "🧠 How the ML Works":
        st.markdown('<p class="main-header">🧠 How the Machine Learning Works</p>', unsafe_allow_html=True)

        tab1, tab2, tab3, tab4 = st.tabs(["🌲 Random Forest Explained", "🔄 Training Process", "📐 Prediction Flow", "🎯 Why This Model"])

        # ── Tab 1: What is RF ─────────────────────────────────────────────
        with tab1:
            c1, c2 = st.columns([1.2, 0.8])
            with c1:
                st.markdown("""
### What is a Random Forest?

A **Random Forest** is an ensemble machine learning algorithm — it builds **many decision trees** and combines their answers.

#### 🌳 Step 1: One Decision Tree
Think of it like a flowchart of questions:
- *"Is Renewable Energy > 40%?"* → Yes → go right → *"Is Wood % > 15?"* → ...
- Each path ends in a predicted score.

#### 🌲🌲🌲 Step 2: Many Trees (The "Forest")
Instead of trusting one tree, we build **200 trees**, each trained on:
- A **random subset of rows** (bootstrap sampling)
- A **random subset of features** at each split

This prevents overfitting — no single pattern dominates.

#### 🗳️ Step 3: Vote / Average
All 200 trees make a prediction. The **final answer is the average**.
> If 150 trees say 85 and 50 say 60, the prediction is about **78.5**

---
### Why is this powerful?
| Single Decision Tree | Random Forest |
|---|---|
| Can memorise training data | Generalises better |
| One perspective | 200 perspectives |
| Fragile to noise | Robust to outliers |
                """)
            with c2:
                # Draw a simple decision tree diagram
                fig_tree, ax_tree = plt.subplots(figsize=(5, 6))
                ax_tree.set_xlim(0, 10)
                ax_tree.set_ylim(0, 10)
                ax_tree.axis('off')
                ax_tree.set_title("A Decision Tree", fontweight='bold', fontsize=11)

                # Root
                root = plt.Rectangle((3.5, 8.5), 3, 0.9, color='#1b5e20', alpha=0.9)
                ax_tree.add_patch(root)
                ax_tree.text(5, 8.95, 'Renewable\nEnergy > 40%?', ha='center', va='center', color='white', fontsize=7, fontweight='bold')

                # Level 2 nodes
                n_yes = plt.Rectangle((1, 6.3), 3, 0.9, color='#2e7d32', alpha=0.8)
                n_no  = plt.Rectangle((6, 6.3), 3, 0.9, color='#c62828', alpha=0.8)
                ax_tree.add_patch(n_yes)
                ax_tree.add_patch(n_no)
                ax_tree.text(2.5, 6.75, 'Recycled\nMat > 30%?', ha='center', va='center', color='white', fontsize=7)
                ax_tree.text(7.5, 6.75, 'Carbon\nScore < 60?', ha='center', va='center', color='white', fontsize=7)

                # Leaf nodes
                for x, y, label, color in [(0.5,4.1,'Score: 90','#43a047'), (3,4.1,'Score: 72','#66bb6a'),
                                            (5.5,4.1,'Score: 45','#ef5350'), (8,4.1,'Score: 28','#b71c1c')]:
                    leaf = plt.Rectangle((x-0.1, y), 2.2, 0.8, color=color, alpha=0.85)
                    ax_tree.add_patch(leaf)
                    ax_tree.text(x+1, y+0.4, label, ha='center', va='center', color='white', fontsize=8, fontweight='bold')

                # Lines
                for start, end in [((5,8.5),(2.5,7.2)), ((5,8.5),(7.5,7.2)),
                                    ((2.5,6.3),(1.5,4.9)), ((2.5,6.3),(4,4.9)),
                                    ((7.5,6.3),(6.5,4.9)), ((7.5,6.3),(9,4.9))]:
                    ax_tree.annotate('', xy=end, xytext=start,
                                     arrowprops=dict(arrowstyle='->', color='grey', lw=1.2))

                ax_tree.text(3.8, 7.8, 'YES', color='#2e7d32', fontsize=8, fontweight='bold')
                ax_tree.text(6.2, 7.8, 'NO', color='#c62828', fontsize=8, fontweight='bold')
                st.pyplot(fig_tree, use_container_width=True)
                plt.close()

        # ── Tab 2: Training Process ───────────────────────────────────────
        with tab2:
            st.markdown("""
### How was the model trained?

#### 📦 The Dataset
- **30 real construction projects** with 17 inputs and 5 outputs each
- Types range from Residential homes to Smart Cities to Solar Power Plants

#### 🔀 Bootstrap Sampling
For each of the 200 trees, we randomly pick 30 rows *with replacement* (some rows repeat, some are left out). This means each tree sees a slightly different version of the data.

#### ✂️ Feature Splitting
At each node in a tree, the algorithm:
1. Randomly selects a subset of features
2. Tests every possible split value
3. Picks the split that reduces prediction error the most

#### 📏 How We Measured Accuracy: Leave-One-Out Cross-Validation
""")
            st.markdown('<div class="highlight-box">🎯 <b>Leave-One-Out (LOO) CV</b> — the most honest evaluation for small datasets:<br><br>1. Take out project #1 → train on 29 → predict #1<br>2. Take out project #2 → train on 29 → predict #2<br>3. Repeat for all 30 projects<br>4. Compare all 30 predictions to actual values → compute MAE</div>', unsafe_allow_html=True)

            # Visualise LOO
            fig_loo, ax_loo = plt.subplots(figsize=(10, 2.5))
            ax_loo.set_xlim(0, 31)
            ax_loo.set_ylim(0, 3)
            ax_loo.axis('off')
            ax_loo.set_title("Leave-One-Out Cross-Validation (30 iterations)", fontweight='bold')
            for i in range(30):
                color = '#ef5350' if i == 5 else '#4caf50'
                rect = plt.Rectangle((i + 0.1, 1.5), 0.8, 0.8, color=color, alpha=0.85)
                ax_loo.add_patch(rect)
                ax_loo.text(i + 0.5, 1.9, str(i+1), ha='center', va='center', fontsize=6, color='white', fontweight='bold')
            ax_loo.text(15.5, 0.9, 'GREEN = Train   RED = Test (left out)', ha='center', fontsize=9, color='#444')
            ax_loo.text(15.5, 0.4, '← One iteration shown — repeated 30 times →', ha='center', fontsize=8, color='#888', style='italic')
            st.pyplot(fig_loo, use_container_width=True)
            plt.close()

        # ── Tab 3: Prediction Flow ────────────────────────────────────────
        with tab3:
            st.markdown("""
### What happens when you click "Predict"?

**Step 1 — You enter 17 inputs:**
""")
            st.code("""
Project Type, Area, Floors, Concrete%, Steel%, Wood%, Glass%,
Recycled Material%, Demolition Waste Reused%, Prefabrication%,
Modular Level, Construction Method, Waste Segregation,
Renewable Energy%, Water Reuse%, Energy Consumption, Material Wastage%
""", language="text")

            st.markdown("**Step 2 — Categorical inputs are encoded:**")
            st.code("""
# Example encoding (learned from training data)
Project Type: "Green Building" → 8
Modular Level: "High"          → 0
Construction Method: "Modular" → 1
Waste Segregation: "Yes"       → 1
""", language="python")

            st.markdown("**Step 3 — Each of the 5 models runs independently:**")
            st.code("""
# Simplified view
for each of 200 trees:
    tree traverses your inputs (left/right splits)
    reaches a leaf node → outputs a number

final_prediction = average of 200 tree outputs
""", language="python")

            st.markdown("**Step 4 — 5 predictions returned:**")
            st.code("""
Recyclable Waste %      → e.g. 84
Carbon Emission Score   → e.g. 42
Waste Generated (kg)    → e.g. 4800
Material Recovery Rate  → e.g. 81
Circular Economy Score  → e.g. 86  ← Main output
""", language="python")

        # ── Tab 4: Why this model ─────────────────────────────────────────
        with tab4:
            st.markdown("""
### Why Random Forest instead of other algorithms?

| Algorithm | Pros | Cons | Suitable? |
|---|---|---|---|
| **Random Forest** ✅ | Handles small data, no feature scaling needed, handles mixed data types, interpretable importance | Slower than linear models | ✅ Best fit |
| Linear Regression | Simple, fast | Assumes linear relationships, bad for complex patterns | ⚠️ Too simple |
| Neural Network | Very powerful for large data | Needs 1000s of samples, black box | ❌ Too complex for 30 rows |
| SVM | Good for small data | Hard to tune, needs scaling | ⚠️ Possible but harder |
| Gradient Boosting | Often more accurate | Prone to overfit on 30 rows | ⚠️ Risky |

### Why 5 separate models instead of 1?
Each output has a **different pattern** in the data. A single model predicting all 5 at once would need to make compromises. Five separate Random Forests, each specialised for one output, gives better accuracy.

### Model Hyperparameters Used
""")
            st.code("""
RandomForestRegressor(
    n_estimators=200,   # 200 decision trees in the forest
    random_state=42,    # Reproducible results (same every run)
    n_jobs=-1           # Use all CPU cores for speed
)
""", language="python")

    # ════════════════════════════════════════════════════════════════════════
    #  PAGE 4: DATA EXPLORER
    # ════════════════════════════════════════════════════════════════════════
    elif page == "📈 Data Explorer":
        st.markdown('<p class="main-header">📈 Training Data Explorer</p>', unsafe_allow_html=True)

        tab_raw, tab_corr, tab_dist = st.tabs(["📋 Raw Data", "🔗 Correlations", "📊 Distributions"])

        with tab_raw:
            st.markdown(f"**{len(df)} construction projects** used for training")
            st.dataframe(df, use_container_width=True, height=450)

        with tab_corr:
            st.markdown("**Correlation matrix** — how strongly are features related to the Circular Economy Score?")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            corr_with_ce = df[numeric_cols].corr()['CIRCULAR_ECONOMY_SCORE'].drop('CIRCULAR_ECONOMY_SCORE').sort_values()

            fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
            colors = ['#c62828' if v < 0 else '#2e7d32' for v in corr_with_ce.values]
            corr_with_ce.plot(kind='barh', ax=ax_corr, color=colors)
            ax_corr.axvline(0, color='black', linewidth=0.8)
            ax_corr.set_title('Correlation with Circular Economy Score', fontweight='bold')
            ax_corr.set_xlabel('Pearson Correlation')
            ax_corr.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_corr, use_container_width=True)
            plt.close()

        with tab_dist:
            st.markdown("Distribution of the **Circular Economy Score** across the 30 projects:")
            fig_dist, axes_dist = plt.subplots(1, 2, figsize=(12, 4))

            axes_dist[0].hist(df['CIRCULAR_ECONOMY_SCORE'], bins=12, color='#2e7d32', edgecolor='white', alpha=0.85)
            axes_dist[0].set_title('CE Score Distribution', fontweight='bold')
            axes_dist[0].set_xlabel('Score')
            axes_dist[0].set_ylabel('Count')
            axes_dist[0].grid(axis='y', alpha=0.3)

            df_sorted = df.sort_values('CIRCULAR_ECONOMY_SCORE')
            bar_colors = ['#c62828' if s < 40 else '#f9a825' if s < 70 else '#2e7d32' for s in df_sorted['CIRCULAR_ECONOMY_SCORE']]
            axes_dist[1].bar(range(len(df_sorted)), df_sorted['CIRCULAR_ECONOMY_SCORE'], color=bar_colors)
            axes_dist[1].set_title('CE Score by Project (sorted)', fontweight='bold')
            axes_dist[1].set_xlabel('Project Index')
            axes_dist[1].set_ylabel('Score')
            axes_dist[1].axhline(df['CIRCULAR_ECONOMY_SCORE'].mean(), color='navy', linestyle='--', label=f"Mean={df['CIRCULAR_ECONOMY_SCORE'].mean():.0f}")
            axes_dist[1].legend()
            axes_dist[1].grid(axis='y', alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig_dist, use_container_width=True)
            plt.close()

    # ════════════════════════════════════════════════════════════════════════
    #  PAGE 5: ABOUT
    # ════════════════════════════════════════════════════════════════════════
    elif page == "📋 About the Project":
        st.markdown('<p class="main-header">📋 About This Project</p>', unsafe_allow_html=True)

        st.markdown("""
### What this app does
This application uses **supervised machine learning** to predict sustainability outcomes for construction projects.
Given 17 input parameters about a project's design, materials, and practices, it predicts 5 key outputs:

| Output | What It Measures | Direction |
|---|---|---|
| ♻️ **Recyclable Waste %** | Proportion of waste that can be recycled | Higher = Better |
| 🌫️ **Carbon Emission Score** | Environmental carbon impact | Lower = Better |
| 🗑️ **Waste Generated (kg)** | Total construction waste | Lower = Better |
| 🔄 **Material Recovery Rate %** | How much material is recovered/reused | Higher = Better |
| 🏆 **Circular Economy Score** | Overall sustainability score | Higher = Better |

---
### What is Circular Economy?
The circular economy in construction focuses on:
- **Reducing waste** at every stage
- **Reusing and recycling** materials
- **Designing for deconstruction** — buildings that can be dismantled and reused
- **Minimising carbon footprint**

---
### Tech Stack
- **Python** — core language
- **scikit-learn** — Random Forest ML models
- **Streamlit** — interactive web interface
- **Pandas + NumPy** — data processing
- **Matplotlib + Seaborn** — visualisations

        """)


if __name__ == "__main__":
    main()
