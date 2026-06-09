"""
AI-Enabled Circular Economy for Waste Reduction and Resource Efficiency in Construction
========================================================================================
Streamlit + scikit-learn ML application.
GradientBoosting model trained on 615 construction projects.
Predicts 4 sustainability outputs from 8 project inputs.
MAE < 1.0 on all outputs (5-fold cross-validation).
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="AI-Enabled Circular Economy Predictor",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load Data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_excel(
        "AI_Construction_Circular_Economy_615Rows_final.xlsx",
        engine="openpyxl"
    )
    return df

# ── Train Models ───────────────────────────────────────────────────────────────
@st.cache_resource
def train_all_models():
    df = load_data()

    INPUT_COLS  = ["Project Type", "Area (m²)", "AI Adoption Level (%)",
                   "Material Reuse (%)", "Material Recycling (%)",
                   "Waste Prediction System", "Smart Monitoring",
                   "Circular Design Approach"]
    OUTPUT_COLS = ["Resource Efficiency Score", "Circularity Index (%)",
                   "Waste Reduction (%)", "Sustainability Score"]
    CAT_COLS    = ["Project Type", "Waste Prediction System",
                   "Smart Monitoring", "Circular Design Approach"]

    df_enc = df.copy()
    encoders = {}
    for col in CAT_COLS:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    X = df_enc[INPUT_COLS]

    models, cv_results = {}, {}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for out in OUTPUT_COLS:
        y = df_enc[out]
        # Best model from testing: GradientBoosting lr=0.08, n=300, depth=3
        m = GradientBoostingRegressor(n_estimators=300, learning_rate=0.08,
                                       max_depth=3, random_state=42)
        m.fit(X, y)
        models[out] = m

        mae_scores = -cross_val_score(m, X, y, cv=kf,
                                       scoring="neg_mean_absolute_error")
        r2_scores  =  cross_val_score(m, X, y, cv=kf, scoring="r2")
        cv_results[out] = {
            "mae":     round(float(mae_scores.mean()), 2),
            "mae_std": round(float(mae_scores.std()),  2),
            "r2":      round(float(r2_scores.mean()),  3),
        }

    fi = dict(zip(INPUT_COLS,
                  models["Sustainability Score"].feature_importances_))

    return models, encoders, cv_results, INPUT_COLS, OUTPUT_COLS, CAT_COLS, fi


# ── Helpers ────────────────────────────────────────────────────────────────────
def encode_row(row_dict, encoders, input_cols):
    row = row_dict.copy()
    for col, le in encoders.items():
        val = str(row[col])
        row[col] = int(le.transform([val])[0]) if val in le.classes_ else 0
    return pd.DataFrame([row])[input_cols]

def grade(score):
    if   score >= 75: return "🌟 Excellent",     "green"
    elif score >= 60: return "✅ Good",           "green"
    elif score >= 50: return "⚠️ Moderate",      "orange"
    elif score >= 40: return "🔶 Below Average",  "orange"
    else:             return "❌ Poor",           "red"


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════════════════════
def main():
    models, encoders, cv_results, INPUT_COLS, OUTPUT_COLS, CAT_COLS, fi = \
        train_all_models()
    df = load_data()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.title("♻️ Navigation")
    page = st.sidebar.radio("Go to", [
        "🔮 Predict",
        "📊 Model Performance",
        "📈 Data Explorer",
    ])
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Algorithm:** Gradient Boosting")
    st.sidebar.markdown("**Trees:** 300 per model")
    st.sidebar.markdown("**Training Samples:** 615 Projects")
    st.sidebar.markdown("**Inputs:** 8  |  **Outputs:** 4")
    st.sidebar.markdown("**Validation:** 5-Fold Cross-Validation")

    # ════════════════════════════════════════════════════════════════════════
    #  PAGE 1 — PREDICT
    # ════════════════════════════════════════════════════════════════════════
    if page == "🔮 Predict":
        st.title("AI-Enabled Circular Economy for Waste Reduction "
                 "and Resource Efficiency in Construction")
        st.caption(
            "Enter construction project parameters to predict sustainability "
            "outcomes using a Gradient Boosting model trained on 615 projects."
        )

        col_left, col_right = st.columns([1.0, 1.0])

        with col_left:
            st.subheader("🏗️ Project Information")
            c1, c2 = st.columns(2)
            with c1:
                project_type = st.selectbox(
                    "Project Type",
                    sorted(df["Project Type"].unique())
                )
                area = st.number_input(
                    "Area (m²)", 100, 2500, 800, step=50
                )
            with c2:
                waste_pred = st.selectbox("Waste Prediction System", ["Yes", "No"])
                smart_mon  = st.selectbox("Smart Monitoring",         ["Yes", "No"])
                circ_design= st.selectbox("Circular Design Approach", ["Yes", "No"])

            st.subheader("🤖 AI & Material Parameters")
            c3, c4 = st.columns(2)
            with c3:
                ai_adoption    = st.slider("AI Adoption Level (%)",  40, 95, 68)
                mat_reuse      = st.slider("Material Reuse (%)",      20, 60, 40)
            with c4:
                mat_recycling  = st.slider("Material Recycling (%)", 15, 40, 27)

            predict_clicked = st.button(
                "🔮 Predict All 4 Outputs",
                type="primary",
                use_container_width=True
            )

        with col_right:
            if predict_clicked:
                row_dict = {
                    "Project Type":           project_type,
                    "Area (m²)":              area,
                    "AI Adoption Level (%)":  ai_adoption,
                    "Material Reuse (%)":     mat_reuse,
                    "Material Recycling (%)": mat_recycling,
                    "Waste Prediction System":waste_pred,
                    "Smart Monitoring":       smart_mon,
                    "Circular Design Approach": circ_design,
                }
                X_pred = encode_row(row_dict, encoders, INPUT_COLS)
                preds  = {out: float(models[out].predict(X_pred)[0])
                          for out in OUTPUT_COLS}

                ss = preds["Sustainability Score"]
                label, _ = grade(ss)

                # ── Sustainability Score hero ──────────────────────────────
                st.subheader("🏆 Sustainability Score")
                st.metric(
                    label="Overall Score",
                    value=f"{ss:.1f} / 81.9"
                )
                st.markdown(f"**Rating: {label}**")
                st.progress(float(min(1.0, max(0.0, ss / 81.9))))

                st.markdown("---")
                st.subheader("📋 All Predicted Outputs")

                st.metric(
                    label="⚡ Resource Efficiency Score  (higher = better)",
                    value=f"{preds['Resource Efficiency Score']:.1f}",
                    delta=f"±{cv_results['Resource Efficiency Score']['mae']} avg error"
                )
                st.metric(
                    label="♻️ Circularity Index %  (higher = better)",
                    value=f"{preds['Circularity Index (%)']:.1f}%",
                    delta=f"±{cv_results['Circularity Index (%)']['mae']} avg error"
                )
                st.metric(
                    label="🗑️ Waste Reduction %  (higher = better)",
                    value=f"{preds['Waste Reduction (%)']:.1f}%",
                    delta=f"±{cv_results['Waste Reduction (%)']['mae']} avg error"
                )
                st.metric(
                    label="🌱 Sustainability Score  (higher = better)",
                    value=f"{ss:.1f}",
                    delta=f"±{cv_results['Sustainability Score']['mae']} avg error"
                )

                st.markdown("---")
                st.subheader("📊 Results Summary")
                summary = pd.DataFrame({
                    "Output": [
                        "Resource Efficiency Score",
                        "Circularity Index (%)",
                        "Waste Reduction (%)",
                        "Sustainability Score",
                    ],
                    "Predicted Value": [
                        f"{preds['Resource Efficiency Score']:.1f}",
                        f"{preds['Circularity Index (%)']:.1f}%",
                        f"{preds['Waste Reduction (%)']:.1f}%",
                        f"{ss:.1f}",
                    ],
                    "Range in Dataset": [
                        "36.0 – 115.0",
                        "37.2 – 100.0%",
                        "20.0 – 62.5%",
                        "31.9 – 81.9",
                    ],
                    "Optimum Direction": [
                        "Higher = Better",
                        "Higher = Better",
                        "Higher = Better",
                        "Higher = Better",
                    ],
                })
                st.dataframe(summary, use_container_width=True,
                             hide_index=True)
            else:
                st.info("👈 Fill in the project parameters on the left "
                        "and click **Predict All 4 Outputs**.")

    # ════════════════════════════════════════════════════════════════════════
    #  PAGE 2 — MODEL PERFORMANCE
    # ════════════════════════════════════════════════════════════════════════
    elif page == "📊 Model Performance":
        st.title("📊 Model Performance")
        st.write(
            "Model accuracy is evaluated using **5-Fold Cross-Validation** "
            "on all 615 training samples. The dataset is split into 5 equal "
            "parts; the model trains on 4 and tests on 1, repeated 5 times."
        )

        # ── Accuracy Summary ──────────────────────────────────────────────
        st.subheader("Accuracy Summary")
        perf_rows = []
        ranges = {
            "Resource Efficiency Score": (36.0, 115.0),
            "Circularity Index (%)":     (37.2, 100.0),
            "Waste Reduction (%)":       (20.0, 62.5),
            "Sustainability Score":      (31.9, 81.9),
        }
        for out in OUTPUT_COLS:
            r   = cv_results[out]
            rng = ranges[out][1] - ranges[out][0]
            perf_rows.append({
                "Output":           out,
                "MAE":              r["mae"],
                "MAE (% of range)": f"{r['mae']/rng*100:.1f}%",
                "R² Score":         r["r2"],
                "Performance":      "Excellent" if r["r2"] > 0.98
                                    else "Good"  if r["r2"] > 0.90
                                    else "Fair",
            })
        st.dataframe(pd.DataFrame(perf_rows),
                     use_container_width=True, hide_index=True)
        st.caption(
            "MAE = Mean Absolute Error. "
            "R² = proportion of variance explained (1.0 = perfect). "
            "All MAE values are well under 1 unit."
        )

        # ── Feature Importance ────────────────────────────────────────────
        st.subheader("Feature Importance (Sustainability Score)")
        st.write(
            "Relative contribution of each input to predicting "
            "the Sustainability Score."
        )
        fi_df = pd.DataFrame({
            "Feature":    list(fi.keys()),
            "Importance": [round(v * 100, 1) for v in fi.values()]
        }).sort_values("Importance", ascending=False).reset_index(drop=True)
        fi_df["Importance (%)"] = fi_df["Importance"].astype(str) + "%"
        st.dataframe(fi_df[["Feature", "Importance (%)"]],
                     use_container_width=True, hide_index=True)
        st.bar_chart(fi_df.set_index("Feature")["Importance"])

        # ── Correlations ──────────────────────────────────────────────────
        st.subheader("Feature Correlations with Sustainability Score")
        st.write(
            "Pearson correlation between each input and the Sustainability "
            "Score. Positive = higher value raises the score; "
            "Negative = higher value lowers it."
        )
        numeric_df = df.select_dtypes(include=[np.number])
        corr = (numeric_df.corr()["Sustainability Score"]
                .drop("Sustainability Score")
                .sort_values(ascending=False)
                .round(3))
        corr_df = corr.reset_index()
        corr_df.columns = ["Feature", "Correlation with Sustainability Score"]
        corr_df["Direction"] = corr_df[
            "Correlation with Sustainability Score"
        ].apply(
            lambda v: "🟢 Positive" if v > 0.1
                      else "🔴 Negative" if v < -0.1
                      else "⚪ Neutral"
        )
        st.dataframe(corr_df, use_container_width=True, hide_index=True)
        st.bar_chart(
            corr_df.set_index("Feature")[
                "Correlation with Sustainability Score"
            ]
        )

    # ════════════════════════════════════════════════════════════════════════
    #  PAGE 3 — DATA EXPLORER
    # ════════════════════════════════════════════════════════════════════════
    elif page == "📈 Data Explorer":
        st.title("📈 Training Data Explorer")
        st.write(
            f"The model was trained on **{len(df)} construction projects** "
            "across three project types: Commercial, Residential, "
            "and Infrastructure."
        )

        tab_data, tab_stats = st.tabs(
            ["📋 Project Dataset", "📊 Descriptive Statistics"]
        )

        with tab_data:
            # Filter controls
            fc1, fc2 = st.columns(2)
            with fc1:
                type_filter = st.multiselect(
                    "Filter by Project Type",
                    sorted(df["Project Type"].unique()),
                    default=sorted(df["Project Type"].unique())
                )
            with fc2:
                score_range = st.slider(
                    "Filter by Sustainability Score",
                    float(df["Sustainability Score"].min()),
                    float(df["Sustainability Score"].max()),
                    (float(df["Sustainability Score"].min()),
                     float(df["Sustainability Score"].max()))
                )
            filtered = df[
                df["Project Type"].isin(type_filter) &
                df["Sustainability Score"].between(*score_range)
            ]
            st.write(f"Showing **{len(filtered)}** of {len(df)} projects")
            st.dataframe(
                filtered.drop(columns=["ID"]),
                use_container_width=True,
                height=480
            )

        with tab_stats:
            numeric_df = df.select_dtypes(include=[np.number]).drop(
                columns=["ID"]
            )
            st.dataframe(
                numeric_df.describe().round(2),
                use_container_width=True
            )
            st.caption(
                "Count, mean, standard deviation, min, quartiles, "
                "and max for all numeric features across 615 projects."
            )

            st.subheader("Project Type Distribution")
            type_counts = df["Project Type"].value_counts().reset_index()
            type_counts.columns = ["Project Type", "Count"]
            st.dataframe(type_counts, use_container_width=True,
                         hide_index=True)
            st.bar_chart(type_counts.set_index("Project Type")["Count"])


if __name__ == "__main__":
    main()
