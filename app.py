"""Dropout Risk and Counterfactual Recourse Dashboard, version 2 (thesis prototype).

Covers the functions of the base-paper tool (data overview, model performance with colour labels,
global SHAP, individual counterfactuals, group counterfactuals, export, scoring of new data) and adds
the thesis contributions (rule-constrained counterfactuals with feasibility checks, LIME confirmation,
recourse metrics, calibration, fairness check, group recourse with per-student feasibility).

The model and the results of the full-data experiment are loaded from ./data; nothing is re-trained.
"""
import io
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

D = Path(__file__).parent / "data"
if not D.exists():            # works when the data files sit next to app.py (flat upload to GitHub)
    D = Path(__file__).parent
THR = 0.30
NAVY, TEAL, RED, GREEN, AMBER = "#3B82F6", "#14B8A6", "#F43F5E", "#22C55E", "#F59E0B"
S2 = "Curricular units 2nd sem"
A, E, V, W, G = (f"{S2} ({s})" for s in ("approved", "enrolled", "evaluations", "without evaluations", "grade"))
FEES, DEBT = "Tuition fees up to date", "Debtor"
ACTIONABLE = [A, V, W, G, FEES, DEBT]
MODES = ["Constrained (proposed)", "Unconstrained (DiCE random)", "Unconstrained (DiCE genetic)"]

st.set_page_config(page_title="Dropout Risk & Recourse Dashboard", page_icon="🎓", layout="wide")
DARK = st.sidebar.toggle("🌙 Dark mode", value=True, key="dark")
st.sidebar.markdown("**Dropout Risk & Recourse**  \nUse the tabs on the main page. Hover over any chart for exact values; "
                    "click legend items to hide or show series.")
if DARK:
    BG, BG2, CARD, TXT, SUB, BORDER, ACC, ACC2 = "#0B1020", "#121A33", "#172044", "#E8ECF8", "#9FB0D6", "#26335F", "#38BDF8", "#A78BFA"
else:
    BG, BG2, CARD, TXT, SUB, BORDER, ACC, ACC2 = "#F5F8FF", "#FFFFFF", "#FFFFFF", "#14213D", "#5B6B8C", "#DCE4F5", "#2563EB", "#0EA5A4"
st.markdown(f"""<style>
.stApp{{background:{BG};color:{TXT}}}
[data-testid="stHeader"]{{background:transparent}}
[data-testid="stSidebar"]{{background:{BG2}}}
.stApp, .stApp p, .stApp li, .stApp label, .stApp span, .stApp div[data-testid="stMarkdownContainer"], .stApp h1, .stApp h2, .stApp h3, .stApp h4{{color:{TXT}}}
.stApp [data-testid="stCaptionContainer"], .stApp small{{color:{SUB}}}
.block-container{{padding-top:1rem;max-width:1400px}}
div[data-testid="stMetric"]{{background:{CARD};border:1px solid {BORDER};border-radius:14px;padding:12px 16px;box-shadow:0 4px 14px rgba(0,0,0,.12)}}
div[data-testid="stMetricValue"]{{color:{ACC}}}
.hero{{background:linear-gradient(120deg,#1D4ED8 0%,#7C3AED 55%,#06B6D4 100%);border-radius:20px;padding:26px 30px;margin-bottom:14px;color:white;box-shadow:0 10px 30px rgba(37,99,235,.35)}}
.hero h1{{color:white!important;margin:0 0 6px 0;font-size:2rem}}
.hero p{{color:#EAF0FF!important;margin:0;font-size:0.95rem}}
.pill{{display:inline-block;background:rgba(255,255,255,.18);border-radius:20px;padding:3px 12px;margin:10px 6px 0 0;font-size:0.8rem;color:white}}
.note{{background:{CARD};border:1px solid {BORDER};border-left:5px solid {ACC};padding:12px 16px;border-radius:10px;font-size:0.92rem;margin:8px 0}}
.warn{{background:{CARD};border:1px solid {BORDER};border-left:5px solid {AMBER};padding:12px 16px;border-radius:10px;font-size:0.92rem;margin:8px 0}}
.why{{background:{CARD};border:1px dashed {BORDER};padding:8px 14px;border-radius:10px;font-size:0.86rem;color:{SUB};margin:2px 0 14px 0}}
.why b{{color:{ACC2}}}
.guide{{border-left:4px solid {ACC2};padding:4px 12px;margin:0 0 10px 0;color:{SUB};font-size:0.95rem}}
.badge{{display:inline-block;padding:3px 12px;border-radius:12px;color:white;font-size:0.8rem;font-weight:600}}
.card{{background:{CARD};border:1px solid {BORDER};border-radius:14px;padding:12px 16px;box-shadow:0 4px 14px rgba(0,0,0,.12)}}
div[data-testid="stAlert"], div[data-testid="stAlertContainer"], div[data-testid="stAlert"] > div{{background:{CARD}!important;border-radius:10px}}
div[data-testid="stAlert"]{{border:1px solid {BORDER};border-left:5px solid {GREEN}}}
div[data-testid="stAlert"] *, div[data-testid="stAlertContainer"] *{{color:{TXT}!important}}
div[data-testid="stMetricValue"]{{font-size:1.5rem}} div[data-testid="stMetricLabel"] p{{white-space:normal;font-size:0.8rem}}
[data-testid="stFileUploaderDropzone"]{{background:{CARD}!important;border:1px dashed {ACC}!important;border-radius:12px}}
[data-testid="stFileUploaderDropzone"] *{{color:{TXT}!important}}
.stApp button[kind="secondary"], .stApp [data-testid="stBaseButton-secondary"]{{background:{CARD}!important;color:{TXT}!important;border:1px solid {ACC}!important;border-radius:10px}}
.stApp [data-testid="stBaseButton-secondary"] *{{color:{TXT}!important}}
.stTabs [data-baseweb="tab-list"]{{gap:6px;flex-wrap:wrap}}
.stTabs [data-baseweb="tab"]{{background:{CARD};border:1px solid {BORDER};border-radius:12px;padding:6px 14px;height:auto}}
.stTabs [aria-selected="true"]{{background:linear-gradient(120deg,#2563EB,#7C3AED);color:white!important}}
.stTabs [aria-selected="true"] p{{color:white!important}}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"]{{display:none}}
.stApp div[data-baseweb="select"] > div, .stApp input{{background:{CARD};color:{TXT}}}
{"[data-testid=stDataFrame]{filter:invert(0.9) hue-rotate(180deg);border-radius:10px}" if DARK else "[data-testid=stDataFrame]{border:1px solid "+BORDER+";border-radius:10px}"}
</style>""", unsafe_allow_html=True)

PX = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=TXT, size=13),
          margin=dict(l=10, r=10, t=50, b=10), legend=dict(bgcolor="rgba(0,0,0,0)"))
GRID = BORDER


def show(fig, h=360, key=None):
    base = dict(PX)
    if fig.layout.margin.l is not None:
        base.pop("margin")
    if fig.layout.legend.orientation is not None:
        base.pop("legend")
    fig.update_layout(height=h, **base)
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, automargin=True)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, automargin=True)
    st.plotly_chart(fig, width="stretch", key=key, theme=None, config={"displaylogo": False})


def why(txt):
    st.markdown(f"<div class='why'><b>What this shows:</b> {txt}</div>", unsafe_allow_html=True)


def guide(txt):
    st.markdown(f"<div class='guide'>{txt}</div>", unsafe_allow_html=True)


def shot(path, caption, text):
    """Static figure inside a framed card, with a plain-language explanation."""
    st.image(str(path), width="stretch")
    st.caption(caption)
    why(text)


# ------------------------------------------------------------------ loading
@st.cache_resource(show_spinner="Loading saved model and results...")
def load():
    m = joblib.load(D / "final_model.pkl")
    s = joblib.load(D / "shap_part2.pkl")
    lw = joblib.load(D / "lime_part2.pkl")["lime_weights"]
    cf = joblib.load(D / "cf_results.pkl")
    sp = joblib.load(D / "data_splits.pkl")
    grid = np.load(D / "group_cf_grid.npz")
    acts = pd.read_csv(D / "group_cf_actions.csv")
    return m, s, lw, cf, sp, grid, acts


model, shp, lime_w, cfr, splits, grid, acts = load()
pipe = model["final_model"]
X, P, y, expl = shp["X_explain"], shp["p_explain"], shp["y_explain"], shp["shap_exp"]
Xtr, Xte = splits["X_train"], splits["X_test"]
feat = list(X.columns)
cat_all = model["categorical_cols"] + model["binary_cols"]
max_obs = Xtr.max()
flagged_pos = sorted(int(i) for i in cfr["chosen"])
ROW_IDS = list(X.index)


def to_model_frame(d):
    d = d[feat].copy()
    for c in cat_all:
        d[c] = pd.to_numeric(d[c]).astype(Xtr[c].dtype)
    for c in model["numeric_cols"]:
        d[c] = pd.to_numeric(d[c]).astype(float)
        if str(Xtr[c].dtype).startswith("int"):
            d[c] = d[c].round().astype(Xtr[c].dtype)
    return d


def predict(d):
    return pipe.predict_proba(to_model_frame(d))[:, 1]


def rules_ok(f):
    return (f[A] <= f[E]) & (f[A] <= f[V]) & ((f[A] > 0) == (f[G] > 0)) & ((f[A] == 0) | (f[G] >= 10))


def rule_flags(r):
    return {"R1 approved ≤ enrolled": r[A] <= r[E], "R2 approved ≤ evaluations": r[A] <= r[V],
            "R3 grade > 0 only if approved > 0": (r[A] > 0) == (r[G] > 0),
            "R4 grade ≥ 10 when approved > 0": (r[A] == 0) or (r[G] >= 10)}


def csv(name):
    return pd.read_csv(D / name)


def status(p):
    return "At risk" if p >= THR else "Not flagged"


def label_metric(name, v):
    """Rule-of-thumb labels (Sokolova & Lapalme benchmarks, as used in the base paper's Table A1)."""
    if name == "Log-loss":
        return ("Weak", RED) if v > 0.5 else (("Acceptable", AMBER) if v >= 0.3 else ("Good", GREEN))
    lo, hi = (0.7, 0.8) if name == "ROC-AUC" else (0.5, 0.8)
    return ("Weak", RED) if v < lo else (("Acceptable", AMBER) if v <= hi else ("Good", GREEN))


def badge(txt, col):
    return f"<span class='badge' style='background:{col}'>{txt}</span>"


def slide(lbl, lo, hi, val, step=1):
    if hi <= lo:
        st.caption(f"{lbl}: **{val:g}** (already at the permitted limit)")
        return val
    return st.slider(lbl, lo, hi, val, step=step)


def feasible_vs(cand, q):
    """Directions + immutables + data rules, for rows of `cand` against the original student q."""
    imm = [c for c in feat if c not in ACTIONABLE]
    ok = pd.Series(True, index=cand.index)
    for c in imm:
        ok &= cand[c].values == q[c]
    ok &= (cand[A] >= q[A]) & (cand[V] >= q[V]) & (cand[G] >= q[G]) & (cand[W] <= q[W]) & (cand[FEES] >= q[FEES]) & (cand[DEBT] <= q[DEBT])
    return ok & rules_ok(cand)


risk_df = pd.DataFrame({"Student (row ID)": X.index, "Dropout probability": np.round(P, 4),
                        "Flagged (≥ 0.30)": P >= THR, "Actual outcome": np.where(y.values == 1, "Dropout", "Non-dropout"),
                        "Counterfactual searched": [i in set(flagged_pos) for i in range(len(X))]})
default_id = int(risk_df.sort_values("Dropout probability", ascending=False)["Student (row ID)"].iloc[0])

# ------------------------------------------------------------------ header
st.markdown("""<div class='hero'><h1>🎓 Dropout Risk &amp; Counterfactual Recourse Dashboard</h1>
<p>From Prediction to Prescription: Counterfactual-Driven Dropout Risk Intervention Using a Stacking Ensemble and Multi-Layer XAI</p>
<span class='pill'>4,424 students</span><span class='pill'>Stacking RF + XGB + SVM</span><span class='pill'>SHAP + LIME</span>
<span class='pill'>Constrained counterfactuals</span><span class='pill'>Group recourse</span><br>
<p style='margin-top:10px;font-size:0.82rem'>Khansa Munir (2024-ag-1876), MS CS, UAF &nbsp;|&nbsp; Supervisor: Ms Nabeela Ashraf</p></div>""", unsafe_allow_html=True)
st.markdown("<div class='note'><b>Prototype for decision support.</b> Built on the anonymised UCI dataset (4,424 students). "
            "It displays the saved results of the full-data experiment; the final decision on any student rests with a teacher or "
            "advisor. It has not been evaluated with real users.</div>", unsafe_allow_html=True)

tabs = st.tabs(["📊 Data", "📈 Model", "🔍 Explainability", "🧑‍🎓 Student risk", "🎯 Individual CF", "👥 Group CF",
                "🎛️ What-if", "📤 Score new data", "⚖️ Fairness & calibration", "🧪 Method quality", "💾 Export"])

# ================================================================== 1 DATA
with tabs[0]:
    st.subheader("Data overview and preprocessing summary")
    guide("Start here: what data the model learned from and how it was prepared.")
    c = st.columns(5)
    c[0].metric("Students", "4,424")
    c[1].metric("Predictors", f"{len(feat)}")
    c[2].metric("Train / test", f"{len(Xtr):,} / {len(X):,}")
    c[3].metric("Dropout share", "32.1%")
    c[4].metric("Missing / duplicates", "0 / 0")
    l, r = st.columns([1, 1])
    with l:
        shot(D / "fig_class_distribution.png", "Outcome classes (Dropout vs Graduate vs Enrolled)",
             "About one student in three drops out, so the classes are imbalanced. This is why recall and the 0.30 threshold matter more than plain accuracy.")
    with r:
        st.markdown("**Preprocessing summary**")
        st.dataframe(pd.DataFrame({"Step": ["Target", "Split", "Categorical (9)", "Binary (8)", "Numeric", "Imbalance", "Threshold", "Model"],
                                   "Choice": ["Dropout = 1; Graduate and Enrolled = 0", "Stratified 80/20, seed 42 (test set used once)",
                                              "One-hot encoding (fitted on training data)", "Kept as 0/1", "Standard scaling (fitted on training data)",
                                              "Compared: class weights, SMOTENC, threshold tuning", "0.30 from out-of-fold training predictions",
                                              "Stacking: RF + XGBoost + SVM, Logistic Regression meta-learner"]}),
                     hide_index=True, width="stretch")
    st.markdown("**Predictors by type**")
    t1, t2, t3 = st.columns(3)
    t1.markdown("Categorical\n\n" + "\n".join(f"- {c}" for c in model["categorical_cols"]))
    t2.markdown("Binary\n\n" + "\n".join(f"- {c}" for c in model["binary_cols"]))
    t3.markdown("Numeric\n\n" + "\n".join(f"- {c}" for c in model["numeric_cols"]))

# ================================================================== 2 MODEL
with tabs[1]:
    st.subheader("Model performance (test set, used once)")
    guide("How well does the model detect students who will drop out? Pick an operating point and read the colour badges.")
    t5 = csv("table5_stacking_threshold_test.csv").set_index("Setting")
    t4 = csv("table4_final_test.csv").set_index("Model")
    sel_thr = st.radio("Operating point", ["Stacking @ 0.30", "Stacking @ 0.50"], horizontal=True)
    row = t5.loc[sel_thr]
    vals = {"ROC-AUC": float(t4.loc["Stacking Ensemble", "ROC-AUC"]), "Log-loss": float(t4.loc["Stacking Ensemble", "Log-Loss"]),
            "Precision": float(row["Precision"]), "Recall": float(row["Recall"]), "F1": float(row["F1"])}
    cols = st.columns(5)
    for cc, (k, v) in zip(cols, vals.items()):
        lab, col = label_metric(k, v)
        cc.markdown(f"<div class='card'><div style='font-size:0.85rem'>{k}</div><div style='font-size:1.7rem;color:{NAVY}'>{v:.3f}</div>{badge(lab, col)}</div>",
                    unsafe_allow_html=True)
    st.caption("Labels follow the rule-of-thumb benchmarks used by the base paper (Table A1): Weak / Acceptable / Good.")
    cl, cr = st.columns([1, 1])
    with cl:
        cm = np.array([[row["TN"], row["FP"]], [row["FN"], row["TP"]]], dtype=int)
        lbl = np.array([["True negatives", "False positives"], ["False negatives", "True positives"]])
        fig = go.Figure(go.Heatmap(z=cm, x=["Predicted non-dropout", "Predicted dropout"], y=["Actual non-dropout", "Actual dropout"],
                                   colorscale="Blues" if not DARK else "Tealgrn", showscale=False,
                                   text=[[f"{lbl[i][j]}<br><b>{cm[i][j]}</b>" for j in range(2)] for i in range(2)], texttemplate="%{text}",
                                   hovertemplate="%{y} / %{x}: %{z}<extra></extra>", textfont=dict(size=15)))
        fig.update_yaxes(autorange="reversed"); fig.update_layout(title=f"Confusion matrix ({sel_thr})")
        show(fig, 380)
        why("Rows are what really happened, columns what the model predicted. Dark/strong cells on the diagonal are correct predictions; "
            "the off-diagonal cells are the mistakes (false alarms and missed dropouts).")
    with cr:
        st.markdown("**All models (test set, 95% bootstrap CIs)**")
        st.dataframe(csv("table4_final_test.csv"), hide_index=True, width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Imbalance handling (ablation)**")
        st.dataframe(csv("table3_imbalance_ablation.csv"), hide_index=True, width="stretch")
        st.markdown("**Paired bootstrap comparison**")
        st.dataframe(csv("table6_paired_bootstrap.csv"), hide_index=True, width="stretch")
    with c2:
        shot(D / "fig_threshold_curve.png", "Threshold selection from out-of-fold training predictions",
             "Lowering the threshold catches more dropouts (higher recall) but raises false alarms. 0.30 was chosen on training data only, so the test set stays untouched.")
    st.markdown("**AutoML baseline (mljar-supervised, as used by the base paper) vs the stacking ensemble**")
    st.dataframe(csv("table_automl_vs_stacking.csv"), hide_index=True, width="stretch")
    st.dataframe(csv("table_automl_paired_difference.csv"), hide_index=True, width="stretch")
    st.markdown("<div class='note'>Discrimination is statistically indistinguishable (ROC-AUC difference −0.0016, 95% CI [−0.0056, 0.0026]); "
                "AutoML has a slightly lower log-loss (difference +0.0123, 95% CI [0.0021, 0.0227]). The stacking ensemble is therefore "
                "<b>comparable</b>, not better; the thesis contribution lies in the explanation and counterfactual layers.</div>", unsafe_allow_html=True)

# ================================================================== 3 EXPLAINABILITY
with tabs[2]:
    st.subheader("Global explanations (all 885 test students)")
    guide("Why does the model predict dropout? These charts show which features matter overall.")
    sg = csv("table_shap_global.csv").head(15).iloc[::-1]
    sgr = csv("table_shap_groups.csv")
    a, b = st.columns(2)
    with a:
        fig = go.Figure(go.Bar(x=sg["Mean |SHAP|"] * 100, y=sg["Feature"].str.replace(S2 + " ", "2nd sem ").str.replace("Curricular units 1st sem", "1st sem"),
                               orientation="h", marker=dict(color=sg["Mean |SHAP|"], colorscale="Plasma"),
                               hovertemplate="%{y}: %{x:.1f} pp<extra></extra>"))
        fig.update_layout(title="Top 15 features (mean |SHAP|, percentage points)"); fig.update_yaxes(automargin=True); show(fig, 520)
        why("Longer bar = the feature changes the predicted dropout probability more, on average. Second-semester approved units dominate; fees and first-semester results follow.")
    with b:
        fig = go.Figure(go.Pie(labels=sgr["Group"], values=sgr["Share %"], hole=0.55, textinfo="label+percent",
                               marker=dict(colors=["#3B82F6", "#14B8A6", "#A78BFA", "#F59E0B", "#F43F5E", "#22C55E"][:len(sgr)])))
        fig.update_layout(title="Importance by feature group", showlegend=False); show(fig, 520)
        why("The same importance grouped by theme. Academic performance carries most of the signal, financial factors come second.")
    a, b = st.columns(2)
    with a:
        shot(D / "fig_shap_beeswarm.png", "SHAP beeswarm (direction and spread of effects)",
             "Each dot is a student. Dots to the right push the predicted risk up, to the left push it down; colour shows whether the feature value is high or low.")
    with b:
        shot(D / "fig_shap_vs_lime.png", "SHAP vs LIME global ranking",
             "Two independent explanation methods ranked the features. When both agree, the explanation is more trustworthy (the table below quantifies the agreement).")
    st.markdown("**SHAP–LIME agreement (multi-layer XAI check)**")
    st.dataframe(csv("table_consistency_summary.csv").rename(columns={"Unnamed: 0": "Measure"}), hide_index=True, width="stretch")
    st.dataframe(csv("table_shap_global.csv").head(15), hide_index=True, width="stretch")

# ================================================================== 4 STUDENT RISK
with tabs[3]:
    st.subheader("Risk list")
    guide("Find students at risk, then open one student's personal explanation below.")
    c = st.columns(3)
    only_flag = c[0].checkbox("Only flagged at-risk (≥ 0.30)", value=True)
    lo = c[1].slider("Minimum probability", 0.0, 1.0, 0.0, 0.05)
    outcome = c[2].selectbox("Actual outcome", ["All", "Dropout", "Non-dropout"])
    v = risk_df[risk_df["Dropout probability"] >= lo]
    if only_flag:
        v = v[v["Flagged (≥ 0.30)"]]
    if outcome != "All":
        v = v[v["Actual outcome"] == outcome]
    st.caption(f"{len(v)} students shown.")
    st.dataframe(v.sort_values("Dropout probability", ascending=False), hide_index=True, width="stretch", height=300)
    st.subheader("Student risk report")
    rid = st.selectbox("Student (row ID)", ROW_IDS, index=ROW_IDS.index(default_id), key="rep_id")
    pos = ROW_IDS.index(rid)
    p = float(P[pos])
    c = st.columns(3)
    c[0].metric("Dropout probability", f"{p:.1%}")
    c[1].metric("Status", status(p))
    c[2].metric("Actual outcome", "Dropout" if y.iloc[pos] == 1 else "Non-dropout")
    sv = expl.values[pos]
    top = np.argsort(-np.abs(sv))[:5]
    rows = [{"Feature": feat[j], "Student value": X.iloc[pos, j], "SHAP effect (pp)": round(sv[j] * 100, 1),
             "LIME weight": round(float(lime_w[pos, j]), 3),
             "Confirmed by LIME": "yes" if np.sign(lime_w[pos, j]) == np.sign(sv[j]) else "no"} for j in top]
    left, right = st.columns([5, 4])
    with left:
        plt.figure(figsize=(7, 5))
        shap.plots.waterfall(expl[pos], max_display=10, show=False)
        st.pyplot(plt.gcf(), clear_figure=True)
        why("Starts from the average risk and shows how each feature of this student pushes the prediction up (red) or down (blue) to the final probability.")
    with right:
        st.markdown("**Top 5 drivers (SHAP) with LIME check**")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        act = sorted([(f, sv[feat.index(f)] * 100) for f in ACTIONABLE if sv[feat.index(f)] * 100 >= 1], key=lambda t: -t[1])
        st.markdown("**Actionable risk drivers** (≥ +1 pp): " + (", ".join(f"{f} (+{x:.1f} pp)" for f, x in act) if act else "none"))
        st.caption("SHAP effects are on the probability scale (percentage points). A driver is 'confirmed' when LIME agrees on its direction.")

# ================================================================== 5 INDIVIDUAL CF
with tabs[4]:
    st.subheader("Individual counterfactual scenarios (flagged students)")
    guide("For one at-risk student: what realistic changes would bring the risk below the 0.30 threshold?")
    flagged_ids = [ROW_IDS[i] for i in flagged_pos]
    ok_pos = [i for i in flagged_pos
              if (lambda t: t is not None and bool((t["valid"] & t["feasible"]).any()))(cfr["cf_store"].get((np.int64(i), MODES[0])))]
    first_ok = ROW_IDS[ok_pos[0]] if ok_pos else flagged_ids[0]
    st.caption(f"The constrained method found a valid and feasible option for {len(ok_pos)} of {len(flagged_ids)} flagged students.")
    c1, c2 = st.columns([1, 2])
    rid5 = c1.selectbox("Flagged student (row ID)", flagged_ids, index=flagged_ids.index(first_ok), key="cf_id")
    mode = c2.radio("Method", MODES, horizontal=True)
    pos5 = ROW_IDS.index(rid5)
    q = X.iloc[pos5]
    st.write(f"Current dropout probability: **{P[pos5]:.1%}**  (target: below {THR:.0%})")
    cf = cfr["cf_store"].get((np.int64(pos5), mode))
    if cf is None or len(cf) == 0:
        st.markdown("<div class='warn'>No counterfactual was generated for this student by this method within the search budget. "
                    "This does not prove that none exists (see tab 10).</div>", unsafe_allow_html=True)
    else:
        cf = cf.copy()
        only_ok = st.checkbox("Show only valid and feasible options" if mode == MODES[0] else "Show only valid (P<0.30) options", value=True)
        sel = cf[cf["valid"] & cf["feasible"]] if (only_ok and mode == MODES[0]) else (cf[cf["valid"]] if only_ok else cf)
        sel = sel.sort_values(["n_changed", "distance"]).head(5).reset_index(drop=True)
        st.caption(f"{len(cf)} candidates generated; {int((cf['valid'] & cf['feasible']).sum())} are valid and feasible.")
        if len(sel) == 0:
            st.markdown("<div class='warn'>No valid and feasible counterfactual found for this student. It is reported as a limitation "
                        "(search budget and permitted actions).</div>", unsafe_allow_html=True)
        else:
            names = [f"CF{i + 1}" for i in range(len(sel))]
            changed = [c for c in feat if (sel[c].astype(float) != float(q[c])).any()]
            disp = pd.DataFrame({"Original": [f"{q[c]:g}" for c in changed]}, index=changed)
            mask = pd.DataFrame(False, index=changed, columns=["Original"] + names)
            for n_, (_, r_) in zip(names, sel.iterrows()):
                disp[n_] = [f"{r_[c]:g}" for c in changed]
                mask[n_] = [float(r_[c]) != float(q[c]) for c in changed]
            disp.loc["P(Dropout)"] = [f"{P[pos5]:.3f}"] + [f"{x:.3f}" for x in sel["P"]]
            disp.loc["Feasible"] = ["-"] + ["yes" if f else "no" for f in sel["feasible"]]
            mask.loc["P(Dropout)"] = False; mask.loc["Feasible"] = False
            st.markdown("**Original vs counterfactual values** (changed values highlighted)")
            st.dataframe(disp.style.apply(lambda d_: mask.map(lambda b: "background-color:#FFF2CC;font-weight:bold" if b else ""), axis=None),
                         width="stretch")
            g1, g2, g3 = st.columns(3)
            with g1:
                fig = go.Figure(go.Bar(x=["Original"] + names, y=[P[pos5]] + list(sel["P"]), marker=dict(color=[RED] + [TEAL] * len(names)),
                                       text=[f"{v_:.2f}" for v_ in [P[pos5]] + list(sel["P"])], textposition="outside"))
                fig.add_hline(y=THR, line_dash="dash", line_color=AMBER, annotation_text="threshold 0.30")
                fig.update_yaxes(range=[0, 1.05], title="Dropout probability"); fig.update_layout(title="Probability before and after")
                show(fig, 340)
                why("Red = today's risk. Teal bars = predicted risk after applying each counterfactual. Below the dashed line = no longer flagged.")
            with g2:
                rng_ = {c: float(max(max_obs[c], 1)) for c in ACTIONABLE}
                cats = [c.replace(S2 + " ", "").replace("Tuition fees up to date", "Fees paid") for c in ACTIONABLE]
                fig = go.Figure()
                for n_, (_, r_) in zip(names, sel.iterrows()):
                    v_ = [abs(float(r_[c]) - float(q[c])) / rng_[c] for c in ACTIONABLE]
                    fig.add_trace(go.Scatterpolar(r=v_ + v_[:1], theta=cats + cats[:1], fill="toself", opacity=0.45, name=n_))
                fig.update_layout(title="Size of changes (share of feature range)", margin=dict(l=60, r=60, t=50, b=40),
                                  legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center"),
                                  polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(gridcolor=GRID, showticklabels=False),
                                             angularaxis=dict(gridcolor=GRID, tickfont=dict(size=10))))
                show(fig, 340)
                why("Each shape is one option. A bigger spike means that feature must change more. Small, thin shapes are easier for the student to achieve.")
            with g3:
                pick = st.selectbox("Contribution of changes in", names, key="cf_pick")
                r_ = sel.iloc[names.index(pick)]
                eff = []
                for c in changed:
                    if float(r_[c]) == float(q[c]):
                        continue
                    t_ = q.copy(); t_[c] = r_[c]
                    eff.append((c, float(P[pos5] - predict(pd.DataFrame([t_]))[0])))
                eff.sort(key=lambda t: t[1])
                fig = go.Figure(go.Bar(y=[e[0].replace(S2 + " ", "2nd sem ") for e in eff], x=[e[1] * 100 for e in eff], orientation="h",
                                       marker=dict(color=TEAL), hovertemplate="%{y}: %{x:.1f} pp<extra></extra>"))
                fig.update_layout(title=f"Effect of each change in {pick}"); fig.update_xaxes(title="Risk reduction alone (pp)")
                show(fig, 340)
                why("How much of the risk reduction each single change would deliver on its own (indicative only).")
            if mode == MODES[0]:
                b_ = sel.iloc[0]
                ch = [f"{c}: {q[c]:g} → {b_[c]:g}" for c in changed if c in ACTIONABLE and float(b_[c]) != float(q[c])]
                st.success("Suggested action set (fewest changes): " + "; ".join(ch) + f"  →  predicted risk {b_['P']:.1%}")
    st.caption("Constrained counterfactuals vary only the 6 actionable features, within permitted directions and data rules R1–R4. "
               "'Effect when applied alone' is indicative: some single changes alone would break rules R1–R4.")

# ================================================================== 6 GROUP CF
with tabs[5]:
    st.subheader("Group counterfactuals (clusters of flagged students)")
    guide("Instead of one student at a time: one shared action for a whole group of similar at-risk students.")
    Pn, Cn, P0g, gpos = grid["Pn"], grid["C"], grid["P0"], grid["pos"]
    Qg = Xte.iloc[gpos]
    ccols = model["numeric_cols"] + model["binary_cols"]

    @st.cache_data(show_spinner=False)
    def cluster_scores():
        Xs_ = StandardScaler().fit_transform(Qg[ccols].astype(float))
        out = []
        for k_ in range(2, 9):
            l_ = KMeans(k_, n_init=10, random_state=42).fit_predict(Xs_)
            out.append({"k": k_, "Silhouette": round(silhouette_score(Xs_, l_), 3), "Davies-Bouldin": round(davies_bouldin_score(Xs_, l_), 3),
                        "Calinski-Harabasz": round(calinski_harabasz_score(Xs_, l_), 1)})
        return pd.DataFrame(out)

    @st.cache_data(show_spinner=False)
    def labels_for(k_):
        Xs_ = StandardScaler().fit_transform(Qg[ccols].astype(float))
        return KMeans(k_, n_init=10, random_state=42).fit_predict(Xs_)

    cs = cluster_scores()
    best_k = int(cs.sort_values("Silhouette", ascending=False).iloc[0]["k"])
    c1, c2, c3 = st.columns(3)
    mode_k = c1.radio("Number of clusters", ["Automatic (best Silhouette)", "Manual"], horizontal=False)
    k = best_k if mode_k.startswith("Auto") else c2.slider("k", 2, 8, 4)
    lab = labels_for(k)
    sizes = np.bincount(lab, minlength=k)
    cl_id = c3.selectbox("Cluster to analyse", list(range(k)), format_func=lambda i: f"Cluster {i} (n={sizes[i]})")
    ok = Pn < THR
    mem = np.where(lab == cl_id)[0]
    left, right = st.columns([1, 1])
    with left:
        fig = go.Figure(go.Bar(x=[f"Cluster {i}" for i in range(k)], y=sizes, marker=dict(color=[ACC if i == cl_id else "#64748B" for i in range(k)]),
                               text=sizes, textposition="outside"))
        fig.update_layout(title=f"Cluster distribution (k = {k})"); fig.update_yaxes(title="Flagged students"); show(fig, 320)
        why("How many flagged students fall in each group of similar students. The highlighted bar is the cluster analysed below.")
    with right:
        st.markdown("**Cluster validity (k = 2 to 8)**")
        st.dataframe(cs, hide_index=True, width="stretch")
        st.caption("Silhouette values near 0.14 mean weak cluster structure: the groups are a practical segmentation, not sharply separated types.")
    if len(mem) < 20:
        st.markdown(f"<div class='warn'>This cluster has only {len(mem)} students, so its percentages are unstable.</div>", unsafe_allow_html=True)
    sh_all = ok[:, mem].mean(1); cost_all = Cn[:, mem].mean(1)
    best_share = float(sh_all.max())
    target = st.slider("Share of the cluster that must be brought below the 0.30 threshold", 10, 95, 50, 5) / 100
    feasible_actions = np.where(sh_all >= target)[0]
    if len(feasible_actions) == 0:
        st.markdown(f"<div class='warn'>No tested group action reaches {target:.0%} for this cluster. The best tested action reaches {best_share:.0%}; "
                    "showing that one.</div>", unsafe_allow_html=True)
        ai = int(np.argmax(sh_all))
    else:
        ai = int(feasible_actions[np.argmin(cost_all[feasible_actions])])
    act_ = acts.iloc[ai]
    st.markdown("**Recommended group action** (the cheapest tested action that meets the target; every student is changed only in the permitted "
                "direction, never above their enrolled units, and rules R1–R4 hold for every student):")
    parts = []
    if act_["a"] > 0:
        parts.append(f"raise approved units to at least {int(act_['a'])} (never above the student's enrolled units)")
        parts.append(f"raise grade to at least {int(act_['g'])}")
    if act_["vt"] > 0:
        parts.append(f"raise evaluations to at least {int(act_['vt'])}")
    if act_["fix_w"]:
        parts.append("remove units without evaluation")
    if act_["fix_fees"]:
        parts.append("bring tuition fees up to date")
    if act_["fix_debt"]:
        parts.append("clear debts")
    st.success("; ".join(parts) if parts else "no change")
    mm = st.columns(5)
    mm[0].metric("Students", len(mem))
    mm[1].metric("Mean risk", f"{P0g[mem].mean():.1%} → {Pn[ai, mem].mean():.1%}")
    mm[2].metric("Brought below 0.30", f"{sh_all[ai]:.0%}")
    mm[3].metric("Features changed / student", f"{cost_all[ai]:.1f}")
    ind = cfr["rec"]; ind = ind[ind["mode"] == MODES[0]].set_index("k")["valid_feasible"]
    mm[4].metric("Individual DiCE recourse (same students)", f"{np.mean([ind.loc[int(gpos[j])] > 0 for j in mem]):.0%}")
    a1, a2, a3 = st.columns(3)
    with a1:
        before = int((P0g[mem] >= THR).sum()); after = int((Pn[ai, mem] >= THR).sum())
        fig = go.Figure()
        fig.add_bar(x=["Before", "After"], y=[before, after], name="Predicted dropout", marker_color=RED)
        fig.add_bar(x=["Before", "After"], y=[len(mem) - before, len(mem) - after], name="Not flagged", marker_color=GREEN, opacity=0.6)
        fig.update_layout(barmode="stack", title="Predictions before / after"); fig.update_yaxes(title="Students"); show(fig, 330)
        why("The red part shrinks when the group action works: students who move below 0.30 turn green.")
    with a2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cost_all, y=sh_all * 100, mode="markers", name="tested actions", marker=dict(size=7, color="#64748B", opacity=0.7)))
        fig.add_trace(go.Scatter(x=[cost_all[ai]], y=[sh_all[ai] * 100], mode="markers", name="chosen", marker=dict(size=16, color=RED, line=dict(width=2, color="white"))))
        fig.update_layout(title="Cost vs effect (all tested actions)"); fig.update_xaxes(title="Mean features changed per student")
        fig.update_yaxes(title="% brought below 0.30"); show(fig, 330)
        why("Every dot is one of the 600 tested group actions. Up and to the left is best: more students helped with fewer changes.")
    with a3:
        base_ = Qg.iloc[mem]
        st.markdown("**Cluster profile (mean values)**")
        prof = base_[[A, E, V, W, G, FEES, DEBT, "Age at enrollment", "Scholarship holder"]].astype(float).mean().round(2)
        st.dataframe(prof.rename("Mean").to_frame(), width="stretch", height=250)
    st.markdown("<div class='note'>Method note. The base paper derives one counterfactual for each cluster centroid (CF-Group with K-means). Here a single "
                "group action is applied to every member, and each member is checked individually for feasibility (permitted directions, units cap and "
                "rules R1–R4), so the share brought below 0.30 is measured on real students, not on a centroid. The action grid is small (600 actions), "
                "so these results are a conservative, search-based estimate.</div>", unsafe_allow_html=True)


# ================================================================== 7 WHAT-IF
with tabs[6]:
    st.subheader("What-if analysis (actionable features only)")
    guide("Move the sliders to test your own scenario and watch the predicted risk change instantly.")
    rid6 = st.selectbox("Student (row ID)", ROW_IDS, index=ROW_IDS.index(default_id), key="wi_id")
    r0 = X.loc[rid6].copy()
    base_p = float(predict(X.loc[[rid6]])[0])
    st.markdown("<div class='note'>Only the six actionable features can change, and only in the permitted direction (more approved units, evaluations "
                "and grade; fewer units without evaluation; fees paid; no debt). All other student attributes stay fixed.</div>", unsafe_allow_html=True)
    r1 = r0.copy()
    c1, c2 = st.columns(2)
    with c1:
        r1[A] = slide(A, int(r0[A]), int(max_obs[A]), int(r0[A]))
        r1[V] = slide(V, int(r0[V]), int(max_obs[V]), int(r0[V]))
        r1[W] = slide(W, 0, int(r0[W]), int(r0[W]))
    with c2:
        r1[G] = slide(G, float(r0[G]), float(max_obs[G]), float(r0[G]), step=0.5)
        r1[FEES] = st.selectbox(FEES, [int(r0[FEES])] if int(r0[FEES]) == 1 else [0, 1], index=0)
        r1[DEBT] = st.selectbox(DEBT, [0] if int(r0[DEBT]) == 0 else [1, 0], index=0)
    bad = [k_ for k_, ok_ in rule_flags(r1).items() if not ok_]
    new_p = float(predict(pd.DataFrame([r1]))[0])
    m_ = st.columns(3)
    m_[0].metric("Current probability", f"{base_p:.1%}")
    m_[1].metric("Scenario probability", f"{new_p:.1%}", delta=f"{(new_p - base_p) * 100:+.1f} pp", delta_color="inverse")
    m_[2].metric("Scenario status", status(new_p))
    if bad:
        st.markdown("<div class='warn'><b>Not a feasible scenario.</b> Data rule violated: " + "; ".join(bad) +
                    ". Adjust the inputs (approved ≤ enrolled and ≤ evaluations; grade ≥ 10 once approved units > 0).</div>", unsafe_allow_html=True)
    elif new_p < THR <= base_p:
        st.success("This scenario is feasible and brings the student below the 0.30 threshold.")

# ================================================================== 8 SCORE NEW DATA
with tabs[7]:
    st.subheader("Score new students")
    st.markdown("Upload a CSV or Excel file with the same predictor columns as the dataset. The saved model gives each student a dropout probability; "
                "for a chosen student a live constrained search proposes actions under the same rules (R1–R4, permitted directions).")
    st.caption("Required columns: " + ", ".join(feat))
    up = st.file_uploader("CSV or Excel file", type=["csv", "xlsx"])
    use_demo = st.checkbox("Use 10 sample test students instead", value=up is None)
    new = None
    if up is not None:
        try:
            new = pd.read_csv(up) if up.name.lower().endswith(".csv") else pd.read_excel(up)
        except Exception as ex:
            st.error(f"The file could not be read: {ex}")
    elif use_demo:
        new = Xte.iloc[:10].copy().reset_index(drop=True)
    if new is not None:
        missing = [c for c in feat if c not in new.columns]
        if missing:
            st.error("Missing required columns: " + ", ".join(missing))
        else:
            data = new[feat].apply(pd.to_numeric, errors="coerce")
            nbad = int(data.isna().any(axis=1).sum())
            if nbad:
                st.markdown(f"<div class='warn'>{nbad} row(s) have missing or non-numeric values and were removed.</div>", unsafe_allow_html=True)
            data = data.dropna().reset_index(drop=True)
            if len(data) == 0:
                st.error("No valid rows to score.")
            else:
                ps = predict(data)
                out = pd.DataFrame({"Row": data.index, "Dropout probability": np.round(ps, 4), "Status": [status(x) for x in ps]})
                st.dataframe(out, hide_index=True, width="stretch", height=260)
                st.download_button("Download scores (CSV)", out.to_csv(index=False).encode(), "scored_students.csv", "text/csv")
                rr = st.selectbox("Student for live recommendations", list(out["Row"]), key="live_row")
                q = data.iloc[int(rr)]
                if ps[int(rr)] < THR:
                    st.success("This student is not flagged; no recommendation is needed.")
                else:
                    rs = np.random.RandomState(42); n_s = 4000
                    cand = pd.DataFrame([q.values] * n_s, columns=feat)
                    cand[A] = rs.randint(int(q[A]), int(max(q[E], q[A])) + 1, n_s)
                    cand[V] = np.maximum(rs.randint(int(q[V]), int(max_obs[V]) + 1, n_s), cand[A])
                    cand[W] = np.where(rs.rand(n_s) < 0.5, q[W], 0)
                    cand[G] = np.where(cand[A] > 0, np.maximum(q[G], rs.uniform(10, float(max_obs[G]), n_s)), q[G])
                    cand[FEES] = np.where(rs.rand(n_s) < 0.5, q[FEES], 1)
                    cand[DEBT] = np.where(rs.rand(n_s) < 0.5, q[DEBT], 0)
                    cand = cand[feasible_vs(cand, q).values].drop_duplicates().reset_index(drop=True)
                    if len(cand):
                        cand["P"] = predict(cand[feat]); cand["n_changed"] = sum((cand[c].astype(float) != float(q[c])).astype(int) for c in ACTIONABLE)
                        good = cand[cand["P"] < THR].sort_values(["n_changed", "P"]).head(5).reset_index(drop=True)
                    else:
                        good = cand
                    if len(good) == 0:
                        st.markdown("<div class='warn'>No valid and feasible option was found within the live search budget (4,000 random candidates). "
                                    "This does not prove none exists.</div>", unsafe_allow_html=True)
                    else:
                        ch = [c for c in ACTIONABLE if (good[c].astype(float) != float(q[c])).any()]
                        show = good[ch + ["P"]].rename(columns={"P": "New probability"})
                        show.index = [f"Option {i + 1}" for i in range(len(show))]
                        st.markdown("**Recommended options** (valid, feasible, fewest changes first)")
                        st.dataframe(show.round(3), width="stretch")
    st.caption("The live search is a random search inside the permitted ranges (same rules as the constrained method in the thesis); it is separate from "
               "the saved DiCE-based results of the experiment.")

# ================================================================== 9 FAIRNESS & CALIBRATION
with tabs[8]:
    st.subheader("Calibration")
    guide("Can the predicted probabilities be trusted, and is the model fair across groups?")
    c1, c2 = st.columns([3, 2])
    with c1:
        shot(D / "fig_calibration.png", "Reliability diagram",
             "Points near the diagonal mean predicted probabilities match what really happened. Below the diagonal = the model under-estimates risk in that bin.")
    c2.dataframe(csv("table_calibration_summary.csv"), hide_index=True, width="stretch")
    c2.markdown("The model is reasonably well calibrated (Brier 0.085, ECE 0.034). A modest under-estimation appears in the 0.2–0.3 bin "
                "(predicted 0.24, observed 0.32), just below the 0.30 threshold. No recalibration was applied.")
    st.subheader("Descriptive fairness check (test set, threshold 0.30)")
    ft = csv("fairness_table.csv")
    st.dataframe(ft, hide_index=True, width="stretch")
    st.markdown("<div class='warn'>Descriptive only, not a fairness guarantee. Recall is lower for students aged 19 or under (0.68) and scholarship "
                "holders (0.62), and the false-positive rate is much higher for debtors (0.37). Group sizes are small in places and no confidence "
                "intervals are shown. The model should not be used as the only basis for decisions about any group.</div>", unsafe_allow_html=True)

# ================================================================== 10 METHOD QUALITY
with tabs[9]:
    st.subheader("Counterfactual quality (all 284 flagged students)")
    guide("How good are the counterfactuals overall? Includes honest limitations.")
    st.dataframe(csv("table_cf_summary.csv"), hide_index=True, width="stretch")
    shot(D / "fig_cf_comparison.png", "Counterfactual methods compared",
         "Constrained counterfactuals respect real-world rules and permitted actions; unconstrained ones can look better on paper but suggest changes that are not realistic.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Data rules checked on all 4,424 real students**")
        st.dataframe(csv("table_rule_check.csv").rename(columns={"Unnamed: 0": "Rule"}), hide_index=True, width="stretch")
    with c2:
        st.markdown("**Search-budget sensitivity of the recourse rate**")
        st.dataframe(csv("table_cf_search_sensitivity.csv"), hide_index=True, width="stretch")
    reach = int(((Pn < THR).any(0)).sum())
    st.markdown(f"<div class='note'><b>How to read the recourse rate.</b> Recourse rate = share of flagged students with at least one counterfactual that is "
                f"both valid (P &lt; 0.30) and feasible. The DiCE-based constrained search gives 58.8%. It is stable across search settings (58.5–58.8%), "
                f"but it is a <b>lower bound</b>: a best-case feasible profile reaches 70.8% (201 of 284) and the group-action grid of tab 6 reaches "
                f"{reach / len(gpos):.1%} ({reach} of {len(gpos)}), because every profile in those checks is feasible by construction. Students for whom no option "
                f"was found are not proven to have none.</div>", unsafe_allow_html=True)

# ================================================================== 11 EXPORT
with tabs[10]:
    st.subheader("Export")
    st.download_button("Download risk list (CSV)", risk_df.to_csv(index=False).encode(), "risk_list.csv", "text/csv")
    rows_ = []
    for (kk, mo), cfm in cfr["cf_store"].items():
        if mo != MODES[0]:
            continue
        g_ = cfm[cfm["valid"] & cfm["feasible"]].copy()
        if len(g_):
            g_.insert(0, "Student (row ID)", ROW_IDS[int(kk)])
            rows_.append(g_)
    if rows_:
        allcf = pd.concat(rows_)
        st.download_button("Download all valid and feasible constrained counterfactuals (CSV)", allcf.to_csv(index=False).encode(), "constrained_counterfactuals.csv", "text/csv")
    gs = []
    for ci in range(k):
        mi = np.where(lab == ci)[0]; sh_ = ok[:, mi].mean(1); co_ = Cn[:, mi].mean(1); cand_ = np.where(sh_ >= 0.5)[0]
        aj = int(cand_[np.argmin(co_[cand_])]) if len(cand_) else int(np.argmax(sh_))
        gs.append({"k": k, "Cluster": ci, "Students": len(mi), "Mean risk before": round(float(P0g[mi].mean()), 3),
                   "Mean risk after": round(float(Pn[aj, mi].mean()), 3), "Share below 0.30": round(float(sh_[aj]), 3),
                   "Mean features changed": round(float(co_[aj]), 2), **{f"action_{a_}": acts.iloc[aj][a_] for a_ in acts.columns}})
    st.download_button(f"Download group counterfactual summary for k = {k} (CSV, 50% target)", pd.DataFrame(gs).to_csv(index=False).encode(), "group_cf_summary.csv", "text/csv")
    rid8 = st.selectbox("Student for text report", ROW_IDS, key="ex_id")
    p8 = ROW_IDS.index(rid8)
    sv8 = expl.values[p8]
    t8 = np.argsort(-np.abs(sv8))[:5]
    buf = io.StringIO()
    buf.write(f"RISK REPORT | student row {rid8}\nPredicted dropout probability: {P[p8]:.1%} | status: {status(P[p8])}\n\nTop drivers (SHAP, pp):\n")
    for j in t8:
        buf.write(f"- {feat[j]} = {X.iloc[p8, j]}: {sv8[j] * 100:+.1f} pp\n")
    st.code(buf.getvalue())
    st.download_button("Download this report (TXT)", buf.getvalue().encode(), f"risk_report_{rid8}.txt")
