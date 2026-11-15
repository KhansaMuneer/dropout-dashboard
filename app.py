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
NAVY, TEAL, RED, GREEN, AMBER = "#1F3864", "#1C7C8C", "#B5413C", "#3B6D11", "#D98324"
S2 = "Curricular units 2nd sem"
A, E, V, W, G = (f"{S2} ({s})" for s in ("approved", "enrolled", "evaluations", "without evaluations", "grade"))
FEES, DEBT = "Tuition fees up to date", "Debtor"
ACTIONABLE = [A, V, W, G, FEES, DEBT]
MODES = ["Constrained (proposed)", "Unconstrained (DiCE random)", "Unconstrained (DiCE genetic)"]

st.set_page_config(page_title="Dropout Risk & Recourse Dashboard", page_icon="🎓", layout="wide")
st.markdown(f"""<style>
h1,h2,h3{{color:{NAVY}}} .block-container{{padding-top:1.4rem}}
div[data-testid="stMetric"]{{background:#EEF2F8;border-radius:10px;padding:10px 14px}}
.note{{background:#EAF1FB;border-left:4px solid {NAVY};padding:10px 14px;border-radius:6px;font-size:0.92rem;margin:6px 0}}
.warn{{background:#FFF4E5;border-left:4px solid {AMBER};padding:10px 14px;border-radius:6px;font-size:0.92rem;margin:6px 0}}
.badge{{display:inline-block;padding:2px 10px;border-radius:12px;color:white;font-size:0.8rem;font-weight:600}}
.card{{background:#EEF2F8;border-radius:10px;padding:10px 14px}}
</style>""", unsafe_allow_html=True)


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
st.title("🎓 Dropout Risk and Counterfactual Recourse Dashboard")
st.caption("From Prediction to Prescription: Counterfactual-Driven Dropout Risk Intervention Using a Stacking Ensemble and "
           "Multi-Layer XAI  |  Khansa Munir (2024-ag-1876), MS CS, UAF  |  Supervisor: Ms Nabeela Ashraf")
st.markdown("<div class='note'><b>Prototype for decision support.</b> Built on the anonymised UCI dataset (4,424 students). "
            "It displays the saved results of the full-data experiment; the final decision on any student rests with a teacher or "
            "advisor. It has not been evaluated with real users.</div>", unsafe_allow_html=True)

tabs = st.tabs(["1 Data", "2 Model", "3 Explainability", "4 Student risk", "5 Individual CF", "6 Group CF",
                "7 What-if", "8 Score new data", "9 Fairness & calibration", "10 Method quality", "11 Export"])

# ================================================================== 1 DATA
with tabs[0]:
    st.subheader("Data overview and preprocessing summary")
    c = st.columns(5)
    c[0].metric("Students", "4,424")
    c[1].metric("Predictors", f"{len(feat)}")
    c[2].metric("Train / test", f"{len(Xtr):,} / {len(X):,}")
    c[3].metric("Dropout share", "32.1%")
    c[4].metric("Missing / duplicates", "0 / 0")
    l, r = st.columns([1, 1])
    with l:
        st.image(str(D / "fig_class_distribution.png"), caption="Outcome classes (Dropout vs Graduate vs Enrolled)", width="stretch")
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
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v}", ha="center", va="center", color="white" if v > cm.max() / 2 else "black", fontsize=14)
        ax.set_xticks([0, 1], ["Non-dropout", "Dropout"]); ax.set_yticks([0, 1], ["Non-dropout", "Dropout"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"Confusion matrix ({sel_thr})")
        st.pyplot(fig, clear_figure=True)
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
        st.image(str(D / "fig_threshold_curve.png"), caption="Threshold selection from out-of-fold training predictions", width="stretch")
    st.markdown("**AutoML baseline (mljar-supervised, as used by the base paper) vs the stacking ensemble**")
    st.dataframe(csv("table_automl_vs_stacking.csv"), hide_index=True, width="stretch")
    st.dataframe(csv("table_automl_paired_difference.csv"), hide_index=True, width="stretch")
    st.markdown("<div class='note'>Discrimination is statistically indistinguishable (ROC-AUC difference −0.0016, 95% CI [−0.0056, 0.0026]); "
                "AutoML has a slightly lower log-loss (difference +0.0123, 95% CI [0.0021, 0.0227]). The stacking ensemble is therefore "
                "<b>comparable</b>, not better; the thesis contribution lies in the explanation and counterfactual layers.</div>", unsafe_allow_html=True)

# ================================================================== 3 EXPLAINABILITY
with tabs[2]:
    st.subheader("Global explanations (all 885 test students)")
    a, b = st.columns(2)
    a.image(str(D / "fig_shap_bar.png"), caption="Global SHAP importance (probability scale)", width="stretch")
    b.image(str(D / "fig_shap_beeswarm.png"), caption="SHAP beeswarm (direction and spread of effects)", width="stretch")
    a, b = st.columns(2)
    a.image(str(D / "fig_shap_groups.png"), caption="Importance by feature group", width="stretch")
    b.image(str(D / "fig_shap_vs_lime.png"), caption="SHAP vs LIME global ranking", width="stretch")
    st.markdown("**SHAP–LIME agreement (multi-layer XAI check)**")
    st.dataframe(csv("table_consistency_summary.csv").rename(columns={"Unnamed: 0": "Measure"}), hide_index=True, width="stretch")
    st.dataframe(csv("table_shap_global.csv").head(15), hide_index=True, width="stretch")

# ================================================================== 4 STUDENT RISK
with tabs[3]:
    st.subheader("Risk list")
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
    with right:
        st.markdown("**Top 5 drivers (SHAP) with LIME check**")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        act = sorted([(f, sv[feat.index(f)] * 100) for f in ACTIONABLE if sv[feat.index(f)] * 100 >= 1], key=lambda t: -t[1])
        st.markdown("**Actionable risk drivers** (≥ +1 pp): " + (", ".join(f"{f} (+{x:.1f} pp)" for f, x in act) if act else "none"))
        st.caption("SHAP effects are on the probability scale (percentage points). A driver is 'confirmed' when LIME agrees on its direction.")

# ================================================================== 5 INDIVIDUAL CF
with tabs[4]:
    st.subheader("Individual counterfactual scenarios (flagged students)")
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
                fig, ax = plt.subplots(figsize=(4.4, 3.4))
                ax.bar(["Original"] + names, [P[pos5]] + list(sel["P"]), color=[RED] + [TEAL] * len(names))
                ax.axhline(THR, color="grey", ls="--"); ax.set_ylim(0, 1); ax.set_ylabel("Dropout probability")
                ax.set_title("Probability before and after"); st.pyplot(fig, clear_figure=True)
            with g2:
                rng_ = {c: float(max(max_obs[c], 1)) for c in ACTIONABLE}
                ang = np.linspace(0, 2 * np.pi, len(ACTIONABLE), endpoint=False).tolist(); ang += ang[:1]
                fig = plt.figure(figsize=(4.4, 3.6)); ax = plt.subplot(111, polar=True)
                for n_, (_, r_) in zip(names, sel.iterrows()):
                    v_ = [abs(float(r_[c]) - float(q[c])) / rng_[c] for c in ACTIONABLE]; v_ += v_[:1]
                    ax.plot(ang, v_, label=n_); ax.fill(ang, v_, alpha=0.08)
                ax.set_xticks(ang[:-1], [c.replace(S2 + " ", "2nd sem ").replace("Tuition fees up to date", "Fees paid") for c in ACTIONABLE], fontsize=6)
                ax.set_title("Size of changes (share of feature range)", fontsize=9); ax.legend(fontsize=6, loc="upper right", bbox_to_anchor=(1.3, 1.1))
                st.pyplot(fig, clear_figure=True)
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
                fig, ax = plt.subplots(figsize=(4.6, 3.4))
                ax.barh([e[0].replace(S2 + " ", "2nd sem ") for e in eff], [e[1] * 100 for e in eff], color=TEAL)
                ax.set_xlabel("Risk reduction when applied alone (pp)"); ax.set_title(f"Effect of each change in {pick}", fontsize=9)
                ax.tick_params(axis="y", labelsize=7); st.pyplot(fig, clear_figure=True)
            if mode == MODES[0]:
                b_ = sel.iloc[0]
                ch = [f"{c}: {q[c]:g} → {b_[c]:g}" for c in changed if c in ACTIONABLE and float(b_[c]) != float(q[c])]
                st.success("Suggested action set (fewest changes): " + "; ".join(ch) + f"  →  predicted risk {b_['P']:.1%}")
    st.caption("Constrained counterfactuals vary only the 6 actionable features, within permitted directions and data rules R1–R4. "
               "'Effect when applied alone' is indicative: some single changes alone would break rules R1–R4.")

# ================================================================== 6 GROUP CF
with tabs[5]:
    st.subheader("Group counterfactuals (clusters of flagged students)")
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
        fig, ax = plt.subplots(figsize=(4.8, 3.2))
        ax.bar([f"C{i}" for i in range(k)], sizes, color=[NAVY if i == cl_id else "#9AA7BD" for i in range(k)])
        ax.set_ylabel("Flagged students"); ax.set_title(f"Cluster distribution (k = {k})"); st.pyplot(fig, clear_figure=True)
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
        fig, ax = plt.subplots(figsize=(4.2, 3.2))
        ax.bar(["Before", "After"], [before, after], color=RED, label="Predicted dropout")
        ax.bar(["Before", "After"], [len(mem) - before, len(mem) - after], bottom=[before, after], color=GREEN, alpha=0.45, label="Not flagged")
        ax.legend(fontsize=7); ax.set_ylabel("Students"); ax.set_title("Predictions before / after"); st.pyplot(fig, clear_figure=True)
    with a2:
        fig, ax = plt.subplots(figsize=(4.4, 3.2))
        ax.scatter(cost_all, sh_all * 100, s=10, color="#9AA7BD"); ax.scatter([cost_all[ai]], [sh_all[ai] * 100], s=60, color=RED, label="chosen")
        ax.set_xlabel("Mean features changed per student"); ax.set_ylabel("% brought below 0.30"); ax.set_title("Cost vs effect (all tested actions)", fontsize=9)
        ax.legend(fontsize=7); st.pyplot(fig, clear_figure=True)
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
    c1, c2 = st.columns([3, 2])
    c1.image(str(D / "fig_calibration.png"), width="stretch")
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
    st.dataframe(csv("table_cf_summary.csv"), hide_index=True, width="stretch")
    st.image(str(D / "fig_cf_comparison.png"), width=760)
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
