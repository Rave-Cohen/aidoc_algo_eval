import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import streamlit as st

st.set_page_config(
    page_title="Aidoc Algorithm Evaluation",
    layout="wide",
)

COLOR_MAP = {
    "Radiologist": "#2ca02c",
    "Algo 1": "#d62728",
    "Algo 2": "#1f77b4",
    "Algo 3": "#ff7f0e",
}
ALGO_COLS = ["Algo 1", "Algo 2", "Algo 3"]
AGE_LINE_COLS = ["Radiologist", "Algo 1", "Algo 2", "Algo 3"]

AGE_QUERY = """
WITH binned AS (
    SELECT *,
        CASE
            WHEN age < 3 THEN '00-03' WHEN age < 6 THEN '03-06' WHEN age < 9 THEN '06-09'
            WHEN age < 12 THEN '09-12' WHEN age < 15 THEN '12-15' WHEN age < 18 THEN '15-18'
            WHEN age < 21 THEN '18-21' WHEN age < 24 THEN '21-24' WHEN age < 27 THEN '24-27'
            WHEN age < 30 THEN '27-30' WHEN age < 32 THEN '30-32' WHEN age < 37 THEN '32-37'
            WHEN age < 42 THEN '37-42' WHEN age < 47 THEN '42-47' WHEN age < 52 THEN '47-52'
            WHEN age < 57 THEN '52-57' WHEN age < 62 THEN '57-62' WHEN age < 67 THEN '62-67'
            WHEN age < 70 THEN '67-70' WHEN age < 73 THEN '70-73' WHEN age < 76 THEN '73-76'
            WHEN age < 79 THEN '76-79' WHEN age < 82 THEN '79-82' ELSE '82+'
        END AS age_group,
        CASE WHEN algo1_answer = radiologist_answer THEN 1 ELSE 0 END AS a1_corr,
        CASE WHEN algo2_answer = radiologist_answer THEN 1 ELSE 0 END AS a2_corr,
        CASE WHEN algo3_answer = radiologist_answer THEN 1 ELSE 0 END AS a3_corr
    FROM filtered_df
)
SELECT
    age_group,
    COUNT(*) AS scans,
    ROUND(AVG(CASE WHEN radiologist_answer = 'P' THEN 1.0 ELSE 0.0 END) * 100, 1) AS "Radiologist",
    ROUND(AVG(a1_corr) * 100, 1) AS "Algo 1",
    ROUND(AVG(a2_corr) * 100, 1) AS "Algo 2",
    ROUND(AVG(a3_corr) * 100, 1) AS "Algo 3"
FROM binned
GROUP BY age_group
ORDER BY age_group
"""


def group_accuracy_query(group_col: str) -> str:
    return f"""
    SELECT
        {group_col},
        COUNT(*) AS scans,
        SUM(CASE WHEN radiologist_answer = 'P' THEN 1 ELSE 0 END) AS positives,
        ROUND(
            SUM(CASE WHEN radiologist_answer = 'P' THEN 1.0 ELSE 0.0 END)
            / COUNT(*) * 100, 1
        ) AS prevalence_pct,
        ROUND(AVG(CASE WHEN algo1_answer = radiologist_answer THEN 1.0 ELSE 0.0 END) * 100, 1) AS "Algo 1",
        ROUND(AVG(CASE WHEN algo2_answer = radiologist_answer THEN 1.0 ELSE 0.0 END) * 100, 1) AS "Algo 2",
        ROUND(AVG(CASE WHEN algo3_answer = radiologist_answer THEN 1.0 ELSE 0.0 END) * 100, 1) AS "Algo 3"
    FROM filtered_df
    GROUP BY {group_col}
    ORDER BY {group_col}
    """


def accuracy_bar(df: pd.DataFrame, group_col: str, title: str) -> go.Figure:
    long = df.melt(
        id_vars=group_col,
        value_vars=ALGO_COLS,
        var_name="Algorithm",
        value_name="Accuracy (%)",
    )
    fig = px.bar(
        long,
        y=group_col,
        x="Accuracy (%)",
        color="Algorithm",
        barmode="group",
        orientation="h",
        title=title,
        color_discrete_map=COLOR_MAP,
        labels={group_col: group_col.replace("_", " ").title()},
    )
    fig.update_layout(legend_title_text="", yaxis_title="")
    return fig


def age_subgroup_chart(age_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=age_df["age_group"],
            y=age_df["scans"],
            name="Scan count",
            marker_color="rgba(128, 128, 128, 0.25)",
            hovertemplate="Age %{x}<br>Scans: %{y:,}<extra></extra>",
        ),
        secondary_y=True,
    )
    for col in AGE_LINE_COLS:
        fig.add_trace(
            go.Scatter(
                x=age_df["age_group"],
                y=age_df[col],
                name=col,
                mode="lines+markers",
                line=dict(color=COLOR_MAP[col]),
                marker=dict(color=COLOR_MAP[col]),
                hovertemplate=f"{col}<br>%{{x}}: %{{y:.1f}}%<extra></extra>",
            ),
            secondary_y=False,
        )
    fig.update_layout(
        title="Accuracy & Radiologist Prevalence by Age Group",
        legend_title_text="",
        xaxis=dict(type="category", title="Age Group"),
        barmode="overlay",
    )
    fig.update_yaxes(title_text="Rate (%)", secondary_y=False, range=[0, 100])
    fig.update_yaxes(title_text="Scan count", secondary_y=True, showgrid=False)
    return fig


AGE_GENDER_QUERY = """
WITH binned AS (
    SELECT *,
        CASE
            WHEN age < 18 THEN '00-17'
            WHEN age < 40 THEN '18-39'
            WHEN age < 65 THEN '40-64'
            ELSE '65+'
        END AS age_group
    FROM filtered_df
)
SELECT
    age_group,
    gender,
    COUNT(*) AS scans,
    SUM(CASE WHEN radiologist_answer = 'P' THEN 1 ELSE 0 END) AS positives,
    ROUND(AVG(CASE WHEN radiologist_answer = 'P' THEN 1.0 ELSE 0.0 END) * 100, 1) AS prevalence_pct,
    ROUND(AVG(CASE WHEN algo1_answer = radiologist_answer THEN 1.0 ELSE 0.0 END) * 100, 1) AS "Algo 1 Acc",
    ROUND(AVG(CASE WHEN algo2_answer = radiologist_answer THEN 1.0 ELSE 0.0 END) * 100, 1) AS "Algo 2 Acc",
    ROUND(AVG(CASE WHEN algo3_answer = radiologist_answer THEN 1.0 ELSE 0.0 END) * 100, 1) AS "Algo 3 Acc",
    ROUND(
        SUM(CASE WHEN algo1_answer = 'P' AND radiologist_answer = 'P' THEN 1.0 ELSE 0.0 END)
        / NULLIF(SUM(CASE WHEN radiologist_answer = 'P' THEN 1 ELSE 0 END), 0) * 100, 1
    ) AS "Algo 1 Sens",
    ROUND(
        SUM(CASE WHEN algo2_answer = 'P' AND radiologist_answer = 'P' THEN 1.0 ELSE 0.0 END)
        / NULLIF(SUM(CASE WHEN radiologist_answer = 'P' THEN 1 ELSE 0 END), 0) * 100, 1
    ) AS "Algo 2 Sens",
    ROUND(
        SUM(CASE WHEN algo3_answer = 'P' AND radiologist_answer = 'P' THEN 1.0 ELSE 0.0 END)
        / NULLIF(SUM(CASE WHEN radiologist_answer = 'P' THEN 1 ELSE 0 END), 0) * 100, 1
    ) AS "Algo 3 Sens"
FROM binned
GROUP BY age_group, gender
ORDER BY age_group, gender
"""


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    n = len(x)
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    out = np.empty(n, dtype=np.float64)
    out[order] = ranks
    return out


def delong_aucs(y_true: np.ndarray, scores: np.ndarray):
    """DeLong AUCs and covariance for k classifiers (k, n). Binary scores are valid."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int(y_true.sum())
    n_neg = y_true.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return None, None
    k = scores.shape[0]
    tx = np.empty((k, n_pos))
    ty = np.empty((k, n_neg))
    tz = np.empty((k, y_true.size))
    for r in range(k):
        tx[r] = _midrank(scores[r, y_true == 1])
        ty[r] = _midrank(scores[r, y_true == 0])
        tz[r] = _midrank(scores[r])
    aucs = (tz[:, y_true == 1].sum(axis=1) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    v01 = (tz[:, y_true == 1] - tx) / n_neg
    v10 = 1.0 - (tz[:, y_true == 0] - ty) / n_pos
    sx = np.cov(v01, ddof=1) if n_pos > 1 else np.zeros((k, k))
    sy = np.cov(v10, ddof=1) if n_neg > 1 else np.zeros((k, k))
    if np.ndim(sx) == 0:
        sx = np.array([[sx]])
        sy = np.array([[sy]])
    cov = sx / n_pos + sy / n_neg
    return aucs, cov


def fmt_p(p) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "—"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def delong_omnibus_and_pairs(aucs: np.ndarray, cov: np.ndarray, names: list[str]):
    k = len(names)
    contrast = np.column_stack([np.ones(k - 1), -np.eye(k - 1)])
    delta = contrast @ aucs
    cov_c = contrast @ cov @ contrast.T
    try:
        chi2 = float(delta.T @ np.linalg.pinv(cov_c) @ delta)
        p_omni = float(stats.chi2.sf(chi2, k - 1))
    except np.linalg.LinAlgError:
        chi2, p_omni = np.nan, np.nan

    rows = []
    for i in range(k):
        for j in range(i + 1, k):
            diff = aucs[i] - aucs[j]
            var = cov[i, i] + cov[j, j] - 2 * cov[i, j]
            if var <= 0:
                z, p = np.nan, np.nan
            else:
                z = diff / np.sqrt(var)
                p = float(2 * stats.norm.sf(abs(z)))
            rows.append(
                {
                    "Pair": f"{names[i]} vs {names[j]}",
                    "AUC diff": round(float(diff), 4),
                    "z": "—" if np.isnan(z) else round(float(z), 3),
                    "p-value": fmt_p(p),
                }
            )
    return chi2, p_omni, pd.DataFrame(rows)


def cochran_q(matrix: np.ndarray):
    """matrix shape (n_scans, k_algos) with 0/1 outcomes."""
    x = np.asarray(matrix, dtype=np.float64)
    if x.size == 0 or x.shape[1] < 2:
        return np.nan, np.nan, x.shape[0]
    n, k = x.shape
    col_sums = x.sum(axis=0)
    row_sums = x.sum(axis=1)
    total = col_sums.sum()
    denom = k * total - np.sum(row_sums ** 2)
    if denom <= 0:
        return np.nan, np.nan, n
    q = (k - 1) * (k * np.sum(col_sums ** 2) - total ** 2) / denom
    p = float(stats.chi2.sf(q, k - 1))
    return float(q), p, n


def mcnemar_pairs(matrix: np.ndarray, names: list[str]):
    n, k = matrix.shape
    rows = []
    n_pairs = k * (k - 1) / 2
    alpha = 0.05 / n_pairs if n_pairs else 0.05
    for i in range(k):
        for j in range(i + 1, k):
            a_ok = matrix[:, i].astype(bool)
            b_ok = matrix[:, j].astype(bool)
            b = int(np.sum(~a_ok & b_ok))
            c = int(np.sum(a_ok & ~b_ok))
            n_disc = b + c
            if n_disc == 0:
                p_exact = 1.0
            else:
                p_exact = float(stats.binomtest(min(b, c), n_disc, 0.5).pvalue)
            rows.append(
                {
                    "Pair": f"{names[i]} vs {names[j]}",
                    "A only": c,
                    "B only": b,
                    "discordant": n_disc,
                    "McNemar p": fmt_p(p_exact),
                    "sig (Bonferroni)": "yes" if p_exact < alpha else "no",
                }
            )
    return pd.DataFrame(rows), alpha


def algo_binary_matrix(df: pd.DataFrame, mode: str) -> np.ndarray:
    """0/1 matrix (n, 3) for Cochran/McNemar. mode: accuracy | sensitivity | specificity."""
    y = df["radiologist_answer"].eq("P")
    preds = [
        df["algo1_answer"].eq("P"),
        df["algo2_answer"].eq("P"),
        df["algo3_answer"].eq("P"),
    ]
    if mode == "accuracy":
        cols = [p.eq(y).astype(int).to_numpy() for p in preds]
        return np.column_stack(cols)
    if mode == "sensitivity":
        mask = y.to_numpy()
        cols = [p.to_numpy()[mask].astype(int) for p in preds]
        return np.column_stack(cols) if mask.any() else np.empty((0, 3), dtype=int)
    mask = (~y).to_numpy()
    cols = [(~p).to_numpy()[mask].astype(int) for p in preds]
    return np.column_stack(cols) if mask.any() else np.empty((0, 3), dtype=int)


@st.cache_data
def load_data():
    file_path = "AI_data_analysis_exercise_(4)_(2)_(2)_(4).xlsx"
    df = pd.read_excel(file_path)
    time_cols = [
        "scan_timestamp",
        "radiologist_sign_time",
        "algos_start_run",
        "algo1_finish_run",
        "algo2_finish_run",
        "algo3_finish_run",
    ]
    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


df = load_data()

st.title("Aidoc Algorithm Evaluation Dashboard")
st.markdown("Interactive local analysis of AI performance and turnaround times.")

st.sidebar.header("Filter Data")
sites = st.sidebar.multiselect(
    "Select Hospital Site",
    options=df["site"].unique(),
    default=df["site"].unique(),
)
patient_classes = st.sidebar.multiselect(
    "Select Patient Class",
    options=df["patient_class"].unique(),
    default=df["patient_class"].unique(),
)

filtered_df = duckdb.sql(
    """
    SELECT * FROM df
    WHERE site IN list_transform(?, x -> x)
      AND patient_class IN list_transform(?, x -> x)
    """,
    params=[sites, patient_classes],
).df()

total_scans = len(filtered_df)
positives = int((filtered_df["radiologist_answer"] == "P").sum()) if total_scans else 0
prevalence = (positives / total_scans * 100) if total_scans else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Scans", f"{total_scans:,}")
col2.metric("Positive Cases", f"{positives:,}")
col3.metric("Prevalence Rate", f"{prevalence:.1f}%")
col4.metric("Active Sites", f"{filtered_df['site'].nunique() if total_scans else 0}")

st.divider()

tab1, tab2, tab3 = st.tabs(["Overview", "Timelines", "Metrics"])

# --- TAB 1: OVERVIEW ---
with tab1:
    st.subheader("Patient Class: IN vs ED")
    pc_df = duckdb.sql(group_accuracy_query("patient_class")).df()

    pc_cols = st.columns(len(pc_df) if len(pc_df) else 1)
    for i, row in pc_df.iterrows():
        pc_cols[i].metric(
            f"{row['patient_class']} scans",
            f"{int(row['scans']):,}",
            f"{row['prevalence_pct']}% prevalence",
        )

    st.dataframe(pc_df, use_container_width=True, hide_index=True)
    st.plotly_chart(
        accuracy_bar(pc_df, "patient_class", "Accuracy (%) by Patient Class"),
        use_container_width=True,
    )

    st.divider()
    st.subheader("Gender")
    gender_df = duckdb.sql(group_accuracy_query("gender")).df()

    g_cols = st.columns(len(gender_df) if len(gender_df) else 1)
    for i, row in gender_df.iterrows():
        g_cols[i].metric(
            f"{row['gender']} scans",
            f"{int(row['scans']):,}",
            f"{row['prevalence_pct']}% prevalence",
        )

    st.dataframe(gender_df, use_container_width=True, hide_index=True)
    st.plotly_chart(
        accuracy_bar(gender_df, "gender", "Accuracy (%) by Gender"),
        use_container_width=True,
    )

    st.divider()
    st.subheader("Age Subgroups")
    st.markdown(
        "Radiologist line shows positive rate (prevalence) per age bin. "
        "Grey bars show scan count per bin (right axis)."
    )
    age_df = duckdb.sql(AGE_QUERY).df()
    st.dataframe(age_df, use_container_width=True, hide_index=True)
    st.plotly_chart(age_subgroup_chart(age_df), use_container_width=True)

# --- TAB 2: TIMELINES ---
with tab2:
    st.subheader("Processing Timelines & Turnaround Time (TAT)")
    st.markdown(
        "Completion time in minutes from `algos_start_run`. "
        "Rows with missing timestamps or negative durations are excluded."
    )

    duration_query = """
    WITH duration_calc AS (
        SELECT
            EPOCH(radiologist_sign_time - algos_start_run) / 60.0 AS rad_min,
            EPOCH(algo1_finish_run - algos_start_run) / 60.0 AS algo1_min,
            EPOCH(algo2_finish_run - algos_start_run) / 60.0 AS algo2_min,
            EPOCH(algo3_finish_run - algos_start_run) / 60.0 AS algo3_min
        FROM filtered_df
        WHERE radiologist_sign_time IS NOT NULL
          AND algos_start_run IS NOT NULL
          AND algo1_finish_run IS NOT NULL
          AND algo2_finish_run IS NOT NULL
          AND algo3_finish_run IS NOT NULL
    )
    SELECT
        rad_min AS "Radiologist",
        algo1_min AS "Algo 1",
        algo2_min AS "Algo 2",
        algo3_min AS "Algo 3"
    FROM duration_calc
    WHERE rad_min >= 0
      AND algo1_min >= 0
      AND algo2_min >= 0
      AND algo3_min >= 0
    """
    duration_df = duckdb.sql(duration_query).df()

    n_with_times = duckdb.sql(
        """
        SELECT COUNT(*) AS n FROM filtered_df
        WHERE radiologist_sign_time IS NOT NULL
          AND algos_start_run IS NOT NULL
          AND algo1_finish_run IS NOT NULL
          AND algo2_finish_run IS NOT NULL
          AND algo3_finish_run IS NOT NULL
        """
    ).df()["n"].iloc[0]
    dropped = int(n_with_times) - len(duration_df)
    if dropped:
        st.caption(f"Excluded {dropped:,} rows with negative TAT (finish before start).")

    if duration_df.empty:
        st.info("No valid TAT rows after filtering missing or negative durations.")
    else:
        timeline_df = pd.DataFrame(
            {
                "Entity": duration_df.columns,
                "Mean (min)": duration_df.mean().round(2).values,
                "Median (min)": duration_df.median().round(2).values,
                "Min (min)": duration_df.min().round(2).values,
                "Max (min)": duration_df.max().round(2).values,
            }
        )
        st.dataframe(
            timeline_df.style.format(
                "{:.2f}",
                subset=["Mean (min)", "Median (min)", "Min (min)", "Max (min)"],
            ),
            use_container_width=True,
            hide_index=True,
        )

        tat_long = duration_df.melt(var_name="Entity", value_name="Minutes")
        fig_box = px.box(
            tat_long,
            x="Entity",
            y="Minutes",
            color="Entity",
            color_discrete_map=COLOR_MAP,
            title="TAT Distribution (minutes)",
            points=False,
        )
        fig_box.update_layout(showlegend=False, xaxis_title="")
        st.plotly_chart(fig_box, use_container_width=True)

# --- TAB 3: METRICS ---
with tab3:
    st.subheader("Core Clinical Metrics")
    st.markdown(
        "Confusion-matrix counts plus sensitivity, specificity, precision, and accuracy "
        "(radiologist as ground truth)."
    )

    metrics_query = """
    WITH cm AS (
        SELECT
            'Algo 1' AS Algorithm,
            SUM(CASE WHEN algo1_answer = 'P' AND radiologist_answer = 'P' THEN 1 ELSE 0 END) AS TP,
            SUM(CASE WHEN algo1_answer = 'N' AND radiologist_answer = 'N' THEN 1 ELSE 0 END) AS TN,
            SUM(CASE WHEN algo1_answer = 'P' AND radiologist_answer = 'N' THEN 1 ELSE 0 END) AS FP,
            SUM(CASE WHEN algo1_answer = 'N' AND radiologist_answer = 'P' THEN 1 ELSE 0 END) AS FN
        FROM filtered_df
        UNION ALL
        SELECT
            'Algo 2',
            SUM(CASE WHEN algo2_answer = 'P' AND radiologist_answer = 'P' THEN 1 ELSE 0 END),
            SUM(CASE WHEN algo2_answer = 'N' AND radiologist_answer = 'N' THEN 1 ELSE 0 END),
            SUM(CASE WHEN algo2_answer = 'P' AND radiologist_answer = 'N' THEN 1 ELSE 0 END),
            SUM(CASE WHEN algo2_answer = 'N' AND radiologist_answer = 'P' THEN 1 ELSE 0 END)
        FROM filtered_df
        UNION ALL
        SELECT
            'Algo 3',
            SUM(CASE WHEN algo3_answer = 'P' AND radiologist_answer = 'P' THEN 1 ELSE 0 END),
            SUM(CASE WHEN algo3_answer = 'N' AND radiologist_answer = 'N' THEN 1 ELSE 0 END),
            SUM(CASE WHEN algo3_answer = 'P' AND radiologist_answer = 'N' THEN 1 ELSE 0 END),
            SUM(CASE WHEN algo3_answer = 'N' AND radiologist_answer = 'P' THEN 1 ELSE 0 END)
        FROM filtered_df
    )
    SELECT
        Algorithm,
        TP, TN, FP, FN,
        ROUND((TP * 100.0) / NULLIF(TP + FN, 0), 2) AS "Sensitivity (%)",
        ROUND((TN * 100.0) / NULLIF(TN + FP, 0), 2) AS "Specificity (%)",
        ROUND((TP * 100.0) / NULLIF(TP + FP, 0), 2) AS "Precision (%)",
        ROUND(((TP + TN) * 100.0) / NULLIF(TP + TN + FP + FN, 0), 2) AS "Accuracy (%)"
    FROM cm
    """
    metrics_df = duckdb.sql(metrics_query).df()
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    radar_metrics = ["Sensitivity (%)", "Specificity (%)", "Precision (%)", "Accuracy (%)"]
    fig_radar = go.Figure()
    for _, row in metrics_df.iterrows():
        values = [row[m] for m in radar_metrics]
        fig_radar.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=radar_metrics + [radar_metrics[0]],
                name=row["Algorithm"],
                line=dict(color=COLOR_MAP[row["Algorithm"]]),
                fill="toself",
                fillcolor=COLOR_MAP[row["Algorithm"]],
                opacity=0.45,
            )
        )
    fig_radar.update_layout(
        title="Diagnostic Profile",
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.divider()
    st.subheader("Age × Gender Performance")
    st.markdown(
        "Accuracy and sensitivity by life-stage and gender for each algorithm. "
        "Coarser age bins keep cell counts usable."
    )
    ag_df = duckdb.sql(AGE_GENDER_QUERY).df()
    st.dataframe(ag_df, use_container_width=True, hide_index=True)
    ag_acc = ag_df.melt(
        id_vars=["age_group", "gender"],
        value_vars=["Algo 1 Acc", "Algo 2 Acc", "Algo 3 Acc"],
        var_name="Algorithm",
        value_name="Accuracy (%)",
    )
    ag_acc["Algorithm"] = ag_acc["Algorithm"].str.replace(" Acc", "", regex=False)
    fig_ag = px.bar(
        ag_acc,
        x="age_group",
        y="Accuracy (%)",
        color="Algorithm",
        facet_col="gender",
        barmode="group",
        title="Accuracy by Age Group and Gender",
        color_discrete_map=COLOR_MAP,
        labels={"age_group": "Age group"},
        category_orders={"age_group": ["00-17", "18-39", "40-64", "65+"]},
    )
    fig_ag.update_layout(legend_title_text="")
    st.plotly_chart(fig_ag, use_container_width=True)

    ag_sens = ag_df.melt(
        id_vars=["age_group", "gender"],
        value_vars=["Algo 1 Sens", "Algo 2 Sens", "Algo 3 Sens"],
        var_name="Algorithm",
        value_name="Sensitivity (%)",
    )
    ag_sens["Algorithm"] = ag_sens["Algorithm"].str.replace(" Sens", "", regex=False)
    fig_ag_s = px.bar(
        ag_sens,
        x="age_group",
        y="Sensitivity (%)",
        color="Algorithm",
        facet_col="gender",
        barmode="group",
        title="Sensitivity by Age Group and Gender",
        color_discrete_map=COLOR_MAP,
        labels={"age_group": "Age group"},
        category_orders={"age_group": ["00-17", "18-39", "40-64", "65+"]},
    )
    fig_ag_s.update_layout(legend_title_text="")
    st.plotly_chart(fig_ag_s, use_container_width=True)

    st.divider()
    st.subheader("Statistical Tests")
    st.markdown(
        "Ground truth is `radiologist_answer`. This file only has boolean P/N outputs "
        "(no probability scores), so DeLong is run on those 0/1 predictions. "
        "At a single operating point, AUC equals (sensitivity + specificity) / 2."
    )

    if filtered_df.empty:
        st.info("No rows in the current filter.")
    else:
        y_true = filtered_df["radiologist_answer"].eq("P").astype(int).to_numpy()
        scores = np.vstack(
            [
                filtered_df["algo1_answer"].eq("P").astype(int).to_numpy(),
                filtered_df["algo2_answer"].eq("P").astype(int).to_numpy(),
                filtered_df["algo3_answer"].eq("P").astype(int).to_numpy(),
            ]
        )
        aucs, cov = delong_aucs(y_true, scores)
        st.markdown("**DeLong test (AUC-ROC)**")
        if aucs is None:
            st.warning("DeLong needs both positive and negative ground-truth cases.")
        else:
            auc_table = pd.DataFrame(
                {"Algorithm": ALGO_COLS, "AUC": np.round(aucs, 4)}
            )
            chi2, p_omni, pair_df = delong_omnibus_and_pairs(aucs, cov, ALGO_COLS)
            st.dataframe(auc_table, use_container_width=True, hide_index=True)
            st.caption(
                f"Omnibus DeLong (3 AUCs equal): chi² = {chi2:.3f}, p = {fmt_p(p_omni)}"
                if pd.notna(p_omni)
                else "Omnibus DeLong could not be computed."
            )
            st.dataframe(pair_df, use_container_width=True, hide_index=True)

            fig_auc = px.bar(
                auc_table,
                x="Algorithm",
                y="AUC",
                color="Algorithm",
                color_discrete_map=COLOR_MAP,
                title="AUC at the boolean operating point",
                range_y=[0, 1],
            )
            fig_auc.update_layout(showlegend=False, xaxis_title="")
            st.plotly_chart(fig_auc, use_container_width=True)

        st.markdown("**Cochran's Q (matched binary outcomes)**")
        st.caption(
            "If Q is significant (p < 0.05), pairwise McNemar tests (exact, Bonferroni-corrected) "
            "show which algorithms differ. A only / B only are discordant counts for the pair."
        )
        for mode, label in [
            ("accuracy", "Accuracy (correct vs radiologist)"),
            ("sensitivity", "Sensitivity (among GT-positive scans)"),
            ("specificity", "Specificity (among GT-negative scans)"),
        ]:
            mat = algo_binary_matrix(filtered_df, mode)
            q, p_q, n_used = cochran_q(mat)
            st.markdown(f"*{label}* — n = {n_used:,}")
            if pd.isna(p_q):
                st.caption("Q undefined (algorithms never disagree on this subset).")
                continue
            st.caption(f"Cochran's Q = {q:.3f}, p = {fmt_p(p_q)}")
            if p_q < 0.05:
                mc_df, alpha = mcnemar_pairs(mat, ALGO_COLS)
                st.caption(f"Pairwise McNemar, Bonferroni α = {alpha:.4f}")
                st.dataframe(mc_df, use_container_width=True, hide_index=True)
            else:
                st.caption("No significant difference among the three algorithms; McNemar not run.")

    st.divider()
    st.subheader("Accuracy by Hospital Site")
    site_query = """
    SELECT
        site,
        COUNT(*) AS scans,
        ROUND(AVG(CASE WHEN algo1_answer = radiologist_answer THEN 1.0 ELSE 0.0 END) * 100, 1) AS "Algo 1",
        ROUND(AVG(CASE WHEN algo2_answer = radiologist_answer THEN 1.0 ELSE 0.0 END) * 100, 1) AS "Algo 2",
        ROUND(AVG(CASE WHEN algo3_answer = radiologist_answer THEN 1.0 ELSE 0.0 END) * 100, 1) AS "Algo 3"
    FROM filtered_df
    GROUP BY site
    ORDER BY site
    """
    site_acc = duckdb.sql(site_query).df()
    site_long = site_acc.melt(
        id_vars="site",
        value_vars=ALGO_COLS,
        var_name="Algorithm",
        value_name="Accuracy (%)",
    )
    fig_site = px.bar(
        site_long,
        x="site",
        y="Accuracy (%)",
        color="Algorithm",
        barmode="group",
        title="Accuracy (%) by Hospital Site",
        color_discrete_map=COLOR_MAP,
        labels={"site": "Hospital Site"},
    )
    fig_site.update_layout(legend_title_text="")
    st.plotly_chart(fig_site, use_container_width=True)
