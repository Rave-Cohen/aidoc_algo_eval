import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
ALGO_ANSWER_COLS = ["algo1_answer", "algo2_answer", "algo3_answer"]

AGE_GROUP_ORDER = ["00-17", "18-39", "40-64", "65+"]
PATIENT_CLASS_ORDER = ["ED", "IN"]
GENDER_COLOR_MAP = {"male": "#2a9d8f", "female": "#bc4749"}
DEPT_COLOR_MAP = {"ED": "#bc4749", "IN": "#2a9d8f"}
DEPT_ALGO_COLORS = {
    "ED": {"Algo 1": "#d62728", "Algo 2": "#1f77b4", "Algo 3": "#ff7f0e"},
    "IN": {"Algo 1": "#e07a7f", "Algo 2": "#6baed6", "Algo 3": "#fdb462"},
}
DEPT_LINE_DASH = {"ED": "solid", "IN": "dash"}


def streamlit_chart_bg() -> str:
    try:
        bg = st.get_option("theme.backgroundColor")
        if bg:
            return bg
    except Exception:
        pass
    try:
        return "#0e1117" if st.get_option("theme.base") == "dark" else "#ffffff"
    except Exception:
        return "#ffffff"


def streamlit_chart_fg() -> str:
    try:
        return "#fafafa" if st.get_option("theme.base") == "dark" else "#31333f"
    except Exception:
        return "#31333f"


def streamlit_chart_grid() -> str:
    try:
        return "#3d3d3d" if st.get_option("theme.base") == "dark" else "#d0d0d0"
    except Exception:
        return "#d0d0d0"


def label_bar_traces(
    fig: go.Figure,
    *,
    horizontal: bool = False,
    as_percent: bool = False,
    decimals: int = 1,
    textposition: str = "outside",
) -> go.Figure:
    for trace in fig.data:
        if trace.type != "bar":
            continue
        vals = trace.x if horizontal else trace.y
        if vals is None:
            continue
        labels = []
        for val in vals:
            if val is None or (isinstance(val, float) and np.isnan(val)):
                labels.append("")
            elif as_percent:
                labels.append(f"{float(val):.{decimals}f}%")
            elif decimals == 0 or float(val).is_integer():
                labels.append(f"{int(round(float(val))):,}")
            else:
                labels.append(f"{float(val):.{decimals}f}")
        trace.text = labels
        trace.textposition = textposition
    return fig

FINE_AGE_EDGES = [
    5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56,
    59, 62, 65, 68, 71, 74, 77, 80, 83, np.inf,
]
FINE_AGE_LABELS = [
    "05-08", "08-11", "11-14", "14-17", "17-20", "20-23", "23-26", "26-29",
    "29-32", "32-35", "35-38", "38-41", "41-44", "44-47", "47-50", "50-53",
    "53-56", "56-59", "59-62", "62-65", "65-68", "68-71", "71-74", "74-77",
    "77-80", "80-83", "83+",
]

MONTH_LABELS = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}
DOW_LABELS = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}


def assign_age_group_series(ages: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [ages < 18, ages < 40, ages < 65],
            ["00-17", "18-39", "40-64"],
            default="65+",
        ),
        index=ages.index,
    )


def filter_dataframe(
    df: pd.DataFrame,
    sites: list,
    patient_classes: list,
    genders: list,
    age_lo: int,
    age_hi: int,
) -> pd.DataFrame:
    return df[
        df["site"].isin(sites)
        & df["patient_class"].isin(patient_classes)
        & df["gender"].isin(genders)
        & (df["age"] >= age_lo)
        & (df["age"] <= age_hi)
    ].copy()


def group_distribution_df(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    out = (
        df.groupby(group_col, as_index=False)
        .agg(
            scans=("radiologist_answer", "count"),
            positives=("radiologist_answer", lambda s: (s == "P").sum()),
        )
        .assign(
            prevalence_pct=lambda d: (d["positives"] / d["scans"] * 100).round(1)
        )
        .sort_values(group_col)
    )
    return out


def group_accuracy_df(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    out = group_distribution_df(df, group_col)
    for name, col in zip(ALGO_COLS, ALGO_ANSWER_COLS):
        out[name] = (
            df.groupby(group_col)
            .apply(lambda g, c=col: (g[c] == g["radiologist_answer"]).mean() * 100)
            .round(1)
            .values
        )
    return out


def volume_sunburst_df(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["site", "patient_class", "gender"], as_index=False)
        .agg(
            scans=("radiologist_answer", "count"),
            positives=("radiologist_answer", lambda s: (s == "P").sum()),
        )
        .assign(
            prevalence_pct=lambda d: (d["positives"] / d["scans"] * 100).round(1)
        )
        .sort_values(["site", "patient_class", "gender"])
    )


def age_prevalence_sunburst_df(df: pd.DataFrame) -> pd.DataFrame:
    binned = df.assign(age_group=assign_age_group_series(df["age"]))
    out = (
        binned.groupby(["patient_class", "age_group"], as_index=False)
        .agg(
            scans=("radiologist_answer", "count"),
            positives=("radiologist_answer", lambda s: (s == "P").sum()),
        )
        .assign(
            prevalence_pct=lambda d: (d["positives"] / d["scans"] * 100).round(1)
        )
    )
    out["patient_class"] = pd.Categorical(
        out["patient_class"], categories=PATIENT_CLASS_ORDER, ordered=True
    )
    out["age_group"] = pd.Categorical(
        out["age_group"], categories=AGE_GROUP_ORDER, ordered=True
    )
    return out.sort_values(["patient_class", "age_group"])


def build_temporal_base(df: pd.DataFrame) -> pd.DataFrame:
    base = df[df["scan_timestamp"].notna()].copy()
    base["hour"] = base["scan_timestamp"].dt.hour.astype(int)
    base["dow"] = base["scan_timestamp"].dt.dayofweek.astype(int)
    base["month"] = base["scan_timestamp"].dt.month.astype(int)
    return base


def hourly_total_scans(temporal_base: pd.DataFrame) -> pd.DataFrame:
    return (
        temporal_base.groupby("hour", as_index=False)
        .agg(scans=("radiologist_answer", "count"))
        .sort_values("hour")
    )


def hourly_pos_rate(temporal_base: pd.DataFrame) -> pd.DataFrame:
    return (
        temporal_base.groupby("hour", as_index=False)
        .agg(pos_rate=("radiologist_answer", lambda s: round((s == "P").mean() * 100, 1)))
        .sort_values("hour")
    )


def monthly_pos_rate(temporal_base: pd.DataFrame) -> pd.DataFrame:
    out = (
        temporal_base.groupby("month", as_index=False)
        .agg(pos_rate=("radiologist_answer", lambda s: round((s == "P").mean() * 100, 1)))
        .sort_values("month")
    )
    out["month_name"] = out["month"].map(MONTH_LABELS)
    return out


def heatmap_data(temporal_base: pd.DataFrame) -> pd.DataFrame:
    return (
        temporal_base.groupby(["dow", "hour"], as_index=False)
        .agg(
            scans=("radiologist_answer", "count"),
            pos_rate=("radiologist_answer", lambda s: round((s == "P").mean() * 100, 1)),
        )
        .sort_values(["dow", "hour"])
    )


def count_with_times(df: pd.DataFrame) -> int:
    required = [
        "radiologist_sign_time",
        "algos_start_run",
        "algo1_finish_run",
        "algo2_finish_run",
        "algo3_finish_run",
    ]
    return int(df[required].notna().all(axis=1).sum())


def compute_duration_df(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "radiologist_sign_time",
        "algos_start_run",
        "algo1_finish_run",
        "algo2_finish_run",
        "algo3_finish_run",
    ]
    sub = df[df[required].notna().all(axis=1)].copy()
    duration_df = pd.DataFrame(
        {
            "Radiologist": (
                sub["radiologist_sign_time"] - sub["algos_start_run"]
            ).dt.total_seconds()
            / 60.0,
            "Algo 1": (
                sub["algo1_finish_run"] - sub["algos_start_run"]
            ).dt.total_seconds()
            / 60.0,
            "Algo 2": (
                sub["algo2_finish_run"] - sub["algos_start_run"]
            ).dt.total_seconds()
            / 60.0,
            "Algo 3": (
                sub["algo3_finish_run"] - sub["algos_start_run"]
            ).dt.total_seconds()
            / 60.0,
        }
    )
    valid = (duration_df >= 0).all(axis=1)
    return duration_df.loc[valid]


def hourly_scans_rad_by_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    mask = (
        df["scan_timestamp"].notna()
        & df["radiologist_sign_time"].notna()
        & df["algos_start_run"].notna()
    )
    sub = df.loc[mask].copy()
    sub["hour"] = sub["scan_timestamp"].dt.hour.astype(int)
    sub["rad_min"] = (
        sub["radiologist_sign_time"] - sub["algos_start_run"]
    ).dt.total_seconds() / 60.0
    sub = sub[sub["rad_min"] >= 0]
    return (
        sub.groupby(["hour", group_col], as_index=False)
        .agg(
            scans=("radiologist_answer", "count"),
            mean_rad_min=("rad_min", lambda s: round(s.mean(), 2)),
        )
        .sort_values(["hour", group_col])
    )


def hourly_scans_algo_by_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    required = [
        "scan_timestamp",
        "algos_start_run",
        "algo1_finish_run",
        "algo2_finish_run",
        "algo3_finish_run",
    ]
    sub = df[df[required].notna().all(axis=1)].copy()
    sub["hour"] = sub["scan_timestamp"].dt.hour.astype(int)
    for i, col in enumerate(ALGO_ANSWER_COLS, 1):
        finish_col = f"algo{i}_finish_run"
        sub[f"algo{i}_min"] = (
            sub[finish_col] - sub["algos_start_run"]
        ).dt.total_seconds() / 60.0
    valid = (sub[[f"algo{i}_min" for i in range(1, 4)]] >= 0).all(axis=1)
    sub = sub[valid]
    return (
        sub.groupby(["hour", group_col], as_index=False)
        .agg(
            scans=("radiologist_answer", "count"),
            **{
                f"mean_{name.lower().replace(' ', '')}_min": (
                    f"algo{i}_min",
                    lambda s, n=name: round(s.mean(), 2),
                )
                for i, name in enumerate(ALGO_COLS, 1)
            },
        )
        .sort_values(["hour", group_col])
    )


def compute_metrics_df(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, col in zip(ALGO_COLS, ALGO_ANSWER_COLS):
        tp = int(((df[col] == "P") & (df["radiologist_answer"] == "P")).sum())
        tn = int(((df[col] == "N") & (df["radiologist_answer"] == "N")).sum())
        fp = int(((df[col] == "P") & (df["radiologist_answer"] == "N")).sum())
        fn = int(((df[col] == "N") & (df["radiologist_answer"] == "P")).sum())
        n = tp + tn + fp + fn
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan
        npv = tn / (tn + fn) if (tn + fn) else np.nan
        miss_rate = fn / (fn + tp) if (fn + tp) else np.nan
        fall_out = fp / (fp + tn) if (fp + tn) else np.nan
        fdr = fp / (fp + tp) if (fp + tp) else np.nan
        rows.append(
            {
                "Algorithm": name,
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "Sensitivity (%)": round(tp / (tp + fn) * 100, 2) if (tp + fn) else np.nan,
                "Specificity (%)": round(tn / (tn + fp) * 100, 2) if (tn + fp) else np.nan,
                "Precision / PPV (%)": round(tp / (tp + fp) * 100, 2) if (tp + fp) else np.nan,
                "NPV (%)": round(npv * 100, 2) if pd.notna(npv) else np.nan,
                "Accuracy (%)": round((tp + tn) / n * 100, 2) if n else np.nan,
                "F1-Score": round(f1, 4) if pd.notna(f1) else np.nan,
                "Miss Rate / FNR (%)": round(miss_rate * 100, 2) if pd.notna(miss_rate) else np.nan,
                "Fall-Out / FPR (%)": round(fall_out * 100, 2) if pd.notna(fall_out) else np.nan,
                "FDR (%)": round(fdr * 100, 2) if pd.notna(fdr) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def compute_age_gender_fine_df(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["age"] >= 5].copy()
    binned = sub.assign(
        age_group=pd.cut(
            sub["age"],
            bins=FINE_AGE_EDGES,
            labels=FINE_AGE_LABELS,
            right=False,
        )
    )
    binned = binned[binned["age_group"].notna()].copy()
    rows = []
    for (age_group, gender), grp in binned.groupby(["age_group", "gender"], sort=False):
        scans = len(grp)
        row = {"age_group": str(age_group), "gender": gender, "scans": scans}
        gt_pos = grp[grp["radiologist_answer"] == "P"]
        gt_neg = grp[grp["radiologist_answer"] == "N"]
        for name, col in zip(ALGO_COLS, ALGO_ANSWER_COLS):
            row[f"{name} Sens"] = (
                round((gt_pos[col] == "P").mean() * 100, 1) if len(gt_pos) else np.nan
            )
            row[f"{name} Spec"] = (
                round((gt_neg[col] == "N").mean() * 100, 1) if len(gt_neg) else np.nan
            )
        rows.append(row)
    out = pd.DataFrame(rows)
    out["age_group"] = pd.Categorical(
        out["age_group"], categories=FINE_AGE_LABELS, ordered=True
    )
    return out.sort_values(["age_group", "gender"])


def compute_age_gender_df(df: pd.DataFrame) -> pd.DataFrame:
    binned = df.assign(age_group=assign_age_group_series(df["age"]))
    rows = []
    for (age_group, gender), grp in binned.groupby(["age_group", "gender"], sort=False):
        scans = len(grp)
        positives = int((grp["radiologist_answer"] == "P").sum())
        gt_pos = grp[grp["radiologist_answer"] == "P"]
        row = {
            "age_group": age_group,
            "gender": gender,
            "scans": scans,
            "positives": positives,
            "prevalence_pct": round(positives / scans * 100, 1),
        }
        for name, col in zip(ALGO_COLS, ALGO_ANSWER_COLS):
            row[f"{name} Acc"] = round((grp[col] == grp["radiologist_answer"]).mean() * 100, 1)
            row[f"{name} Sens"] = (
                round((gt_pos[col] == "P").mean() * 100, 1) if len(gt_pos) else np.nan
            )
        rows.append(row)
    out = pd.DataFrame(rows)
    out["age_group"] = pd.Categorical(
        out["age_group"], categories=AGE_GROUP_ORDER, ordered=True
    )
    return out.sort_values(["age_group", "gender"])


def site_accuracy_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.groupby("site", as_index=False).agg(
        scans=("radiologist_answer", "count")
    )
    for name, col in zip(ALGO_COLS, ALGO_ANSWER_COLS):
        out[name] = (
            df.groupby("site")
            .apply(lambda g, c=col: (g[c] == g["radiologist_answer"]).mean() * 100)
            .round(1)
            .values
        )
    return out.sort_values("site")


def compute_age_subgroup_df(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["age"] >= 5].copy()
    binned = sub.assign(
        age_group=pd.cut(
            sub["age"],
            bins=FINE_AGE_EDGES,
            labels=FINE_AGE_LABELS,
            right=False,
        )
    )
    binned = binned[binned["age_group"].notna()].copy()
    for i, col in enumerate(ALGO_ANSWER_COLS, 1):
        binned[f"a{i}_corr"] = (
            binned[col] == binned["radiologist_answer"]
        ).astype(int)
    out = (
        binned.groupby("age_group", as_index=False, observed=False)
        .agg(
            scans=("radiologist_answer", "count"),
            Radiologist=(
                "radiologist_answer",
                lambda s: round((s == "P").mean() * 100, 1),
            ),
            **{
                name: (f"a{i}_corr", lambda s: round(s.mean() * 100, 1))
                for i, name in enumerate(ALGO_COLS, 1)
            },
        )
        .sort_values("age_group")
    )
    return out


def prevalence_sunburst(
    df: pd.DataFrame,
    path: list[str],
    title: str,
    sort_slices: bool = True,
    *,
    compact: bool = False,
    show_title: bool = True,
) -> go.Figure:
    fig = px.sunburst(
        df,
        path=path,
        values="scans",
        title=title if show_title else None,
        color="prevalence_pct",
        color_continuous_scale="Tealrose",
        custom_data=["scans", "positives", "prevalence_pct"],
    )
    fig.update_traces(
        sort=sort_slices,
        textinfo="label+percent parent+value",
        insidetextorientation="radial",
        textfont=dict(size=14 if compact else 15),
        hovertemplate=(
            "<b>%{label}</b><br>Scans: %{customdata[0]:,}"
            "<br>Positives: %{customdata[1]:,}"
            "<br>Prevalence: %{customdata[2]}%<extra></extra>"
        ),
    )
    layout_kwargs = dict(
        height=580 if compact else 700,
        margin=dict(t=20 if compact and not show_title else 60, l=10, r=10, b=10),
        uniformtext_minsize=12,
        coloraxis_colorbar=dict(title="Prevalence %", thickness=12, len=0.75),
    )
    if show_title:
        layout_kwargs["title_font"] = dict(size=18)
    if not compact:
        layout_kwargs["width"] = 900
    fig.update_layout(**layout_kwargs)
    return fig


def age_distribution_by_gender_chart(df: pd.DataFrame, bin_width: int = 5) -> go.Figure:
    ages = df["age"].dropna()
    fig = go.Figure()
    if ages.empty:
        fig.update_layout(title="Age Distribution by Gender")
        return fig

    min_edge = int(np.floor(ages.min() / bin_width) * bin_width)
    max_edge = int(np.ceil(ages.max() / bin_width) * bin_width)
    bin_edges = np.arange(min_edge, max_edge + bin_width, bin_width)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    for gender in ["male", "female"]:
        counts, _ = np.histogram(
            df.loc[df["gender"] == gender, "age"].dropna(),
            bins=bin_edges,
        )
        color = GENDER_COLOR_MAP[gender]
        fig.add_trace(
            go.Bar(
                x=bin_centers,
                y=counts,
                name=gender,
                marker=dict(color=color, line=dict(width=0)),
                opacity=0.72,
                width=bin_width * 0.92,
                text=[str(c) if c else "" for c in counts],
                textposition="outside",
                textfont=dict(color=color, size=11, family="Arial Black, Arial, sans-serif"),
                hovertemplate=(
                    f"{gender}<br>Age %{{customdata}}<br>Count: %{{y:,}}<extra></extra>"
                ),
                customdata=[
                    f"{int(lo)}–{int(hi)}"
                    for lo, hi in zip(bin_edges[:-1], bin_edges[1:])
                ],
            )
        )

    fig.update_layout(
        title="Age Distribution by Gender",
        barmode="overlay",
        xaxis=dict(title="Age (years)", tickmode="linear", dtick=bin_width),
        yaxis=dict(title="Count"),
        legend_title_text="",
        bargap=0.05,
        uniformtext_minsize=9,
        margin=dict(t=70, r=20, b=50, l=50),
    )
    return fig


def tat_dot_plot(duration_df: pd.DataFrame) -> go.Figure:
    entity_order = ["Radiologist", "Algo 1", "Algo 2", "Algo 3"]
    fig = go.Figure()
    rng = np.random.default_rng(42)

    for idx, entity in enumerate(entity_order):
        values = duration_df[entity].dropna().to_numpy()
        mean_val = float(np.mean(values))
        x_jitter = idx + rng.uniform(-0.35, 0.35, len(values))
        color = COLOR_MAP[entity]

        fig.add_trace(
            go.Scatter(
                x=x_jitter,
                y=values,
                mode="markers",
                name=entity,
                marker=dict(color=color, size=4, opacity=0.2),
                hovertemplate=f"{entity}<br>TAT: %{{y:.1f}} min<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[idx - 0.38, idx + 0.38],
                y=[mean_val, mean_val],
                mode="lines",
                line=dict(color=color, width=2.5, dash="dash"),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[idx],
                y=[mean_val],
                mode="markers+text",
                marker=dict(symbol="diamond", size=11, color=color, line=dict(width=1, color="white")),
                text=[f"μ={mean_val:.1f}"],
                textposition="top center",
                textfont=dict(color=color, size=11),
                showlegend=False,
                hovertemplate=f"{entity}<br>Mean: {mean_val:.2f} min<extra></extra>",
            )
        )

    fig.update_layout(
        title="TAT Dot Plot (minutes)",
        xaxis=dict(
            tickvals=list(range(len(entity_order))),
            ticktext=entity_order,
            title="",
        ),
        yaxis_title="Minutes",
        showlegend=False,
        margin=dict(t=50, b=40, l=50, r=20),
    )
    return fig


def tat_win_count_chart(duration_df: pd.DataFrame) -> go.Figure:
    pairs = [
        ("Radiologist", "Algo 1"),
        ("Radiologist", "Algo 2"),
        ("Radiologist", "Algo 3"),
        ("Algo 1", "Algo 2"),
        ("Algo 1", "Algo 3"),
        ("Algo 2", "Algo 3"),
    ]
    fig = go.Figure()
    for name_a, name_b in pairs:
        a_wins = int((duration_df[name_a] < duration_df[name_b]).sum())
        b_wins = int((duration_df[name_a] > duration_df[name_b]).sum())
        total = len(duration_df)
        matchup = f"{name_a} vs {name_b}"
        fig.add_trace(
            go.Bar(
                y=[matchup], x=[a_wins / total * 100],
                orientation="h",
                marker_color=COLOR_MAP.get(name_a, "#636363"),
                text=[f"{name_a}: {a_wins:,} ({a_wins / total * 100:.1f}%)"],
                textposition="inside",
                showlegend=False,
                hovertemplate=f"{name_a} faster: {a_wins:,} ({a_wins / total * 100:.1f}%)<extra></extra>",
            )
        )
        fig.add_trace(
            go.Bar(
                y=[matchup], x=[b_wins / total * 100],
                orientation="h",
                marker_color=COLOR_MAP.get(name_b, "#636363"),
                text=[f"{name_b}: {b_wins:,} ({b_wins / total * 100:.1f}%)"],
                textposition="inside",
                showlegend=False,
                hovertemplate=f"{name_b} faster: {b_wins:,} ({b_wins / total * 100:.1f}%)<extra></extra>",
            )
        )
    fig.update_layout(
        title="Who Finishes First? (% of scans)",
        barmode="stack",
        xaxis_title="% of scans",
        yaxis_title="",
        height=350,
        margin=dict(l=150, t=50, b=40, r=20),
    )
    return fig


def tat_clinical_gap_chart(
    duration_df: pd.DataFrame, threshold: float = 0.5
) -> tuple[go.Figure, float, float]:
    diff = duration_df["Radiologist"] - duration_df["Algo 3"]
    mean_gap = float(diff.mean())
    median_gap = float(diff.median())
    rad_slower = int((diff > threshold).sum())
    algo3_slower = int((diff < -threshold).sum())
    within_threshold = len(diff) - rad_slower - algo3_slower

    clinical = pd.DataFrame([
        {
            "Category": f"Radiologist slower by >{threshold} min",
            "Count": rad_slower,
            "Pct": f"{rad_slower / len(diff) * 100:.1f}%",
        },
        {
            "Category": f"Algo 3 slower by >{threshold} min",
            "Count": algo3_slower,
            "Pct": f"{algo3_slower / len(diff) * 100:.1f}%",
        },
        {
            "Category": f"Within ±{threshold} min",
            "Count": within_threshold,
            "Pct": f"{within_threshold / len(diff) * 100:.1f}%",
        },
    ])
    clinical["Label"] = clinical.apply(
        lambda r: f"{int(r['Count']):,}<br>{r['Pct']}", axis=1
    )
    fig = px.bar(
        clinical,
        x="Category",
        y="Count",
        text="Label",
        title=f"Clinical Threshold: TAT Gap > {threshold} min (Radiologist vs Algo 3)",
        color="Category",
        color_discrete_map={
            f"Radiologist slower by >{threshold} min": COLOR_MAP["Radiologist"],
            f"Algo 3 slower by >{threshold} min": COLOR_MAP["Algo 3"],
            f"Within ±{threshold} min": "rgba(128,128,128,0.5)",
        },
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Scans")
    return fig, mean_gap, median_gap


def hourly_scans_rad_chart(
    df: pd.DataFrame,
    group_col: str,
    color_map: dict[str, str],
    title: str,
) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for group, color in color_map.items():
        sub = df[df[group_col] == group]
        fig.add_trace(
            go.Bar(
                x=sub["hour"], y=sub["scans"],
                name=f"{group} scans",
                marker_color=color, opacity=0.5,
                text=[f"{int(v):,}" for v in sub["scans"]],
                textposition="outside",
                textfont=dict(size=9),
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=sub["hour"], y=sub["mean_rad_min"],
                mode="lines+markers",
                name=f"{group} rad time",
                line=dict(color=color, dash="dot", width=2),
                marker=dict(color=color, size=6),
            ),
            secondary_y=True,
        )
    fig.update_layout(
        title=title,
        barmode="group",
        legend_title_text="",
        margin=dict(t=60, b=50, l=50, r=50),
    )
    fig.update_yaxes(title_text="Total Scans", secondary_y=False)
    fig.update_yaxes(title_text="Mean Rad Time (min)", secondary_y=True)
    fig.update_xaxes(title_text="Hour (0-23)")
    return fig


def hourly_scans_algo_chart(
    df: pd.DataFrame,
    group_col: str,
) -> go.Figure:
    algo_col_map = {
        "Algo 1": "mean_algo1_min",
        "Algo 2": "mean_algo2_min",
        "Algo 3": "mean_algo3_min",
    }
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for dept in PATIENT_CLASS_ORDER:
        dept_color = DEPT_COLOR_MAP[dept]
        sub = df[df[group_col] == dept]
        fig.add_trace(
            go.Bar(
                x=sub["hour"],
                y=sub["scans"],
                name=f"{dept} scans",
                marker_color=dept_color,
                opacity=0.35,
                text=[f"{int(v):,}" for v in sub["scans"]],
                textposition="outside",
                textfont=dict(size=9),
                legendgroup=f"{dept}-bars",
            ),
            secondary_y=False,
        )
        for algo, col in algo_col_map.items():
            algo_color = DEPT_ALGO_COLORS[dept][algo]
            fig.add_trace(
                go.Scatter(
                    x=sub["hour"],
                    y=sub[col],
                    mode="lines+markers",
                    name=f"{dept} — {algo}",
                    line=dict(
                        color=algo_color,
                        dash=DEPT_LINE_DASH[dept],
                        width=2.5 if dept == "ED" else 2,
                    ),
                    marker=dict(color=algo_color, size=5),
                    legendgroup=f"{dept}-{algo}",
                ),
                secondary_y=True,
            )
    fig.update_layout(
        title="Hourly Scan Volume & Algo Time — by Department",
        barmode="group",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            font=dict(size=10),
            tracegroupgap=4,
        ),
        legend_title_text="",
        margin=dict(t=70, b=50, l=50, r=160),
    )
    fig.update_yaxes(title_text="Total Scans", secondary_y=False)
    fig.update_yaxes(title_text="Mean Algo Time (min)", secondary_y=True)
    fig.update_xaxes(title_text="Hour (0-23)")
    return fig


def diagnostic_radar_chart(metrics_df: pd.DataFrame) -> go.Figure:
    radar_metrics = [
        "Sensitivity (%)",
        "Specificity (%)",
        "Precision / PPV (%)",
        "Accuracy (%)",
    ]
    fig = go.Figure()
    for _, row in metrics_df.iterrows():
        values = [row[m] for m in radar_metrics]
        fig.add_trace(
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
    bg = streamlit_chart_bg()
    fg = streamlit_chart_fg()
    grid = streamlit_chart_grid()
    fig.update_layout(
        title=dict(text="Diagnostic Profile", font=dict(color=fg)),
        paper_bgcolor=bg,
        plot_bgcolor=bg,
        font=dict(color=fg),
        polar=dict(
            bgcolor=bg,
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor=grid,
                linecolor=grid,
                tickfont=dict(color=fg, size=9),
            ),
            angularaxis=dict(
                gridcolor=grid,
                linecolor=grid,
                tickfont=dict(color=fg, size=9),
            ),
        ),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.08,
            font=dict(size=10, color=fg),
            bgcolor=bg,
        ),
        height=320,
        margin=dict(t=50, b=20, l=40, r=110),
    )
    return fig


def _cm_cell_text(label: str, count: int, total: int) -> str:
    pct = count / total * 100 if total else 0
    return f"{label}<br>{count:,}<br>({pct:.1f}%)"


def confusion_matrices_chart(metrics_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=list(metrics_df["Algorithm"]),
        horizontal_spacing=0.12,
    )
    for i, row in metrics_df.iterrows():
        tp, tn, fp, fn = int(row["TP"]), int(row["TN"]), int(row["FP"]), int(row["FN"])
        total = tp + tn + fp + fn
        cm = [[tn, fp], [fn, tp]]
        text = [
            [_cm_cell_text("TN", tn, total), _cm_cell_text("FP", fp, total)],
            [_cm_cell_text("FN", fn, total), _cm_cell_text("TP", tp, total)],
        ]
        fig.add_trace(
            go.Heatmap(
                z=cm,
                x=["N", "P"],
                y=["N", "P"],
                text=text,
                texttemplate="%{text}",
                colorscale=[[0, "#f0fdf4"], [0.5, "#86efac"], [1, "#15803d"]],
                showscale=(i == len(metrics_df) - 1),
                hovertemplate="%{text}<extra></extra>",
                xgap=2,
                ygap=2,
            ),
            row=1,
            col=i + 1,
        )

    for col in (1, 2, 3):
        fig.update_xaxes(
            title_text="Predicted" if col == 2 else "",
            title_standoff=8,
            row=1,
            col=col,
        )
        fig.update_yaxes(
            title_text="Actual" if col == 1 else "",
            title_standoff=8,
            showticklabels=(col == 1),
            row=1,
            col=col,
        )

    fig.update_layout(
        title="Confusion Matrices",
        height=380,
        margin=dict(t=60, b=55, l=70, r=50),
    )
    return fig


def f1_score_chart(metrics_df: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        metrics_df,
        x="Algorithm",
        y="F1-Score",
        color="Algorithm",
        color_discrete_map=COLOR_MAP,
        title="F1-Score per Algorithm",
        text=metrics_df["F1-Score"].map(lambda v: f"{v:.4f}"),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        showlegend=False,
        yaxis=dict(range=[0, 1], title="F1-Score"),
        xaxis_title="",
        height=350,
        margin=dict(t=50, b=40),
    )
    return fig


def npv_chart(metrics_df: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        metrics_df,
        x="Algorithm",
        y="NPV (%)",
        color="Algorithm",
        color_discrete_map=COLOR_MAP,
        title="NPV per Algorithm",
        text=metrics_df["NPV (%)"].map(lambda v: f"{v:.2f}%"),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        showlegend=False,
        yaxis=dict(range=[0, 100], title="NPV (%)"),
        xaxis_title="",
        height=350,
        margin=dict(t=50, b=40),
    )
    return fig


def error_rate_chart(metrics_df: pd.DataFrame) -> go.Figure:
    rate_cols = ["Miss Rate / FNR (%)", "Fall-Out / FPR (%)", "FDR (%)"]
    long = metrics_df.melt(
        id_vars="Algorithm",
        value_vars=rate_cols,
        var_name="Metric",
        value_name="Rate (%)",
    )
    fig = px.bar(
        long,
        x="Metric",
        y="Rate (%)",
        color="Algorithm",
        barmode="group",
        color_discrete_map=COLOR_MAP,
        title="Error Rate Metrics (lower = better)",
        text="Rate (%)",
    )
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig.update_layout(
        height=400,
        xaxis_title="",
        yaxis_title="Rate (%)",
        legend_title_text="",
        margin=dict(t=60, b=40),
    )
    return fig


def overall_sensitivity_pct(df: pd.DataFrame, gender: str, algo_col: str) -> float:
    sub = df[(df["gender"] == gender) & (df["age"] >= 5)]
    gt_pos = sub[sub["radiologist_answer"] == "P"]
    if gt_pos.empty:
        return np.nan
    return round((gt_pos[algo_col] == "P").mean() * 100, 1)


def age_gender_sensitivity_line_chart(
    ag_df: pd.DataFrame, source_df: pd.DataFrame
) -> go.Figure:
    x_categories = [str(label) for label in FINE_AGE_LABELS]
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Male", "Female"],
        shared_yaxes=True,
    )
    for col_idx, gender in enumerate(["male", "female"], start=1):
        sub = ag_df[ag_df["gender"] == gender].sort_values("age_group")
        for algo in ALGO_COLS:
            col_name = f"{algo} Sens"
            algo_col = ALGO_ANSWER_COLS[ALGO_COLS.index(algo)]
            color = COLOR_MAP[algo]
            mean_sens = overall_sensitivity_pct(source_df, gender, algo_col)

            fig.add_trace(
                go.Scatter(
                    x=sub["age_group"].astype(str),
                    y=sub[col_name],
                    mode="lines",
                    name=algo,
                    line=dict(color=color, width=2.5, shape="linear"),
                    legendgroup=algo,
                    showlegend=(col_idx == 1),
                    connectgaps=False,
                    hovertemplate=(
                        f"{algo}<br>Age %{{x}}<br>Sensitivity: %{{y:.1f}}%<extra></extra>"
                    ),
                ),
                row=1,
                col=col_idx,
            )
            if pd.notna(mean_sens):
                fig.add_trace(
                    go.Scatter(
                        x=x_categories,
                        y=[mean_sens] * len(x_categories),
                        mode="lines",
                        name=f"{algo} mean ({mean_sens:.1f}%)",
                        line=dict(color=color, width=2, dash="dash"),
                        legendgroup=algo,
                        showlegend=(col_idx == 1),
                        hovertemplate=f"{algo} overall mean<br>Sensitivity: {mean_sens:.1f}%<extra></extra>",
                    ),
                    row=1,
                    col=col_idx,
                )
    fig.update_layout(
        title="Sensitivity by Age Group and Gender",
        height=420,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.14,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(t=110, b=80, l=50, r=20),
    )
    fig.update_yaxes(title_text="Sensitivity (%)", range=[0, 100], row=1, col=1)
    fig.update_yaxes(range=[0, 100], row=1, col=2)
    for col_idx in (1, 2):
        fig.update_xaxes(
            title_text="Age group (years)",
            type="category",
            categoryorder="array",
            categoryarray=x_categories,
            tickangle=-45,
            row=1,
            col=col_idx,
        )
    return fig


def age_gender_metric_chart(
    ag_df: pd.DataFrame,
    metric_suffix: str,
    y_label: str,
    title: str,
) -> go.Figure:
    x_categories = [str(label) for label in FINE_AGE_LABELS]
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Male", "Female"],
        shared_yaxes=True,
    )
    for col_idx, gender in enumerate(["male", "female"], start=1):
        sub = ag_df[ag_df["gender"] == gender].sort_values("age_group")
        for algo in ALGO_COLS:
            col_name = f"{algo} {metric_suffix}"
            fig.add_trace(
                go.Scatter(
                    x=sub["age_group"].astype(str),
                    y=sub[col_name],
                    mode="lines+markers",
                    name=algo,
                    line=dict(color=COLOR_MAP[algo], width=2.5),
                    marker=dict(color=COLOR_MAP[algo], size=7),
                    legendgroup=algo,
                    showlegend=(col_idx == 1),
                    connectgaps=False,
                ),
                row=1,
                col=col_idx,
            )
    fig.update_layout(
        title=title,
        height=420,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.12,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(t=100, b=80, l=50, r=20),
    )
    fig.update_yaxes(title_text=y_label, range=[0, 100], row=1, col=1)
    fig.update_yaxes(range=[0, 100], row=1, col=2)
    for col_idx in (1, 2):
        fig.update_xaxes(
            title_text="Age group (years)",
            type="category",
            categoryorder="array",
            categoryarray=x_categories,
            tickangle=-45,
            row=1,
            col=col_idx,
        )
    return fig


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
        text=long["Accuracy (%)"].map(lambda v: f"{v:.1f}%"),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(legend_title_text="", yaxis_title="")
    return fig


def age_subgroup_chart(age_df: pd.DataFrame) -> go.Figure:
    algo_legend_names = {
        "Radiologist": "Radiologist (prevalence)",
        "Algo 1": "Algo 1 (accuracy)",
        "Algo 2": "Algo 2 (accuracy)",
        "Algo 3": "Algo 3 (accuracy)",
    }
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=age_df["age_group"],
            y=age_df["scans"],
            name="Scan count",
            marker_color="rgba(128, 128, 128, 0.25)",
            text=[f"{int(v):,}" for v in age_df["scans"]],
            textposition="outside",
            textfont=dict(size=9, color="#666"),
            hovertemplate="Age %{x}<br>Scans: %{y:,}<extra></extra>",
        ),
        secondary_y=True,
    )
    for col in AGE_LINE_COLS:
        legend_name = algo_legend_names[col]
        fig.add_trace(
            go.Scatter(
                x=age_df["age_group"],
                y=age_df[col],
                name=legend_name,
                mode="lines+markers",
                line=dict(color=COLOR_MAP[col]),
                marker=dict(color=COLOR_MAP[col]),
                hovertemplate=f"{legend_name}<br>%{{x}}: %{{y:.1f}}%<extra></extra>",
            ),
            secondary_y=False,
        )
    fig.update_layout(
        title="Prevalence & Algorithm Accuracy by Age Group",
        legend_title_text="",
        xaxis=dict(
            type="category",
            title="Age Group (years)",
            categoryorder="array",
            categoryarray=[str(label) for label in FINE_AGE_LABELS],
            tickangle=-45,
        ),
        barmode="overlay",
        margin=dict(b=80),
    )
    fig.update_yaxes(title_text="Rate (%)", secondary_y=False, range=[0, 100])
    fig.update_yaxes(title_text="Scan count", secondary_y=True, showgrid=False)
    return fig


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
genders = st.sidebar.multiselect(
    "Select Gender",
    options=sorted(df["gender"].unique()),
    default=sorted(df["gender"].unique()),
)

age_max = int(df["age"].max())
st.sidebar.markdown("**Age range (years)**")
age_col1, age_col2 = st.sidebar.columns(2)
age_lo = age_col1.number_input("From", min_value=5, max_value=age_max, value=5, step=1)
age_hi = age_col2.number_input("To", min_value=0, max_value=age_max, value=age_max, step=1)
if age_lo > age_hi:
    age_lo, age_hi = age_hi, age_lo

filtered_df = filter_dataframe(df, sites, patient_classes, genders, age_lo, age_hi)

total_scans = len(filtered_df)
positives = int((filtered_df["radiologist_answer"] == "P").sum()) if total_scans else 0
prevalence = (positives / total_scans * 100) if total_scans else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Scans", f"{total_scans:,}")
col2.metric("Positive Cases", f"{positives:,}")
col3.metric("Prevalence Rate", f"{prevalence:.1f}%")
col4.metric("Active Sites", f"{filtered_df['site'].nunique() if total_scans else 0}")

st.divider()

tab1, tab2, tab3 = st.tabs(
    ["Overview", "Timelines", "Algo Performance"]
)

# --- TAB 1: OVERVIEW ---
with tab1:
    st.info(
        "**How to use this dashboard:**\n\n"
        "Use the **sidebar filters** (left) to narrow data by site, department, gender, or age.\n\n"
        "**Hover** over any chart for details.\n\n"
        "**Click** legend items to toggle series on/off.\n\n"
        "Click the 🏠 **house icon** on a chart toolbar to reset zoom and view.",
        icon="💡",
    )
    vol_df = volume_sunburst_df(filtered_df)
    age_vol_df = age_prevalence_sunburst_df(filtered_df)

    pie_col1, pie_col2 = st.columns(2, gap="medium")
    with pie_col1:
        st.markdown("#### Scan Volume: Site → Department → Gender")
        st.plotly_chart(
            prevalence_sunburst(
                vol_df,
                ["site", "patient_class", "gender"],
                "Scan Volume: Site → Department → Gender",
                compact=True,
                show_title=False,
            ),
            use_container_width=True,
        )
    with pie_col2:
        st.markdown("#### Prevalence by Age: Department → Age Group")
        st.plotly_chart(
            prevalence_sunburst(
                age_vol_df,
                ["patient_class", "age_group"],
                "Prevalence by Age: Department → Age Group",
                sort_slices=False,
                compact=True,
                show_title=False,
            ),
            use_container_width=True,
        )

    st.markdown(
        "**Notes**\n\n"
        "- Prevalence seems to be consistently higher in the **IN** department.\n\n"
        "- Prevalence seems to be higher in **younger age groups** — e.g., in the **IN** department, ages 18–39 show **14.1%** prevalence vs. **11.6%** for ages 40–64.\n\n"
        "- There is no notable difference in distributions between the **two sites** provided.\n\n"
        "- **Males** are much more common in the **ED** department."
    )

    st.divider()
    st.subheader("Gender")
    gender_df = group_distribution_df(filtered_df, "gender")

    g_cols = st.columns(len(gender_df) if len(gender_df) else 1)
    for i, row in gender_df.iterrows():
        g_cols[i].metric(
            f"{row['gender']} scans",
            f"{int(row['scans']):,}",
            f"{row['prevalence_pct']}% prevalence",
        )

    st.dataframe(gender_df, use_container_width=True, hide_index=True)

    fig_age_dist = age_distribution_by_gender_chart(filtered_df)
    st.plotly_chart(fig_age_dist, use_container_width=True)
    st.markdown(
        "**Notes:** This can provide insight about which statistical results are more significant."
    )

    st.divider()
    st.subheader("Commonness")
    st.markdown("Hourly, monthly, and weekly patterns of scan volume and positive rates.")

    temporal_base = build_temporal_base(filtered_df)

    st.markdown("##### Hourly Total Scan Volume")
    hourly_total_overall = hourly_total_scans(temporal_base)

    fig_tvm = go.Figure(go.Bar(
        x=hourly_total_overall["hour"], y=hourly_total_overall["scans"],
        marker_color="#6a994e", name="All scans",
        text=[f"{int(v):,}" for v in hourly_total_overall["scans"]],
        textposition="outside",
        textfont=dict(size=9),
    ))
    mean_val = hourly_total_overall["scans"].mean()
    fig_tvm.add_hline(
        y=mean_val, line_dash="dash", line_color="#bc4749",
        annotation_text=f"Mean: {mean_val:.0f}",
    )
    fig_tvm.update_layout(
        title="Hourly Total Scan Volume (All Scans)",
        xaxis_title="Hour (0-23)", yaxis_title="Total Scans",
        margin=dict(t=50),
    )
    st.plotly_chart(fig_tvm, use_container_width=True)
    st.markdown(
        "**Note:** Utilization of scanning seems pretty robust and consistent (on both sites)."
    )

    st.markdown("##### Hourly Positive Rate")
    hourly_pos_overall = hourly_pos_rate(temporal_base)

    fig_prm = go.Figure(go.Scatter(
        x=hourly_pos_overall["hour"], y=hourly_pos_overall["pos_rate"],
        mode="lines+markers", marker_color="#6a994e", name="All scans",
    ))
    mean_pr = hourly_pos_overall["pos_rate"].mean()
    fig_prm.add_hline(
        y=mean_pr, line_dash="dash", line_color="#bc4749",
        annotation_text=f"Mean: {mean_pr:.1f}%",
    )
    fig_prm.update_layout(
        title="Hourly Positive Rate (All Scans)",
        xaxis_title="Hour (0-23)", yaxis_title="Positive Rate (%)",
        margin=dict(t=50),
    )
    st.plotly_chart(fig_prm, use_container_width=True)
    st.markdown(
        "**Note:** \n\n"
        "It can be interesting to hypothesize and do more research on peak times - "
        "e.g. night hours **00:00–02:00** might be correlated with some type of "
        "ICH for a reason."
    )

    st.markdown("##### Monthly Positive Rate")
    monthly_pos_overall = monthly_pos_rate(temporal_base)

    fig_mpm = go.Figure(go.Scatter(
        x=monthly_pos_overall["month_name"], y=monthly_pos_overall["pos_rate"],
        mode="lines+markers", marker_color="#6a994e", name="All scans",
    ))
    mean_mp = monthly_pos_overall["pos_rate"].mean()
    fig_mpm.add_hline(
        y=mean_mp, line_dash="dash", line_color="#bc4749",
        annotation_text=f"Mean: {mean_mp:.1f}%",
    )
    fig_mpm.update_layout(
        title="Monthly Positive Rate (All Scans)",
        xaxis_title="Month", yaxis_title="Positive Rate (%)",
        margin=dict(t=50),
    )
    st.plotly_chart(fig_mpm, use_container_width=True)
    st.markdown(
        "**Note:** Prevalence seems higher in **summer months** — might be worth investigating further. "
        "If a certain ICH state correlates with summer, that could inform operational decisions or "
        "help adjust algorithm weights."
    )

    st.markdown("##### Positive Rate Heatmap: Day-of-Week × Hour")
    heatmap_df = heatmap_data(temporal_base)
    dow_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    heatmap_df["day_name"] = heatmap_df["dow"].map(DOW_LABELS)
    pivot = heatmap_df.pivot(index="day_name", columns="hour", values="pos_rate").reindex(dow_order)

    hottest_idx = heatmap_df.loc[heatmap_df["pos_rate"].idxmax()]
    hottest_day = DOW_LABELS[int(hottest_idx["dow"])]
    hottest_hour = int(hottest_idx["hour"])
    hottest_rate = hottest_idx["pos_rate"]

    coldest_idx = heatmap_df.loc[heatmap_df["pos_rate"].idxmin()]
    coldest_day = DOW_LABELS[int(coldest_idx["dow"])]
    coldest_hour = int(coldest_idx["hour"])
    coldest_rate = coldest_idx["pos_rate"]

    fig_heat = px.imshow(
        pivot, aspect="auto",
        title="Positive Rate (%) — Day × Hour",
        labels={"x": "Hour", "y": "Day", "color": "Pos Rate (%)"},
        color_continuous_scale="RdYlGn_r",
    )
    fig_heat.update_layout(
        height=400,
        margin=dict(t=60, b=40, l=60, r=20),
        coloraxis_colorbar=dict(title="Rate %", thickness=12),
    )
    fig_heat.add_annotation(
        x=hottest_hour, y=hottest_day,
        text="🔥", showarrow=False, font=dict(size=18),
    )
    fig_heat.add_annotation(
        x=coldest_hour, y=coldest_day,
        text="❄️", showarrow=False, font=dict(size=18),
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    st.markdown(
        f"**🔥 Hottest:** {hottest_day} at {hottest_hour}:00 ({hottest_rate}% positive rate)  \n"
        f"**❄️ Coldest:** {coldest_day} at {coldest_hour}:00 ({coldest_rate}% positive rate)"
    )

# --- TAB 2: TIMELINES ---
with tab2:
    st.subheader("Processing Timelines & Turnaround Time (TAT)")
    st.markdown(
        "Completion time in minutes from `algos_start_run`.\n\n"
        "Timestamps are stored to **millisecond** precision and TAT is computed from the full datetime difference.\n\n"
        "Rows with missing timestamps or negative durations are excluded from this tab."
    )

    duration_df = compute_duration_df(filtered_df)

    n_with_times = count_with_times(filtered_df)
    dropped = int(n_with_times) - len(duration_df)
    if dropped:
        st.caption(
            f"Excluded {dropped:,} rows with negative TAT (finish before start). "
            "In the full dataset this is usually a small number of cases where Algo 1 "
            "finishes a few seconds before its recorded start time, or the radiologist "
            "signed slightly before `algos_start_run`."
        )

    if duration_df.empty:
        st.info("No valid TAT rows after filtering missing or negative durations.")
    else:
        tat_long = duration_df.melt(var_name="Entity", value_name="Minutes")
        tat_means = duration_df.mean()
        fig_box = px.box(
            tat_long,
            x="Entity",
            y="Minutes",
            color="Entity",
            color_discrete_map=COLOR_MAP,
            title="TAT Distribution (minutes)",
            points=False,
        )
        for trace in fig_box.data:
            mean_val = tat_means[trace.name]
            trace.boxmean = True
            trace.hovertemplate = (
                f"<b>{trace.name}</b><br>"
                f"Mean: {mean_val:.2f} min<br>"
                "Median: %{median:.2f} min<br>"
                "Q1: %{q1:.2f} min<br>"
                "Q3: %{q3:.2f} min<br>"
                "<extra></extra>"
            )
        fig_box.update_layout(showlegend=False, xaxis_title="")
        st.plotly_chart(fig_box, use_container_width=True)

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

        st.plotly_chart(tat_dot_plot(duration_df), use_container_width=True)

        st.divider()
        st.markdown("##### Who Finishes First?")
        st.plotly_chart(tat_win_count_chart(duration_df), use_container_width=True)

        st.markdown("##### TAT Gap: Radiologist vs Algo 3")
        fig_clin, mean_gap, median_gap = tat_clinical_gap_chart(duration_df)
        gap_col1, gap_col2 = st.columns(2)
        gap_col1.metric("Mean gap (Rad − Algo 3)", f"{mean_gap:.2f} min")
        gap_col2.metric("Median gap", f"{median_gap:.2f} min")
        st.plotly_chart(fig_clin, use_container_width=True)
        st.markdown(
            "**Note:** It seems that most of the time the TAT gaps between Algo 3 and the "
            "Radiologist are minor."
        )

        st.divider()
        st.subheader("Scan Volume & Radiologist Time per Hour")
        hourly_gender_tat = hourly_scans_rad_by_group(filtered_df, "gender")
        hourly_dept_tat = hourly_scans_rad_by_group(filtered_df, "patient_class")

        hour_col1, hour_col2 = st.columns(2, gap="medium")
        with hour_col1:
            st.markdown("##### By Gender")
            st.plotly_chart(
                hourly_scans_rad_chart(
                    hourly_gender_tat,
                    "gender",
                    GENDER_COLOR_MAP,
                    "Hourly Scan Volume & Radiologist Time — by Gender",
                ),
                use_container_width=True,
            )
            st.markdown(
                "**Note:** Female radiologist verdicts take longer than male radiologist verdicts."
            )
        with hour_col2:
            st.markdown("##### By Department")
            st.plotly_chart(
                hourly_scans_rad_chart(
                    hourly_dept_tat,
                    "patient_class",
                    DEPT_COLOR_MAP,
                    "Hourly Scan Volume & Radiologist Time — by Department",
                ),
                use_container_width=True,
            )
            st.markdown(
                "**Note:** IN radiologist verdicts take around ~1 minute longer than ED — "
                "ED workflows likely prioritize faster turnaround given the urgency of "
                "acute triage decisions."
            )

        st.divider()
        st.subheader("Algo Time by Department")
        st.caption(
            "Bars show hourly scan volume; lines show mean algo completion time. "
            "ED algos use solid lines with standard hues; IN algos use dashed lines with lighter shades."
        )
        hourly_dept_algo = hourly_scans_algo_by_group(filtered_df, "patient_class")
        st.plotly_chart(
            hourly_scans_algo_chart(hourly_dept_algo, "patient_class"),
            use_container_width=True,
        )

# --- TAB 3: ALGO PERFORMANCE ---
with tab3:
    st.subheader("Core Clinical Metrics")
    st.markdown(
        "Confusion-matrix counts plus sensitivity, specificity, precision, and accuracy "
        "(radiologist as ground truth)."
    )

    metrics_df = compute_metrics_df(filtered_df)

    cm_col, radar_col = st.columns([1.2, 1], gap="medium")
    with cm_col:
        st.plotly_chart(
            confusion_matrices_chart(metrics_df),
            use_container_width=True,
        )
    with radar_col:
        st.plotly_chart(
            diagnostic_radar_chart(metrics_df),
            use_container_width=True,
        )

    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    f1_col, npv_col = st.columns(2, gap="medium")
    with f1_col:
        st.plotly_chart(f1_score_chart(metrics_df), use_container_width=True)
    with npv_col:
        st.plotly_chart(npv_chart(metrics_df), use_container_width=True)

    st.plotly_chart(error_rate_chart(metrics_df), use_container_width=True)

    st.divider()
    st.subheader("Age Subgroups")
    st.markdown(
        "Green line: **Radiologist (prevalence)** — positive rate per 3-year bin. "
        "Algo lines: **accuracy** vs. radiologist ground truth. "
        "Grey bars: scan count (right axis)."
    )
    age_df = compute_age_subgroup_df(filtered_df)
    st.plotly_chart(age_subgroup_chart(age_df), use_container_width=True)
    st.dataframe(age_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Age × Gender Sensitivity Performance")
    st.markdown(
        "Sensitivity by 3-year age bin and gender — share of GT-positive scans "
        "correctly flagged by each algorithm."
    )
    ag_fine_df = compute_age_gender_fine_df(filtered_df)
    st.dataframe(ag_fine_df, use_container_width=True, hide_index=True)
    st.plotly_chart(
        age_gender_sensitivity_line_chart(ag_fine_df, filtered_df),
        use_container_width=True,
    )

    st.subheader("Age × Gender Specificity Performance")
    st.markdown(
        "Specificity by 3-year age bin and gender — share of GT-negative scans "
        "correctly cleared by each algorithm."
    )
    st.plotly_chart(
        age_gender_metric_chart(
            ag_fine_df, "Spec", "Specificity (%)", "Specificity by Age Group and Gender"
        ),
        use_container_width=True,
    )

    st.divider()
    st.subheader("Accuracy by Hospital Site")
    site_acc = site_accuracy_df(filtered_df)
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
        text=site_long["Accuracy (%)"].map(lambda v: f"{v:.1f}%"),
    )
    fig_site.update_traces(textposition="outside")
    fig_site.update_layout(legend_title_text="")
    st.plotly_chart(fig_site, use_container_width=True)
