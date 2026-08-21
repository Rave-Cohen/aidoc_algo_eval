import duckdb
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


def group_distribution_query(group_col: str) -> str:
    return f"""
    SELECT
        {group_col},
        COUNT(*) AS scans,
        SUM(CASE WHEN radiologist_answer = 'P' THEN 1 ELSE 0 END) AS positives,
        ROUND(
            SUM(CASE WHEN radiologist_answer = 'P' THEN 1.0 ELSE 0.0 END)
            / COUNT(*) * 100, 1
        ) AS prevalence_pct
    FROM filtered_df
    GROUP BY {group_col}
    ORDER BY {group_col}
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


AGE_GROUP_BIN = """
    CASE
        WHEN age < 18 THEN '00-17'
        WHEN age < 40 THEN '18-39'
        WHEN age < 65 THEN '40-64'
        ELSE '65+'
    END AS age_group
"""
AGE_GROUP_ORDER = ["00-17", "18-39", "40-64", "65+"]
PATIENT_CLASS_ORDER = ["ED", "IN"]
GENDER_COLOR_MAP = {"male": "#2a9d8f", "female": "#bc4749"}
DEPT_COLOR_MAP = {"ED": "#bc4749", "IN": "#2a9d8f"}


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
        col_a, col_b = name_a, name_b
        a_wins = int((duration_df[col_a] < duration_df[col_b]).sum())
        b_wins = int((duration_df[col_a] > duration_df[col_b]).sum())
        total = len(duration_df)
        matchup = f"{name_a} vs {name_b}"
        fig.add_trace(
            go.Bar(
                y=[matchup], x=[a_wins / total * 100],
                orientation="h",
                marker_color=COLOR_MAP.get(name_a, "#636363"),
                text=[f"{name_a}: {a_wins:,}"], textposition="inside",
                showlegend=False,
                hovertemplate=f"{name_a} faster: {a_wins:,} ({a_wins / total * 100:.1f}%)<extra></extra>",
            )
        )
        fig.add_trace(
            go.Bar(
                y=[matchup], x=[b_wins / total * 100],
                orientation="h",
                marker_color=COLOR_MAP.get(name_b, "#636363"),
                text=[f"{name_b}: {b_wins:,}"], textposition="inside",
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
    fig = px.bar(
        clinical,
        x="Category",
        y="Count",
        text="Pct",
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
age_lo = age_col1.number_input("From", min_value=0, max_value=age_max, value=0, step=1)
age_hi = age_col2.number_input("To", min_value=0, max_value=age_max, value=age_max, step=1)
if age_lo > age_hi:
    age_lo, age_hi = age_hi, age_lo

filtered_df = duckdb.sql(
    """
    SELECT * FROM df
    WHERE site IN list_transform(?, x -> x)
      AND patient_class IN list_transform(?, x -> x)
      AND gender IN list_transform(?, x -> x)
      AND age >= ?
      AND age <= ?
    """,
    params=[sites, patient_classes, genders, age_lo, age_hi],
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

tab1, tab2, tab3, tab4 = st.tabs(
    ["Overview", "Timelines", "Metrics", "Algo Performance"]
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
    vol_query = """
    SELECT site, patient_class, gender,
           COUNT(*) AS scans,
           SUM(CASE WHEN radiologist_answer = 'P' THEN 1 ELSE 0 END) AS positives,
           ROUND(SUM(CASE WHEN radiologist_answer = 'P' THEN 1.0 ELSE 0.0 END)
                 / COUNT(*) * 100, 1) AS prevalence_pct
    FROM filtered_df
    GROUP BY site, patient_class, gender
    ORDER BY site, patient_class, gender
    """
    vol_df = duckdb.sql(vol_query).df()

    age_vol_query = f"""
    WITH binned AS (
        SELECT *, {AGE_GROUP_BIN}
        FROM filtered_df
    )
    SELECT patient_class, age_group,
           COUNT(*) AS scans,
           SUM(CASE WHEN radiologist_answer = 'P' THEN 1 ELSE 0 END) AS positives,
           ROUND(SUM(CASE WHEN radiologist_answer = 'P' THEN 1.0 ELSE 0.0 END)
                 / COUNT(*) * 100, 1) AS prevalence_pct
    FROM binned
    GROUP BY patient_class, age_group
    ORDER BY patient_class, age_group
    """
    age_vol_df = duckdb.sql(age_vol_query).df()
    age_vol_df["patient_class"] = pd.Categorical(
        age_vol_df["patient_class"], categories=PATIENT_CLASS_ORDER, ordered=True
    )
    age_vol_df["age_group"] = pd.Categorical(
        age_vol_df["age_group"], categories=AGE_GROUP_ORDER, ordered=True
    )
    age_vol_df = age_vol_df.sort_values(["patient_class", "age_group"])

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
        "- Prevalence seems to be higher in **younger age groups**.\n\n"
        "- There is no notable difference in distributions between the **two sites** provided.\n\n"
        "- **Males** are much more common in the **ED** department."
    )

    st.divider()
    st.subheader("Gender")
    gender_df = duckdb.sql(group_distribution_query("gender")).df()

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

    # ---- temporal base query ----
    temporal_base = duckdb.sql("""
    SELECT *,
        EXTRACT(HOUR FROM scan_timestamp) AS hour,
        EXTRACT(DOW FROM scan_timestamp) AS dow,
        EXTRACT(MONTH FROM scan_timestamp) AS month
    FROM filtered_df
    WHERE scan_timestamp IS NOT NULL
    """).df()

    MONTH_LABELS = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
    DOW_LABELS = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}

    st.markdown("##### Hourly Total Scan Volume")
    hourly_total_overall = duckdb.sql("""
    SELECT CAST(hour AS INTEGER) AS hour,
           COUNT(*) AS scans
    FROM temporal_base
    GROUP BY hour ORDER BY hour
    """).df()

    fig_tvm = go.Figure(go.Bar(
        x=hourly_total_overall["hour"], y=hourly_total_overall["scans"],
        marker_color="#6a994e", name="All scans",
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
    hourly_pos_overall = duckdb.sql("""
    SELECT CAST(hour AS INTEGER) AS hour,
           ROUND(SUM(CASE WHEN radiologist_answer = 'P' THEN 1.0 ELSE 0.0 END) / COUNT(*) * 100, 1) AS pos_rate
    FROM temporal_base
    GROUP BY hour ORDER BY hour
    """).df()

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
    monthly_pos_overall = duckdb.sql("""
    SELECT CAST(month AS INTEGER) AS month,
           ROUND(SUM(CASE WHEN radiologist_answer = 'P' THEN 1.0 ELSE 0.0 END) / COUNT(*) * 100, 1) AS pos_rate
    FROM temporal_base
    GROUP BY month ORDER BY month
    """).df()
    monthly_pos_overall["month_name"] = monthly_pos_overall["month"].map(MONTH_LABELS)

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

    # ---- Heatmap: Hour × Day-of-Week ----
    st.markdown("##### Positive Rate Heatmap: Day-of-Week × Hour")
    heatmap_df = duckdb.sql("""
    SELECT CAST(dow AS INTEGER) AS dow, CAST(hour AS INTEGER) AS hour,
           COUNT(*) AS scans,
           ROUND(SUM(CASE WHEN radiologist_answer = 'P' THEN 1.0 ELSE 0.0 END) / COUNT(*) * 100, 1) AS pos_rate
    FROM temporal_base
    GROUP BY dow, hour
    ORDER BY dow, hour
    """).df()
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
        st.caption(
            f"Excluded {dropped:,} rows with negative TAT (finish before start). "
            "In the full dataset this is usually a small number of cases where Algo 1 "
            "finishes a few seconds before its recorded start time, or the radiologist "
            "signed slightly before `algos_start_run`."
        )

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
        hourly_gender_tat = duckdb.sql("""
        SELECT
            CAST(EXTRACT(HOUR FROM scan_timestamp) AS INTEGER) AS hour,
            gender,
            COUNT(*) AS scans,
            ROUND(
                AVG(EPOCH(radiologist_sign_time - algos_start_run) / 60.0), 2
            ) AS mean_rad_min
        FROM filtered_df
        WHERE scan_timestamp IS NOT NULL
          AND radiologist_sign_time IS NOT NULL
          AND algos_start_run IS NOT NULL
          AND EPOCH(radiologist_sign_time - algos_start_run) >= 0
        GROUP BY hour, gender
        ORDER BY hour, gender
        """).df()

        hourly_dept_tat = duckdb.sql("""
        SELECT
            CAST(EXTRACT(HOUR FROM scan_timestamp) AS INTEGER) AS hour,
            patient_class,
            COUNT(*) AS scans,
            ROUND(
                AVG(EPOCH(radiologist_sign_time - algos_start_run) / 60.0), 2
            ) AS mean_rad_min
        FROM filtered_df
        WHERE scan_timestamp IS NOT NULL
          AND radiologist_sign_time IS NOT NULL
          AND algos_start_run IS NOT NULL
          AND EPOCH(radiologist_sign_time - algos_start_run) >= 0
        GROUP BY hour, patient_class
        ORDER BY hour, patient_class
        """).df()

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
                "**Note:** ED radiologist verdicts take around ~1 minute longer relative to IN."
            )

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

# --- TAB 4: ALGO PERFORMANCE ---
with tab4:
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
    gender_algo_df = duckdb.sql(group_accuracy_query("gender")).df()

    g_cols = st.columns(len(gender_algo_df) if len(gender_algo_df) else 1)
    for i, row in gender_algo_df.iterrows():
        g_cols[i].metric(
            f"{row['gender']} scans",
            f"{int(row['scans']):,}",
            f"{row['prevalence_pct']}% prevalence",
        )

    st.dataframe(gender_algo_df, use_container_width=True, hide_index=True)
    st.plotly_chart(
        accuracy_bar(gender_algo_df, "gender", "Accuracy (%) by Gender"),
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
