#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2026/4/14 09:12
# @Author  : Jennifer
# @Software: PyCharm
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2026/4/14 09:09
# @Software: PyCharm

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import altair as alt
import json
from pathlib import Path

import folium
from folium.plugins import FastMarkerCluster
from streamlit_folium import st_folium


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Official Inspections", layout="wide")

st.markdown("""
<style>
[data-testid="stSidebarNav"]::before {
    content: "🍽️\\A NYC\\A Restaurant\\A Quality\\A\\A QMSS Final Project";
    white-space: pre-line;
    font-size: 20px;
    font-weight: bold;
    display: block;
    margin-bottom: 20px;
    padding-left: 1.5rem;
    line-height: 1.2;
}
</style>
""", unsafe_allow_html=True)

# =========================
# PATHS
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent

CSV_PATH = BASE_DIR / "data" / "inspection_sample.csv"
GEOJSON_PATH = BASE_DIR / "data" / "Borough_Boundaries_20260416.geojson"


# =========================
# LOAD + PREP DATA
# =========================
@st.cache_data
def load_and_prepare_data():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")

    if not GEOJSON_PATH.exists():
        raise FileNotFoundError(f"GeoJSON file not found: {GEOJSON_PATH}")

    inspections = pd.read_csv(CSV_PATH)
    boroughs = gpd.read_file(GEOJSON_PATH)

    boroughs["boroname"] = boroughs["boroname"].astype(str).str.strip().str.title()
    boroughs = boroughs[["boroname", "geometry"]].copy()

    if boroughs.crs is not None and str(boroughs.crs) != "EPSG:4326":
        boroughs = boroughs.to_crs(epsg=4326)

    df = inspections[
        [
            "CAMIS",
            "DBA",
            "BORO",
            "CUISINE DESCRIPTION",
            "INSPECTION DATE",
            "SCORE",
            "GRADE",
            "VIOLATION CODE",
            "CRITICAL FLAG",
            "Latitude",
            "Longitude",
        ]
    ].copy()

    df["INSPECTION DATE"] = pd.to_datetime(df["INSPECTION DATE"], errors="coerce")
    df["SCORE"] = pd.to_numeric(df["SCORE"], errors="coerce")
    df["BORO"] = df["BORO"].astype(str).str.strip().str.title()
    df["GRADE"] = df["GRADE"].astype(str).str.strip()
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")

    valid_boros = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
    df = df[df["BORO"].isin(valid_boros)].copy()
    df = df.dropna(subset=["CAMIS", "INSPECTION DATE"])

    latest_dates = (
        df.groupby("CAMIS")["INSPECTION DATE"]
        .max()
        .reset_index()
        .rename(columns={"INSPECTION DATE": "LATEST_INSPECTION_DATE"})
    )

    df_latest = df.merge(latest_dates, on="CAMIS", how="left")
    df_latest = df_latest[
        df_latest["INSPECTION DATE"] == df_latest["LATEST_INSPECTION_DATE"]
    ].copy()

    df_latest["critical_binary"] = np.where(
        df_latest["CRITICAL FLAG"].astype(str).str.strip().str.lower() == "critical",
        1,
        0,
    )

    df_latest["violation_present"] = np.where(
        df_latest["VIOLATION CODE"].notna(),
        1,
        0,
    )

    restaurant_level = (
        df_latest.groupby("CAMIS")
        .agg(
            restaurant_name=("DBA", "first"),
            borough=("BORO", "first"),
            cuisine=("CUISINE DESCRIPTION", "first"),
            latest_inspection_date=("INSPECTION DATE", "first"),
            score=("SCORE", "first"),
            grade=("GRADE", "first"),
            violation_count=("violation_present", "sum"),
            critical_violation_count=("critical_binary", "sum"),
            latitude=("Latitude", "first"),
            longitude=("Longitude", "first"),
        )
        .reset_index()
    )

    restaurant_level["has_critical_violation"] = np.where(
        restaurant_level["critical_violation_count"] > 0,
        1,
        0,
    )

    return restaurant_level, boroughs


def filter_data(data: pd.DataFrame, cuisine: str = "All", borough: str = "All") -> pd.DataFrame:
    temp = data.copy()

    if cuisine != "All":
        temp = temp[temp["cuisine"] == cuisine]

    if borough != "All":
        temp = temp[temp["borough"] == borough]

    return temp


def summarize_borough(filtered_df: pd.DataFrame) -> pd.DataFrame:
    borough_summary = (
        filtered_df.groupby("borough")
        .agg(
            n_restaurants=("CAMIS", "count"),
            avg_score=("score", "mean"),
            median_score=("score", "median"),
            pct_grade_A=("grade", lambda x: (x == "A").mean() * 100),
            pct_grade_B=("grade", lambda x: (x == "B").mean() * 100),
            pct_grade_C=("grade", lambda x: (x == "C").mean() * 100),
            critical_violation_rate=("has_critical_violation", "mean"),
            avg_violation_count=("violation_count", "mean"),
        )
        .reset_index()
    )

    borough_summary["critical_violation_rate"] = (
        borough_summary["critical_violation_rate"] * 100
    )

    return borough_summary


# =========================
# CHART BUILDERS
# =========================
def build_metric_chart(borough_summary: pd.DataFrame):
    interactive_df = borough_summary[
        ["borough", "avg_score", "pct_grade_A", "critical_violation_rate", "n_restaurants"]
    ].copy()

    plot_df = interactive_df.melt(
        id_vars=["borough", "n_restaurants"],
        value_vars=["avg_score", "pct_grade_A", "critical_violation_rate"],
        var_name="metric",
        value_name="value",
    )

    label_map = {
        "avg_score": "Average Inspection Score",
        "pct_grade_A": "Grade A Share (%)",
        "critical_violation_rate": "Critical Violation Rate (%)",
    }

    plot_df["metric_label"] = plot_df["metric"].map(label_map)

    metric_dropdown = alt.param(
        name="Metric",
        bind=alt.binding_select(
            options=[
                "Average Inspection Score",
                "Grade A Share (%)",
                "Critical Violation Rate (%)",
            ],
            name="Choose metric: ",
        ),
        value="Average Inspection Score",
    )

    bars = (
        alt.Chart(plot_df)
        .add_params(metric_dropdown)
        .transform_filter(alt.datum.metric_label == metric_dropdown)
        .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(
            y=alt.Y("borough:N", sort="-x", title="Borough"),
            x=alt.X("value:Q", title="Value"),
            color=alt.Color(
                "metric_label:N",
                legend=None,
                scale=alt.Scale(
                    domain=[
                        "Average Inspection Score",
                        "Grade A Share (%)",
                        "Critical Violation Rate (%)",
                    ],
                    range=["#3B82F6", "#10B981", "#EF4444"],
                ),
            ),
            tooltip=[
                alt.Tooltip("borough:N", title="Borough"),
                alt.Tooltip("metric_label:N", title="Metric"),
                alt.Tooltip("value:Q", title="Value", format=".2f"),
                alt.Tooltip("n_restaurants:Q", title="Number of Restaurants"),
            ],
        )
        .properties(
            width=650,
            height=280,
            title="Comparison of Restaurant Inspection Metrics Across NYC Boroughs",
        )
    )

    labels = (
        alt.Chart(plot_df)
        .add_params(metric_dropdown)
        .transform_filter(alt.datum.metric_label == metric_dropdown)
        .mark_text(align="left", baseline="middle", dx=4)
        .encode(
            y=alt.Y("borough:N", sort="-x"),
            x=alt.X("value:Q"),
            text=alt.Text("value:Q", format=".2f"),
        )
    )

    return bars + labels

def build_map_chart(boroughs, borough_summary):
    borough_map = boroughs.copy()

    borough_map["boroname"] = borough_map["boroname"].astype(str).str.strip().str.title()
    borough_summary = borough_summary.copy()
    borough_summary["borough"] = borough_summary["borough"].astype(str).str.strip().str.title()

    borough_map = borough_map.merge(
        borough_summary[
            ["borough", "avg_score", "pct_grade_A", "critical_violation_rate", "n_restaurants"]
        ],
        left_on="boroname",
        right_on="borough",
        how="left",
    )

    borough_map = borough_map.dropna(subset=["avg_score"]).copy()

    if borough_map.crs is not None and str(borough_map.crs) != "EPSG:4326":
        borough_map = borough_map.to_crs(epsg=4326)

    borough_json = json.loads(borough_map.to_json())

    fig = px.choropleth_mapbox(
        borough_map,
        geojson=borough_json,
        locations="boroname",
        featureidkey="properties.boroname",
        color="avg_score",
        hover_name="boroname",
        hover_data={
            "avg_score": ":.2f",
            "pct_grade_A": ":.2f",
            "critical_violation_rate": ":.2f",
            "n_restaurants": True,
        },
        color_continuous_scale="Blues",
        mapbox_style="carto-positron",
        center={"lat": 40.7128, "lon": -74.0060},
        zoom=9,
        opacity=0.75,
        title="Restaurant Inspection Outcomes Across NYC Boroughs",
    )

    fig.update_layout(
        height=520,
        margin=dict(l=0, r=0, t=50, b=0),
        coloraxis_colorbar=dict(title="Average Score"),
    )

    return fig

def build_grade_distribution_chart(filtered_df: pd.DataFrame):
    grade_df = filtered_df[filtered_df["grade"].isin(["A", "B", "C"])].copy()

    grade_dist = (
        grade_df.groupby(["borough", "grade"])
        .size()
        .reset_index(name="count")
    )

    grade_dist["pct"] = (
        grade_dist["count"] / grade_dist.groupby("borough")["count"].transform("sum")
    ) * 100

    return (
        alt.Chart(grade_dist)
        .mark_bar()
        .encode(
            x=alt.X("borough:N", title="Borough"),
            y=alt.Y("pct:Q", title="Percent of Restaurants"),
            color=alt.Color(
                "grade:N",
                title="Grade",
                sort=["A", "B", "C"],
                scale=alt.Scale(
                    domain=["A", "B", "C"],
                    range=["#0F6CC9", "#7EC8F8", "#FF3B30"],
                ),
            ),
            tooltip=[
                alt.Tooltip("borough:N", title="Borough"),
                alt.Tooltip("grade:N", title="Grade"),
                alt.Tooltip("pct:Q", title="Percent", format=".2f"),
            ],
        )
        .properties(
            width=500,
            height=320,
            title="Grade Distribution by Borough",
        )
    )


def build_avg_score_chart(borough_summary: pd.DataFrame):
    score_plot = borough_summary.sort_values("avg_score")

    return (
        alt.Chart(score_plot)
        .mark_bar(color="#3B82F6")
        .encode(
            x=alt.X("borough:N", sort=None, title="Borough"),
            y=alt.Y("avg_score:Q", title="Average Score"),
            tooltip=[
                alt.Tooltip("borough:N", title="Borough"),
                alt.Tooltip("avg_score:Q", title="Average Score", format=".2f"),
            ],
        )
        .properties(
            width=500,
            height=320,
            title="Average Inspection Score by Borough",
        )
    )


def build_critical_violation_chart(borough_summary: pd.DataFrame):
    critical_plot = borough_summary.sort_values("critical_violation_rate", ascending=False)

    return (
        alt.Chart(critical_plot)
        .mark_bar(color="#EF4444")
        .encode(
            x=alt.X("borough:N", sort=None, title="Borough"),
            y=alt.Y(
                "critical_violation_rate:Q",
                title="Percent of Restaurants with Critical Violations",
            ),
            tooltip=[
                alt.Tooltip("borough:N", title="Borough"),
                alt.Tooltip(
                    "critical_violation_rate:Q",
                    title="Critical Violation Rate (%)",
                    format=".2f",
                ),
            ],
        )
        .properties(
            width=500,
            height=320,
            title="Critical Violation Rate by Borough",
        )
    )


def build_grade_a_cluster_map(filtered_df: pd.DataFrame):
    grade_a_df = filtered_df[
        (filtered_df["grade"] == "A")
        & (filtered_df["latitude"].notna())
        & (filtered_df["longitude"].notna())
    ].copy()

    grade_a_points = grade_a_df[["latitude", "longitude"]].values.tolist()

    m = folium.Map(
        location=[40.7128, -74.0060],
        zoom_start=10,
        tiles="CartoDB positron",
    )

    if len(grade_a_points) > 0:
        FastMarkerCluster(grade_a_points).add_to(m)

    return m, len(grade_a_points)


# =========================
# LOAD DATA
# =========================
restaurant_level, boroughs = load_and_prepare_data()


# =========================
# SIDEBAR FILTERS
# =========================
with st.sidebar:
    st.markdown("### Filters")

    all_boroughs = sorted(restaurant_level["borough"].dropna().unique().tolist())
    all_cuisines = sorted(
        [c for c in restaurant_level["cuisine"].dropna().unique().tolist() if str(c) != "nan"]
    )

    selected_borough = st.selectbox("Borough", ["All"] + all_boroughs)
    selected_cuisine = st.selectbox("Cuisine", ["All"] + all_cuisines)


filtered_df = filter_data(
    restaurant_level,
    cuisine=selected_cuisine,
    borough=selected_borough,
)

borough_summary = summarize_borough(filtered_df)


# =========================
# PAGE CONTENT
# =========================
st.title("📝 Official Inspections")

col1, col2, col3 = st.columns(3)

with col1:
    with st.container(border=True):
        st.metric("🍕 Total Restaurants", f"{len(filtered_df):,}")

with col2:
    with st.container(border=True):
        st.metric("📋 Average Score", f"{filtered_df['score'].mean():.2f}")

with col3:
    with st.container(border=True):
        pct_a = filtered_df["grade"].eq("A").mean() * 100
        st.metric("🏆 Percent Grade A", f"{pct_a:.1f}%")


with st.container(border=True):
    st.altair_chart(build_metric_chart(borough_summary), use_container_width=True)


with st.container(border=True):
    st.markdown(
        """
To compare differences across boroughs consistently, we aggregated the data to the restaurant
level and kept only the latest inspection record for each restaurant. This ensures each restaurant
is included only once in the analysis. We define the critical violation rate as the percentage of
restaurants with at least one recorded critical violation in the latest inspection record. We also
consider the average inspection score as another important indicator, with higher scores indicating
worse inspection outcomes.
"""
    )


col_left, col_right = st.columns(2)

with col_left:
    with st.container(border=True):
        st.plotly_chart(build_map_chart(boroughs, borough_summary), use_container_width=True)
        st.caption(
            "This map compares average inspection scores across NYC boroughs. "
            "Because higher scores indicate worse inspection outcomes, darker boroughs represent areas "
            "with relatively weaker recent inspection performance."
        )

with col_right:
    with st.container(border=True):
        st.altair_chart(build_grade_distribution_chart(filtered_df), use_container_width=True)
        st.caption(
            "Grade A restaurants make up the majority in each borough, with Staten Island and "
            "Manhattan accounting for the highest proportions of Grade A restaurants. Queens has "
            "the highest proportion of Grade B and Grade C restaurants. This chart shows the full "
            "grade composition within each borough."
        )


with st.container(border=True):
    st.markdown("### Spatial Cluster of Grade A Restaurants")

    grade_a_map, n_grade_a_points = build_grade_a_cluster_map(filtered_df)

    st_folium(
        grade_a_map,
        width=None,
        height=520,
        returned_objects=[],
    )

    st.caption(
        f"This interactive marker-cluster map shows the spatial concentration of "
        f"**{n_grade_a_points:,} Grade A restaurants** under the current filters. "
        "Dense clusters appear in areas with higher restaurant concentration, especially in central Manhattan "
        "and nearby commercial districts."
    )


with st.container(border=True):
    st.markdown("**Key Takeaway**")
    st.markdown(
        """
Overall, restaurant inspection performance across NYC is not clearly dominated by any one borough; 
instead, the results are mixed across the different metrics. Manhattan performs relatively well overall, 
with a high Grade A share and a better average inspection score. Queens appears weaker because it has 
the worst average inspection score and a lower Grade A share. Staten Island has both the highest critical 
violation rate and a high Grade A share, which may be partly affected by its smaller restaurant sample size. 
Brooklyn is generally in the middle across the metrics, while the Bronx performs relatively better on 
average inspection score but still shows a notable critical violation pattern.
"""
    )