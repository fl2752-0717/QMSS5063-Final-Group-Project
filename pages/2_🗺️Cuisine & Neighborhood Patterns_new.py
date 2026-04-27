#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2026/4/14 09:13
# @Author  : Jennifer
# @Software: PyCharm
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2026/4/14 09:09
# @Software: PyCharm
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

# =========================
# Page config + style
# =========================
st.set_page_config(page_title="Cuisine Patterns", layout="wide")

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
# File path
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "inspection_sample.csv"

if not CSV_PATH.exists():
    raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")

# =========================
# Load data
# =========================
@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv(CSV_PATH)
    df = df.copy()

    df = df[df["BORO"].notna()]
    df = df[df["CUISINE DESCRIPTION"].notna()]
    df = df[df["INSPECTION DATE"].notna()]
    df = df[df["BORO"] != "0"]

    df["INSPECTION DATE"] = pd.to_datetime(df["INSPECTION DATE"], errors="coerce")
    df = df[df["INSPECTION DATE"].notna()]

    df["YEAR"] = df["INSPECTION DATE"].dt.year
    df["SCORE"] = pd.to_numeric(df["SCORE"], errors="coerce")

    df["BORO"] = df["BORO"].astype(str).str.strip().str.title()

    return df


def recode_cuisine(x):
    if x == "American":
        return "American"
    elif x == "Chinese":
        return "Chinese"
    elif x in ["Italian", "Pizza"]:
        return "Italian"
    elif x == "Japanese":
        return "Japanese"
    elif x == "Korean":
        return "Korean"
    elif x == "Thai":
        return "Thai"
    elif x in ["Mexican", "Tex-Mex"]:
        return "Mexican"
    elif x in ["Latin American", "Spanish"]:
        return "Latin American"
    elif x == "Caribbean":
        return "Caribbean"
    elif x == "Indian":
        return "Indian"
    elif x == "Asian/Asian Fusion":
        return "Asian Fusion"
    else:
        return "Other"


def apply_filters(data, boroughs=None, cuisines=None, years=None, grades=None):
    filtered = data.copy()

    if boroughs and "All" not in boroughs:
        filtered = filtered[filtered["BORO"].isin(boroughs)]

    if cuisines and "All" not in cuisines:
        filtered = filtered[filtered["CUISINE_CLEAN"].isin(cuisines)]

    if years and "All" not in years:
        filtered = filtered[filtered["YEAR"].isin(years)]

    if grades and "All" not in grades and "GRADE" in filtered.columns:
        filtered = filtered[filtered["GRADE"].isin(grades)]

    return filtered


def summarize_by_cuisine(inspection_df, violation_df, metric):
    if metric == "Avg Score":
        return (
            inspection_df.groupby("CUISINE_CLEAN")["SCORE"]
            .mean()
            .reset_index(name="value")
            .sort_values("value")
        )

    elif metric == "Restaurant Count":
        return (
            inspection_df.groupby("CUISINE_CLEAN")["CAMIS"]
            .nunique()
            .reset_index(name="value")
            .sort_values("value", ascending=False)
        )

    elif metric == "Inspection Count":
        return (
            inspection_df.groupby("CUISINE_CLEAN")
            .size()
            .reset_index(name="value")
            .sort_values("value", ascending=False)
        )

    elif metric == "Violation Count":
        return (
            violation_df.groupby("CUISINE_CLEAN")
            .size()
            .reset_index(name="value")
            .sort_values("value", ascending=False)
        )

    elif metric == "Grade A %":
        temp = inspection_df.copy()
        temp["is_A"] = (temp["GRADE"] == "A").astype(int)
        return (
            temp.groupby("CUISINE_CLEAN")["is_A"]
            .mean()
            .reset_index(name="value")
            .sort_values("value")
        )

    else:
        raise ValueError("Unknown metric")


def make_heatmap_data(inspection_df, violation_df, metric):
    if metric == "Avg Score":
        return inspection_df.groupby(["BORO", "CUISINE_CLEAN"])["SCORE"].mean().unstack()

    elif metric == "Restaurant Count":
        return inspection_df.groupby(["BORO", "CUISINE_CLEAN"])["CAMIS"].nunique().unstack()

    elif metric == "Inspection Count":
        return inspection_df.groupby(["BORO", "CUISINE_CLEAN"]).size().unstack()

    elif metric == "Violation Count":
        return violation_df.groupby(["BORO", "CUISINE_CLEAN"]).size().unstack()

    elif metric == "Grade A %":
        temp = inspection_df.copy()
        temp["is_A"] = (temp["GRADE"] == "A").astype(int)
        return temp.groupby(["BORO", "CUISINE_CLEAN"])["is_A"].mean().unstack()

    else:
        raise ValueError("Unknown metric")


# =========================
# Data prep
# =========================
df = load_and_prepare_data()
df["CUISINE_CLEAN"] = df["CUISINE DESCRIPTION"].apply(recode_cuisine)

VALID_BOROUGHS = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]
metric = "Avg Score"

# =========================
# Sidebar filters
# =========================
st.sidebar.header("Filters")

selected_borough = st.sidebar.selectbox(
    "Borough",
    ["All"] + VALID_BOROUGHS,
)

if selected_borough == "All":
    filtered_df = df.copy()
else:
    filtered_df = df[df["BORO"] == selected_borough].copy()

if filtered_df.empty:
    st.warning("No data available for the selected borough.")
    st.stop()

filtered_violation_df = filtered_df.copy()
filtered_inspection_df = filtered_violation_df.drop_duplicates(
    subset=["CAMIS", "INSPECTION DATE"]
).copy()

cuisine_summary = summarize_by_cuisine(
    filtered_inspection_df,
    filtered_violation_df,
    metric,
)

score_df = filtered_df.dropna(subset=["BORO", "SCORE"]).copy()

borough_perf = (
    score_df.groupby("BORO", as_index=False)
    .agg(
        avg_score=("SCORE", "mean"),
        n=("SCORE", "size"),
    )
)

city_avg = score_df["SCORE"].mean()
borough_perf["diff_from_city"] = borough_perf["avg_score"] - city_avg
borough_perf = borough_perf.sort_values("diff_from_city")

borough_perf["performance"] = np.where(
    borough_perf["diff_from_city"] <= 0,
    "Better than average",
    "Worse than average",
)

heatmap_data = make_heatmap_data(
    filtered_inspection_df,
    filtered_violation_df,
    metric,
)


# =========================
# Chart builders
# =========================
def build_borough_relative_chart():
    plot_df = borough_perf.copy()
    plot_df = plot_df.sort_values("diff_from_city")

    fig = px.bar(
        plot_df,
        x="BORO",
        y="diff_from_city",
        color="performance",
        template="plotly_white",
        title="Borough Inspection Performance Relative to City Average",
        labels={
            "BORO": "Borough",
            "diff_from_city": "Difference from city average score",
            "performance": "",
        },
        color_discrete_map={
            "Better than average": "#22c55e",
            "Worse than average": "#ef4444",
        },
    )

    fig.add_hline(y=0, line_color="gray", line_width=1)

    fig.update_layout(
        height=420,
        margin=dict(l=40, r=30, t=70, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    fig.update_xaxes(tickangle=0)
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>Difference: %{y:.2f}<extra></extra>"
    )

    return fig


def build_cuisine_score_chart():
    cuisine_plot = cuisine_summary.dropna(subset=["value"]).copy()
    cuisine_plot = cuisine_plot.sort_values("value", ascending=True)

    fig = px.bar(
        cuisine_plot,
        x="value",
        y="CUISINE_CLEAN",
        orientation="h",
        template="plotly_white",
        title="Average Inspection Score by Cuisine",
        labels={
            "value": "Avg Score",
            "CUISINE_CLEAN": "Cuisine",
        },
        color_discrete_sequence=["#357BA8"],
    )

    fig.update_layout(
        height=560,
        showlegend=False,
        margin=dict(l=40, r=20, t=60, b=40),
    )

    return fig


def build_borough_cuisine_heatmap():
    fig = plt.figure(figsize=(12, 6))
    sns.heatmap(
        heatmap_data,
        cmap="Blues",
        annot=True,
        fmt=".1f",
        linewidths=0.4,
    )
    plt.title(f"{metric} by Borough and Cuisine")
    plt.xlabel("Cuisine")
    plt.ylabel("Borough")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    return fig


def build_violation_heatmap():
    viol_df = df[["BORO", "VIOLATION DESCRIPTION"]].copy()
    viol_df = viol_df.dropna(subset=["BORO", "VIOLATION DESCRIPTION"])
    viol_df["BORO"] = viol_df["BORO"].astype(str).str.strip().str.title()
    viol_df = viol_df[viol_df["BORO"].isin(VALID_BOROUGHS)]

    if selected_borough != "All":
        viol_df = viol_df[viol_df["BORO"] == selected_borough].copy()

    top_violations = viol_df["VIOLATION DESCRIPTION"].value_counts().head(10).index
    viol_top = viol_df[viol_df["VIOLATION DESCRIPTION"].isin(top_violations)].copy()

    borough_violation = pd.crosstab(
        viol_top["VIOLATION DESCRIPTION"],
        viol_top["BORO"],
    )

    borough_violation_pct = borough_violation.div(
        borough_violation.sum(axis=0),
        axis=1,
    ) * 100

    label_map = {
        "Anti-siphonage or back-flow prevention device not provided where required; equipment or floor not properly drained; sewage disposal system in disrepair or not functioning properly. Condensation or liquid waste improperly disposed of.": "Plumbing / drainage",
        "Cold TCS food item held above 41 °F; smoked or processed fish held above 38 °F; intact raw eggs held above 45 °F; or reduced oxygen packaged (ROP) TCS foods held above required temperatures except during active necessary preparation.": "Cold food temp",
        "Establishment is not free of harborage or conditions conducive to rodents, insects or other pests.": "Pest conditions",
        "Evidence of mice or live mice in establishment's food or non-food areas.": "Rodent evidence",
        "Filth flies or food/refuse/sewage associated with (FRSA) flies or other nuisance pests in establishment’s food and/or non-food areas. FRSA flies include house flies, blow flies, bottle flies, flesh flies, drain flies, Phorid flies and fruit flies.": "Flies / pests",
        "Food Protection Certificate (FPC) not held by manager or supervisor of food operations.": "No FPC",
        "Food contact surface not properly washed, rinsed and sanitized after each use and following any activity when contamination may have occurred.": "Surface sanitation",
        "Food, supplies, or equipment not protected from potential source of contamination during storage, preparation, transportation, display, service or from customer’s refillable, reusable container. Condiments not in single-service containers or dispensed directly by the vendor.": "Improper protection",
        "Hot TCS food item not held at or above 140 °F.": "Hot food temp",
        "Non-food contact surface or equipment made of unacceptable material, not kept clean, or not properly sealed, raised, spaced or movable to allow accessibility for cleaning on all sides, above and underneath the unit.": "Non-food surface(cleanliness)",
    }

    borough_violation_pct.index = [
        label_map.get(v, v) for v in borough_violation_pct.index
    ]

    borough_violation_pct = borough_violation_pct.loc[
        borough_violation_pct.mean(axis=1).sort_values(ascending=False).index
    ]

    existing_borough_order = [b for b in VALID_BOROUGHS if b in borough_violation_pct.columns]
    borough_violation_pct = borough_violation_pct[existing_borough_order]

    fig = plt.figure(figsize=(10, 6))
    sns.heatmap(
        borough_violation_pct,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        linewidths=0.5,
        annot_kws={"size": 9},
    )

    plt.title(
        "Distribution of Top 10 Restaurant Health Violations Across NYC Boroughs (%)",
        fontsize=13,
    )
    plt.xlabel("Borough")
    plt.ylabel("Violation Type")
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()

    return fig, borough_violation_pct


def build_interactive_map():
    map_df = filtered_df.copy()

    map_df = map_df.dropna(
        subset=[
            "Latitude",
            "Longitude",
            "BORO",
            "DBA",
            "CUISINE DESCRIPTION",
            "CRITICAL FLAG",
        ]
    ).copy()

    map_df = map_df[
        (map_df["Latitude"] != 0)
        & (map_df["Longitude"] != 0)
    ].copy()

    map_df["BORO"] = map_df["BORO"].astype(str).str.strip().str.title()
    map_df["CRITICAL FLAG"] = map_df["CRITICAL FLAG"].astype(str).str.strip()
    map_df["DBA"] = map_df["DBA"].astype(str).str.strip()
    map_df["CUISINE DESCRIPTION"] = map_df["CUISINE DESCRIPTION"].astype(str).str.strip()

    map_df["is_critical"] = (map_df["CRITICAL FLAG"] == "Critical").astype(int)

    risk_map = (
        map_df.groupby(
            ["DBA", "BORO", "CUISINE DESCRIPTION", "Latitude", "Longitude"],
            as_index=False,
        )
        .agg(
            critical_violation_count=("is_critical", "sum"),
            total_records=("is_critical", "size"),
        )
    )

    risk_map = risk_map[risk_map["critical_violation_count"] > 0].copy()

    if risk_map.empty:
        return None, risk_map

    risk_map["critical_violation_rate"] = (
        risk_map["critical_violation_count"] / risk_map["total_records"]
    )

    upper = risk_map["critical_violation_rate"].quantile(0.95)
    risk_map["rate_capped"] = risk_map["critical_violation_rate"].clip(upper=upper)

    fig = px.scatter_mapbox(
        risk_map,
        lat="Latitude",
        lon="Longitude",
        color="rate_capped",
        color_continuous_scale="OrRd",
        hover_name="DBA",
        hover_data={
            "BORO": True,
            "CUISINE DESCRIPTION": True,
            "critical_violation_count": True,
            "total_records": True,
            "critical_violation_rate": ":.2f",
            "Latitude": False,
            "Longitude": False,
        },
        zoom=10,
        height=650,
        opacity=0.30,
        title="Spatial Distribution of Restaurant-Level Critical Violation Risk in NYC",
    )

    fig.update_layout(mapbox_style="carto-positron")
    fig.update_traces(marker=dict(size=5))
    fig.update_coloraxes(colorbar_title="Critical violation rate")

    return fig, risk_map


# =========================
# Build charts
# =========================
fig1 = build_borough_relative_chart()
fig2 = build_cuisine_score_chart()
fig3 = build_borough_cuisine_heatmap()
fig4, borough_violation_pct_short = build_violation_heatmap()
fig5, risk_map = build_interactive_map()


# =========================
# Render page
# =========================
st.markdown(
    "<h1 style='white-space: nowrap;'>🗺️ Cuisine & Neighborhood Patterns</h1>",
    unsafe_allow_html=True,
)

best_boro = borough_perf.sort_values("diff_from_city").iloc[0]
worst_boro = borough_perf.sort_values("diff_from_city").iloc[-1]

with st.container(border=True):
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown(
        f"""
**Key takeaway:** Relative to the filtered city average, **{best_boro['BORO']}** performs best, while **{worst_boro['BORO']}** performs worst. Lower inspection scores indicate better performance.
"""
    )

with st.container(border=True):
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown(
        """
**Key takeaway:** Inspection outcomes differ noticeably across cuisine types. 
Indian and Caribbean cuisines have the highest average inspection scores, indicating higher violation levels, 
while American cuisine shows the lowest average score and better overall performance. 
Most other cuisines fall in a middle range, suggesting moderate inspection outcomes across categories.
"""
    )

with st.container(border=True):
    st.pyplot(fig3, use_container_width=True)
    st.markdown(
        """
**Key takeaway:** Inspection outcomes vary jointly by borough and cuisine. The heatmap highlights which cuisine categories perform relatively better or worse across boroughs.
"""
    )

with st.container(border=True):
    st.pyplot(fig4, use_container_width=True)

    if not borough_violation_pct_short.empty:
        top_violation = borough_violation_pct_short.mean(axis=1).sort_values(ascending=False).index[0]
        st.markdown(
            f"""
**Key takeaway:** The most prevalent violation category across boroughs is **{top_violation}**. The relative composition of violation types differs somewhat by borough, but several categories remain consistently common citywide.
"""
        )
    else:
        st.markdown("**Key takeaway:** No violation data are available for the selected filter.")

with st.container(border=True):
    if fig5 is not None:
        st.plotly_chart(fig5, use_container_width=True)
        st.markdown(
            """
**Key takeaway:** The spatial distribution of restaurant-level critical violation risk in NYC shows clear clustering rather than random dispersion. Denser clusters appear in parts of Manhattan, Brooklyn, and Queens, suggesting that inspection outcomes are shaped by both location and restaurant characteristics.
"""
        )
    else:
        st.markdown("**Key takeaway:** No map data are available for the selected filter.")
