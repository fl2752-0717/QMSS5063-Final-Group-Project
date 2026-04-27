#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2026/4/14 09:13
# @Author  : Jennifer
# @Software: PyCharm
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2026/4/14 09:09
# @Software: PyCharm
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from rapidfuzz import fuzz, process
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pathlib import Path

st.set_page_config(page_title="Conclusion", layout="wide")

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

PLOT_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}

GRADE_ORDER = ["A", "B", "C"]
GRADE_LABELS = {"A": "Grade A", "B": "Grade B", "C": "Grade C"}
GRADE_COLORS = {"A": "#2ecc71", "B": "#f39c12", "C": "#e74c3c"}
BORO_ORDER = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
MATCH_THRESHOLD = 85

def locate_inspection_csv():
    base_dir = Path(__file__).resolve().parents[1]
    candidates = [
        base_dir / "DOHMH_New_York_City_Restaurant_Inspection_Results_20260414.csv",
        base_dir / "data" / "DOHMH_New_York_City_Restaurant_Inspection_Results_20260414.csv",
        Path("/mnt/data/DOHMH_New_York_City_Restaurant_Inspection_Results_20260414.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Inspection CSV not found.")

def locate_tripadvisor_csv():
    base_dir = Path(__file__).resolve().parents[1]
    candidates = [
        base_dir / "trip advisor restaurents  10k - trip_rest_neywork_1(1).csv",
        base_dir / "data" / "trip advisor restaurents  10k - trip_rest_neywork_1(1).csv",
        Path("/mnt/data/trip advisor restaurents  10k - trip_rest_neywork_1(1).csv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("TripAdvisor CSV not found.")

@st.cache_data(show_spinner=True)
def build_merged_data():
    insp = pd.read_csv(locate_inspection_csv(), engine="python", on_bad_lines="skip")
    insp["INSPECTION DATE"] = pd.to_datetime(insp["INSPECTION DATE"], errors="coerce")
    insp["SCORE"] = pd.to_numeric(insp["SCORE"], errors="coerce")
    insp["BORO"] = insp["BORO"].astype(str).str.strip().str.title()
    insp["GRADE"] = insp["GRADE"].astype(str).str.strip()
    insp["DBA"] = insp["DBA"].astype(str).str.strip()

    valid_boros = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
    insp = insp[insp["BORO"].isin(valid_boros)].copy()
    insp = insp.dropna(subset=["CAMIS", "INSPECTION DATE", "DBA"])

    latest = (
        insp.groupby("CAMIS")["INSPECTION DATE"].max()
        .reset_index()
        .rename(columns={"INSPECTION DATE": "LATEST_DATE"})
    )
    insp_latest = insp.merge(latest, on="CAMIS")
    insp_latest = insp_latest[insp_latest["INSPECTION DATE"] == insp_latest["LATEST_DATE"]]

    restaurant_level = (
        insp_latest.groupby("CAMIS")
        .agg(
            name=("DBA", "first"),
            borough=("BORO", "first"),
            cuisine=("CUISINE DESCRIPTION", "first"),
            score=("SCORE", "first"),
            grade=("GRADE", "first"),
        )
        .reset_index()
    )
    restaurant_level = restaurant_level[restaurant_level["grade"].isin(["A", "B", "C"])].copy()
    restaurant_level["name_clean"] = restaurant_level["name"].astype(str).str.lower().str.strip()

    ta = pd.read_csv(locate_tripadvisor_csv())
    ta["comment_clean"] = (
        ta["Reveiw Comment"].astype(str).str.strip()
        .str.strip('"').str.strip("'")
        .str.strip('\u201c\u201d')
        .str.strip()
    )
    ta["title_clean"] = ta["Title"].astype(str).str.lower().str.strip()

    sia = SentimentIntensityAnalyzer()
    ta["sentiment"] = ta["comment_clean"].apply(lambda x: sia.polarity_scores(str(x))["compound"])

    insp_names = restaurant_level["name_clean"].tolist()
    insp_lookup = restaurant_level.set_index("name_clean")

    matched_rows = []
    for _, ta_row in ta.iterrows():
        result = process.extractOne(
            ta_row["title_clean"],
            insp_names,
            scorer=fuzz.token_sort_ratio,
        )
        if result and result[1] >= MATCH_THRESHOLD:
            matched_name = result[0]
            insp_info = insp_lookup.loc[matched_name]
            if isinstance(insp_info, pd.DataFrame):
                insp_info = insp_info.iloc[0]

            matched_rows.append({
                "ta_title": ta_row["Title"],
                "insp_name": insp_info["name"],
                "match_score": result[1],
                "borough": insp_info["borough"],
                "cuisine": insp_info["cuisine"],
                "grade": insp_info["grade"],
                "insp_score": insp_info["score"],
                "sentiment": ta_row["sentiment"],
                "comment": ta_row["comment_clean"],
            })

    merged = pd.DataFrame(matched_rows).drop_duplicates(subset=["ta_title"]).copy()
    merged = merged[merged["grade"].isin(["A", "B", "C"])].copy()
    merged["borough"] = merged["borough"].astype(str).str.strip().str.title()
    merged["insp_score"] = pd.to_numeric(merged["insp_score"], errors="coerce")
    return merged

def build_violin_chart(df):
    fig = go.Figure()

    for g in GRADE_ORDER:
        sub = df[df["grade"] == g]["sentiment"].dropna()
        if len(sub) == 0:
            continue

        fig.add_trace(
            go.Violin(
                y=sub,
                x=[GRADE_LABELS[g]] * len(sub),
                name=GRADE_LABELS[g],
                box_visible=True,
                meanline_visible=True,
                line_color=GRADE_COLORS[g],
                fillcolor=GRADE_COLORS[g],
                opacity=0.55,
                points="all",
                jitter=0.08,
                pointpos=0,
                marker=dict(size=4, opacity=0.25, color=GRADE_COLORS[g]),
            )
        )

    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
    fig.update_layout(
        title="Review Sentiment Distribution by Inspection Grade",
        template="plotly_white",
        height=460,
        xaxis_title="Inspection Grade",
        yaxis_title="VADER Compound Sentiment Score",
        showlegend=False,
        margin=dict(l=40, r=20, t=60, b=20),
    )
    fig.update_yaxes(range=[-1.05, 1.05])
    return fig

def build_avg_sentiment_chart(df):
    summary = (
        df.groupby("grade")["sentiment"]
        .agg(["mean", "sem", "count"])
        .reindex(GRADE_ORDER)
        .reset_index()
    )

    a_scores = df[df["grade"] == "A"]["sentiment"].values
    bc_scores = df[df["grade"].isin(["B", "C"])]["sentiment"].values

    p_val = np.nan
    d_val = np.nan
    if len(bc_scores) > 0:
        _, p_val = stats.mannwhitneyu(a_scores, bc_scores, alternative="two-sided")
        d_val = (a_scores.mean() - bc_scores.mean()) / np.sqrt((a_scores.var() + bc_scores.var()) / 2)

    fig = go.Figure()
    for _, row in summary.iterrows():
        g = row["grade"]
        fig.add_trace(
            go.Bar(
                x=[GRADE_LABELS[g]],
                y=[row["mean"]],
                error_y=dict(type="data", array=[row["sem"] if pd.notna(row["sem"]) else 0], visible=True),
                marker_color=GRADE_COLORS[g],
                text=[f"{row['mean']:.3f}<br>(n={int(row['count'])})"],
                textposition="outside",
                showlegend=False,
            )
        )

    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    fig.update_layout(
        title=f"Average Review Sentiment by Inspection Grade (A vs B/C: p={p_val:.3g}, {sig})",
        template="plotly_white",
        height=430,
        xaxis_title="Inspection Grade",
        yaxis_title="Average Sentiment Score",
        margin=dict(l=40, r=20, t=60, b=20),
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)

    return fig, summary, p_val, d_val

def build_scatter_chart(df):
    sub = df.dropna(subset=["insp_score"]).copy()
    sub = sub[sub["insp_score"].between(0, 150)]

    pearson_r, pearson_p = stats.pearsonr(sub["insp_score"], sub["sentiment"])
    spearman_r, spearman_p = stats.spearmanr(sub["insp_score"], sub["sentiment"])

    fig = px.scatter(
        sub,
        x="insp_score",
        y="sentiment",
        color="grade",
        category_orders={"grade": GRADE_ORDER},
        color_discrete_map=GRADE_COLORS,
        opacity=0.45,
        template="plotly_white",
        title=f"Review Sentiment vs Inspection Score (Pearson r={pearson_r:.3f})",
        labels={"insp_score": "Inspection Score (higher = worse)", "sentiment": "VADER Sentiment Score"},
    )

    x = sub["insp_score"].values
    y = sub["sentiment"].values
    m, b = np.polyfit(x, y, 1)
    xline = np.linspace(x.min(), x.max(), 100)
    fig.add_trace(
        go.Scatter(
            x=xline,
            y=m * xline + b,
            mode="lines",
            line=dict(color="black", dash="dash", width=2),
            name=f"Linear fit (slope={m:.4f})",
        )
    )

    fig.add_hline(y=0, line_dash="dot", line_color="gray", line_width=1)
    fig.update_layout(
        height=460,
        margin=dict(l=40, r=20, t=60, b=20),
        legend_title="",
    )

    return fig, pearson_r, spearman_r, m

def takeaway_violin(df):
    medians = df.groupby("grade")["sentiment"].median().reindex(GRADE_ORDER)
    return (
        "Sentiment distributions look similar across all three grades, with medians clustered at similar levels. "
        f"Grade A has the widest spread, while Grade B and C are narrower partly because the matched sample is much smaller."
    )

def takeaway_avg(summary, p_val, d_val):
    means = summary.set_index("grade")["mean"]
    return (
        f"Average sentiment is nearly identical across grades (A: {means['A']:.3f}, "
        f"B: {means['B']:.3f}, C: {means['C']:.3f}), and the A vs B/C difference is not statistically significant "
        f"(p={p_val:.4f}). The effect size is negligible (Cohen's d={d_val:.3f}). "
        "This suggests public review language does not reliably track official inspection outcomes."
    )

def takeaway_scatter(pearson_r, spearman_r, slope):
    return (
        f"The correlation between review sentiment and inspection score is extremely weak "
        f"(Pearson r={pearson_r:.3f}, Spearman r={spearman_r:.3f}). "
        f"The trend line is nearly flat (slope={slope:.4f}), meaning worse inspection scores are not associated with meaningfully worse reviews."
    )

def show_conclusion():
    st.title("📊 Conclusion: Public Reviews vs Official Inspections")

    merged = build_merged_data()

    with st.sidebar:
        st.markdown("## Conclusion Filters")
        borough_options = ["All"] + BORO_ORDER
        selected_borough = st.selectbox("Borough", borough_options, index=0)

    filtered = merged.copy()
    if selected_borough != "All":
        filtered = filtered[filtered["borough"] == selected_borough]

    if filtered.empty:
        st.warning("No matched review–inspection records are available for the current filters.")
        return

    # ---- Violin ----
    with st.container(border=True):
        st.subheader("Review Sentiment Distribution by Inspection Grade")
        fig1 = build_violin_chart(filtered)
        st.plotly_chart(fig1, use_container_width=True, config=PLOT_CONFIG)
        st.markdown(f"**Key takeaway:** {takeaway_violin(filtered)}")

    # ---- Avg ----
    with st.container(border=True):
        st.subheader("Average Sentiment by Inspection Grade")
        fig2, avg_summary, p_val, d_val = build_avg_sentiment_chart(filtered)
        st.plotly_chart(fig2, use_container_width=True, config=PLOT_CONFIG)
        st.markdown(f"**Key takeaway:** {takeaway_avg(avg_summary, p_val, d_val)}")

    # ---- Scatter ----
    with st.container(border=True):
        st.subheader("Sentiment vs Inspection Score")
        fig3, pearson_r, spearman_r, slope = build_scatter_chart(filtered)
        st.plotly_chart(fig3, use_container_width=True, config=PLOT_CONFIG)
        st.markdown(f"**Key takeaway:** {takeaway_scatter(pearson_r, spearman_r, slope)}")

    # ---- Text Blocks ----
    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.markdown("## 💡 Key Findings")
            st.markdown("""
- Public reviews do not meaningfully distinguish restaurant inspection quality.
    Sentiment remains consistently positive across Grade A, B, and C establishments.
- Differences in sentiment across grades are small and inconsistent.
    In some cases, lower-graded restaurants receive comparable or even higher sentiment than Grade A.
- Inspection outcomes and customer perception are weakly aligned.
    Statistical patterns show little to no relationship between inspection scores and review sentiment.
- Review content reflects experience—not compliance.
    Language in reviews overwhelmingly focuses on food, service, and atmosphere, with minimal reference to hygiene or safety.
""")

    with col2:
        with st.container(border=True):
            st.markdown("## 🔄 Where Signals Diverge")
            st.markdown("""
Restaurant inspections and public reviews measure fundamentally different dimensions of quality:
Inspections capture regulatory compliance (sanitation, violations, safety risks)
Reviews capture subjective experience (taste, service, ambiance)

As a result:
Restaurants can fail inspections but still receive strong reviews
High-performing (Grade A) restaurants are not guaranteed higher sentiment
Consumers are effectively blind to hygiene signals when forming opinions
""")

show_conclusion()