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
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import re
from rapidfuzz import fuzz, process

st.set_page_config(page_title="Public Reviews", layout="wide")

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

STOPWORDS = {
    'the','a','an','and','or','but','is','was','were','it','its','in','on',
    'at','to','for','of','with','this','that','we','i','my','our','they',
    'he','she','had','have','has','be','been','are','as','so','by','from',
    'not','no','very','also','just','food','restaurant','place','came','went',
    'got','get','go','us','their','there','here','one','all','more','would',
    'will','like','out','up','about','really','well',
    'time','back','did','do','even','made','make','menu','you','your',
}

MATCH_THRESHOLD = 85
BORO_ORDER = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
GRADE_ORDER = ["A", "B", "C"]


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


def locate_inspection_csv():
    base_dir = Path(__file__).resolve().parents[1]
    candidates = [
        base_dir / "DOHMH_New_York_City_Restaurant_Inspection_Results_20260414.csv",
        base_dir / "DOHMH_New_York_City_Restaurant_Inspection_Results_20260416.csv",
        base_dir / "data" / "DOHMH_New_York_City_Restaurant_Inspection_Results_20260414.csv",
        base_dir / "data" / "DOHMH_New_York_City_Restaurant_Inspection_Results_20260416.csv",
        Path("/mnt/data/DOHMH_New_York_City_Restaurant_Inspection_Results_20260414.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Inspection CSV not found.")


@st.cache_data(show_spinner=False)
def load_public_reviews():
    ta = pd.read_csv(locate_tripadvisor_csv())

    ta["Title"] = ta["Title"].astype(str).str.strip()
    ta["Catagory"] = ta["Catagory"].astype(str).str.strip()
    ta["comment_clean"] = (
        ta["Reveiw Comment"].astype(str).str.strip()
        .str.strip('"').str.strip("'")
        .str.strip('\u201c\u201d')
        .str.strip()
    )

    sia = SentimentIntensityAnalyzer()
    ta["sentiment"] = ta["comment_clean"].apply(
        lambda x: sia.polarity_scores(str(x))["compound"]
    )

    def primary_category(x):
        parts = [p.strip() for p in str(x).split(",") if p.strip()]
        return parts[0] if parts else "Unknown"

    ta["primary_category"] = ta["Catagory"].apply(primary_category)
    return ta


@st.cache_data(show_spinner=True)
def build_merged_data():
    ta = load_public_reviews().copy()
    ta["title_clean"] = ta["Title"].astype(str).str.lower().str.strip()

    insp = pd.read_csv(locate_inspection_csv(), engine="python", on_bad_lines="skip")
    insp["INSPECTION DATE"] = pd.to_datetime(insp["INSPECTION DATE"], errors="coerce")
    insp["SCORE"] = pd.to_numeric(insp["SCORE"], errors="coerce")
    insp["BORO"] = insp["BORO"].astype(str).str.strip().str.title()
    insp["GRADE"] = insp["GRADE"].astype(str).str.strip()
    insp["DBA"] = insp["DBA"].astype(str).str.strip()

    valid_boros = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
    insp = insp[insp["BORO"].isin(valid_boros)].copy()
    insp = insp.dropna(subset=["CAMIS", "INSPECTION DATE", "DBA"])

    latest_dates = (
        insp.groupby("CAMIS")["INSPECTION DATE"]
        .max()
        .reset_index()
        .rename(columns={"INSPECTION DATE": "LATEST_DATE"})
    )

    insp_latest = insp.merge(latest_dates, on="CAMIS", how="left")
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

    insp_names = restaurant_level["name_clean"].tolist()
    insp_lookup = restaurant_level.set_index("name_clean")

    matched_rows = []

    for _, row in ta.iterrows():
        result = process.extractOne(
            row["title_clean"],
            insp_names,
            scorer=fuzz.token_sort_ratio,
        )

        if result and result[1] >= MATCH_THRESHOLD:
            matched_name = result[0]
            match_score = result[1]
            insp_info = insp_lookup.loc[matched_name]

            if isinstance(insp_info, pd.DataFrame):
                insp_info = insp_info.iloc[0]

            matched_rows.append({
                "ta_title": row["Title"],
                "comment": row["comment_clean"],
                "sentiment": row["sentiment"],
                "category": row["primary_category"],
                "insp_name": insp_info["name"],
                "grade": insp_info["grade"],
                "borough": insp_info["borough"],
                "cuisine": insp_info["cuisine"],
                "insp_score": insp_info["score"],
                "match_score": match_score,
            })

    merged = pd.DataFrame(matched_rows)
    merged = merged[merged["grade"].isin(["A", "B", "C"])].copy()
    merged["borough"] = merged["borough"].astype(str).str.strip().str.title()

    return merged

def get_word_freq(texts):
    words = []
    for text in texts:
        tokens = re.findall(r"[a-z]+", str(text).lower())
        words.extend([w for w in tokens if w not in STOPWORDS and len(w) > 2])
    return dict(Counter(words))


def build_sentiment_hist(ta):
    fig = px.histogram(
        ta,
        x="sentiment",
        nbins=50,
        template="plotly_white",
        title="Distribution of Sentiment Across All TripAdvisor Snippets",
        color_discrete_sequence=["#3498db"],
    )
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        height=430,
        xaxis_title="VADER Compound Sentiment",
        yaxis_title="Count",
        margin=dict(l=30, r=20, t=60, b=20),
    )
    return fig


def build_category_sentiment_chart(ta, top_n=10):
    summary = (
        ta.groupby("primary_category", as_index=False)
        .agg(avg_sentiment=("sentiment", "mean"), n=("Title", "count"))
        .sort_values("n", ascending=False)
        .head(top_n)
        .sort_values("avg_sentiment", ascending=True)
    )

    fig = px.bar(
        summary,
        x="avg_sentiment",
        y="primary_category",
        orientation="h",
        color="avg_sentiment",
        color_continuous_scale=["#dce8ff", "#7da7ff", "#3557b7"],
        template="plotly_white",
        title="Average Review Sentiment by Top Review Categories",
    )

    fig.update_layout(
        height=470,
        xaxis_title="Average Sentiment",
        yaxis_title="Primary TripAdvisor Category",
        coloraxis_showscale=False,
        margin=dict(l=40, r=20, t=60, b=20),
    )

    return fig, summary

def build_borough_grade_heatmap(merged):
    heatmap = (
        merged.groupby(["borough", "grade"])["sentiment"]
        .mean()
        .unstack("grade")
        .reindex(columns=GRADE_ORDER)
    )

    boro_order = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
    heatmap = heatmap.reindex([b for b in boro_order if b in heatmap.index])

    counts = (
        merged.groupby(["borough", "grade"])
        .size()
        .unstack("grade")
        .reindex(columns=GRADE_ORDER)
        .fillna(0)
        .astype(int)
    )
    counts = counts.reindex([b for b in boro_order if b in counts.index])

    text = []
    for i in range(len(heatmap.index)):
        row_text = []
        for j in range(len(heatmap.columns)):
            val = heatmap.iloc[i, j]
            n = counts.iloc[i, j]
            if pd.notna(val):
                row_text.append(f"{val:.2f}<br>(n={n})")
            else:
                row_text.append("n/a")
        text.append(row_text)

    # 关键：Plotly heatmap 默认把第一个 y 放在下面，所以这里反转
    y_labels = heatmap.index.tolist()[::-1]
    z_values = heatmap.values[::-1]
    text_values = text[::-1]

    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=[f"Grade {c}" for c in heatmap.columns],
            y=y_labels,
            colorscale="Blues",
            zmin=-0.2,
            zmax=0.9,
            text=text_values,
            texttemplate="%{text}",
            textfont={"size": 12},
            hovertemplate="Borough: %{y}<br>%{x}<br>Mean Sentiment: %{z:.2f}<extra></extra>",
            colorbar=dict(title="Mean VADER Sentiment"),
        )
    )

    fig.update_layout(
        title="Review Sentiment: Borough × Grade",
        template="plotly_white",
        height=520,
        xaxis_title="Inspection Grade",
        yaxis_title="Borough",
        margin=dict(l=40, r=20, t=60, b=20),
    )

    return fig, heatmap

def build_wordcloud_figure(merged):
    grade_a_texts = merged[merged["grade"] == "A"]["comment"].tolist()
    grade_bc_texts = merged[merged["grade"].isin(["B", "C"])]["comment"].tolist()

    freq_a = get_word_freq(grade_a_texts)
    freq_bc = get_word_freq(grade_bc_texts)

    wc_a = WordCloud(
        width=800,
        height=600,
        background_color="white",
        colormap="Greens",
        max_words=80,
        relative_scaling=0.5,
        min_font_size=12,
        prefer_horizontal=0.9,
        random_state=42,
    ).generate_from_frequencies(freq_a)

    wc_bc = WordCloud(
        width=800,
        height=600,
        background_color="white",
        colormap="Reds",
        max_words=80,
        relative_scaling=0.5,
        min_font_size=12,
        prefer_horizontal=0.9,
        random_state=42,
    ).generate_from_frequencies(freq_bc)

    fig, (ax_a, ax_bc) = plt.subplots(1, 2, figsize=(16, 7))

    ax_a.imshow(wc_a, interpolation="bilinear")
    ax_a.axis("off")
    ax_a.set_title(
        f"Grade A Restaurants\n(n={len(grade_a_texts)} reviews)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    ax_bc.imshow(wc_bc, interpolation="bilinear")
    ax_bc.axis("off")
    ax_bc.set_title(
        f"Grade B / C Restaurants\n(n={len(grade_bc_texts)} reviews)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    fig.suptitle(
        "Most Common Review Words: Grade A vs. Grade B/C",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    return fig


def sentiment_takeaway(ta):
    return (
        f"Public review sentiment is heavily skewed positive, with an average VADER score of "
        f"**{ta['sentiment'].mean():.3f}**. Strongly negative reviews are relatively rare, "
        "which suggests a ceiling effect in public review data."
    )


def category_takeaway(summary):
    if summary.empty:
        return "Not enough data for this chart."

    best = summary.iloc[-1]
    worst = summary.iloc[0]

    return (
        f"Review sentiment varies somewhat by category. Among the most common categories shown, "
        f"**{best['primary_category']}** has the highest average sentiment, while "
        f"**{worst['primary_category']}** has the lowest."
    )


def heatmap_takeaway(heatmap):
    grade_a = heatmap["A"].dropna()

    if grade_a.empty:
        return (
            "Public review sentiment varies by borough and inspection grade, but the sample is uneven across cells."
        )

    top_boro = grade_a.idxmax()
    low_boro = grade_a.idxmin()

    return (
        f"Among Grade A restaurants, **{top_boro}** has the highest mean sentiment "
        f"({grade_a.loc[top_boro]:.2f}), while **{low_boro}** is lower "
        f"({grade_a.loc[low_boro]:.2f}). Across boroughs, Grade B sentiment is often comparable to or higher "
        "than Grade A, suggesting a disconnect between hygiene-based grading and diner-perceived experience."
    )


def wordcloud_takeaway():
    return (
        'Both groups share the same top words — **"great", "good", "best", "delicious", and "nice"** '
        "dominate in reviews of Grade A and Grade B/C restaurants alike. No clear negative or hygiene-related "
        'terms such as **"dirty", "rude", or "wait"** appear prominently. Grade B/C reviews also feature words '
        'like **"find", "gem", "hidden", and "new"**, suggesting these restaurants are often framed as discoveries '
        "rather than flawed establishments."
    )


def show_public_reviews():
    st.title("🧾 Public Review Insights")

    ta = load_public_reviews()
    merged = build_merged_data()

    with st.sidebar:
        st.markdown("## Public Review Filters")

        category_options = ["All"] + sorted(ta["primary_category"].dropna().unique().tolist())
        selected_category = st.selectbox("Category", category_options, index=0)

        borough_options = ["All"] + BORO_ORDER
        selected_borough = st.selectbox("Borough for matched review analysis", borough_options, index=0)

    filtered_ta = ta.copy()
    if selected_category != "All":
        filtered_ta = filtered_ta[filtered_ta["primary_category"] == selected_category].copy()

    filtered_merged = merged.copy()
    if selected_borough != "All":
        filtered_merged = filtered_merged[filtered_merged["borough"] == selected_borough].copy()

    if filtered_ta.empty:
        st.warning("No public review records available for the selected category.")
        return

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            fig1 = build_sentiment_hist(filtered_ta)
            st.plotly_chart(fig1, use_container_width=True, config=PLOT_CONFIG)
            st.markdown(f"**Key takeaway:** {sentiment_takeaway(filtered_ta)}")

    with col2:
        with st.container(border=True):
            fig2, category_summary = build_category_sentiment_chart(filtered_ta)
            st.plotly_chart(fig2, use_container_width=True, config=PLOT_CONFIG)
            st.markdown(f"**Key takeaway:** {category_takeaway(category_summary)}")

    if filtered_merged.empty:
        st.warning("No matched inspection-review records available for the selected borough.")
        return

    with st.container(border=True):
        fig3, heatmap = build_borough_grade_heatmap(filtered_merged)
        st.plotly_chart(fig3, use_container_width=True, config=PLOT_CONFIG)
        st.markdown(f"**Key takeaway:** {heatmap_takeaway(heatmap)}")

    with st.container(border=True):
        wc_fig = build_wordcloud_figure(filtered_merged)
        st.pyplot(wc_fig, use_container_width=True)
        st.markdown(f"**Key takeaway:** {wordcloud_takeaway()}")


show_public_reviews()
