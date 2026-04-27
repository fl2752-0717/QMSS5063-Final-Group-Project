#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2026/4/14 09:22
# @Author  : Jennifer
# @Software: PyCharm
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2026/4/14 09:09
# @Software: PyCharm
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import streamlit as st
from PIL import Image
import os

st.set_page_config(page_title="Overview", layout="wide")

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


def load_image(image_path):
    if os.path.exists(image_path):
        try:
            return Image.open(image_path)
        except Exception as e:
            st.error(f"无法加载图片: {e}")
            return None
    return None


LOCAL_IMAGE_PATH = "pic.png"


def show_overview():
    with st.container(border=True):
        col = st.columns(2)

        with col[0]:
            st.markdown("""
            <div style="line-height: 0.8; margin: 0; padding: 0;">
                <h1 style="margin: 0; padding: 0;">Mapping</h1>
                <h1 style="margin: 0; padding: 0;">Restaurant Quality</h1>
                <h1 style="margin: 0; padding: 0;">Across NYC</h1>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("")
            st.markdown("""
            Explore the intersection of official health inspections and public reviews across New York City's diverse restaurant landscape. Discover patterns in quality, compliance, and customer satisfaction from Manhattan to the Bronx.
            """)

            st.page_link(
                "pages/1_📝Official Inspections.py",
                label="Explore official inspections →",
                icon="📝"
            )

        with col[1]:
            img = load_image(LOCAL_IMAGE_PATH)
            if img:
                st.image(img, use_container_width=True)
            else:
                st.info(f"💡 未找到本地图片，请确认图片路径：{LOCAL_IMAGE_PATH}")

    with st.container(border=True):
        st.markdown("## 📊 About the Project")
        st.markdown("""
        This project investigates the relationship between official restaurant inspection outcomes and public review patterns across New York City, exploring how regulatory compliance and customer satisfaction align or diverge across neighborhoods and cuisine types.
        """)

        st.markdown("### Research Questions")
        st.markdown("""
        - How do restaurant inspection outcomes vary across New York City?
        - Are some cuisine categories or neighborhoods associated with systematically better or worse inspection outcomes?
        - Does the language used in public reviews align with official inspection outcomes?
        """)

    with st.container(border=True):
        st.markdown("## 📁 Data Sources")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            ### 🏛️ NYC Inspection Data
            - NYC DOHMH restaurant inspection records
            - Inspection scores, grades, dates, and boroughs
            - Violation descriptions and critical flags
            - Restaurant cuisine, location, and CAMIS ID
            """)

        with col2:
            st.markdown("""
            ### 🗺️ Geographic Data
            - NYC borough boundary GeoJSON
            - Restaurant latitude and longitude
            - Borough-level spatial aggregation
            - Interactive choropleth and point maps
            """)

        with col3:
            st.markdown("""
            ### 🧾 Public Review Data
            - TripAdvisor restaurant review dataset
            - Review text and restaurant names
            - VADER sentiment scores
            - Fuzzy matching between reviews and inspection records
            """)

        st.markdown("""
        Data were cleaned, standardized, aggregated, and matched across sources to compare **official inspection outcomes**, **spatial patterns**, and **public review sentiment**.
        """)


show_overview()
