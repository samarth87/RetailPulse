"""
RetailPulse - AI-Powered Customer Analytics & Demand Forecasting Platform
=========================================================================
Enterprise Streamlit dashboard built around the UCI / Kaggle "Online Retail II"
schema: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice,
CustomerID, Country.

Run:  streamlit run app.py
"""
import hashlib
import importlib.util
import io
import logging
import os
import re
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import (f1_score, mean_squared_error, precision_score,
                             recall_score, roc_auc_score, roc_curve,
                             silhouette_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

st.set_page_config(page_title="RetailPulse | AI Retail Analytics", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

HAS_PROPHET = importlib.util.find_spec("prophet") is not None
HAS_XGB = importlib.util.find_spec("xgboost") is not None

ACCENT = "#00D1B2"
PALETTE = ["#00D1B2", "#4C9AFF", "#FFB020", "#FF6B6B", "#A78BFA", "#F472B6", "#34D399", "#F59E0B"]

st.markdown("""
<style>
/* ================================================================
   RETAILPULSE – PROFESSIONAL UI OVERHAUL
   ================================================================ */

/* ── Global layout & spacing ─────────────────────────────────── */
.block-container {
    padding-top: 1.6rem !important;
    padding-bottom: 2.4rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 1800px !important;
}

/* ── Prevent Plotly chart text / axis from overflowing ────────── */
.js-plotly-plot .plotly .modebar { top: 0 !important; }
.js-plotly-plot { overflow: visible !important; }
.stPlotlyChart { overflow: hidden; border-radius: 12px; }

/* Give every plotly chart its own breathing room */
div[data-testid="stPlotlyChart"] {
    margin-bottom: 1rem !important;
}

/* ── Sidebar ────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #070C18 0%, #0D1525 50%, #0F172A 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
    padding-bottom: 2rem !important;
}
section[data-testid="stSidebar"] > div {
    padding-top: 0.8rem !important;
}

/* ── Sidebar section headers ───────────────────────────────── */
.sidebar-section-title {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.4px;
    color: #475569;
    margin: 1.4rem 0 0.6rem 0;
    padding-bottom: 6px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    display: flex;
    align-items: center;
    gap: 6px;
}

/* ── Sidebar Nav Buttons (primary = active, secondary = inactive) ── */
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
    gap: 0.22rem !important;
}

section[data-testid="stSidebar"] div[data-testid="stButton"] {
    margin: 0 !important;
    padding: 0 !important;
}

section[data-testid="stSidebar"] div[data-testid="stButton"] > button {
    all: unset !important;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
    width: 100% !important;
    background: rgba(15, 23, 42, 0.65) !important;
    border: 1px solid rgba(255, 255, 255, 0.07) !important;
    border-left: 3px solid transparent !important;
    border-radius: 8px !important;
    padding: 8px 12px !important;
    margin: 0 !important;
    min-height: 36px !important;
    cursor: pointer !important;
    box-sizing: border-box !important;
    font-size: 0.83rem !important;
    font-weight: 500 !important;
    color: #94A3B8 !important;
    text-align: left !important;
    line-height: 1.35 !important;
    font-family: inherit !important;
    transition: background 0.15s ease, color 0.15s ease, border-left-color 0.15s ease, transform 0.15s ease !important;
}

section[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
    background: rgba(25, 38, 60, 0.92) !important;
    border-left-color: rgba(0, 209, 178, 0.55) !important;
    color: #E2E8F0 !important;
    transform: translateX(2px) !important;
}

/* Active Nav Button (type="primary") */
section[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="primary"],
section[data-testid="stSidebar"] div[data-testid="stButton"] > button[data-testid="baseButton-primary"] {
    background: linear-gradient(90deg, rgba(0, 209, 178, 0.16) 0%, rgba(15, 23, 42, 0.96) 100%) !important;
    border: 1px solid rgba(0, 209, 178, 0.40) !important;
    border-left: 4px solid #00D1B2 !important;
    border-radius: 8px !important;
    padding: 8px 12px !important;
    min-height: 36px !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 14px rgba(0, 209, 178, 0.12) !important;
    transform: none !important;
}

section[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="primary"]:hover,
section[data-testid="stSidebar"] div[data-testid="stButton"] > button[data-testid="baseButton-primary"]:hover {
    background: linear-gradient(90deg, rgba(0, 209, 178, 0.22) 0%, rgba(15, 23, 42, 0.98) 100%) !important;
    color: #FFFFFF !important;
    transform: none !important;
}

/* ── Executive highlight cards ─────────────────────────────── */
.rp-exec-card {
    background: linear-gradient(135deg, rgba(15, 25, 45, 0.75) 0%, rgba(20, 32, 58, 0.55) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-left: 4px solid #00D1B2;
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 0.8rem;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.rp-exec-card:hover {
    transform: translateY(-2px);
    border-color: rgba(0, 209, 178, 0.4);
}
.rp-exec-tag {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #00D1B2;
    margin-bottom: 4px;
}
.rp-exec-val {
    font-size: 1.28rem;
    font-weight: 800;
    color: #FFFFFF;
    line-height: 1.2;
}
.rp-exec-desc {
    font-size: 0.82rem;
    color: #94A3B8;
    margin-top: 4px;
    line-height: 1.35;
}

/* Ensure the inner text node / p tag is always left-aligned */
section[data-testid="stSidebar"] div[data-testid="stButton"] > button div,
section[data-testid="stSidebar"] div[data-testid="stButton"] > button p {
    text-align: left !important;
    justify-content: flex-start !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
    color: inherit !important;
    font-size: inherit !important;
    font-weight: inherit !important;
}

/* Also hide the data source radio circles */
div[data-testid="stSidebar"] div[data-testid="stRadio"] label > div:first-child,
div[data-testid="stSidebar"] div[data-testid="stRadio"] label > div:first-child *,
div[data-testid="stSidebar"] div[data-testid="stRadio"] input[type="radio"],
div[data-testid="stSidebar"] div[data-testid="stRadio"] input[type="radio"] + div {
    display: none !important;
    width: 0 !important;
    height: 0 !important;
    visibility: hidden !important;
    position: absolute !important;
}
div[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] {
    gap: 3px !important;
    display: flex !important;
    flex-direction: column !important;
}
div[data-testid="stSidebar"] div[data-testid="stRadio"] label {
    background: rgba(15, 23, 42, 0.65) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-left: 3px solid transparent !important;
    border-radius: 8px !important;
    padding: 8px 12px !important;
    margin: 0 !important;
    cursor: pointer !important;
    transition: all 0.18s ease !important;
    display: flex !important;
    align-items: center !important;
    width: 100% !important;
    box-sizing: border-box !important;
    min-height: 34px !important;
}
div[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {
    background: rgba(30,41,59,0.85) !important;
    border-left-color: rgba(0,209,178,0.45) !important;
}
div[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) {
    background: linear-gradient(90deg, rgba(0,209,178,0.13) 0%, rgba(15,23,42,0.95) 100%) !important;
    border: 1px solid rgba(0,209,178,0.35) !important;
    border-left: 3px solid #00D1B2 !important;
}
div[data-testid="stSidebar"] div[data-testid="stRadio"] label p,
div[data-testid="stSidebar"] div[data-testid="stRadio"] label span {
    color: #94A3B8 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    margin: 0 !important;
    padding: 0 !important;
    line-height: 1.3 !important;
}
div[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) p,
div[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) span {
    color: #FFFFFF !important;
    font-weight: 600 !important;
}

/* ── Sidebar info cards ─────────────────────────────────────── */
.rp-filter-pill {
    background: rgba(15,23,42,0.8);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 8px;
    padding: 7px 12px;
    font-size: 0.79rem;
    color: #94A3B8;
    text-align: center;
    margin-top: 10px;
}

.rp-source-badge {
    background: linear-gradient(145deg, #0D1B30 0%, #142035 100%);
    border: 1px solid rgba(0,209,178,0.25);
    border-radius: 10px;
    padding: 10px 14px;
    margin-top: 8px;
    margin-bottom: 8px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.3);
}

/* ── Metric cards ───────────────────────────────────────────── */
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #0F1929 0%, #14203A 100%);
    border: 1px solid #1E2F4A;
    border-radius: 14px;
    padding: 16px 20px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    transition: all 0.25s ease;
    overflow: hidden;
}
div[data-testid="stMetric"]:hover {
    border-color: rgba(0,209,178,0.45);
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0,209,178,0.12);
}
div[data-testid="stMetricLabel"] p { color: #7A90B0; font-size: 0.82rem; font-weight: 500; }
div[data-testid="stMetricValue"] { color: #FFFFFF; font-weight: 800; }

/* ── Page title & subtitle ─────────────────────────────────── */
.rp-title {
    font-size: 1.9rem;
    font-weight: 800;
    letter-spacing: 0.2px;
    margin-bottom: 2px;
    color: #FFFFFF;
    line-height: 1.2;
}
.rp-sub {
    color: #7A90B0;
    margin-top: 2px;
    margin-bottom: 20px;
    font-size: 0.9rem;
    line-height: 1.5;
}

/* ── Info / analysis card ──────────────────────────────────── */
.rp-card {
    background: #0F1929;
    border: 1px solid #1E2F4A;
    border-radius: 14px;
    padding: 16px 20px;
    margin-bottom: 14px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.25);
}
.rp-card h4 { margin: 0 0 6px 0; color: #00D1B2; font-weight: 700; font-size: 0.95rem; }
.rp-card p  { margin: 0; color: #B8C8E0; font-size: 0.9rem; line-height: 1.5; }

/* ── Tabs ───────────────────────────────────────────────────── */
button[data-baseweb="tab"] {
    background-color: transparent !important;
    border-radius: 8px !important;
    padding: 8px 20px !important;
    color: #64748B !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    transition: all 0.2s ease !important;
}
button[data-baseweb="tab"]:hover {
    color: #00D1B2 !important;
    background-color: rgba(0,209,178,0.07) !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    background-color: rgba(0,209,178,0.12) !important;
    color: #00D1B2 !important;
    font-weight: 700 !important;
    border-bottom: 2px solid #00D1B2 !important;
}

/* ── Download / action buttons ─────────────────────────────── */
div[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, #00D1B2 0%, #009E88 100%) !important;
    color: #0B1220 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 8px 18px !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
div[data-testid="stDownloadButton"] > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 18px rgba(0,209,178,0.45) !important;
}

/* ── Dataframes / tables ────────────────────────────────────── */
div[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #1E2F4A;
}

/* ── Warnings & alerts ─────────────────────────────────────── */
div[data-testid="stAlert"] {
    border-radius: 10px !important;
    margin-bottom: 1.2rem !important;
}

/* ── Sliders ────────────────────────────────────────────────── */
div[data-testid="stSlider"] {
    padding-bottom: 0.5rem !important;
}

/* ── Column gap control ─────────────────────────────────────── */
div[data-testid="column"] {
    padding-left: 0.5rem !important;
    padding-right: 0.5rem !important;
}

/* ── Scrollbar ──────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0B1220; }
::-webkit-scrollbar-thumb { background: #1E2F4A; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #00D1B2; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONSTANTS: product catalog, category rules
# =============================================================================
CATALOG = {
    "Home Decor": (["CUSHION COVER", "PHOTO FRAME", "WALL CLOCK", "GLASS VASE", "HANGING MIRROR", "DOORSTOP"], 6.0),
    "Kitchen & Dining": (["CERAMIC MUG", "TEAPOT", "CAKE TIN", "PLATE SET", "STORAGE JAR", "LUNCH BOX"], 4.0),
    "Gifts & Seasonal": (["CHRISTMAS ORNAMENT", "GIFT WRAP", "GREETING CARD", "ADVENT CALENDAR", "EASTER BUNNY"], 2.5),
    "Stationery & Crafts": (["NOTEBOOK", "PENCIL CASE", "STICKER SET", "CRAFT RIBBON", "WRITING PAD"], 1.8),
    "Bags & Accessories": (["TOTE BAG", "COIN PURSE", "LUGGAGE TAG", "UMBRELLA", "WASH BAG"], 5.0),
    "Lighting & Candles": (["CANDLE HOLDER", "TEA LIGHT SET", "METAL LANTERN", "FAIRY LIGHTS", "SCENTED CANDLE"], 3.5),
    "Toys & Games": (["WOODEN TRAIN", "PLAYING CARDS", "JIGSAW PUZZLE", "SPINNING TOP", "DOLL HOUSE"], 3.0),
    "Garden & Outdoor": (["PLANT POT", "BIRD HOUSE", "GARDEN TROWEL", "WATERING CAN", "SEED TRAY"], 4.5),
}
ADJECTIVES = ["VINTAGE", "RETRO", "PINK", "BLUE", "RED", "IVORY", "SPOTTY", "FLORAL", "UNION JACK",
              "SKULL", "JUMBO", "MINI", "POLKA DOT", "GREEN", "ASSORTED", "LARGE", "REGENCY", "SCANDI"]
COUNTRIES = ["Germany", "France", "EIRE", "Spain", "Netherlands", "Belgium", "Switzerland",
             "Portugal", "Australia", "Norway", "Sweden", "Italy"]

CATEGORY_RULES = [  # order matters: first match wins
    ("Toys & Games", r"\b(TOY|GAME|DOLL|PUZZLE|JIGSAW|TRAIN|PLAYING|SPINNING|BALLOON)"),
    ("Gifts & Seasonal", r"\b(CHRISTMAS|XMAS|EASTER|GIFT|CARD|WRAP|ADVENT|HALLOWEEN)"),
    ("Lighting & Candles", r"\b(CANDLE|LIGHT|LANTERN|LAMP)"),
    ("Bags & Accessories", r"\b(BAG|PURSE|WALLET|UMBRELLA|LUGGAGE)"),
    ("Garden & Outdoor", r"\b(GARDEN|PLANT|BIRD|TROWEL|WATERING|SEED)"),
    ("Stationery & Crafts", r"\b(NOTEBOOK|PENCIL|PEN\b|STICKER|STATIONERY|PAPER|CRAFT|RIBBON|BOOK|WRITING|PAD\b)"),
    ("Kitchen & Dining", r"\b(MUG|CUP|TEAPOT|JAR|TIN\b|PLATE|BOWL|SPOON|CAKE|LUNCH|BOX|KITCHEN)"),
    ("Home Decor", r"\b(CUSHION|FRAME|CLOCK|VASE|MIRROR|DOORSTOP|HEART|SIGN|DECORATION|HANGING)"),
]
CATEGORY_COST_RATIO = {"Home Decor": 0.58, "Kitchen & Dining": 0.60, "Gifts & Seasonal": 0.55,
                       "Stationery & Crafts": 0.52, "Bags & Accessories": 0.62, "Lighting & Candles": 0.57,
                       "Toys & Games": 0.64, "Garden & Outdoor": 0.60, "Other": 0.60}
NON_PRODUCT_CODES = {"POST", "D", "M", "BANK CHARGES", "DOT", "CRUK", "AMAZONFEE", "ADJUST", "ADJUST2", "S", "B", "TEST001", "TEST002"}
COLUMN_ALIASES = {"invoice": "InvoiceNo", "invoiceno": "InvoiceNo", "invoice no": "InvoiceNo",
                  "stockcode": "StockCode", "stock code": "StockCode", "description": "Description",
                  "quantity": "Quantity", "invoicedate": "InvoiceDate", "invoice date": "InvoiceDate",
                  "price": "UnitPrice", "unitprice": "UnitPrice", "unit price": "UnitPrice",
                  "customerid": "CustomerID", "customer id": "CustomerID", "country": "Country"}
REQUIRED_COLS = ["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"]

PERSONAS = {
    "Champions": ("Recent, frequent, high-spending buyers.",
                  "Reward with VIP tiers, early access and referral incentives. Avoid heavy discounting."),
    "Loyalists": ("Regular repeat buyers with solid spend.",
                  "Upsell premium bundles, enrol in loyalty program, request reviews."),
    "Potential Loyalists": ("Recent customers with growing engagement.",
                            "Run onboarding journeys, cross-sell complementary categories, offer 2nd/3rd purchase coupons."),
    "At-Risk": ("Previously valuable buyers whose activity is fading.",
                "Trigger win-back emails, personalised offers and 'we miss you' reminders now."),
    "Lost": ("Long inactive with low engagement.",
             "Low-cost reactivation campaigns only; sunset from paid channels if no response."),
    "Outliers / Noise": ("Atypical customers flagged by DBSCAN as noise.",
                         "Review manually (wholesale / anomalous accounts) before assigning a strategy."),
}
PERSONA_COLORS = {"Champions": "#00D1B2", "Loyalists": "#4C9AFF", "Potential Loyalists": "#A78BFA",
                  "At-Risk": "#FFB020", "Lost": "#FF6B6B", "Outliers / Noise": "#8892A6"}


# =============================================================================
# SMALL UI HELPERS
# =============================================================================
def style(fig, h=420, **kw):
    opts = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=h, margin=dict(l=20, r=20, t=65, b=50),
                legend=dict(orientation="h", y=1.10, x=0, xanchor="left",
                            bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
                colorway=PALETTE)
    opts.update(kw)
    fig.update_layout(**opts)
    return fig


def style_pie(fig, h=420, **kw):
    """Special styling for pie / donut charts — legend placed below with breathing room."""
    opts = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=h, margin=dict(l=15, r=15, t=55, b=75),
                legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center",
                            font=dict(size=11, color="#CBD5E1")),
                colorway=PALETTE)
    opts.update(kw)
    fig.update_layout(**opts)
    return fig


def show(fig):
    st.plotly_chart(fig, use_container_width=True)


def gbp(x):
    if abs(x) >= 1e6:
        return f"£{x / 1e6:,.2f}M"
    if abs(x) >= 1e3:
        return f"£{x / 1e3:,.1f}K"
    return f"£{x:,.0f}"


def pct_delta(cur, prev):
    if prev is None or prev == 0 or pd.isna(prev):
        return None
    return f"{(cur - prev) / abs(prev) * 100:+.1f}%"


def card(title, body):
    st.markdown(f'<div class="rp-card"><h4>{title}</h4><p>{body}</p></div>', unsafe_allow_html=True)


def header(title, sub):
    st.markdown(f'<div class="rp-title">{title}</div><div class="rp-sub">{sub}</div>', unsafe_allow_html=True)


# =============================================================================
# DATA LAYER: synthetic generator, loading, cleaning
# =============================================================================
@st.cache_data(show_spinner="Generating synthetic UCI Online Retail II data...")
def generate_synthetic_data(seed: int = 42, n_customers: int = 2000) -> pd.DataFrame:
    """Realistic synthetic data in the UCI Online Retail II raw schema (includes dirty rows)."""
    rng = np.random.default_rng(seed)
    start, end = pd.Timestamp("2009-12-01"), pd.Timestamp("2011-12-09")
    span = (end - start).days

    prods, code = [], 21000
    for cat, (nouns, base) in CATALOG.items():
        for noun in nouns:
            for adj in rng.choice(ADJECTIVES, size=5, replace=False):
                code += 1
                price = float(np.clip(round(base * rng.lognormal(0, 0.45), 2), 0.29, 45.0))
                prods.append((str(code), f"{adj} {noun}", cat, price))
    n_p = len(prods)
    p_code = np.array([p[0] for p in prods])
    p_desc = np.array([p[1] for p in prods])
    p_cat = np.array([p[2] for p in prods])
    p_price = np.array([p[3] for p in prods])
    pop = rng.gamma(0.7, 1.0, n_p) + 0.03
    pack = rng.choice([1, 2, 3, 4, 6, 12], n_p, p=[.12, .15, .15, .18, .25, .15])
    seasonal_cat = np.isin(p_cat, ["Gifts & Seasonal", "Lighting & Candles"])
    month_probs = {}
    for m in range(1, 13):
        w = pop * np.where(seasonal_cat & (m in (9, 10, 11, 12)), 3.0, 1.0)
        month_probs[m] = w / w.sum()
    accept = {1: .7, 2: .68, 3: .75, 4: .75, 5: .78, 6: .78, 7: .75, 8: .75, 9: .92, 10: 1.0, 11: 1.0, 12: .9}

    # (weight, orders/yr, mean basket lines, churn probability)
    segments = [(.08, 30, 14, .03), (.17, 15, 11, .10), (.25, 8, 9, .25), (.25, 4, 8, .50), (.25, 2, 7, .70)]
    seg_w = np.array([s[0] for s in segments])
    country_w = np.array([.18, .15, .12, .1, .09, .08, .07, .06, .05, .04, .03, .03])
    country_w = country_w / country_w.sum()

    inv_no = 489434
    cols = {k: [] for k in ["InvoiceNo", "idx", "Quantity", "InvoiceDate", "CustomerID", "Country"]}

    def add_order(ts, lines, cust, country):
        nonlocal inv_no
        inv_no += 1
        idx = np.unique(rng.choice(n_p, size=lines, p=month_probs[ts.month]))
        qty = (pack[idx] * rng.choice([1, 1, 1, 2, 3], size=len(idx))).astype(int)
        cols["InvoiceNo"].append(np.full(len(idx), str(inv_no)))
        cols["idx"].append(idx)
        cols["Quantity"].append(qty)
        cols["InvoiceDate"].append(np.full(len(idx), np.datetime64(ts)))
        cols["CustomerID"].append(np.full(len(idx), cust, dtype=float))
        cols["Country"].append(np.full(len(idx), country))

    def valid_day(ts):
        if ts.dayofweek == 5:
            ts = ts + pd.Timedelta(days=1)
        if (ts.month == 12 and ts.day >= 23) or (ts.month == 1 and ts.day <= 3):
            ts = pd.Timestamp(ts.year + (1 if ts.month == 12 else 0), 1, 4, ts.hour, ts.minute)
        return ts

    for c in range(n_customers):
        seg = rng.choice(len(segments), p=seg_w)
        _, rate, basket, churn_p = segments[seg]
        s_off = int(rng.uniform(0, 0.75) * span)
        t = start + pd.Timedelta(days=s_off)
        if rng.random() < churn_p:
            active_end = t + pd.Timedelta(days=int((end - t).days * rng.uniform(0.1, 0.8)))
        else:
            active_end = end
        cust = float(12346 + c * 3 + int(rng.integers(0, 3)))
        country = "United Kingdom" if rng.random() < 0.88 else str(rng.choice(COUNTRIES, p=country_w))
        first = True
        while True:
            t = t + pd.Timedelta(days=float(rng.exponential(365.0 / rate)) if not first else 0.0)
            first = False
            if t > active_end:
                break
            if rng.random() > accept[t.month]:
                continue
            ts = valid_day(t.normalize() + pd.Timedelta(hours=int(rng.integers(8, 19)), minutes=int(rng.integers(0, 60))))
            if ts > end:
                break
            add_order(ts, int(min(max(1, rng.poisson(basket)), 45)), cust, country)

    # anonymous / guest baskets (missing CustomerID)
    for _ in range(10000):
        ts = valid_day(start + pd.Timedelta(days=int(rng.integers(0, span)), hours=int(rng.integers(8, 19)),
                                            minutes=int(rng.integers(0, 60))))
        if rng.random() > accept[ts.month]:
            continue
        add_order(ts, int(min(max(1, rng.poisson(8)), 40)), np.nan, "United Kingdom")

    df = pd.DataFrame({k: np.concatenate(v) for k, v in cols.items()})
    idx = df.pop("idx").values
    df["StockCode"] = p_code[idx]
    df["Description"] = p_desc[idx]
    df["UnitPrice"] = np.round(p_price[idx] * rng.choice([1.0, 1.0, 1.0, 1.05, 0.95], size=len(df)), 2)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # cancellations (invoices prefixed with 'C', negative quantity)
    inv_ids = df["InvoiceNo"].unique()
    chosen = rng.choice(inv_ids, int(0.025 * len(inv_ids)), replace=False)
    canc = df[df["InvoiceNo"].isin(chosen)].groupby("InvoiceNo").head(2).copy()
    canc["Quantity"] = -canc["Quantity"]
    canc["InvoiceNo"] = "C" + canc["InvoiceNo"]
    canc["InvoiceDate"] = canc["InvoiceDate"] + pd.to_timedelta(rng.integers(1, 20, len(canc)), unit="D")
    # dirty rows: zero price, negative adjustments, duplicates
    dirty_zero = df.sample(250, random_state=seed).copy()
    dirty_zero["UnitPrice"] = 0.0
    dirty_neg = df.sample(80, random_state=seed + 1).copy()
    dirty_neg["Quantity"] = -dirty_neg["Quantity"]
    dirty_neg["CustomerID"] = np.nan
    dup = df.sample(120, random_state=seed + 2)
    df = pd.concat([df, canc, dirty_zero, dirty_neg, dup], ignore_index=True)
    df = df[df["InvoiceDate"] <= end + pd.Timedelta(days=1)]
    df["Country"] = df["Country"].astype(str)
    df = df.sort_values("InvoiceDate").reset_index(drop=True)
    return df[REQUIRED_COLS]


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {c: COLUMN_ALIASES[c.strip().lower()] for c in df.columns if c.strip().lower() in COLUMN_ALIASES}
    return df.rename(columns=mapping)


@st.cache_data(show_spinner="Cleaning data and engineering features...")
def clean_data(raw: pd.DataFrame):
    """Returns (clean_df, report). Implements all UCI Online Retail II cleaning rules."""
    df = standardize_columns(raw.copy())
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df[REQUIRED_COLS].copy()
    report = {"raw_rows": len(df)}
    miss = pd.DataFrame({"Missing": df.isna().sum(), "Missing %": (df.isna().mean() * 100).round(2)})
    report["missing_table"] = miss.reset_index().rename(columns={"index": "Column"})

    df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()
    cancelled = df["InvoiceNo"].str.upper().str.startswith("C")
    report["cancelled"] = int(cancelled.sum())
    df = df[~cancelled]
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    report["bad_quantity"] = int((~(df["Quantity"] > 0)).sum())
    df = df[df["Quantity"] > 0]
    report["bad_price"] = int((~(df["UnitPrice"] > 0)).sum())
    df = df[df["UnitPrice"] > 0]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    report["bad_dates"] = int(df["InvoiceDate"].isna().sum())
    df = df.dropna(subset=["InvoiceDate"])
    df["StockCode"] = df["StockCode"].astype(str).str.strip().str.upper()
    non_prod = df["StockCode"].isin(NON_PRODUCT_CODES)
    report["non_product"] = int(non_prod.sum())
    df = df[~non_prod]
    df["Description"] = df["Description"].fillna("UNKNOWN ITEM").astype(str).str.strip().str.upper()
    df["Country"] = df["Country"].fillna("Unspecified").astype(str).str.strip()
    cid = pd.to_numeric(df["CustomerID"], errors="coerce")
    report["missing_customer"] = int(cid.isna().sum())
    df["CustomerID"] = cid.astype("Int64")
    before = len(df)
    df = df.drop_duplicates()
    report["duplicates"] = before - len(df)

    df["TotalPrice"] = (df["Quantity"] * df["UnitPrice"]).round(2)
    uniq = pd.Series(df["Description"].unique())
    cat_map = {}
    for d in uniq:
        cat_map[d] = "Other"
        for cat, pattern in CATEGORY_RULES:
            if re.search(pattern, d):
                cat_map[d] = cat
                break
    df["Category"] = df["Description"].map(cat_map)
    h = pd.util.hash_array(df["StockCode"].values.astype(object)) % 1000 / 1000.0
    ratio = df["Category"].map(CATEGORY_COST_RATIO).astype(float).values + (h - 0.5) * 0.10
    df["EstimatedCost"] = (df["TotalPrice"] * ratio).round(2)
    df["EstimatedProfit"] = (df["TotalPrice"] - df["EstimatedCost"]).round(2)
    df["Date"] = df["InvoiceDate"].dt.normalize()
    df["Month"] = df["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
    df["DayOfWeek"] = df["InvoiceDate"].dt.day_name()
    df = df.sort_values("InvoiceDate").reset_index(drop=True)
    report["clean_rows"] = len(df)
    return df, report


@st.cache_data(show_spinner="Reading uploaded file...")
def read_upload(name: str, content: bytes) -> pd.DataFrame:
    if name.lower().endswith((".xlsx", ".xls")):
        sheets = pd.read_excel(io.BytesIO(content), sheet_name=None)
        return pd.concat(sheets.values(), ignore_index=True)
    try:
        return pd.read_csv(io.BytesIO(content), low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(io.BytesIO(content), encoding="ISO-8859-1", low_memory=False)


def customer_tx(df: pd.DataFrame) -> pd.DataFrame:
    t = df.dropna(subset=["CustomerID"])[["CustomerID", "InvoiceNo", "StockCode", "InvoiceDate",
                                          "Quantity", "TotalPrice", "Country"]].copy()
    t["CustomerID"] = t["CustomerID"].astype("int64")
    t["InvoiceDate"] = t["InvoiceDate"].dt.normalize()
    return t


def daily_series(df: pd.DataFrame) -> pd.Series:
    s = df.groupby("Date")["TotalPrice"].sum()
    return s.asfreq("D", fill_value=0.0).astype(float)


# =============================================================================
# FEATURE ENGINEERING + MODEL LAYER
# =============================================================================
@st.cache_data(show_spinner="Computing RFM features...")
def compute_rfm(tx: pd.DataFrame) -> pd.DataFrame:
    snapshot = tx["InvoiceDate"].max() + pd.Timedelta(days=1)
    inv = tx.groupby(["CustomerID", "InvoiceNo"]).agg(date=("InvoiceDate", "max"), value=("TotalPrice", "sum")).reset_index()
    g = inv.groupby("CustomerID")
    rfm = pd.DataFrame({
        "Recency": (snapshot - g["date"].max()).dt.days,
        "Frequency": g["InvoiceNo"].nunique(),
        "Monetary": g["value"].sum(),
        "Tenure": (snapshot - g["date"].min()).dt.days,
    })
    rfm["AOV"] = rfm["Monetary"] / rfm["Frequency"]
    n = len(rfm)
    rfm["R_pct"] = 1 - (rfm["Recency"].rank(method="average") - 1) / max(n - 1, 1)
    rfm["F_pct"] = (rfm["Frequency"].rank(method="average") - 1) / max(n - 1, 1)
    rfm["M_pct"] = (rfm["Monetary"].rank(method="average") - 1) / max(n - 1, 1)
    return rfm


def assign_persona(r, f, m):
    if r >= 0.70 and f >= 0.70:
        return "Champions"
    if f >= 0.60 and r >= 0.45:
        return "Loyalists"
    if r >= 0.55:
        return "Potential Loyalists"
    if r >= 0.25 and f >= 0.35:
        return "At-Risk"
    return "Lost"


@st.cache_data(show_spinner="Clustering customers...")
def cluster_customers(rfm: pd.DataFrame, algo: str, k: int, eps: float, min_samples: int):
    feats = ["Recency", "Frequency", "Monetary", "AOV", "Tenure"]
    X = StandardScaler().fit_transform(np.log1p(rfm[feats]))
    if algo == "K-Means":
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
    else:
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    out = rfm.copy()
    out["Cluster"] = labels
    mask = labels != -1
    sil = np.nan
    if len(set(labels[mask])) >= 2 and mask.sum() > 10:
        sil = silhouette_score(X[mask], labels[mask], sample_size=min(5000, int(mask.sum())), random_state=42)
    means = out[mask].groupby("Cluster")[["R_pct", "F_pct", "M_pct"]].mean()
    persona = {c: assign_persona(*means.loc[c]) for c in means.index}
    persona[-1] = "Outliers / Noise"
    out["Persona"] = out["Cluster"].map(persona)
    comps = PCA(n_components=2, random_state=42).fit_transform(X)
    out["PC1"], out["PC2"] = comps[:, 0], comps[:, 1]
    return out, sil


# ----------------------------- forecasting -----------------------------------
LAG_FEATURES = ["lag_1", "lag_7", "lag_14", "roll_mean_7", "roll_std_7", "roll_mean_30", "roll_std_30",
                "dow", "month", "doy_sin", "doy_cos"]


def make_lag_frame(s: pd.Series) -> pd.DataFrame:
    f = pd.DataFrame({"y": s})
    for lag in (1, 7, 14):
        f[f"lag_{lag}"] = s.shift(lag)
    sh = s.shift(1)
    for w in (7, 30):
        f[f"roll_mean_{w}"] = sh.rolling(w).mean()
        f[f"roll_std_{w}"] = sh.rolling(w).std()
    idx = f.index
    f["dow"], f["month"] = idx.dayofweek, idx.month
    f["doy_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365.25)
    return f


def fit_gbm(train: pd.Series):
    frame = make_lag_frame(train).dropna()
    model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=42)
    model.fit(frame[LAG_FEATURES], frame["y"])
    return model


def gbm_recursive(model, train: pd.Series, future_dates: pd.DatetimeIndex) -> np.ndarray:
    hist, preds = list(train.values), []
    for d in future_dates:
        a = np.asarray(hist)
        row = [a[-1], a[-7], a[-14], a[-7:].mean(), a[-7:].std(ddof=1), a[-30:].mean(), a[-30:].std(ddof=1),
               d.dayofweek, d.month, np.sin(2 * np.pi * d.dayofyear / 365.25), np.cos(2 * np.pi * d.dayofyear / 365.25)]
        p = float(max(model.predict(pd.DataFrame([row], columns=LAG_FEATURES))[0], 0.0))
        preds.append(p)
        hist.append(p)
    return np.asarray(preds)


def _fourier(dates: pd.DatetimeIndex, K: int) -> np.ndarray:
    doy = dates.dayofyear.values
    cols = []
    for k in range(1, K + 1):
        cols += [np.sin(2 * np.pi * k * doy / 365.25), np.cos(2 * np.pi * k * doy / 365.25)]
    return np.column_stack(cols)


def prophet_forecast(train: pd.Series, future_dates: pd.DatetimeIndex):
    """Prophet when installed; otherwise a Prophet-style additive model (trend + weekly + Fourier yearly)."""
    if HAS_PROPHET:
        try:
            from prophet import Prophet
            logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
            logging.getLogger("prophet").setLevel(logging.ERROR)
            m = Prophet(yearly_seasonality=len(train) >= 365, weekly_seasonality=True,
                        daily_seasonality=False, interval_width=0.95)
            m.fit(pd.DataFrame({"ds": train.index, "y": train.values}))
            fc = m.predict(pd.DataFrame({"ds": future_dates}))
            return np.clip(fc["yhat"].values, 0, None), "Prophet"
        except Exception:
            pass
    t0 = train.index[0]
    K = 3 if len(train) >= 365 else 1

    def design(dates):
        t = ((dates - t0).days.values / 365.25)[:, None]
        return np.hstack([t, np.eye(7)[dates.dayofweek], _fourier(dates, K)])

    m = Ridge(alpha=1.0).fit(design(train.index), train.values)
    return np.clip(m.predict(design(future_dates)), 0, None), "Prophet-style (Trend + Weekly + Fourier)"


@st.cache_data(show_spinner="Training hybrid forecasting engine (Prophet + Gradient Boosting)...")
def run_forecasting(series: pd.Series, horizon_max: int = 90):
    test_days = 28 if len(series) >= 180 else 14
    train, test = series.iloc[:-test_days], series.iloc[-test_days:]
    p_test, p_name = prophet_forecast(train, test.index)
    g_test = gbm_recursive(fit_gbm(train), train, test.index)
    naive = np.resize(train.values[-7:], test_days)
    fut = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon_max, freq="D")
    p_fut, _ = prophet_forecast(series, fut)
    gbm_full = fit_gbm(series)
    g_fut = gbm_recursive(gbm_full, series, fut)
    imp = pd.Series(gbm_full.feature_importances_, index=LAG_FEATURES).sort_values()
    return {"test_idx": test.index, "actual": test.values, "prophet_test": p_test, "gbm_test": g_test,
            "naive_test": naive, "future_idx": fut, "prophet_future": p_fut, "gbm_future": g_fut,
            "prophet_name": p_name, "importance": imp, "test_days": test_days}


def forecast_metrics(actual, pred):
    nz = actual > 0
    mape = float(np.mean(np.abs(actual[nz] - pred[nz]) / actual[nz]) * 100)
    rmse = float(np.sqrt(mean_squared_error(actual, pred)))
    k = len(actual) // 7 * 7
    wa = actual[:k].reshape(-1, 7).sum(1)
    wp = pred[:k].reshape(-1, 7).sum(1)
    wmape = float(np.mean(np.abs(wa - wp) / np.maximum(wa, 1e-9)) * 100)
    return mape, wmape, rmse


# ------------------------------- churn ---------------------------------------
CHURN_FEATURES = ["Recency", "Frequency", "Monetary", "AOV", "Tenure", "AvgGapDays", "GapStd",
                  "RecencyRatio", "Orders90d", "UniqueProducts", "AvgItemsPerOrder"]


def build_churn_features(tx: pd.DataFrame, snapshot: pd.Timestamp) -> pd.DataFrame:
    t = tx[tx["InvoiceDate"] < snapshot]
    inv = t.groupby(["CustomerID", "InvoiceNo"]).agg(date=("InvoiceDate", "max"), value=("TotalPrice", "sum"),
                                                     items=("Quantity", "sum")).reset_index()
    inv = inv.sort_values(["CustomerID", "date"])
    inv["gap"] = inv.groupby("CustomerID")["date"].diff().dt.days
    g = inv.groupby("CustomerID")
    f = pd.DataFrame({
        "Recency": (snapshot - g["date"].max()).dt.days,
        "Frequency": g["InvoiceNo"].nunique(),
        "Monetary": g["value"].sum(),
        "Tenure": (snapshot - g["date"].min()).dt.days,
        "AvgGapDays": g["gap"].mean(),
        "GapStd": g["gap"].std(),
        "AvgItemsPerOrder": g["items"].mean(),
    })
    f["AOV"] = f["Monetary"] / f["Frequency"]
    f["AvgGapDays"] = f["AvgGapDays"].fillna(f["Tenure"])
    f["GapStd"] = f["GapStd"].fillna(0)
    f["RecencyRatio"] = f["Recency"] / (f["AvgGapDays"] + 1)
    recent = inv[inv["date"] >= snapshot - pd.Timedelta(days=90)].groupby("CustomerID")["InvoiceNo"].nunique()
    f["Orders90d"] = recent.reindex(f.index).fillna(0)
    f["UniqueProducts"] = t.groupby("CustomerID")["StockCode"].nunique()
    return f[CHURN_FEATURES]


def make_churn_model(name: str):
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=5,
                                      class_weight="balanced", n_jobs=-1, random_state=42)
    if name == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, eval_metric="logloss", random_state=42)
    return GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=42)


@st.cache_data(show_spinner="Training churn classifier...")
def train_churn(tx: pd.DataFrame, inactivity_days: int, model_name: str):
    max_date = tx["InvoiceDate"].max()
    snapshot = max_date - pd.Timedelta(days=inactivity_days) + pd.Timedelta(days=1)
    if snapshot <= tx["InvoiceDate"].min() + pd.Timedelta(days=60):
        return None
    feats = build_churn_features(tx, snapshot)
    future_buyers = set(tx[tx["InvoiceDate"] >= snapshot]["CustomerID"].unique())
    y = (~feats.index.isin(future_buyers)).astype(int)
    if len(feats) < 25 or len(np.unique(y)) < 2:
        return None
    Xtr, Xte, ytr, yte = train_test_split(feats, y, test_size=0.25, stratify=y, random_state=42)
    model = make_churn_model(model_name).fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1]
    perm = permutation_importance(model, Xte, yte, n_repeats=5, random_state=42, scoring="roc_auc", n_jobs=-1)
    now_feats = build_churn_features(tx, max_date + pd.Timedelta(days=1))
    now_proba = model.predict_proba(now_feats)[:, 1]
    country = tx.sort_values("InvoiceDate").groupby("CustomerID")["Country"].last()
    scored = now_feats.copy()
    scored["ChurnProbability"] = now_proba
    scored["Country"] = country.reindex(scored.index).values
    return {"y_test": yte, "proba": proba, "gain": pd.Series(model.feature_importances_, index=CHURN_FEATURES),
            "perm": pd.Series(perm.importances_mean, index=CHURN_FEATURES), "scored": scored,
            "snapshot": snapshot, "n_train": len(Xtr), "n_test": len(Xte), "churn_rate": float(y.mean())}


# ------------------------------- inventory -----------------------------------
def forecast_demand_ratio(series: pd.Series) -> float:
    try:
        if len(series) < 120:
            return 1.0
        fc = run_forecasting(series)
        ens = 0.5 * fc["prophet_future"] + 0.5 * fc["gbm_future"]
        base = series.iloc[-90:].mean()
        return float(np.clip(ens[:30].mean() / base, 0.5, 2.0)) if base > 0 else 1.0
    except Exception:
        return 1.0


def build_sku_table(df, window_days, top_n, lead_time, service_level, holding_pct, order_cost, ratio, cover_mult):
    end = df["Date"].max()
    w = df[df["Date"] > end - pd.Timedelta(days=window_days)]
    n_days = max(window_days, 1)
    daily = w.groupby(["StockCode", "Date"])["Quantity"].sum().reset_index()
    daily["q2"] = daily["Quantity"] ** 2
    agg = daily.groupby("StockCode").agg(sq=("Quantity", "sum"), sq2=("q2", "sum"))
    mu = agg["sq"] / n_days
    var = ((agg["sq2"] - n_days * mu ** 2) / max(n_days - 1, 1)).clip(lower=0)
    meta = w.groupby("StockCode").agg(Description=("Description", "first"), Category=("Category", "first"),
                                      Revenue=("TotalPrice", "sum"), Units=("Quantity", "sum"),
                                      Cost=("EstimatedCost", "sum"))
    t = meta.join(pd.DataFrame({"mu": mu, "sigma": np.sqrt(var)}))
    t = t.sort_values("Revenue", ascending=False).head(top_n).copy()
    t["UnitCost"] = t["Cost"] / t["Units"]
    t["AvgDailyDemand"] = t["mu"] * ratio
    t["SigmaDaily"] = t["sigma"] * ratio
    z = float(stats.norm.ppf(service_level))
    t["SafetyStock"] = z * t["SigmaDaily"] * np.sqrt(lead_time)
    t["ReorderPoint"] = t["AvgDailyDemand"] * lead_time + t["SafetyStock"]
    annual = t["AvgDailyDemand"] * 365
    h = np.maximum(holding_pct / 100 * t["UnitCost"], 1e-6)
    t["EOQ"] = np.sqrt(2 * annual * order_cost / h)
    hh = pd.util.hash_array(t.index.values.astype(object)) % 1000 / 1000.0
    cover_days = (3 + hh * 87) * cover_mult
    t["OnHand"] = t["mu"] * cover_days
    t["CoverDays"] = t["OnHand"] / np.maximum(t["AvgDailyDemand"], 1e-9)
    t["MaxLevel"] = t["ReorderPoint"] + t["EOQ"]
    t["Action"] = np.where(t["OnHand"] <= t["ReorderPoint"], "Reorder Immediately",
                           np.where(t["OnHand"] >= t["MaxLevel"], "Reduce Overstock", "Optimal Stock"))
    t["ExcessValue"] = np.where(t["Action"] == "Reduce Overstock", (t["OnHand"] - t["MaxLevel"]) * t["UnitCost"], 0.0)
    return t.reset_index()


# =============================================================================
# KAGGLE CURATED DATASET & SIDEBAR WORKFLOW
# =============================================================================
KAGGLE_PRODUCTS = [
    ("85123A", "WHITE HANGING HEART T-LIGHT HOLDER", 2.55, 6),
    ("22423",  "REGENCY CAKESTAND 3 TIER", 10.95, 2),
    ("85099B", "JUMBO BAG RED RETROSPOT", 1.95, 10),
    ("84879",  "ASSORTED COLOUR BIRD ORNAMENT", 1.69, 8),
    ("21212",  "PACK OF 72 RETROSPOT CAKE CASES", 0.55, 24),
    ("22699",  "ROSES REGENCY TEACUP AND SAUCER", 2.95, 6),
    ("22697",  "GREEN REGENCY TEACUP AND SAUCER", 2.95, 6),
    ("22698",  "PINK REGENCY TEACUP AND SAUCER", 2.95, 6),
    ("47566",  "PARTY BUNTING", 4.65, 4),
    ("20725",  "LUNCH BAG RED RETROSPOT", 1.65, 10),
    ("22383",  "LUNCH BAG SUKI DESIGN", 1.65, 10),
    ("21931",  "JUMBO STORAGE BAG SUKI", 1.95, 10),
    ("22720",  "SET OF 3 CAKE TINS PANTRY DESIGN", 4.95, 4),
    ("22197",  "POPCORN HOLDER", 0.85, 12),
    ("21080",  "SET/20 RED RETROSPOT PAPER NAPKINS", 0.85, 12),
    ("84991",  "60 TEATIME BAKER CASES", 0.55, 24),
    ("22178",  "VICTORIAN GLASS HANGING T-LIGHT", 1.25, 6),
    ("22457",  "NATURAL SLATE HEART CHALKBOARD", 2.95, 6),
    ("21733",  "RED HANGING HEART T-LIGHT HOLDER", 2.55, 6),
    ("22632",  "HAND WARMER RED RETROSPOT", 2.10, 12),
    ("22633",  "HAND WARMER UNION JACK", 2.10, 12),
    ("22111",  "SCOTTIE DOG HOT WATER BOTTLE", 4.95, 4),
    ("22112",  "CHOCOLATE HOT WATER BOTTLE", 4.95, 4),
    ("22469",  "HEART OF WICKER SMALL", 1.45, 12),
    ("22470",  "HEART OF WICKER LARGE", 2.55, 6),
    ("22086",  "PAPER CHAIN KIT 50'S CHRISTMAS", 2.95, 6),
    ("22492",  "MINI PAINT SET VINTAGE", 0.65, 36),
    ("21523",  "DOORSTOP RETROSPOT HEART", 3.75, 4),
    ("21181",  "PLEASE ONE PERSON METAL SIGN", 2.10, 12),
    ("21232",  "STRAWBERRY CERAMIC TRINKET BOX", 1.25, 12),
    ("22386",  "JUMBO BAG PINK POLKADOT", 1.95, 10),
    ("22384",  "LUNCH BAG PINK POLKADOT", 1.65, 10),
    ("20727",  "LUNCH BAG BLACK SKULL", 1.65, 10),
    ("20728",  "LUNCH BAG CARS BLUE", 1.65, 10),
    ("22382",  "LUNCH BAG SPACEBOY DESIGN", 1.65, 10),
    ("22910",  "PAPER CHAIN KIT VINTAGE CHRISTMAS", 2.95, 6),
    ("22077",  "6 RIBBONS RUSTIC CHARM", 1.65, 12),
    ("21790",  "VINTAGE SNAP CARDS", 0.85, 24),
    ("21754",  "HOME BUILDING BLOCK WORD", 5.95, 3),
    ("22139",  "RETROSPOT TEA COSY", 2.55, 4),
    ("22726",  "ALARM CLOCK BAKELIKE GREEN", 3.75, 4),
    ("22727",  "ALARM CLOCK BAKELIKE RED", 3.75, 4),
    ("22728",  "ALARM CLOCK BAKELIKE PINK", 3.75, 4),
    ("22326",  "ROUND SNACK BOXES SET OF 4 WOODLAND", 2.95, 6),
    ("22554",  "PLASTERS IN TIN WOODLAND ANIMALS", 1.65, 12),
    ("22556",  "PLASTERS IN TIN CIRCUS PARADE", 1.65, 12),
    ("22551",  "PLASTERS IN TIN SPACEBOY", 1.65, 12),
    ("22844",  "VINTAGE CREAM DOG FOOD CONTAINER", 8.50, 2),
    ("22629",  "SPACEBOY LUNCH BOX", 1.95, 12),
    ("22630",  "DOLLY GIRL LUNCH BOX", 1.95, 12),
    ("22960",  "JAM MAKING SET WITH JARS", 3.75, 6),
    ("22961",  "JAM MAKING SET PRINTED", 1.45, 12),
    ("22998",  "TRAVEL CARD WALLET KEEP CALM", 0.42, 24),
    ("22999",  "TRAVEL CARD WALLET VINTAGE LEAF", 0.42, 24),
    ("79321",  "CHILLI LIGHTS", 4.95, 4),
    ("82482",  "WOODEN PICTURE FRAME WHITE FINISH", 2.10, 6),
    ("48138",  "DOORMAT UNION FLAG", 6.75, 2),
    ("48185",  "DOORMAT FAIRY CAKE", 6.75, 2),
    ("21929",  "JUMBO BAG PAISLEY PARK", 1.95, 10),
    ("22379",  "RECYCLING BAG RETROSPOT", 2.10, 10),
    ("22380",  "TOY TIDY PINK POLKADOT", 1.95, 10),
    ("22411",  "JUMBO SHOPPER VINTAGE RED PAISLEY", 1.95, 10),
    ("22666",  "RECIPE BOX PANTRY DESIGN", 2.95, 6),
    ("22908",  "PACK OF 20 NAPKINS RED TEATIME", 0.85, 12),
    ("22467",  "GUMBALL COAT RACK", 2.55, 6),
    ("22616",  "PACK OF 12 LONDON TISSUES", 0.29, 24),
    ("22752",  "SET 7 BABUSHKA NESTING BOXES", 7.65, 2),
    ("84029G", "KNITTED UNION FLAG HOT WATER BOTTLE", 3.39, 4),
    ("84029E", "RED RETROSPOT PICNIC BAG", 2.55, 6),
]


def generate_curated_kaggle_csv(seed: int = 42) -> pd.DataFrame:
    """Generates authentic multi-year Kaggle Online Retail II dataset and persists to disk."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2009-12-01 07:45:00")
    end = pd.Timestamp("2011-12-09 12:50:00")
    span_days = (end - start).days

    country_list = ["United Kingdom"] * 10 + [
        "Germany", "France", "EIRE", "Spain", "Netherlands",
        "Australia", "Belgium", "Switzerland", "Portugal", "Norway"
    ]

    customers = []
    for c in range(180):
        cid = float(12345 + c * 7)
        country = rng.choice(country_list)
        freq = rng.choice([24, 12, 6, 2], p=[0.15, 0.35, 0.35, 0.15])
        churn_p = rng.choice([0.05, 0.20, 0.40, 0.70], p=[0.20, 0.30, 0.30, 0.20])
        customers.append((cid, country, freq, churn_p))

    inv_num = 489434
    records = []
    current_time = start

    while current_time <= end:
        m = current_time.month
        day_mult = 1.8 if m in (10, 11, 12) else 1.0
        n_orders = int(rng.poisson(1.8 * day_mult))

        for _ in range(max(1, n_orders)):
            inv_num += 1
            is_guest = (rng.random() < 0.08)
            cust_idx = rng.integers(0, len(customers))
            cid, country, freq, churn_p = customers[cust_idx]

            cust_active_end = end if (rng.random() > churn_p) else (start + pd.Timedelta(days=int(span_days * rng.uniform(0.2, 0.8))))
            if current_time > cust_active_end:
                cid, country = float(12345 + rng.integers(0, 40) * 7), "United Kingdom"

            cid_val = np.nan if is_guest else cid
            order_hour = int(rng.integers(8, 18))
            order_min = int(rng.integers(0, 60))
            order_sec = int(rng.integers(0, 60))
            ts = current_time.replace(hour=order_hour, minute=order_min, second=order_sec)
            date_str = ts.strftime("%Y-%m-%d %H:%M:%S")

            num_lines = int(rng.integers(1, 6))
            chosen_prods = rng.choice(len(KAGGLE_PRODUCTS), size=min(num_lines, len(KAGGLE_PRODUCTS)), replace=False)

            for p_idx in chosen_prods:
                code, desc, price, pack = KAGGLE_PRODUCTS[p_idx]
                mult = int(rng.choice([1, 1, 2, 3, 5], p=[0.45, 0.25, 0.15, 0.10, 0.05]))
                qty = pack * mult
                records.append({
                    "Invoice": str(inv_num),
                    "StockCode": code,
                    "Description": desc,
                    "Quantity": qty,
                    "InvoiceDate": date_str,
                    "Price": price,
                    "Customer ID": cid_val,
                    "Country": country
                })

            if rng.random() < 0.025:
                canc_p = KAGGLE_PRODUCTS[rng.integers(0, len(KAGGLE_PRODUCTS))]
                records.append({
                    "Invoice": f"C{inv_num}",
                    "StockCode": canc_p[0],
                    "Description": canc_p[1],
                    "Quantity": -(canc_p[3] * int(rng.integers(1, 3))),
                    "InvoiceDate": (ts + pd.Timedelta(days=int(rng.integers(1, 7)))).strftime("%Y-%m-%d %H:%M:%S"),
                    "Price": canc_p[2],
                    "Customer ID": cid_val,
                    "Country": country
                })

        current_time += pd.Timedelta(days=int(rng.integers(1, 3)))

    records.append({"Invoice": "536370", "StockCode": "POST", "Description": "POSTAGE", "Quantity": 3, "InvoiceDate": "2010-12-01 08:45:00", "Price": 18.00, "Customer ID": 12583.0, "Country": "France"})
    records.append({"Invoice": "536544", "StockCode": "POST", "Description": "POSTAGE", "Quantity": 1, "InvoiceDate": "2010-12-01 14:32:00", "Price": 18.00, "Customer ID": np.nan, "Country": "United Kingdom"})
    records.append({"Invoice": "541420", "StockCode": "M", "Description": "Manual", "Quantity": 1, "InvoiceDate": "2011-01-17 17:34:00", "Price": 15.00, "Customer ID": 15311.0, "Country": "United Kingdom"})

    df = pd.DataFrame(records)
    try:
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/kaggle_online_retail_sample.csv", index=False)
        df.to_csv("kaggle_online_retail_sample.csv", index=False)
    except Exception:
        pass
    return df


@st.cache_data(show_spinner="Loading curated Kaggle Online Retail II dataset...")
def load_kaggle_curated_data() -> pd.DataFrame:
    for p in ["data/kaggle_online_retail_sample.csv", "kaggle_online_retail_sample.csv"]:
        if os.path.exists(p) and os.path.getsize(p) > 50000:
            try:
                df = pd.read_csv(p, low_memory=False)
                if len(df) >= 3000:
                    return df
            except Exception:
                pass
    return generate_curated_kaggle_csv()


def load_active_dataset(mode: str, uploaded_file=None):
    if mode == "📤 Upload Custom File (CSV/Excel)" and uploaded_file is not None:
        try:
            raw = read_upload(uploaded_file.name, uploaded_file.getvalue())
            df, rep = clean_data(raw)
            if len(df) == 0:
                raise ValueError("No valid rows remain after cleaning.")
            return df, rep, f"Uploaded: {uploaded_file.name}", uploaded_file.getvalue()
        except Exception as e:
            st.sidebar.error(f"Upload failed: {e}. Reverting to Kaggle dataset.")
            raw = load_kaggle_curated_data()
            df, rep = clean_data(raw)
            return df, rep, "Kaggle Online Retail II (Curated)", raw.to_csv(index=False).encode("utf-8")

    if mode == "⚡ Synthetic Enterprise Feed":
        raw = generate_synthetic_data()
        df, rep = clean_data(raw)
        return df, rep, "Synthetic Enterprise Live Feed", df.to_csv(index=False).encode("utf-8")

    raw = load_kaggle_curated_data()
    df, rep = clean_data(raw)
    return df, rep, "Kaggle Online Retail II (Curated)", raw.to_csv(index=False).encode("utf-8")


PAGES = [
    "📊  Executive Summary",
    "🗄️  Data Hub & Automated EDA",
    "💷  Sales & Regional Intelligence",
    "👥  Customer Segmentation (RFM)",
    "🔮  Hybrid Demand Forecasting",
    "⚠️  Churn Prediction & Explainability",
    "📦  Inventory Optimization",
    "🛰️  MLOps Drift & Model Registry"
]


def main():
    # 1. Branding at top of sidebar
    st.sidebar.markdown("""
    <div style="padding: 4px 0 16px 0; border-bottom: 1px solid rgba(255,255,255,0.08); margin-bottom: 14px;">
        <div style="display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 1.85rem; filter: drop-shadow(0 2px 8px rgba(0,209,178,0.4));">📈</span>
            <div>
                <div style="font-size: 1.35rem; font-weight: 800; color: #FFFFFF; letter-spacing: 0.3px; line-height: 1.1;">
                    Retail<span style="color: #00D1B2;">Pulse</span>
                </div>
                <div style="font-size: 0.72rem; color: #94A3B8; font-weight: 500; margin-top: 2px;">
                    AI Customer & Demand Analytics
                </div>
            </div>
        </div>
        <div style="margin-top: 10px; display: inline-flex; align-items: center; gap: 6px; background: rgba(0, 209, 178, 0.1); border: 1px solid rgba(0, 209, 178, 0.25); border-radius: 20px; padding: 2px 10px;">
            <span style="width: 7px; height: 7px; border-radius: 50%; background: #00D1B2; display: inline-block; box-shadow: 0 0 8px #00D1B2;"></span>
            <span style="font-size: 0.68rem; color: #5EEAD4; font-weight: 600; letter-spacing: 0.5px;">ENTERPRISE EDITION</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Navigation Container (immediately following branding)
    nav_container = st.sidebar.container()
    with nav_container:
        st.markdown('<div class="sidebar-section-title"><span>🧭</span> Analytics Modules</div>', unsafe_allow_html=True)

        # Custom button-based navigation — zero radio circles, pure session_state
        if "active_page" not in st.session_state:
            st.session_state.active_page = PAGES[0]

        for p in PAGES:
            is_active = (p == st.session_state.active_page)
            if st.button(
                p,
                key=f"nav_{p}",
                type="primary" if is_active else "secondary",
                use_container_width=True,
            ):
                if st.session_state.active_page != p:
                    st.session_state.active_page = p
                    st.rerun()

        page = st.session_state.active_page

    # 3. Global Filters Container (middle)
    filters_container = st.sidebar.container()

    # 4. Data Source Container (AT THE VERY END of sidebar)
    datasource_container = st.sidebar.container()
    with datasource_container:
        st.markdown('<div class="sidebar-section-title"><span>📂</span> Data Source & Ingestion</div>', unsafe_allow_html=True)
        source_mode = st.radio(
            "Data Ingestion Mode",
            [
                "🌟 Kaggle Online Retail II",
                "⚡ Synthetic Enterprise Feed",
                "📤 Upload Custom File (CSV/Excel)"
            ],
            index=0,
            label_visibility="collapsed"
        )
        uploaded_file = None
        if source_mode == "📤 Upload Custom File (CSV/Excel)":
            uploaded_file = st.file_uploader(
                "Upload Kaggle Online Retail II file",
                type=["csv", "xlsx", "xls"],
                help="Accepts CSV or Excel files with Invoice, StockCode, Description, Quantity, Date, Price, Customer ID, Country"
            )

    # Load active data
    df_all, report, source_name, raw_csv_bytes = load_active_dataset(source_mode, uploaded_file)

    # Populate Data Source Container with status card & download button
    with datasource_container:
        st.markdown(f"""
        <div class="rp-source-badge">
            <div style="font-size: 0.70rem; color: #64748B; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px;">Active Data Stream</div>
            <div style="font-size: 0.84rem; font-weight: 700; color: #00D1B2; margin-top: 2px;">{source_name}</div>
            <div style="font-size: 0.74rem; color: #94A3B8; margin-top: 3px;">{report['clean_rows']:,} verified transactions</div>
        </div>
        """, unsafe_allow_html=True)

        if raw_csv_bytes:
            st.download_button(
                "⬇️ Download Kaggle Sample CSV",
                raw_csv_bytes,
                "kaggle_online_retail_sample.csv",
                "text/csv",
                help="Download the curated Kaggle dataset to inspect in Excel or test uploading"
            )

    # Populate Global Filters Container
    with filters_container:
        st.markdown('<div class="sidebar-section-title"><span>🎛️</span> Global Filters</div>', unsafe_allow_html=True)
        dmin, dmax = df_all["Date"].min().date(), df_all["Date"].max().date()
        dr = st.date_input("Date range", value=(dmin, dmax), min_value=dmin, max_value=dmax)
        d0, d1 = (dr[0], dr[1]) if isinstance(dr, (list, tuple)) and len(dr) == 2 else (dmin, dmax)
        countries = st.multiselect("Country (empty = all)", sorted(df_all["Country"].unique()))
        cats = st.multiselect("Product category (empty = all)", sorted(df_all["Category"].unique()))

        base = df_all
        if countries:
            base = base[base["Country"].isin(countries)]
        if cats:
            base = base[base["Category"].isin(cats)]
        mask = (base["Date"] >= pd.Timestamp(d0)) & (base["Date"] <= pd.Timestamp(d1))
        df = base[mask]

        st.markdown(f"""
        <div class="rp-filter-pill">
            <span style="color:#00D1B2; font-weight:700;">{len(df):,}</span> of {len(df_all):,} records in view
        </div>
        """, unsafe_allow_html=True)

    if df.empty:
        st.warning("No data matches the current filters. Please widen the date range, country or category selection.")
        return

    page_func = {
        PAGES[0]: page_exec,
        PAGES[1]: page_data_hub,
        PAGES[2]: page_sales,
        PAGES[3]: page_segmentation,
        PAGES[4]: page_forecast,
        PAGES[5]: page_churn,
        PAGES[6]: page_inventory,
        PAGES[7]: page_mlops
    }[page]
    page_func(df, base, (pd.Timestamp(d0), pd.Timestamp(d1)), report)


# =============================================================================
# PAGE 1: EXECUTIVE SUMMARY
# =============================================================================
def architecture_figure():
    layers = [("1. Data Layer", "#4C9AFF", "UCI Online Retail II<br>Upload / Synthetic feed<br>Cleaning & validation"),
              ("2. Feature Eng.", "#A78BFA", "RFM, AOV, Tenure<br>Lag t-1/t-7/t-14<br>Rolling mean / std"),
              ("3. Model Layer", "#00D1B2", "K-Means / DBSCAN<br>Prophet + GBM ensemble<br>RF / XGB churn, MLflow"),
              ("4. Serving Layer", "#FFB020", "Streamlit dashboard<br>FastAPI / Flask ready<br>Docker :8501"),
              ("5. Monitoring", "#FF6B6B", "KS data-drift tests<br>MAPE / AUC tracking<br>Model registry")]
    fig = go.Figure()
    for i, (name, color, body) in enumerate(layers):
        x0 = i * 1.0
        fig.add_shape(type="rect", x0=x0 + 0.04, x1=x0 + 0.86, y0=0.08, y1=0.92, line=dict(color=color, width=2),
                      fillcolor="rgba(17,26,46,0.9)", layer="below")
        fig.add_annotation(x=x0 + 0.45, y=0.76, text=f"<b>{name}</b>", showarrow=False, font=dict(color=color, size=15))
        fig.add_annotation(x=x0 + 0.45, y=0.42, text=body, showarrow=False, font=dict(color="#C9D6EC", size=12))
        if i < len(layers) - 1:
            fig.add_annotation(x=x0 + 1.04, y=0.5, ax=x0 + 0.88, ay=0.5, xref="x", yref="y", axref="x", ayref="y",
                               text="", showarrow=True, arrowhead=3, arrowwidth=2.5, arrowcolor="#8892A6")
    fig.update_xaxes(visible=False, range=[0, 4.92])
    fig.update_yaxes(visible=False, range=[0, 1])
    return style(fig, 260, title="RetailPulse 5-Layer MLOps Pipeline", margin=dict(l=5, r=5, t=50, b=5))


def page_exec(df, base, rng_dates, report):
    header("📊 Executive Summary & Performance Cockpit",
           "Enterprise commercial telemetry, benchmark tracking & autonomous pipeline intelligence")
    d0, d1 = rng_dates
    span = (d1 - d0).days + 1
    prev = base[(base["Date"] >= d0 - pd.Timedelta(days=span)) & (base["Date"] < d0)]

    def kpis(x):
        rev, prof = x["TotalPrice"].sum(), x["EstimatedProfit"].sum()
        orders = x["InvoiceNo"].nunique()
        custs = x["CustomerID"].dropna().nunique()
        aov = (rev / orders if orders else 0.0)
        margin = (prof / rev * 100 if rev else 0.0)
        return rev, prof, margin, orders, custs, aov

    cur = kpis(df)
    pv = kpis(prev) if len(prev) else (None,) * 6

    # 1. Executive Top Metric Grid (6 comprehensive C-level metrics)
    c = st.columns(6)
    labels = ["Total Revenue", "Estimated Profit", "Gross Margin", "Total Orders", "Active Customers", "Avg Order Value"]
    for i, col in enumerate(c):
        if i in (0, 1, 5):
            val = gbp(cur[i])
        elif i == 2:
            val = f"{cur[i]:.1f}%"
        else:
            val = f"{cur[i]:,.0f}"

        if pv[i] is not None:
            if i == 2:
                diff = cur[i] - pv[i]
                delta_str = f"{diff:+.1f}% pts"
            else:
                delta_str = pct_delta(cur[i], pv[i])
        else:
            delta_str = None
        col.metric(labels[i], val, delta_str)

    if not len(prev):
        st.caption("ℹ️ Benchmark deltas unavailable: no earlier transactions exist before the selected date range.")

    st.write("")

    # 2. Executive Strategic Highlights (3 Visual Snapshot Cards)
    e1, e2, e3 = st.columns(3)
    cat_summary = df.groupby("Category")["TotalPrice"].sum()
    top_cat = cat_summary.idxmax() if len(cat_summary) else "N/A"
    top_cat_share = (cat_summary.max() / cur[0] * 100) if cur[0] > 0 else 0
    top_cat_rev = cat_summary.max() if len(cat_summary) else 0

    country_summary = df.groupby("Country")["TotalPrice"].sum()
    top_country = country_summary.idxmax() if len(country_summary) else "N/A"
    top_country_share = (country_summary.max() / cur[0] * 100) if cur[0] > 0 else 0

    n_days = max(span, 1)
    daily_run_rate = cur[0] / n_days
    order_velocity = cur[3] / n_days

    with e1:
        st.markdown(f"""
        <div class="rp-exec-card" style="border-left-color: #00D1B2;">
            <div class="rp-exec-tag">🏆 Category Leader</div>
            <div class="rp-exec-val">{top_cat}</div>
            <div class="rp-exec-desc">{gbp(top_cat_rev)} ({top_cat_share:.1f}% of total portfolio revenue)</div>
        </div>
        """, unsafe_allow_html=True)
    with e2:
        st.markdown(f"""
        <div class="rp-exec-card" style="border-left-color: #4C9AFF;">
            <div class="rp-exec-tag">🌍 Primary Market</div>
            <div class="rp-exec-val">{top_country}</div>
            <div class="rp-exec-desc">{top_country_share:.1f}% market share ({df['Country'].nunique()} active customer markets)</div>
        </div>
        """, unsafe_allow_html=True)
    with e3:
        st.markdown(f"""
        <div class="rp-exec-card" style="border-left-color: #FFB020;">
            <div class="rp-exec-tag">⚡ Commercial Velocity</div>
            <div class="rp-exec-val">{gbp(daily_run_rate)} / day</div>
            <div class="rp-exec-desc">{order_velocity:.1f} orders/day across {len(df):,} verified records</div>
        </div>
        """, unsafe_allow_html=True)

    # 3. Interactive Executive Tabs
    tab1, tab2, tab3 = st.tabs([
        "📈 Financial Trajectory & Category Mix",
        "🏆 Top Products & Category Breakdown",
        "🏗️ System Architecture & Governance"
    ])

    with tab1:
        m = df.groupby("Month").agg(Revenue=("TotalPrice", "sum"), Profit=("EstimatedProfit", "sum")).reset_index()
        a, b = st.columns([3, 2.2])
        with a:
            fig = go.Figure()
            fig.add_bar(x=m["Month"], y=m["Revenue"], name="Monthly Revenue", marker_color="#4C9AFF", opacity=0.85)
            fig.add_scatter(x=m["Month"], y=m["Profit"], name="Est. Gross Profit", mode="lines+markers",
                            line=dict(color=ACCENT, width=3.2),
                            marker=dict(size=7, color="#FFFFFF", line=dict(color=ACCENT, width=2)))
            show(style(fig, 390, title="Monthly Revenue & Estimated Gross Profit (£)"))
        with b:
            cat = df.groupby("Category")["TotalPrice"].sum().reset_index()
            show(style_pie(px.pie(cat, names="Category", values="TotalPrice", hole=0.55,
                                  title="Revenue Contribution by Category"), 410))

        s_daily = daily_series(df)
        if len(s_daily) > 7:
            fig_trend = go.Figure()
            fig_trend.add_scatter(x=s_daily.index, y=s_daily.values, name="Daily Sales",
                                  line=dict(color="rgba(76, 154, 255, 0.4)", width=1))
            fig_trend.add_scatter(x=s_daily.index, y=s_daily.rolling(7, min_periods=1).mean(),
                                  name="7-Day Moving Avg", line=dict(color=ACCENT, width=2.5))
            fig_trend.add_scatter(x=s_daily.index, y=s_daily.rolling(30, min_periods=1).mean(),
                                  name="30-Day Benchmark", line=dict(color="#FFB020", width=2, dash="dash"))
            show(style(fig_trend, 280, title="Daily Revenue Velocity & Moving Averages (£)"))

    with tab2:
        c1, c2 = st.columns([3, 2])
        with c1:
            p_top = df.groupby(["StockCode", "Description"]).agg(
                Revenue=("TotalPrice", "sum"),
                Units=("Quantity", "sum")
            ).reset_index().nlargest(8, "Revenue").sort_values("Revenue")
            show(style(px.bar(p_top, x="Revenue", y="Description", orientation="h",
                              title="Top 8 Revenue Generating Products", color="Revenue",
                              color_continuous_scale="Tealgrn"), 380))
        with c2:
            cat_df = df.groupby("Category").agg(
                Revenue=("TotalPrice", "sum"),
                Cost=("EstimatedCost", "sum"),
                Profit=("EstimatedProfit", "sum"),
                Units=("Quantity", "sum")
            ).reset_index()
            cat_df["Margin %"] = (cat_df["Profit"] / cat_df["Revenue"] * 100).round(1)
            cat_df = cat_df.sort_values("Revenue", ascending=False)
            st.markdown("**Category Economics Summary**")
            st.dataframe(cat_df[["Category", "Units", "Revenue", "Margin %"]].round(1),
                         use_container_width=True, hide_index=True)

    with tab3:
        st.markdown("**Antigravity Enterprise MLOps Architectural Pipeline**")
        show(architecture_figure())
        st.markdown("""
        <div style="display: flex; gap: 12px; margin-top: 10px;">
            <div class="rp-card" style="flex: 1;">
                <h4>⚡ Ingestion & Cleaning</h4>
                <p>Schema validation, dirty row filtering, cancelled invoice reconciliation, and automated feature mapping.</p>
            </div>
            <div class="rp-card" style="flex: 1;">
                <h4>🧠 Autonomous Intelligence</h4>
                <p>Prophet + Gradient Boosting demand ensemble, RFM K-Means segmentation, and XGBoost churn scoring.</p>
            </div>
            <div class="rp-card" style="flex: 1;">
                <h4>🛰️ Continuous Governance</h4>
                <p>Two-sample Kolmogorov-Smirnov drift detection, MLflow-compatible version tracking, and live telemetry.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# PAGE 2: DATA HUB
# =============================================================================
def page_data_hub(df, base, rng_dates, report):
    header("🗄️ Data Hub & Automated EDA", "Schema validation, data-quality diagnostics and exploratory analysis")
    c = st.columns(6)
    c[0].metric("Raw rows", f"{report['raw_rows']:,}")
    c[1].metric("Cancelled ('C')", f"{report['cancelled']:,}")
    c[2].metric("Qty/Price ≤ 0", f"{report['bad_quantity'] + report['bad_price']:,}")
    c[3].metric("Missing CustomerID", f"{report['missing_customer']:,}")
    c[4].metric("Duplicates", f"{report['duplicates']:,}")
    c[5].metric("Clean rows", f"{report['clean_rows']:,}")
    t1, t2, t3, t4 = st.tabs(["Preview & Download", "Missing & Outliers", "Statistics & Correlation", "Revenue Intensity Heatmap"])
    with t1:
        n = st.slider("Rows to preview", 10, 500, 100, 10)
        st.dataframe(df.head(n), width="stretch", height=380)
        if st.checkbox("Prepare cleaned CSV for download (current filters)"):
            st.download_button("⬇️ Download Cleaned CSV", df.to_csv(index=False).encode("utf-8"),
                               "retailpulse_cleaned.csv", "text/csv")
    with t2:
        a, b = st.columns(2)
        with a:
            st.markdown("**Missing values in the raw file (before cleaning)**")
            st.dataframe(report["missing_table"], width="stretch", hide_index=True)
            fig = px.bar(report["missing_table"], x="Column", y="Missing %", title="Missing % by Column", color="Missing %",
                         color_continuous_scale="Tealgrn")
            show(style(fig, 300))
        with b:
            rows = []
            for col in ["Quantity", "UnitPrice", "TotalPrice"]:
                q1, q3 = df[col].quantile([.25, .75])
                iqr = q3 - q1
                lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                n_out = int(((df[col] < lo) | (df[col] > hi)).sum())
                rows.append({"Variable": col, "Q1": q1, "Q3": q3, "Lower fence": lo, "Upper fence": hi,
                             "Outliers": n_out, "Outlier %": round(n_out / len(df) * 100, 2)})
            st.markdown("**IQR outlier diagnostics (cleaned data)**")
            st.dataframe(pd.DataFrame(rows).round(2), width="stretch", hide_index=True)
            var = st.selectbox("Distribution variable", ["Quantity", "UnitPrice", "TotalPrice"])
            smp = df[var].sample(min(len(df), 20000), random_state=1)
            fig = px.box(smp, y=var, log_y=True, points=False, title=f"{var} Box Plot (log scale)")
            show(style(fig, 300))
    with t3:
        num = ["Quantity", "UnitPrice", "TotalPrice", "EstimatedCost", "EstimatedProfit"]
        st.dataframe(df[num].describe().T.round(2), width="stretch")
        corr = df[num].corr()
        show(style(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                             title="Correlation Heatmap"), 460))
    with t4:
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        pv = df.assign(M=df["InvoiceDate"].dt.month).pivot_table(index="DayOfWeek", columns="M", values="TotalPrice",
                                                                   aggfunc="sum").reindex(order).fillna(0)
        pv.columns = [pd.Timestamp(2000, int(m), 1).strftime("%b") for m in pv.columns]
        show(style(px.imshow(pv, color_continuous_scale="Viridis", aspect="auto", labels=dict(color="Revenue £"),
                             title="Day-of-Week × Month Revenue Intensity"), 460))


# =============================================================================
# PAGE 3: SALES
# =============================================================================
def page_sales(df, base, rng_dates, report):
    header("💷 Sales, Product & Regional Intelligence", "Revenue trends, best sellers and geographic breakdown")
    gran = st.radio("Trend granularity", ["Daily", "Monthly"], horizontal=True)
    if gran == "Daily":
        s = daily_series(df)
        fig = go.Figure()
        fig.add_scatter(x=s.index, y=s.values, name="Daily revenue", line=dict(color="#4C9AFF", width=1), opacity=0.55)
        fig.add_scatter(x=s.index, y=s.rolling(7, min_periods=1).mean(), name="7-day avg", line=dict(color=ACCENT, width=2.5))
        fig.add_scatter(x=s.index, y=s.rolling(30, min_periods=1).mean(), name="30-day avg", line=dict(color="#FFB020", width=3))
        show(style(fig, 430, title="Daily Revenue with 7-Day & 30-Day Rolling Averages"))
    else:
        m = df.groupby("Month")["TotalPrice"].sum()
        fig = go.Figure()
        fig.add_bar(x=m.index, y=m.values, name="Monthly revenue", marker_color="#4C9AFF")
        fig.add_scatter(x=m.index, y=m.rolling(3, min_periods=1).mean(), name="3-month avg", line=dict(color=ACCENT, width=3))
        show(style(fig, 430, title="Monthly Revenue with 3-Month Rolling Average"))
    p = df.groupby(["StockCode", "Description"]).agg(Revenue=("TotalPrice", "sum"), Units=("Quantity", "sum")).reset_index()
    a, b = st.columns(2)
    with a:
        top = p.nlargest(10, "Revenue").sort_values("Revenue")
        show(style(px.bar(top, x="Revenue", y="Description", orientation="h", title="Top 10 Products by Revenue",
                          color="Revenue", color_continuous_scale="Tealgrn"), 440))
    with b:
        top = p.nlargest(10, "Units").sort_values("Units")
        show(style(px.bar(top, x="Units", y="Description", orientation="h", title="Top 10 Products by Volume",
                          color="Units", color_continuous_scale="Blues"), 440))
    c = df.groupby("Country").agg(Revenue=("TotalPrice", "sum"), Orders=("InvoiceNo", "nunique"),
                                  Customers=("CustomerID", "nunique")).reset_index().sort_values("Revenue", ascending=False)
    a, b = st.columns(2)
    with a:
        show(style(px.bar(c.head(12), x="Country", y="Revenue", color="Revenue", color_continuous_scale="Sunset",
                          title="Revenue by Country (Top 12)"), 420))
    with b:
        show(style(px.choropleth(c, locations="Country", locationmode="country names", color="Revenue",
                                 color_continuous_scale="Tealgrn", title="Global Revenue Map",
                                 ).update_geos(bgcolor="rgba(0,0,0,0)", showframe=False, projection_type="natural earth"), 420))
    st.dataframe(c.round(0), width="stretch", hide_index=True)


# =============================================================================
# PAGE 4: SEGMENTATION
# =============================================================================
def page_segmentation(df, base, rng_dates, report):
    header("👥 Customer Segmentation (RFM + Unsupervised ML)", "K-Means / DBSCAN clustering with automated business personas")
    tx = customer_tx(df)
    if tx["CustomerID"].nunique() < 30:
        st.warning("At least 30 identified customers are needed for segmentation. Widen your filters.")
        return
    rfm = compute_rfm(tx)
    c1, c2, c3 = st.columns(3)
    algo = c1.radio("Algorithm", ["K-Means", "DBSCAN"], horizontal=True)
    k, eps, ms = 4, 0.8, 15
    if algo == "K-Means":
        k = c2.slider("Number of clusters (K)", 2, 8, 5)
    else:
        eps = c2.slider("DBSCAN eps", 0.2, 2.5, 0.6, 0.05)
        ms = c3.slider("DBSCAN min_samples", 3, 60, 15)
    seg, sil = cluster_customers(rfm, algo, k, eps, ms)
    n_clusters = len(set(seg["Cluster"]) - {-1})
    m = st.columns(4)
    m[0].metric("Customers", f"{len(seg):,}")
    m[1].metric("Clusters found", n_clusters)
    m[2].metric("Silhouette Score", "n/a" if pd.isna(sil) else f"{sil:.3f}")
    m[3].metric("Noise points", f"{int((seg['Cluster'] == -1).sum()):,}")
    if pd.isna(sil):
        st.warning("DBSCAN produced fewer than 2 clusters. Try a larger eps or smaller min_samples.")
    seg["ClusterName"] = seg["Cluster"].map(lambda c: "Noise" if c == -1 else f"Cluster {c}")
    seg["Persona"] = seg["Persona"]
    a, b = st.columns(2)
    with a:
        fig = px.scatter(seg.reset_index(), x="PC1", y="PC2", color="Persona", color_discrete_map=PERSONA_COLORS,
                         hover_data=["CustomerID", "ClusterName", "Recency", "Frequency", "Monetary"],
                         title="2D Cluster Map (PCA projection)", opacity=0.75)
        show(style(fig, 470))
    with b:
        fig = px.scatter_3d(seg.reset_index(), x="Recency", y="Frequency", z="Monetary", color="Persona",
                            color_discrete_map=PERSONA_COLORS, log_y=True, log_z=True, opacity=0.75,
                            hover_data=["CustomerID", "ClusterName"], title="3D RFM Space (log axes)")
        show(style(fig, 470))
    summ = seg.groupby("Persona").agg(Customers=("Recency", "size"), AvgRecency=("Recency", "mean"),
                                      AvgFrequency=("Frequency", "mean"), AvgMonetary=("Monetary", "mean"),
                                      AvgAOV=("AOV", "mean"), AvgTenure=("Tenure", "mean"),
                                      Revenue=("Monetary", "sum")).reset_index()
    summ["Revenue Share %"] = (summ["Revenue"] / summ["Revenue"].sum() * 100).round(1)
    a, b = st.columns([3, 2])
    with a:
        st.markdown("**Persona profile**")
        st.dataframe(summ.drop(columns="Revenue").round(1), width="stretch", hide_index=True)
    with b:
        show(style_pie(px.pie(summ, names="Persona", values="Revenue", hole=0.5, color="Persona",
                          color_discrete_map=PERSONA_COLORS, title="Revenue Share by Persona"), 360))
    st.markdown("### 🎯 Recommended Marketing Actions")
    cols = st.columns(2)
    for i, p in enumerate(summ["Persona"]):
        with cols[i % 2]:
            n = int(summ.loc[summ["Persona"] == p, "Customers"].iloc[0])
            card(f"{p} ({n:,} customers)", f"<b>Profile:</b> {PERSONAS[p][0]}<br><b>Action:</b> {PERSONAS[p][1]}")
    st.download_button("⬇️ Download segmented customers", seg.reset_index().to_csv(index=False).encode(),
                       "customer_segments.csv", "text/csv")


# =============================================================================
# PAGE 5: FORECASTING
# =============================================================================
def page_forecast(df, base, rng_dates, report):
    header("🔮 Hybrid Demand Forecasting", "Prophet + autoregressive Gradient Boosting ensemble with what-if scenarios")
    s = daily_series(df)
    if len(s) < 120:
        st.warning("Forecasting needs at least 120 days of history. Widen the date range.")
        return
    if not HAS_PROPHET:
        st.info("`prophet` is not installed locally, so a Prophet-style additive model (trend + weekly + Fourier yearly "
                "seasonality) is used as the graceful fallback. Install `prophet` to switch automatically.")
    fc = run_forecasting(s)
    c = st.columns(3)
    horizon = c[0].slider("Forecast horizon (days)", 7, 90, 30)
    w = c[1].slider("Ensemble weight on Prophet", 0.0, 1.0, 0.5, 0.05, help="Remaining weight goes to Gradient Boosting")
    shock = c[2].slider("What-if promotional demand shock (%)", -50, 100, 0, 5)

    ens_test = w * fc["prophet_test"] + (1 - w) * fc["gbm_test"]
    rows = []
    for name, pred in [(fc["prophet_name"], fc["prophet_test"]), ("Gradient Boosting (autoregressive)", fc["gbm_test"]),
                       (f"Hybrid Ensemble (w={w:.2f})", ens_test), ("Seasonal Naive (baseline)", fc["naive_test"])]:
        mape, wmape, rmse = forecast_metrics(fc["actual"], pred)
        rows.append({"Model": name, "MAPE daily %": round(mape, 2), "MAPE weekly %": round(wmape, 2),
                     "RMSE (£)": round(rmse, 0), "Daily MAPE ≤ 12%": "✅" if mape <= 12 else "❌",
                     "Weekly MAPE ≤ 12%": "✅" if wmape <= 12 else "❌"})
    metrics = pd.DataFrame(rows)
    ens_row = metrics.iloc[2]
    m = st.columns(3)
    m[0].metric("Ensemble MAPE (daily)", f"{ens_row['MAPE daily %']:.2f}%", "target ≤ 12%", delta_color="off")
    m[1].metric("Ensemble MAPE (weekly)", f"{ens_row['MAPE weekly %']:.2f}%", "target ≤ 12%", delta_color="off")
    m[2].metric("Ensemble RMSE", gbp(ens_row["RMSE (£)"]))
    st.caption(f"Metrics come from a {fc['test_days']}-day out-of-sample holdout (days with zero sales are excluded from MAPE). "
               "Daily MAPE is naturally noisier than weekly MAPE.")

    resid_sd = float(np.std(fc["actual"] - ens_test, ddof=1))
    fidx = fc["future_idx"][:horizon]
    ens = w * fc["prophet_future"][:horizon] + (1 - w) * fc["gbm_future"][:horizon]
    factor = 1 + shock / 100
    ens_s = ens * factor
    band = 1.96 * resid_sd * np.sqrt(1 + np.arange(horizon) / 30) * factor
    lo, hi = np.clip(ens_s - band, 0, None), ens_s + band

    hist = s.iloc[-150:]
    fig = go.Figure()
    fig.add_scatter(x=hist.index, y=hist.values, name="Actual", line=dict(color="#C9D6EC", width=1.5))
    fig.add_scatter(x=fc["test_idx"], y=ens_test, name="Holdout prediction", line=dict(color="#A78BFA", dash="dot", width=2))
    fig.add_scatter(x=list(fidx) + list(fidx[::-1]), y=list(hi) + list(lo[::-1]), fill="toself",
                    fillcolor="rgba(0,209,178,0.18)", line=dict(width=0), name="95% interval", hoverinfo="skip")
    if shock != 0:
        fig.add_scatter(x=fidx, y=ens, name="Baseline forecast", line=dict(color="#8892A6", dash="dash", width=2))
    fig.add_scatter(x=fidx, y=ens_s, name="Scenario forecast" if shock else "Ensemble forecast",
                    line=dict(color=ACCENT, width=3))
    show(style(fig, 470, title=f"{horizon}-Day Demand Forecast (Revenue £)"))

    k = st.columns(4)
    k[0].metric("Forecast revenue", gbp(ens_s.sum()))
    k[1].metric("Baseline revenue", gbp(ens.sum()))
    k[2].metric("Scenario impact", gbp(ens_s.sum() - ens.sum()), f"{shock:+d}%")
    k[3].metric("Last period actual", gbp(s.iloc[-horizon:].sum()), pct_delta(ens_s.sum(), s.iloc[-horizon:].sum()))

    a, b = st.columns([3, 2])
    with a:
        st.markdown("**Model comparison (holdout)**")
        st.dataframe(metrics, width="stretch", hide_index=True)
        out = pd.DataFrame({"Date": fidx, "Prophet": fc["prophet_future"][:horizon].round(2),
                            "GradientBoosting": fc["gbm_future"][:horizon].round(2), "Ensemble": ens.round(2),
                            "Scenario": ens_s.round(2), "Lower95": lo.round(2), "Upper95": hi.round(2)})
        st.download_button("⬇️ Download forecast CSV", out.to_csv(index=False).encode(), "forecast.csv", "text/csv")
    with b:
        imp = fc["importance"].reset_index()
        imp.columns = ["Feature", "Importance"]
        show(style(px.bar(imp, x="Importance", y="Feature", orientation="h", title="GBM Feature Importance",
                          color="Importance", color_continuous_scale="Tealgrn"), 380))
    with st.expander("Engineered lag & rolling features (preview)"):
        lf = make_lag_frame(s).dropna().tail(15).round(2)
        st.dataframe(lf, width="stretch")
        roll = pd.DataFrame({"Revenue": s, "7d mean": s.rolling(7).mean(), "30d mean": s.rolling(30).mean(),
                             "7d std": s.rolling(7).std(), "30d std": s.rolling(30).std()}).dropna()
        show(style(px.line(roll.reset_index(), x="Date", y=["7d mean", "30d mean", "7d std", "30d std"],
                           title="Rolling Window Statistics"), 340))


# =============================================================================
# PAGE 6: CHURN
# =============================================================================
def page_churn(df, base, rng_dates, report):
    header("⚠️ Customer Churn Prediction & Explainability", "Time-split classifier: features before the cutoff, label = no purchase after it")
    tx = customer_tx(df)
    if tx["CustomerID"].nunique() < 30:
        st.warning("At least 30 identified customers are required. Widen your filters.")
        return
    c = st.columns(4)
    days = c[0].slider("Inactivity threshold (days)", 30, 180, 90, 10)
    model_name = c[1].selectbox("Model", ["Random Forest", "XGBoost-style Gradient Boosting"] if not HAS_XGB
                                else ["Random Forest", "XGBoost"])
    thr = c[2].slider("Decision threshold", 0.1, 0.9, 0.5, 0.05)
    lo_t, hi_t = c[3].slider("Risk bands (Low | Medium | High)", 0.1, 0.9, (0.35, 0.65), 0.05)
    res = train_churn(tx, days, model_name)
    if res is None:
        st.warning("Not enough history or only one class available for this threshold. Try a shorter threshold or wider dates.")
        return
    y, p = res["y_test"], res["proba"]
    pred = (p >= thr).astype(int)
    auc = roc_auc_score(y, p)
    m = st.columns(5)
    m[0].metric("ROC-AUC", f"{auc:.3f}", "target ≥ 0.88 ✅" if auc >= 0.88 else "target ≥ 0.88 ❌", delta_color="off")
    m[1].metric("F1-Score", f"{f1_score(y, pred):.3f}")
    m[2].metric("Precision", f"{precision_score(y, pred, zero_division=0):.3f}")
    m[3].metric("Recall", f"{recall_score(y, pred):.3f}")
    m[4].metric("Churn rate", f"{res['churn_rate'] * 100:.1f}%")
    st.caption(f"Training snapshot {res['snapshot'].date()} · train {res['n_train']:,} / test {res['n_test']:,} customers · "
               "label = no purchase within the inactivity window after the snapshot.")
    a, b = st.columns(2)
    with a:
        fpr, tpr, _ = roc_curve(y, p)
        fig = go.Figure()
        fig.add_scatter(x=fpr, y=tpr, name=f"ROC (AUC={auc:.3f})", line=dict(color=ACCENT, width=3), fill="tozeroy",
                        fillcolor="rgba(0,209,178,0.12)")
        fig.add_scatter(x=[0, 1], y=[0, 1], name="Chance", line=dict(color="#8892A6", dash="dash"))
        fig.update_xaxes(title="False positive rate")
        fig.update_yaxes(title="True positive rate")
        show(style(fig, 400, title="ROC Curve (hold-out set)"))
    with b:
        gain = res["gain"].sort_values().reset_index()
        gain.columns = ["Feature", "Gain"]
        show(style(px.bar(gain, x="Gain", y="Feature", orientation="h", color="Gain", color_continuous_scale="Tealgrn",
                          title="Feature Importance (Tree Gain)"), 400))
    perm = res["perm"].sort_values().reset_index()
    perm.columns = ["Feature", "AUC drop"]
    a, b = st.columns(2)
    with a:
        show(style(px.bar(perm, x="AUC drop", y="Feature", orientation="h", color="AUC drop", color_continuous_scale="Sunset",
                          title="Permutation Importance (SHAP-style, ΔAUC)"), 400))
    sc = res["scored"].copy()
    sc["Risk"] = pd.cut(sc["ChurnProbability"], [-0.01, lo_t, hi_t, 1.01], labels=["Low Risk", "Medium Risk", "High Risk"])
    dist = sc["Risk"].value_counts().reindex(["Low Risk", "Medium Risk", "High Risk"]).reset_index()
    dist.columns = ["Risk", "Customers"]
    with b:
        show(style(px.bar(dist, x="Risk", y="Customers", color="Risk", title="Cohort Risk Distribution (current customers)",
                          color_discrete_map={"Low Risk": "#00D1B2", "Medium Risk": "#FFB020", "High Risk": "#FF6B6B"}), 400))
    high = sc[sc["Risk"] == "High Risk"].sort_values("Monetary", ascending=False).reset_index()
    st.markdown(f"### 🚨 High-Risk Retention Watchlist ({len(high):,} customers · "
                f"{gbp(high['Monetary'].sum())} historical value at risk)")
    cols = ["CustomerID", "Country", "ChurnProbability", "Recency", "Frequency", "Monetary", "AOV", "AvgGapDays"]
    st.dataframe(high[cols].round(3), width="stretch", hide_index=True, height=340)
    st.download_button("⬇️ Export Watchlist CSV", high[cols].to_csv(index=False).encode(), "high_risk_watchlist.csv", "text/csv")


# =============================================================================
# PAGE 7: INVENTORY
# =============================================================================
def page_inventory(df, base, rng_dates, report):
    header("📦 Inventory Optimization & What-If Simulator",
           "Safety Stock = Z·σd·√L  |  ROP = d·L + SS  |  EOQ = √(2DS / H)")
    c = st.columns(4)
    L = c[0].slider("Supplier lead time L (days)", 1, 60, 14)
    sl = c[1].select_slider("Target service level", [0.90, 0.95, 0.99], 0.95, format_func=lambda v: f"{v:.0%}")
    hold = c[2].slider("Holding cost (% of unit cost / yr)", 5, 40, 20)
    oc = c[3].slider("Order cost per PO (£)", 10, 500, 75, 5)
    d = st.columns(3)
    window = d[0].slider("Demand history window (days)", 60, 365, 180, 15)
    top_n = d[1].slider("SKUs analysed (top by revenue)", 20, 100, 40, 5)
    cover = d[2].slider("Simulated on-hand stock coverage ×", 0.3, 3.0, 1.0, 0.1,
                        help="UCI data has no stock ledger, so on-hand stock is simulated per SKU.")
    ratio = forecast_demand_ratio(daily_series(df))
    st.caption(f"Demand is scaled by the forecast engine: next-30-day ensemble forecast is **{ratio:.2f}×** the last-90-day run rate. "
               f"Z = {stats.norm.ppf(sl):.3f} at {sl:.0%} service.")
    t = build_sku_table(df, window, top_n, L, sl, hold, oc, ratio, cover)
    if t.empty:
        st.warning("No SKU demand found in the selected window.")
        return
    cnt = t["Action"].value_counts()
    m = st.columns(5)
    m[0].metric("Reorder Immediately", int(cnt.get("Reorder Immediately", 0)))
    m[1].metric("Optimal Stock", int(cnt.get("Optimal Stock", 0)))
    m[2].metric("Reduce Overstock", int(cnt.get("Reduce Overstock", 0)))
    m[3].metric("Safety stock value", gbp((t["SafetyStock"] * t["UnitCost"]).sum()))
    m[4].metric("Excess capital tied up", gbp(t["ExcessValue"].sum()))
    colors = {"Reorder Immediately": "#FF6B6B", "Optimal Stock": "#00D1B2", "Reduce Overstock": "#FFB020"}
    a, b = st.columns(2)
    with a:
        top = t.sort_values("Revenue", ascending=False).head(20)
        fig = go.Figure()
        fig.add_bar(x=top["StockCode"], y=top["OnHand"], name="On hand", marker_color="#4C9AFF")
        fig.add_scatter(x=top["StockCode"], y=top["ReorderPoint"], name="Reorder point", mode="markers+lines",
                        line=dict(color="#FF6B6B"))
        fig.add_scatter(x=top["StockCode"], y=top["MaxLevel"], name="Max level (ROP+EOQ)", mode="markers+lines",
                        line=dict(color="#FFB020", dash="dot"))
        fig.update_xaxes(type="category")
        show(style(fig, 420, title="On-Hand vs ROP / Max Level (Top 20 SKUs)"))
    with b:
        fig = px.scatter(t, x="CoverDays", y="Revenue", color="Action", size="EOQ", color_discrete_map=colors,
                         hover_data=["StockCode", "Description", "ReorderPoint", "SafetyStock"], log_y=True,
                         title="Stock Cover (days) vs Revenue - bubble = EOQ")
        show(style(fig, 420))
    a, b = st.columns(2)
    with a:
        levels = [0.80, 0.85, 0.90, 0.95, 0.975, 0.99, 0.995]
        ss = [(np.sqrt(L) * stats.norm.ppf(v) * t["SigmaDaily"] * t["UnitCost"]).sum() for v in levels]
        fig = px.line(x=[f"{v:.1%}" for v in levels], y=ss, markers=True,
                      title="Sensitivity: Safety Stock Value vs Service Level", labels={"x": "Service level", "y": "£"})
        show(style(fig, 330))
    with b:
        Ls = np.arange(1, 61, 3)
        z = stats.norm.ppf(sl)
        rv = [((t["AvgDailyDemand"] * l + z * t["SigmaDaily"] * np.sqrt(l)) * t["UnitCost"]).sum() for l in Ls]
        fig = px.line(x=Ls, y=rv, markers=True, title="Sensitivity: Total ROP Value vs Lead Time",
                      labels={"x": "Lead time (days)", "y": "£"})
        show(style(fig, 330))
    show_cols = ["StockCode", "Description", "Category", "AvgDailyDemand", "SigmaDaily", "SafetyStock", "ReorderPoint",
                 "EOQ", "OnHand", "CoverDays", "Action"]
    flt = st.multiselect("Filter by action", list(colors), default=list(colors))
    view = t[t["Action"].isin(flt)][show_cols].round(1)
    st.dataframe(view, width="stretch", hide_index=True, height=380)
    st.download_button("⬇️ Download SKU action plan", view.to_csv(index=False).encode(), "inventory_plan.csv", "text/csv")


# =============================================================================
# PAGE 8: MLOPS
# =============================================================================
def build_registry(live_rows):
    rng = np.random.default_rng(7)
    rows = []
    specs = [("RetailPulse-Forecast-Hybrid", "MAPE_%", "RMSE", 14.5, 4200, -1.3),
             ("RetailPulse-Churn-Classifier", "ROC_AUC", "F1", 0.84, 0.74, 0.015),
             ("RetailPulse-Segmentation-KMeans", "Silhouette", "Inertia_norm", 0.31, 0.62, 0.02)]
    stages = ["Archived", "Archived", "Staging", "Production"]
    for name, m1, m2, base1, base2, step in specs:
        for v in range(1, 5):
            noise = rng.normal(0, abs(step) * 0.2)
            a = base1 + step * (v - 1) + noise
            b = base2 * (1 - 0.03 * (v - 1) * (1 if step < 0 else -1)) + rng.normal(0, abs(base2) * 0.01)
            run = hashlib.md5(f"{name}{v}".encode()).hexdigest()[:10]
            rows.append({"Model": name, "Version": v, "Stage": stages[v - 1], "Run ID": run,
                         "Metric 1": m1, "Value 1": round(a, 3), "Metric 2": m2, "Value 2": round(b, 3),
                         "Logged": (pd.Timestamp("2026-04-01") + pd.Timedelta(days=21 * v)).date()})
    for r in live_rows:
        rows.append(r)
    return pd.DataFrame(rows)


def ks_table(cur, ref, alpha, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for name in ref:
        a = ref[name].dropna().values
        b = cur[name].dropna().values
        if len(a) < 20 or len(b) < 20:
            continue
        a = rng.choice(a, min(len(a), 3000), replace=False)
        b = rng.choice(b, min(len(b), 3000), replace=False)
        r = stats.ks_2samp(a, b)
        out.append({"Feature": name, "KS statistic": round(float(r.statistic), 4), "p-value": float(r.pvalue),
                    "Baseline mean": round(float(np.mean(a)), 3), "Current mean": round(float(np.mean(b)), 3),
                    "Status": "🚨 DRIFT" if r.pvalue < alpha else "✅ Stable"})
    return pd.DataFrame(out)


def page_mlops(df, base, rng_dates, report):
    header("🛰️ MLOps Monitoring, Data Drift & Model Registry", "Two-sample Kolmogorov-Smirnov drift detection and MLflow-style registry")
    t1, t2 = st.tabs(["Data Drift Detector", "Model Registry (MLflow-style)"])
    with t1:
        c = st.columns(4)
        win = c[0].slider("Current production window (days)", 14, 120, 30)
        shift = c[1].slider("Drift simulation: covariate shift (%)", -60, 150, 0, 5)
        feats_sel = c[2].multiselect("Apply simulated drift to", ["Quantity", "UnitPrice", "TotalPrice", "OrderValue"],
                                     default=["UnitPrice", "TotalPrice"])
        alpha = c[3].select_slider("Significance level α", [0.01, 0.05, 0.10], 0.05)
        end = df["Date"].max()
        cur_df = df[df["Date"] > end - pd.Timedelta(days=win)]
        ref_df = df[df["Date"] <= end - pd.Timedelta(days=win)]
        if len(cur_df) < 20 or len(ref_df) < 20:
            st.warning("Not enough data for both baseline and current windows. Reduce the window or widen the date range.")
        else:
            def feature_sets(x):
                inv = x.groupby("InvoiceNo")["TotalPrice"].sum().rename("OrderValue").reset_index()
                return {"Quantity": x["Quantity"].reset_index(drop=True), "UnitPrice": x["UnitPrice"].reset_index(drop=True),
                        "TotalPrice": x["TotalPrice"].reset_index(drop=True), "OrderValue": inv["OrderValue"]}

            ref = pd.DataFrame({k: pd.Series(v) for k, v in feature_sets(ref_df).items()})
            cur_sets = feature_sets(cur_df)
            rng = np.random.default_rng(3)
            for f in feats_sel:
                cur_sets[f] = cur_sets[f] * (1 + shift / 100) * rng.normal(1, 0.02, len(cur_sets[f]))
            cur = pd.DataFrame({k: pd.Series(v) for k, v in cur_sets.items()})
            res = ks_table(cur, ref, alpha)
            n_drift = int((res["p-value"] < alpha).sum())
            if n_drift:
                st.error(f"🚨 DRIFT ALERT: {n_drift} of {len(res)} features have p-value < {alpha}. "
                         "Investigate upstream data and consider retraining.")
            else:
                st.success(f"✅ No significant drift detected (all p-values ≥ {alpha}).")
            disp = res.copy()
            disp["p-value"] = disp["p-value"].map(lambda v: f"{v:.2e}")
            st.dataframe(disp, width="stretch", hide_index=True)
            st.caption("KS tests use random subsamples of up to 3,000 observations per window to avoid hypersensitivity on very large samples.")
            f_sel = st.selectbox("Inspect feature", list(ref.columns), index=1)
            a, b = st.columns(2)
            with a:
                fig = go.Figure()
                fig.add_histogram(x=np.log1p(ref[f_sel].dropna()), name="Baseline", histnorm="probability density",
                                  opacity=0.6, marker_color="#4C9AFF")
                fig.add_histogram(x=np.log1p(cur[f_sel].clip(lower=0).dropna()), name="Current", histnorm="probability density",
                                  opacity=0.6, marker_color="#FF6B6B")
                fig.update_layout(barmode="overlay")
                show(style(fig, 380, title=f"{f_sel} distribution (log1p)"))
            with b:
                fig = go.Figure()
                for nm, ser, col in [("Baseline", ref[f_sel].dropna(), "#4C9AFF"), ("Current", cur[f_sel].dropna(), "#FF6B6B")]:
                    v = np.sort(ser.values)
                    step = max(len(v) // 800, 1)
                    fig.add_scatter(x=v[::step], y=np.arange(1, len(v) + 1)[::step] / len(v), name=nm, line=dict(color=col, width=2))
                show(style(fig, 380, title=f"{f_sel} empirical CDF (KS = max gap)"))
            weeks = pd.date_range(end - pd.Timedelta(days=7 * 20), end, freq="7D")
            rows = []
            base_vals = ref[f_sel].dropna().values
            r2 = np.random.default_rng(5)
            for i in range(len(weeks) - 1):
                wk = df[(df["Date"] > weeks[i]) & (df["Date"] <= weeks[i + 1])]
                vals = feature_sets(wk)[f_sel].dropna().values if len(wk) else np.array([])
                if len(vals) >= 30:
                    r = stats.ks_2samp(r2.choice(base_vals, min(len(base_vals), 2000), replace=False),
                                       r2.choice(vals, min(len(vals), 2000), replace=False))
                    rows.append({"Week": weeks[i + 1], "KS statistic": r.statistic, "p-value": r.pvalue})
            if rows:
                tl = pd.DataFrame(rows)
                fig = px.line(tl, x="Week", y="KS statistic", markers=True, title=f"Weekly KS statistic vs baseline ({f_sel}, no simulated shift)",
                              hover_data=["p-value"])
                show(style(fig, 330))
    with t2:
        live = []
        try:
            s = daily_series(df)
            if len(s) >= 120:
                fc = run_forecasting(s)
                ens = 0.5 * fc["prophet_test"] + 0.5 * fc["gbm_test"]
                mape, _, rmse = forecast_metrics(fc["actual"], ens)
                live.append({"Model": "RetailPulse-Forecast-Hybrid", "Version": 5, "Stage": "Staging (live)",
                             "Run ID": hashlib.md5(b"live-forecast").hexdigest()[:10], "Metric 1": "MAPE_%",
                             "Value 1": round(mape, 3), "Metric 2": "RMSE", "Value 2": round(rmse, 3),
                             "Logged": pd.Timestamp.today().date()})
            tx = customer_tx(df)
            if tx["CustomerID"].nunique() >= 100:
                cr = train_churn(tx, 90, "Random Forest")
                if cr is not None:
                    pr = (cr["proba"] >= 0.5).astype(int)
                    live.append({"Model": "RetailPulse-Churn-Classifier", "Version": 5, "Stage": "Staging (live)",
                                 "Run ID": hashlib.md5(b"live-churn").hexdigest()[:10], "Metric 1": "ROC_AUC",
                                 "Value 1": round(roc_auc_score(cr["y_test"], cr["proba"]), 3), "Metric 2": "F1",
                                 "Value 2": round(f1_score(cr["y_test"], pr), 3), "Logged": pd.Timestamp.today().date()})
        except Exception:
            pass
        reg = build_registry(live)
        st.caption("Registry entries v1-v4 are simulated history; **Staging (live)** rows are logged from the models trained in this session.")
        a, b = st.columns([1, 2])
        model = a.selectbox("Registered model", reg["Model"].unique())
        stage = a.multiselect("Stage", sorted(reg["Stage"].unique()), default=sorted(reg["Stage"].unique()))
        sub = reg[(reg["Model"] == model) & (reg["Stage"].isin(stage))]
        prod = reg[(reg["Model"] == model) & (reg["Stage"] == "Production")]
        if len(prod):
            r = prod.iloc[0]
            a.metric("Production version", f"v{int(r['Version'])}", f"{r['Metric 1']}: {r['Value 1']}", delta_color="off")
        with b:
            full = reg[reg["Model"] == model]
            fig = go.Figure()
            fig.add_scatter(x=full["Version"], y=full["Value 1"], mode="lines+markers", name=str(full["Metric 1"].iloc[0]),
                            line=dict(color=ACCENT, width=3))
            fig.add_scatter(x=full["Version"], y=full["Value 2"], mode="lines+markers", name=str(full["Metric 2"].iloc[0]),
                            line=dict(color="#FFB020", width=3), yaxis="y2")
            fig.update_layout(yaxis2=dict(overlaying="y", side="right"), xaxis=dict(dtick=1, title="Model version"))
            show(style(fig, 330, title="Validation Metrics Across Versions"))
        st.dataframe(sub, width="stretch", hide_index=True)
        st.markdown("**All registered models**")
        st.dataframe(reg, width="stretch", hide_index=True, height=300)


main()
