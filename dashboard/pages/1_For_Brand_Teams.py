"""
BAG Framework Dashboard -- For Brand Teams
Plain-language companion to the Analyst Dashboard. No statistics, no jargon --
built for someone on a brand/marketing team with no data science background.

Run as part of the multipage app: streamlit run app.py
(this file lives in pages/ and is reached via the sidebar)
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="BAG | For Brand Teams", layout="wide", page_icon="\U0001F3F7\uFE0F")

NAVY = "#F0ECE3"
BG = "#12141C"
CARD = "#1D2029"
BORDER = "#2E323F"
MUTED = "#9B958A"
BRASS = "#D8B27E"
GOOD = "#7FE0AC"
WARN = "#F0B86B"
BAD = "#F0988C"
HERO_START = "#1A2038"
HERO_END = "#2A3355"

CATEGORY = {
    "Chanel": "luxury", "Dior": "luxury", "Gucci": "luxury",
    "Supreme": "streetwear", "Off-White": "streetwear", "Palace": "streetwear",
}
DIMENSIONS = ["sentiment_gap", "emotion_gap", "semantic_gap", "topic_gap"]

# Plain-language translations -- no "gap", "contribution", "z-test", etc. anywhere on this page
PLAIN_LABELS = {
    "sentiment_gap": ("Tone Match", "\U0001F5E3\uFE0F",
                       "Whether your official voice sounds as positive (or as critical) as customers actually feel."),
    "emotion_gap": ("Emotional Connection", "\u2764\uFE0F",
                     "Whether the emotions in your messaging match the emotions customers express."),
    "semantic_gap": ("Message Alignment", "\U0001F3AF",
                      "Whether you and your customers are, broadly, talking about the same brand in the same way."),
    "topic_gap": ("Conversation Overlap", "\U0001F4AC",
                   "Whether the specific things you talk about are the same things customers actually discuss."),
}
ASPECT_LABELS = {
    "price_value": "Pricing & Value", "exclusivity": "Exclusivity", "quality": "Quality",
    "originality": "Originality & Design", "resale_culture": "Resale Market",
    "overexposure": "Overexposure", "heritage_history": "Heritage & History",
}

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@400;500;600&display=swap');
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
    .stApp {{ background-color: {BG}; }}
    h1, h2, h3 {{ font-family: 'Playfair Display', Georgia, serif !important; color: {NAVY} !important; }}
    p, span, div, label {{ color: {NAVY}; }}
    .hero {{
        background: linear-gradient(135deg, {HERO_START} 0%, {HERO_END} 100%);
        border: 1px solid {BORDER};
        padding: 40px 44px; border-radius: 14px; margin-bottom: 28px;
    }}
    .hero .eyebrow {{
        color: {BRASS}; font-size: 12px; letter-spacing: 0.18em; text-transform: uppercase;
        font-weight: 600; margin-bottom: 10px;
    }}
    .hero h1 {{ color: #FFFFFF !important; margin: 0; font-size: 34px; }}
    .hero p {{ color: #C7CEDC !important; margin: 10px 0 0 0; font-size: 15px; max-width: 640px; }}

    /* Hide Streamlit's auto-generated page list (duplicates our own
       sidebar page_link buttons below) */
    [data-testid="stSidebarNav"] {{ display: none; }}
    .insight-card {{
        background-color: {CARD}; border: 1px solid {BORDER}; border-radius: 12px;
        padding: 20px 22px; height: 100%; margin-bottom: 14px;
    }}
    .insight-card .title {{ font-size: 17px; font-weight: 700; color: {NAVY}; margin-bottom: 6px; }}
    .insight-card .body {{ font-size: 14px; color: {MUTED}; line-height: 1.5; }}
    .verdict-banner {{
        background-color: {CARD}; border-left: 5px solid {BRASS}; border-radius: 8px;
        padding: 20px 24px; font-size: 17px; font-weight: 600; margin-bottom: 24px;
    }}
    .rank-row {{
        display: flex; align-items: center; padding: 10px 16px; border-radius: 8px;
        margin-bottom: 6px; background-color: {CARD};
    }}
    .action-number {{
        background-color: {BRASS}; color: white; border-radius: 50%; width: 28px; height: 28px;
        display: inline-flex; align-items: center; justify-content: center; font-weight: 700;
        font-size: 14px; margin-right: 12px; flex-shrink: 0;
    }}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    data = {}
    files = {
        "bag": "bag_composite_scores.csv",
        "contrib": "bag_score_contributions.csv",
        "aspect_gap": "aspect_bag_gap.csv",
    }
    missing = []
    for key, fname in files.items():
        try:
            data[key] = pd.read_csv(fname, index_col=0 if key in ("bag", "contrib") else None)
        except FileNotFoundError:
            missing.append(fname)
            data[key] = None
    return data, missing


data, missing_files = load_data()

st.markdown("""
<div class="hero">
    <div class="eyebrow">Brand Voice Check</div>
    <h1>Your Brand, Through Customers' Eyes</h1>
    <p>A plain-language look at how closely your official messaging matches the way
    customers actually talk about you online.</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.page_link("app.py", label="\U0001F4CA Analyst Dashboard")
st.sidebar.page_link("pages/1_For_Brand_Teams.py", label="\U0001F3F7\uFE0F For Brand Teams")
st.sidebar.divider()
st.sidebar.caption("This page avoids statistics and technical terms by design. "
                    "For full methodology, see the Analyst Dashboard.")

if missing_files:
    st.warning("Some data files are missing -- this page may be incomplete. "
               "Make sure all CSVs are in the same folder as app.py.")
    st.stop()

bag = data["bag"]
contrib = data["contrib"]

brand_choice = st.selectbox("Choose a brand", sorted(bag.index.tolist()), key="plain_brand")

brand_bag = bag.loc[brand_choice, "BAG_score_equal_weight"]
alignment_pct = max(0, min(100, round((1 - brand_bag) * 100)))
category = bag.loc[brand_choice, "category"]
rank = int(bag["BAG_score_equal_weight"].rank(method="min").loc[brand_choice])
n_brands = len(bag)

# ---------------------------------------------------------------------------
# Hero score gauge -- the single "wow" visual moment of this page
# ---------------------------------------------------------------------------
gauge_color = GOOD if alignment_pct >= 70 else (WARN if alignment_pct >= 45 else BAD)

col_gauge, col_verdict = st.columns([1, 1.3])

with col_gauge:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=alignment_pct,
        number={"suffix": "%", "font": {"size": 44, "color": NAVY, "family": "Playfair Display, serif"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": MUTED, "tickfont": {"color": MUTED}},
            "bar": {"color": gauge_color, "thickness": 0.28},
            "bgcolor": CARD, "borderwidth": 0,
            "steps": [
                {"range": [0, 45], "color": "#4A2B28"},
                {"range": [45, 70], "color": "#4A3B22"},
                {"range": [70, 100], "color": "#2A4536"},
            ],
        },
        title={"text": "Customer Alignment Score", "font": {"size": 15, "color": MUTED, "family": "Inter, sans-serif"}},
    ))
    fig.update_layout(height=280, margin=dict(t=50, b=10, l=30, r=30),
                       paper_bgcolor="rgba(0,0,0,0)", font_family="Inter, sans-serif")
    st.plotly_chart(fig, use_container_width=True)

with col_verdict:
    st.write("")
    st.write("")
    if alignment_pct >= 70:
        verdict = (f"**{brand_choice}'s** official voice closely matches how customers actually "
                   f"talk about the brand. This is a strong foundation \u2014 customers are hearing "
                   f"an authentic reflection of what you're saying about yourselves.")
    elif alignment_pct >= 45:
        verdict = (f"**{brand_choice}'s** official voice partly matches how customers talk about "
                   f"the brand, but there's a noticeable gap in specific areas. Closing it could "
                   f"meaningfully strengthen how authentic the brand feels.")
    else:
        verdict = (f"**{brand_choice}'s** official voice differs substantially from how customers "
                   f"actually talk about the brand. This is the largest gap of the six brands "
                   f"studied, and represents a real opportunity to realign messaging with "
                   f"customer reality.")
    st.markdown(f'<div class="verdict-banner">{verdict}</div>', unsafe_allow_html=True)
    st.caption(f"Ranked {rank} of {n_brands} brands studied (1 = closest match to customers) \u00b7 "
               f"{category.title()} category")

st.divider()

st.divider()

# ---------------------------------------------------------------------------
# What this means -- plain-language dimension cards, ranked
# ---------------------------------------------------------------------------
st.markdown("### What This Means For You")
st.caption("The areas below are ranked by how much each one is currently holding back your alignment score.")

brand_contrib = contrib.loc[brand_choice, DIMENSIONS].sort_values(ascending=False)

for i, (dim, val) in enumerate(brand_contrib.items()):
    label, icon, plain_desc = PLAIN_LABELS[dim]
    # severity is magnitude-based (this dimension's own contribution value),
    # not tied to its rank position -- so each card is honest on its own terms
    if val >= 0.15:
        severity, severity_color = "Needs attention", BAD
    elif val >= 0.07:
        severity, severity_color = "Worth reviewing", WARN
    else:
        severity, severity_color = "In good shape", GOOD
    st.markdown(f"""
    <div class="insight-card">
        <div class="title">{icon} {label}
            <span style="float:right; font-size:12px; font-weight:700; color:{severity_color};
            background:white; padding:4px 10px; border-radius:12px;">{severity}</span>
        </div>
        <div class="body">{plain_desc}</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------------------------
# What you can do -- plain-language action items
# ---------------------------------------------------------------------------
st.markdown("### What You Can Do")

top_dim = brand_contrib.index[0]
top_label, top_icon, _ = PLAIN_LABELS[top_dim]
reference_brand = bag[top_dim].idxmin()

action_text = {
    "sentiment_gap": f"Review how positive or critical your official messaging sounds compared to real customer conversations. Aim to reflect the tone customers actually use, not just an idealised version of it.",
    "emotion_gap": f"Look at the emotional register of your campaigns and product copy. If your messaging is consistently upbeat but customers express a wider range of feelings, consider acknowledging that range rather than only projecting positivity.",
    "semantic_gap": f"Your messaging may be more abstract or campaign-driven than how customers actually describe the brand. Try grounding official copy in more concrete, specific language \u2014 closer to how real conversations describe the product experience.",
    "topic_gap": f"Customers are talking about things your official messaging doesn't cover much. Identify the specific topics that dominate real conversation (e.g. via the Analyst Dashboard) and consider addressing them directly rather than relying solely on heritage or campaign themes.",
}

st.markdown(f"""
<div class="insight-card" style="border-left: 4px solid {BRASS};">
    <div style="margin-bottom:10px;"><span class="action-number">1</span>
    <b>Focus first on {top_label.lower()}.</b></div>
    <div class="body" style="margin-left:40px;">{action_text[top_dim]}</div>
</div>
""", unsafe_allow_html=True)

if reference_brand != brand_choice:
    st.markdown(f"""
    <div class="insight-card" style="border-left: 4px solid {BRASS};">
        <div style="margin-bottom:10px;"><span class="action-number">2</span>
        <b>Look at what {reference_brand} does differently.</b></div>
        <div class="body" style="margin-left:40px;">{reference_brand} performs best on {top_label.lower()}
        among the six brands studied. Comparing your approach to theirs \u2014 in tone, subject matter,
        or messaging style \u2014 is a concrete starting point.</div>
    </div>
    """, unsafe_allow_html=True)

# plain-language aspect insight, if available
if data["aspect_gap"] is not None:
    ag = data["aspect_gap"]
    brand_aspects = (ag[ag["brand"] == brand_choice].dropna(subset=["aspect_gap"])
                      .sort_values("aspect_gap", ascending=False))
    if len(brand_aspects) > 0:
        top_aspect_row = brand_aspects.iloc[0]
        aspect_name = ASPECT_LABELS.get(top_aspect_row["aspect"], top_aspect_row["aspect"])
        st.markdown(f"""
        <div class="insight-card" style="border-left: 4px solid {BRASS};">
            <div style="margin-bottom:10px;"><span class="action-number">3</span>
            <b>Pay attention to how you talk about {aspect_name.lower()}.</b></div>
            <div class="body" style="margin-left:40px;">This is the specific topic where your official
            messaging and real customer conversation diverge the most for {brand_choice}.</div>
        </div>
        """, unsafe_allow_html=True)

st.divider()
st.caption(
    "This page summarises findings from a computational analysis of official brand messaging and "
    "real customer conversation (Reddit), covering six fashion brands. It is a research project, not "
    "an industry benchmarking service \u2014 figures reflect this specific dataset only."
)
