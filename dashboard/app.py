"""
BAG Framework Dashboard -- Analyst View
Run locally with: streamlit run app.py
(pip install streamlit plotly pandas numpy)

Expects these CSVs in the SAME folder as this script:
  bag_composite_scores.csv, bag_score_contributions.csv, aspect_bag_gap.csv,
  authenticity_dimension_summary.csv, transition_significance_tests.csv,
  monthly_sentiment_trend.csv, originality_overclaim_index.csv,
  commercial_avoidance_index.csv, topic_summary.csv

Also requires .streamlit/config.toml (forces light theme so this never
renders unreadable in a dark-mode browser again) and pages/1_For_Brand_Teams.py
(the plain-language companion view).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="BAG | Analyst Dashboard", layout="wide",
                    initial_sidebar_state="expanded", page_icon="\U0001F4CA")

# ---------------------------------------------------------------------------
# Fashion-editorial design system -- dark mode
# ---------------------------------------------------------------------------
BG = "#12141C"
CARD = "#1D2029"
BORDER = "#2E323F"
NAVY = "#F0ECE3"       # primary text -- kept variable name NAVY for minimal diff, now light
MUTED = "#9B958A"
BRASS = "#D8B27E"
LUXURY = "#7593C9"
STREET = "#E0946B"
GOOD = "#8CB69B"
BAD = "#D98E84"
HERO_START = "#1A2038"
HERO_END = "#2A3355"
PLOTLY_TEMPLATE = "plotly_dark"

CATEGORY = {
    "Chanel": "luxury", "Dior": "luxury", "Gucci": "luxury",
    "Supreme": "streetwear", "Off-White": "streetwear", "Palace": "streetwear",
}
DIMENSIONS = ["sentiment_gap", "emotion_gap", "semantic_gap", "topic_gap"]
DIM_LABELS = {
    "sentiment_gap": "Sentiment", "emotion_gap": "Emotion",
    "semantic_gap": "Semantic Similarity", "topic_gap": "Topic Overlap",
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
        font-weight: 600; margin-bottom: 10px; font-family: 'Inter', sans-serif;
    }}
    .hero h1 {{ color: #FFFFFF !important; margin: 0; font-size: 34px; }}
    .hero p {{ color: #B7BECF !important; margin: 10px 0 0 0; font-size: 15px; max-width: 640px; }}

    .metric-card {{
        background-color: {CARD}; border: 1px solid {BORDER}; border-radius: 10px;
        padding: 16px 18px; height: 100%;
    }}
    .metric-card .label {{
        font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase;
        color: {MUTED}; margin-bottom: 6px; font-weight: 600;
    }}
    .metric-card .value {{
        font-size: 26px; font-weight: 700; color: {NAVY} !important;
        font-family: 'Playfair Display', serif; line-height: 1.2;
    }}
    .metric-card .sub {{ font-size: 13px; margin-top: 4px; font-weight: 500; }}

    .stTabs [data-baseweb="tab-list"] {{ gap: 28px; border-bottom: 1px solid {BORDER}; }}
    .stTabs [data-baseweb="tab"] {{
        background-color: transparent; padding: 10px 2px; font-weight: 600;
        color: {MUTED} !important; font-size: 15px;
    }}
    .stTabs [data-baseweb="tab"] p {{ color: inherit !important; }}
    .stTabs [aria-selected="true"] {{
        color: {NAVY} !important; border-bottom: 2px solid {BRASS} !important;
    }}
    .stTabs [aria-selected="true"] p {{ color: {NAVY} !important; }}

    /* Hide Streamlit's auto-generated page list (duplicates our own
       sidebar page_link buttons below) */
    [data-testid="stSidebarNav"] {{ display: none; }}

    div[data-testid="stExpander"] {{
        background-color: {CARD}; border: 1px solid {BORDER}; border-radius: 10px;
    }}
    hr {{ border-color: {BORDER} !important; }}
</style>
""", unsafe_allow_html=True)


def metric_card(label, value, sub=None, sub_color=NAVY):
    sub_html = f'<div class="sub" style="color:{sub_color};">{sub}</div>' if sub else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)


def style_fig(fig, height=400):
    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=height, title_font_size=16,
        title_font_family="Playfair Display, Georgia, serif",
        font_family="Inter, sans-serif", font_color=NAVY,
        paper_bgcolor=CARD, plot_bgcolor=CARD,
        margin=dict(t=60, l=10, r=10, b=10),
    )
    return fig


@st.cache_data
def load_data():
    data = {}
    files = {
        "bag": "bag_composite_scores.csv",
        "contrib": "bag_score_contributions.csv",
        "aspect_gap": "aspect_bag_gap.csv",
        "auth_dim": "authenticity_dimension_summary.csv",
        "transitions": "transition_significance_tests.csv",
        "monthly": "monthly_sentiment_trend.csv",
        "orig_overclaim": "originality_overclaim_index.csv",
        "comm_avoidance": "commercial_avoidance_index.csv",
        "topics": "topic_summary.csv",
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
    <div class="eyebrow">MSc Dissertation \u00b7 Computational Brand Research</div>
    <h1>Brand Authenticity Gap</h1>
    <p>Measuring the divergence between brand-official language and consumer discourse
    across six fashion houses \u2014 three luxury, three streetwear.</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.page_link("app.py", label="\U0001F4CA Analyst Dashboard")
st.sidebar.page_link("pages/1_For_Brand_Teams.py", label="\U0001F3F7\uFE0F For Brand Teams")
st.sidebar.divider()
st.sidebar.caption("The Analyst view (this page) shows full methodology and statistics. "
                    "The Brand Teams view translates findings into plain language.")

if missing_files:
    st.warning(f"Missing files (some tabs will be incomplete): {', '.join(missing_files)}. "
               f"Make sure all CSVs are in the same folder as app.py.")

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Dimension Breakdown",
                                     "Temporal Analysis", "Recommendation Engine"])

# ===========================================================================
# TAB 1: OVERVIEW
# ===========================================================================
with tab1:
    if data["bag"] is not None:
        bag = data["bag"].copy()
        # sort by category first, then score within category, so the two groups
        # are always cleanly separated (not just an accidental side-effect of
        # Plotly's color-grouping behavior)
        bag_sorted = bag.sort_values(["category", "BAG_score_equal_weight"])
        brand_order = list(bag_sorted.index)
        n_first_group = int((bag_sorted["category"] == bag_sorted["category"].iloc[0]).sum())

        col1, col2 = st.columns([2, 1])

        with col1:
            fig = px.bar(
                bag_sorted, x="BAG_score_equal_weight", y=bag_sorted.index,
                color="category", orientation="h",
                color_discrete_map={"luxury": LUXURY, "streetwear": STREET},
                category_orders={"y": brand_order},
                labels={"BAG_score_equal_weight": "BAG Score", "y": "Brand"},
                title="Composite BAG Score by Brand", text_auto=".3f",
            )
            # divider line marking the exact boundary between the two category groups
            fig.add_hline(y=n_first_group - 0.5, line_dash="dot", line_color="#FFFFFF",
                           opacity=0.35, line_width=1.5)
            st.plotly_chart(style_fig(fig, 420), use_container_width=True)

        with col2:
            global_sorted = bag.sort_values("BAG_score_equal_weight")  # true overall min/max, independent of category grouping used for the chart
            metric_card("Most aligned brand", global_sorted.index[0],
                        f"{global_sorted['BAG_score_equal_weight'].iloc[0]:.3f}", GOOD)
            st.write("")
            metric_card("Most divergent brand", global_sorted.index[-1],
                        f"{global_sorted['BAG_score_equal_weight'].iloc[-1]:.3f}", BAD)
            st.write("")
            cat_means = bag.groupby("category")["BAG_score_equal_weight"].mean()
            mc1, mc2 = st.columns(2)
            with mc1:
                metric_card("Luxury mean", f"{cat_means.get('luxury', float('nan')):.3f}")
            with mc2:
                metric_card("Streetwear mean", f"{cat_means.get('streetwear', float('nan')):.3f}")

        st.divider()
        st.subheader("Category Comparison")
        cat_stats = bag.groupby("category")["BAG_score_equal_weight"].agg(["mean", "std", "count"]).reindex(
            ["luxury", "streetwear"])
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=cat_stats.index, y=cat_stats["mean"],
            error_y=dict(type="data", array=cat_stats["std"]),
            marker_color=[LUXURY, STREET],
        ))
        fig2.update_layout(title=f"Mean BAG Score by Category (n={int(cat_stats['count'].iloc[0])} brands each)",
                            yaxis_title="Mean BAG Score")
        st.plotly_chart(style_fig(fig2, 380), use_container_width=True)
        st.caption("Note: n=3 brands per category. Reported as a descriptive pattern in this "
                   "sample, not a statistically tested claim.")

        with st.expander("Full BAG score table (all weighting schemes)"):
            st.dataframe(bag.round(4))
    else:
        st.error("bag_composite_scores.csv not found.")

# ===========================================================================
# TAB 2: DIMENSION BREAKDOWN
# ===========================================================================
with tab2:
    if data["contrib"] is not None:
        st.subheader("What Drives Each Brand's BAG Score")
        contrib = data["contrib"].copy()
        contrib_order = contrib.sum(axis=1, numeric_only=True).sort_values().index
        fig3 = go.Figure()
        dim_colors = {"sentiment_gap": LUXURY, "emotion_gap": STREET, "semantic_gap": GOOD, "topic_gap": BAD}
        for dim in DIMENSIONS:
            fig3.add_trace(go.Bar(
                y=contrib_order, x=contrib.loc[contrib_order, dim],
                name=DIM_LABELS[dim], orientation="h", marker_color=dim_colors[dim],
            ))
        fig3.update_layout(barmode="stack", title="Exact contribution of each dimension to the composite score",
                            xaxis_title="Contribution to BAG Score")
        st.plotly_chart(style_fig(fig3, 420), use_container_width=True)
        st.caption("This is an exact linear decomposition of the equal-weighted composite score, "
                   "not an estimated model -- each bar segment shows precisely how much that "
                   "dimension contributed.")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top Aspect Gaps")
        if data["aspect_gap"] is not None:
            ag = data["aspect_gap"].dropna(subset=["aspect_gap"]).sort_values("aspect_gap", ascending=False)
            st.dataframe(ag[["brand", "aspect", "aspect_gap"]].head(10).round(3), hide_index=True)

    with col2:
        st.subheader("Overclaim Indices")
        if data["orig_overclaim"] is not None:
            st.write("**Originality overclaim** (brand claims originality more than consumers frame it)")
            st.dataframe(data["orig_overclaim"][["originality_overclaim", "category"]].round(3))
        if data["comm_avoidance"] is not None:
            st.write("**Commercial-framing avoidance** (consumers frame as commercial more than brand admits)")
            st.dataframe(data["comm_avoidance"][["commercial_avoidance", "category"]].round(3))

# ===========================================================================
# TAB 3: TEMPORAL ANALYSIS
# ===========================================================================
with tab3:
    st.subheader("Creative Director Transition: Significance Tests")
    if data["transitions"] is not None:
        trans = data["transitions"]
        display_cols = ["brand", "n_pre", "n_post", "sentiment_mannwhitney_p", "sentiment_sig",
                         "emotion_gap_bootstrap_p", "emotion_gap_sig",
                         "semantic_sim_bootstrap_p", "semantic_sim_sig",
                         "topic_overlap_ztest_p", "topic_overlap_sig"]
        available_cols = [c for c in display_cols if c in trans.columns]
        st.dataframe(trans[available_cols].round(4), hide_index=True)
        st.caption("*** p<0.001, ** p<0.01, * p<0.05, ns = not significant. "
                   "Supreme and Palace excluded (insufficient/no transition data -- see Methodology).")
    else:
        st.info("transition_significance_tests.csv not found.")

    st.divider()
    st.subheader("Monthly Sentiment Trend")
    if data["monthly"] is not None:
        monthly = data["monthly"].copy()
        monthly["year_month"] = pd.to_datetime(monthly["year_month"])
        brand_choice = st.selectbox("Select brand", sorted(monthly["brand"].unique()), key="monthly_brand")
        b = monthly[(monthly["brand"] == brand_choice) & (monthly["reliable"])].sort_values("year_month")
        if len(b) > 0:
            b = b.set_index("year_month")
            rolling = b["mean_sentiment"].rolling("180D", min_periods=2).mean()
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(x=b.index, y=b["mean_sentiment"], mode="markers",
                                       marker=dict(color=MUTED, size=5), name="Monthly value"))
            fig4.add_trace(go.Scatter(x=rolling.index, y=rolling.values, mode="lines",
                                       line=dict(color=BRASS, width=3), name="6-month rolling avg"))
            fig4.update_layout(title=f"{brand_choice}: Monthly Reddit Sentiment",
                                yaxis_title="Mean Sentiment (-1 to +1)")
            st.plotly_chart(style_fig(fig4, 400), use_container_width=True)
        else:
            st.warning(f"No reliable months of data for {brand_choice}.")

# ===========================================================================
# TAB 4: RECOMMENDATION ENGINE
# ===========================================================================
with tab4:
    st.markdown("## Recommendation Engine")
    st.caption("Scoped to the 6 brands studied in this dissertation \u2014 a same-dataset, "
               "nearest-best-performer engine grounded entirely in this project's own measured "
               "data. Not a trained predictive model, and not applicable beyond these 6 brands.")

    if data["contrib"] is not None and data["bag"] is not None:
        contrib = data["contrib"]
        bag = data["bag"]

        brand_choice = st.selectbox("Select a brand to analyse", sorted(contrib.index.tolist()), key="rec_brand")

        brand_contrib = contrib.loc[brand_choice, DIMENSIONS].sort_values(ascending=False)
        brand_bag_score = bag.loc[brand_choice, "BAG_score_equal_weight"]
        brand_category = bag.loc[brand_choice, "category"]
        rank = int(bag["BAG_score_equal_weight"].rank(method="min").loc[brand_choice])
        n_brands = len(bag)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("BAG Score", f"{brand_bag_score:.3f}")
        with c2:
            metric_card("Alignment Rank", f"{rank} of {n_brands}", "1 = most aligned", MUTED)
        with c3:
            metric_card("Category", brand_category.title())
        with c4:
            cat_avg = bag[bag["category"] == brand_category]["BAG_score_equal_weight"].mean()
            diff = brand_bag_score - cat_avg
            diff_color = GOOD if diff < 0 else (BAD if diff > 0 else MUTED)
            metric_card("vs. Category Average", f"{diff:+.3f}",
                        "better than average" if diff < 0 else "worse than average", diff_color)

        st.divider()

        st.markdown("### How This Brand Compares")
        bag_sorted_rec = bag.sort_values("BAG_score_equal_weight")
        for b in bag_sorted_rec.index:
            pct = max(0, min(100, round((1 - bag.loc[b, "BAG_score_equal_weight"]) * 100)))
            is_selected = b == brand_choice
            bar_color = BRASS if is_selected else BORDER
            label_weight = "700" if is_selected else "500"
            st.markdown(f"""
            <div style="display:flex; align-items:center; margin-bottom:8px;">
                <div style="width:110px; font-weight:{label_weight}; color:{NAVY};">{b}{"  \U0001F449" if is_selected else ""}</div>
                <div style="flex:1; background-color:{CARD}; border:1px solid {BORDER}; border-radius:6px; height:22px; margin:0 12px; overflow:hidden;">
                    <div style="width:{pct}%; background-color:{bar_color}; height:100%; border-radius:6px;"></div>
                </div>
                <div style="width:80px; text-align:right; color:{MUTED}; font-size:14px;">{pct}% / {bag.loc[b, 'BAG_score_equal_weight']:.3f}</div>
            </div>
            """, unsafe_allow_html=True)
        st.caption("Alignment % = (1 \u2212 BAG score) \u00d7 100, shown alongside the raw BAG score for reference.")

        st.divider()

        st.markdown("### Where This Brand Stands")
        st.caption("Raw gap value on each dimension, compared against the category average and "
                   "the best performer in the whole sample \u2014 not just the single largest driver.")

        compare_rows = []
        for dim in DIMENSIONS:
            best_brand = bag[dim].idxmin()
            compare_rows.append({"dimension": DIM_LABELS[dim], "series": brand_choice, "value": bag.loc[brand_choice, dim]})
            compare_rows.append({"dimension": DIM_LABELS[dim], "series": f"Category avg ({brand_category})",
                                  "value": bag[bag["category"] == brand_category][dim].mean()})
            compare_rows.append({"dimension": DIM_LABELS[dim], "series": f"Best in sample ({best_brand})",
                                  "value": bag[dim].min()})
        compare_df = pd.DataFrame(compare_rows)

        fig5 = px.bar(compare_df, x="dimension", y="value", color="series", barmode="group",
                      color_discrete_sequence=[NAVY, MUTED, GOOD],
                      title="Gap Comparison Across All Four Dimensions",
                      labels={"value": "Gap value (lower = better aligned)", "dimension": ""})
        fig5.update_layout(legend_title_text="")
        st.plotly_chart(style_fig(fig5, 380), use_container_width=True)

        st.divider()

        st.markdown("### Priority Action Plan")
        st.caption("All four dimensions, ranked by how much each currently contributes to this "
                   "brand's BAG score. Addressing #1 first yields the largest theoretical impact.")

        recommendations = {
            "sentiment_gap": lambda b, r, tv, rv: (
                f"Overall sentiment mismatch between {b}'s official copy and consumer conversation. "
                f"**{r}** shows the smallest sentiment gap in the sample ({rv:.3f} vs {b}'s {tv:.3f}). "
                f"Check the Dimension Breakdown tab for which specific aspects (price, quality, exclusivity, etc.) "
                f"drive this gap, and whether official copy's emotional register matches how consumers "
                f"actually discuss those aspects."
            ),
            "emotion_gap": lambda b, r, tv, rv: (
                f"Emotional tone in {b}'s official copy diverges from how consumers write about it. "
                f"**{r}** achieves the closest emotional alignment in the sample ({rv:.3f} vs {b}'s {tv:.3f}). "
                f"Consider whether official copy leans on uniformly aspirational/positive language while "
                f"real conversation carries a wider emotional range \u2014 closing this gap is not about "
                f"becoming more positive, but about acknowledging the fuller emotional register consumers use."
            ),
            "semantic_gap": lambda b, r, tv, rv: (
                f"{b}'s official copy sits far from consumer language in overall meaning-space. "
                f"**{r}** shows the closest semantic alignment ({rv:.3f} vs {b}'s {tv:.3f}). This typically "
                f"indicates official copy relies on abstract, campaign-driven phrasing rather than the "
                f"concrete, specific language consumers use when discussing the product directly."
            ),
            "topic_gap": lambda b, r, tv, rv: (
                f"Consumer conversation about {b} rarely covers the same themes as its official copy. "
                f"**{r}** achieves the highest topic overlap in the sample ({(1-rv):.1%} vs {b}'s {(1-tv):.1%} "
                f"of consumer conversation overlapping with official themes). Incorporating language closer "
                f"to the specific, concrete themes that already dominate real conversation \u2014 rather than "
                f"relying solely on heritage or campaign framing \u2014 is the most direct lever here."
            ),
        }

        priority_labels = ["Priority 1", "Priority 2", "Priority 3", "Priority 4"]
        for rank_i, (dim, contribution) in enumerate(brand_contrib.items()):
            reference_brand = bag[dim].idxmin()
            reference_value = bag.loc[reference_brand, dim]
            target_value = bag.loc[brand_choice, dim]
            is_self = brand_choice == reference_brand

            with st.expander(f"{priority_labels[rank_i]} \u2014 {DIM_LABELS[dim]}  "
                              f"(contributes {contribution:.3f} of {brand_bag_score:.3f} total)",
                              expanded=(rank_i == 0)):
                if is_self:
                    st.success(f"{brand_choice} is already the best performer on {DIM_LABELS[dim]} "
                               f"in this sample \u2014 no action needed here.")
                else:
                    st.markdown(recommendations[dim](brand_choice, reference_brand, target_value, reference_value))

        st.divider()

        st.markdown("### What-If: Closing the #1 Priority Gap")
        top_dim = brand_contrib.index[0]
        top_contribution = brand_contrib.iloc[0]
        hypothetical_score = max(0.0, brand_bag_score - top_contribution)
        hypothetical_rank = int((bag.loc[bag.index != brand_choice, "BAG_score_equal_weight"] < hypothetical_score).sum()) + 1

        wc1, wc2, wc3 = st.columns([1, 0.3, 1])
        with wc1:
            metric_card("Current BAG Score", f"{brand_bag_score:.3f}", f"Rank {rank} of {n_brands}", MUTED)
        wc2.markdown(f"<div style='text-align:center; font-size:28px; padding-top:24px; color:{BRASS};'>\u2192</div>", unsafe_allow_html=True)
        with wc3:
            metric_card(f"If {DIM_LABELS[top_dim]} Matched Best-in-Sample", f"{hypothetical_score:.3f}",
                        f"Rank {hypothetical_rank} of {n_brands}", GOOD)

        st.caption(
            "Illustrative simplification: assumes this dimension's contribution goes to zero (matching "
            "the best performer) while every other brand's scores stay fixed. This is a transparent "
            "recalculation of the known composite-score formula, not a predictive estimate \u2014 useful for "
            "showing the theoretical ceiling of fixing one dimension, not a forecast."
        )

        st.divider()

        with st.expander("Supporting Evidence from Aspect & Authenticity-Dimension Analysis"):
            ec1, ec2 = st.columns(2)
            with ec1:
                st.write(f"**{brand_choice}'s largest aspect-level gaps** (keyword-based sentiment analysis)")
                if data["aspect_gap"] is not None:
                    brand_aspects = (data["aspect_gap"][data["aspect_gap"]["brand"] == brand_choice]
                                      .dropna(subset=["aspect_gap"])
                                      .sort_values("aspect_gap", ascending=False).head(3))
                    if len(brand_aspects) > 0:
                        st.dataframe(brand_aspects[["aspect", "aspect_gap"]].round(3), hide_index=True)
                    else:
                        st.write("No aspect data available for this brand.")
            with ec2:
                st.write(f"**{brand_choice}'s authenticity-dimension gaps** (zero-shot classification)")
                if data["auth_dim"] is not None:
                    ad = data["auth_dim"]
                    piv = ad.pivot_table(index=["brand", "top_dimension"], columns="source_type", values="pct").reset_index().fillna(0)
                    if "brand_official" in piv.columns and "reddit" in piv.columns:
                        piv["gap"] = (piv["brand_official"] - piv["reddit"]).abs()
                        brand_dims = piv[piv["brand"] == brand_choice].sort_values("gap", ascending=False).head(3)
                        st.dataframe(brand_dims[["top_dimension", "gap"]].round(3), hide_index=True)

            st.caption("These two analyses are independent of the 4-dimension pipeline above (different "
                       "techniques: keyword-tagged sentiment vs. zero-shot entailment classification) \u2014 "
                       "shown here as converging or complementary evidence, not as sub-components of the "
                       "priority plan's dimensions.")

        st.caption("Every number on this page is drawn directly from this dissertation's own measured "
                   "data \u2014 no external benchmarks, no fabricated targets, no data beyond the 6 brands studied.")
    else:
        st.error("Required files (bag_score_contributions.csv, bag_composite_scores.csv) not found.")
