# Powerball.py - OPTIMIZED VERSION
# Enhanced statistical analysis with improved accuracy and additional metrics

import io
import math
import base64
import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from io import StringIO
from scipy import stats

# Import seed analysis module
try:
    from Seeds import render_seed_analysis

    SEEDS_AVAILABLE = True
except ImportError:
    SEEDS_AVAILABLE = False

WHITE_MAX = 69
MEGA_MAX = 26  # Powerball max
NY_OD_URL = "https://data.ny.gov/api/views/d6yy-54nr/rows.csv?accessType=DOWNLOAD"

st.set_page_config(page_title="Powerball Patterns", layout="wide")


# ---------- Utilities ----------
@st.cache_data(show_spinner=False)
def fetch_history_years(years: int) -> pd.DataFrame:
    r = requests.get(NY_OD_URL, timeout=30)
    r.raise_for_status()
    raw = pd.read_csv(StringIO(r.text))
    raw_cols = {c.lower(): c for c in raw.columns}
    date_col = raw_cols.get("draw date", "Draw Date")
    raw[date_col] = pd.to_datetime(raw[date_col])
    raw = raw.sort_values(date_col)
    cutoff = datetime.today() - timedelta(days=int(365.25 * years))
    raw = raw[raw[date_col] >= cutoff].copy()

    win_col = raw_cols.get("winning numbers", "Winning Numbers")
    parts = raw[win_col].astype(str).str.split(" ", expand=True)
    whites = parts.iloc[:, :5].astype(int)
    whites.columns = ["w1", "w2", "w3", "w4", "w5"]

    if parts.shape[1] >= 6:
        red = parts.iloc[:, 5].astype(int)
    else:
        power_col = (
            raw_cols.get("powerball")
            or raw_cols.get("power ball")
            or raw_cols.get("pb")
            or "Powerball"
        )
        red = raw[power_col].astype(int)

    df = pd.DataFrame(
        {
            "date": raw[date_col],
            "w1": whites["w1"],
            "w2": whites["w2"],
            "w3": whites["w3"],
            "w4": whites["w4"],
            "w5": whites["w5"],
            "mega": red,
        }
    ).sort_values("date")
    return df.reset_index(drop=True)


def load_uploaded(csv_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(csv_bytes))
    cols = {c.lower(): c for c in df.columns}
    req = ["date", "w1", "w2", "w3", "w4", "w5", "mega"]
    if not all(k in cols for k in req):
        raise ValueError("CSV must include: date,w1,w2,w3,w4,w5,mega")
    df = df.rename(columns={cols[k]: k for k in req})
    df["date"] = pd.to_datetime(df["date"])
    for k in ["w1", "w2", "w3", "w4", "w5", "mega"]:
        df[k] = df[k].astype(int)
    return df.sort_values("date").reset_index(drop=True)


def recency_weights(dates: pd.Series, half_life_days: float) -> np.ndarray:
    t_max = dates.max()
    age = (t_max - dates).dt.days.astype(float)
    lam = math.log(2) / max(half_life_days, 1.0)
    w = np.exp(-lam * age)
    w /= w.sum()
    return w


def make_freq_tables(df: pd.DataFrame, weights: np.ndarray | None = None):
    if weights is None:
        weights = np.ones(len(df)) / len(df)
    wfreq = np.zeros(WHITE_MAX + 1)
    mfreq = np.zeros(MEGA_MAX + 1)
    for (_, r), wt in zip(df.iterrows(), weights):
        for n in [int(r[k]) for k in ["w1", "w2", "w3", "w4", "w5"]]:
            wfreq[n] += wt
        mfreq[int(r["mega"])] += wt
    return wfreq / wfreq.sum(), mfreq / mfreq.sum()


# NEW: Chi-square goodness of fit test for randomness
def test_randomness(df: pd.DataFrame):
    """Test if number frequencies deviate significantly from uniform distribution"""
    white_counts = np.zeros(WHITE_MAX + 1)
    mega_counts = np.zeros(MEGA_MAX + 1)

    for _, r in df.iterrows():
        for n in [int(r[k]) for k in ["w1", "w2", "w3", "w4", "w5"]]:
            white_counts[n] += 1
        mega_counts[int(r["mega"])] += 1

    # Expected frequencies (uniform distribution)
    n_draws = len(df)
    expected_white = n_draws * 5 / WHITE_MAX
    expected_mega = n_draws / MEGA_MAX

    # Chi-square test
    chi2_white, p_white = stats.chisquare(white_counts[1:], f_exp=expected_white)
    chi2_mega, p_mega = stats.chisquare(mega_counts[1:], f_exp=expected_mega)

    return {
        "white_chi2": chi2_white,
        "white_pvalue": p_white,
        "mega_chi2": chi2_mega,
        "mega_pvalue": p_mega,
    }


# NEW: Moving average frequency (trend detection)
def moving_window_freq(df: pd.DataFrame, window_size: int = 50):
    """Calculate frequencies in sliding windows to detect trends"""
    if len(df) < window_size:
        return None

    windows = []
    for i in range(len(df) - window_size + 1):
        window_df = df.iloc[i : i + window_size]
        w_freq, m_freq = make_freq_tables(window_df)
        windows.append(
            {
                "end_date": window_df["date"].iloc[-1],
                "white_entropy": stats.entropy(w_freq[1:]),
                "mega_entropy": stats.entropy(m_freq[1:]),
            }
        )

    return pd.DataFrame(windows)


# NEW: Streak analysis
def analyze_streaks(df: pd.DataFrame):
    """Find longest gaps and shortest gaps between appearances"""
    white_gaps = {n: [] for n in range(1, WHITE_MAX + 1)}
    mega_gaps = {n: [] for n in range(1, MEGA_MAX + 1)}

    white_last = {n: None for n in range(1, WHITE_MAX + 1)}
    mega_last = {n: None for n in range(1, MEGA_MAX + 1)}

    for idx, r in df.iterrows():
        # White balls
        for n in [int(r[k]) for k in ["w1", "w2", "w3", "w4", "w5"]]:
            if white_last[n] is not None:
                gap = idx - white_last[n]
                white_gaps[n].append(gap)
            white_last[n] = idx

        # Powerball
        m = int(r["mega"])
        if mega_last[m] is not None:
            gap = idx - mega_last[m]
            mega_gaps[m].append(gap)
        mega_last[m] = idx

    # Calculate statistics
    white_stats = {}
    for n in range(1, WHITE_MAX + 1):
        if white_gaps[n]:
            white_stats[n] = {
                "mean_gap": np.mean(white_gaps[n]),
                "std_gap": np.std(white_gaps[n]),
                "max_gap": max(white_gaps[n]),
                "min_gap": min(white_gaps[n]),
            }

    mega_stats = {}
    for n in range(1, MEGA_MAX + 1):
        if mega_gaps[n]:
            mega_stats[n] = {
                "mean_gap": np.mean(mega_gaps[n]),
                "std_gap": np.std(mega_gaps[n]),
                "max_gap": max(mega_gaps[n]),
                "min_gap": min(mega_gaps[n]),
            }

    return white_stats, mega_stats


# NEW: Pair correlation strength
def pair_correlation_strength(df: pd.DataFrame, top_n: int = 20):
    """Calculate which pairs appear together more than expected by chance"""
    C = np.zeros((WHITE_MAX + 1, WHITE_MAX + 1), dtype=float)
    individual_counts = np.zeros(WHITE_MAX + 1)

    n_draws = len(df)

    for _, r in df.iterrows():
        w = sorted([int(r[k]) for k in ["w1", "w2", "w3", "w4", "w5"]])
        for num in w:
            individual_counts[num] += 1
        for i in range(5):
            for j in range(i + 1, 5):
                a, b = w[i], w[j]
                C[a, b] += 1
                C[b, a] += 1

    # Calculate expected co-occurrence under independence
    correlations = []
    for i in range(1, WHITE_MAX + 1):
        for j in range(i + 1, WHITE_MAX + 1):
            observed = C[i, j]
            # Expected if independent: P(i appears) * P(j appears | i appeared) * n_draws
            # Approximation: (count_i/total_slots) * (count_j/(total_slots-1)) * total_pairs
            p_i = individual_counts[i] / (n_draws * 5)
            p_j = individual_counts[j] / (n_draws * 5)
            expected = n_draws * 10 * p_i * p_j  # 10 pairs per draw

            if expected > 0:
                ratio = observed / expected
                correlations.append((i, j, observed, expected, ratio))

    # Sort by deviation from expected
    correlations.sort(key=lambda x: abs(x[4] - 1), reverse=True)
    return correlations[:top_n]


def cooccurrence_matrix(df: pd.DataFrame) -> np.ndarray:
    C = np.zeros((WHITE_MAX + 1, WHITE_MAX + 1), dtype=float)
    for _, r in df.iterrows():
        w = sorted([int(r[k]) for k in ["w1", "w2", "w3", "w4", "w5"]])
        for i in range(5):
            for j in range(i + 1, 5):
                a, b = w[i], w[j]
                C[a, b] += 1
                C[b, a] += 1
    totals = np.zeros(WHITE_MAX + 1)
    for n in range(1, WHITE_MAX + 1):
        totals[n] = ((df[["w1", "w2", "w3", "w4", "w5"]] == n).any(axis=1)).sum()
        C[n, n] = totals[n]
    return C


def days_since_last_seen(df: pd.DataFrame):
    last_seen = {n: None for n in range(1, WHITE_MAX + 1)}
    mega_last = {n: None for n in range(1, MEGA_MAX + 1)}
    for _, r in df.iterrows():
        d = r["date"]
        for n in [int(r[k]) for k in ["w1", "w2", "w3", "w4", "w5"]]:
            last_seen[n] = d
        mega_last[int(r["mega"])] = d
    t_max = df["date"].max()
    white_gap = {
        n: (t_max - last_seen[n]).days if last_seen[n] else None
        for n in range(1, WHITE_MAX + 1)
    }
    mega_gap = {
        n: (t_max - mega_last[n]).days if mega_last[n] else None
        for n in range(1, MEGA_MAX + 1)
    }
    return white_gap, mega_gap


# IMPROVED: Balanced ticket generation using constraint satisfaction
def generate_balanced_tickets(n=10, seed=123):
    """Generate tickets with balanced coverage across number ranges"""
    rng = np.random.default_rng(seed)
    tickets = []

    # Divide white balls into ranges for coverage
    ranges = [
        list(range(1, 15)),  # 1-14
        list(range(15, 29)),  # 15-28
        list(range(29, 43)),  # 29-42
        list(range(43, 57)),  # 43-56
        list(range(57, 70)),  # 57-69
    ]

    for i in range(n):
        # Pick 1 number from each range for good coverage
        whites = []
        for r in ranges:
            whites.append(rng.choice(r))
        whites = tuple(sorted(whites))

        # Cycle through powerballs
        mega = (i % MEGA_MAX) + 1
        tickets.append((whites, mega))

    return tickets


def sample_tickets_from_probs(
    wprob: np.ndarray, mprob: np.ndarray, k_tickets=10, seed=123
):
    rng = np.random.RandomState(seed)
    whites = np.arange(1, WHITE_MAX + 1)
    megas = np.arange(1, MEGA_MAX + 1)
    out = []
    for _ in range(k_tickets):
        w = tuple(
            sorted(
                rng.choice(whites, size=5, replace=False, p=wprob[1:] / wprob[1:].sum())
            )
        )
        m = int(
            rng.choice(megas, size=1, replace=False, p=mprob[1:] / mprob[1:].sum())[0]
        )
        out.append((w, m))
    seen = set()
    uniq = []
    for t in out:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    return uniq


def gridify_1d(vec_1idx: np.ndarray, rows: int, cols: int) -> np.ndarray:
    arr = vec_1idx[1:].copy()
    if len(arr) != rows * cols:
        raise ValueError("Grid shape mismatch")
    return arr.reshape(rows, cols)


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# ---------- Sidebar ----------
st.sidebar.title("Controls")
data_src = st.sidebar.radio("Data source", ["Auto-download", "Upload CSV"])
years = st.sidebar.slider("Years of history (auto)", 1, 20, 5)
half_life = st.sidebar.slider("Recency half-life (days)", 30, 365, 180, step=15)
num_tix = st.sidebar.slider("Candidate tickets", 5, 30, 12, step=1)

seed = st.sidebar.number_input(
    "Random seed",
    value=123,
    step=1,
    key="random_seed",
    help="Change this value to generate different number combinations. Press Enter or click outside the field to update.",
)
show_cooccur = st.sidebar.checkbox(
    "Show pair co-occurrence heatmap (69×69)", value=False
)

uploaded_df = None
if data_src == "Upload CSV":
    up = st.sidebar.file_uploader("CSV (date,w1..w5,mega)", type=["csv"])
    if up is not None:
        uploaded_df = load_uploaded(up.read())

# ---------- Load Data ----------
with st.spinner("Loading data…"):
    if uploaded_df is not None:
        df = uploaded_df
    else:
        df = fetch_history_years(years)

st.title("Powerball • Advanced Statistical Analysis")
st.caption(
    "⚠️ **Educational Tool**: Lottery draws are truly random. Past patterns cannot predict future outcomes."
)
st.markdown(
    f"**Draws loaded:** {len(df)} (from {df['date'].min().date()} to {df['date'].max().date()})"
)

# ---------- Analysis ----------
w_plain, m_plain = make_freq_tables(df, None)
weights = recency_weights(df["date"], half_life)
w_recent, m_recent = make_freq_tables(df, weights)

whites_rank_plain = sorted(
    [(i, float(w_plain[i])) for i in range(1, WHITE_MAX + 1)],
    key=lambda x: x[1],
    reverse=True,
)
whites_rank_recent = sorted(
    [(i, float(w_recent[i])) for i in range(1, WHITE_MAX + 1)],
    key=lambda x: x[1],
    reverse=True,
)
mega_rank_recent = sorted(
    [(i, float(m_recent[i])) for i in range(1, MEGA_MAX + 1)],
    key=lambda x: x[1],
    reverse=True,
)

white_gap, mega_gap = days_since_last_seen(df)

# NEW: Advanced analysis
randomness_test = test_randomness(df)
white_streaks, mega_streaks = analyze_streaks(df)
top_pairs = pair_correlation_strength(df, top_n=15)

# ---------- Layout ----------
tab_labels = [
    "Overview",
    "Randomness Tests",
    "Heat Maps",
    "Advanced Analysis",
    "Tickets",
    "Exports",
]
if SEEDS_AVAILABLE:
    tab_labels.append("Virtual Seeds")

tab1, tab2, tab3, tab4, tab5, tab6, *rest_tabs = st.tabs(tab_labels)
if SEEDS_AVAILABLE:
    tab7 = rest_tabs[0]

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("High-Probability List (Plain Frequency)")
        df_plain = pd.DataFrame(whites_rank_plain[:20], columns=["white", "plain_prob"])
        st.dataframe(df_plain, use_container_width=True, hide_index=True)
    with c2:
        st.subheader("High-Probability List (Recency-Weighted)")
        df_recent = pd.DataFrame(
            whites_rank_recent[:20], columns=["white", "recency_prob"]
        )
        st.dataframe(df_recent, use_container_width=True, hide_index=True)

    st.subheader("Powerball (Recency-Weighted)")
    df_mega = pd.DataFrame(mega_rank_recent[:10], columns=["powerball", "recency_prob"])
    st.dataframe(df_mega, use_container_width=True, hide_index=True)

    st.subheader("Hot / Cold (Days Since Seen)")
    cg = pd.DataFrame(
        {"white": list(white_gap.keys()), "days_since_seen": list(white_gap.values())}
    ).sort_values("days_since_seen", ascending=False)
    st.dataframe(cg.head(20), use_container_width=True, hide_index=True)

with tab2:
    st.subheader("🔬 Statistical Randomness Tests")

    st.markdown("### Chi-Square Goodness of Fit")
    st.markdown(
        "Tests whether number frequencies deviate significantly from uniform distribution (true randomness)."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.metric("White Balls χ² statistic", f"{randomness_test['white_chi2']:.2f}")
        st.metric("White Balls p-value", f"{randomness_test['white_pvalue']:.4f}")
        if randomness_test["white_pvalue"] > 0.05:
            st.success(
                "✅ White balls appear consistent with random distribution (p > 0.05)"
            )
        else:
            st.warning(
                "⚠️ White balls show statistically significant deviation from uniformity"
            )

    with col2:
        st.metric("Powerball χ² statistic", f"{randomness_test['mega_chi2']:.2f}")
        st.metric("Powerball p-value", f"{randomness_test['mega_pvalue']:.4f}")
        if randomness_test["mega_pvalue"] > 0.05:
            st.success(
                "✅ Powerball appears consistent with random distribution (p > 0.05)"
            )
        else:
            st.warning(
                "⚠️ Powerball shows statistically significant deviation from uniformity"
            )

    st.info(
        "**Interpretation**: High p-values (> 0.05) indicate the lottery behaves as expected for a truly random system."
    )

with tab3:
    st.subheader("White Ball Frequency — Plain")
    grid_p = gridify_1d(w_plain, 3, 23)
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    im1 = ax1.imshow(grid_p, aspect="auto", cmap="viridis")
    ax1.set_title("White Frequency (Plain)")
    fig1.colorbar(im1)
    st.pyplot(fig1, clear_figure=True)
    img_plain = fig_to_png_bytes(fig1)

    st.subheader("White Ball Frequency — Recency-Weighted")
    grid_r = gridify_1d(w_recent, 3, 23)
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    im2 = ax2.imshow(grid_r, aspect="auto", cmap="plasma")
    ax2.set_title("White Frequency (Recency-Weighted)")
    fig2.colorbar(im2)
    st.pyplot(fig2, clear_figure=True)
    img_recent = fig_to_png_bytes(fig2)

    st.subheader("Powerball (Recency-Weighted) — Bar")
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    xs = np.arange(1, MEGA_MAX + 1)
    ax3.bar(xs, m_recent[1:], color="coral")
    ax3.set_xlabel("Powerball Number")
    ax3.set_ylabel("Weight")
    ax3.set_title("Powerball Recency-Weighted Frequency")
    ax3.grid(axis="y", alpha=0.3)
    st.pyplot(fig3, clear_figure=True)
    img_mega = fig_to_png_bytes(fig3)

with tab4:
    st.subheader("🔍 Advanced Pattern Analysis")

    st.markdown("### Strongest Pair Correlations")
    st.markdown("Pairs that appear together more/less than expected by chance:")

    pair_df = pd.DataFrame(
        top_pairs, columns=["Num1", "Num2", "Observed", "Expected", "Ratio"]
    )
    pair_df["Deviation"] = ((pair_df["Ratio"] - 1) * 100).round(1)
    st.dataframe(
        pair_df[["Num1", "Num2", "Observed", "Expected", "Deviation"]].head(10),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("Deviation shows % above/below expected if truly independent")

    st.markdown("### Streak Statistics (Top 10 Most Variable)")
    streak_data = []
    for n, stats in white_streaks.items():
        streak_data.append(
            {
                "Number": n,
                "Mean Gap": round(stats["mean_gap"], 1),
                "Std Gap": round(stats["std_gap"], 1),
                "Max Gap": stats["max_gap"],
                "Min Gap": stats["min_gap"],
            }
        )
    streak_df = pd.DataFrame(streak_data).sort_values("Std Gap", ascending=False)
    st.dataframe(streak_df.head(10), use_container_width=True, hide_index=True)

    if show_cooccur:
        st.subheader("White Ball Pair Co-Occurrence Heatmap")
        with st.spinner("Computing co-occurrence…"):
            C = cooccurrence_matrix(df)
        fig4, ax4 = plt.subplots(figsize=(12, 10))
        im4 = ax4.imshow(C[1:, 1:], aspect="auto", cmap="coolwarm")
        ax4.set_title("White Pair Co-Occurrence (1..69)")
        ax4.set_xlabel("White Ball Number")
        ax4.set_ylabel("White Ball Number")
        fig4.colorbar(im4)
        st.pyplot(fig4, clear_figure=True)
        img_co = fig_to_png_bytes(fig4)
    else:
        img_co = None
        st.info("Enable 69×69 heatmap in sidebar (computationally intensive)")

with tab5:
    st.subheader("🎫 Candidate Tickets")

    st.warning(
        "⚠️ **Remember**: All lottery combinations have EQUAL probability. These are for entertainment only."
    )

    # ============================================================
    # TOP 10 GAMES - ADVANCED STATISTICAL ANALYSIS
    # ============================================================

    def generate_statistical_games(n_games=10, seed_value=123):
        """
        Generate games using multiple statistical factors:
        - Recency weighting
        - Historical frequency
        - Streak consistency
        - Pair correlations
        - Range balancing
        """
        games = []

        # Get numbers with favorable streak patterns (consistent gaps)
        consistent_nums = []
        for n, stats in white_streaks.items():
            if stats["std_gap"] < 8:  # Low variability = consistent
                consistent_nums.append(n)
        consistent_nums = consistent_nums[:20]

        # Get numbers from strong pairs
        pair_nums = set()
        for n1, n2, _, _, ratio in top_pairs[:10]:
            if ratio > 1.0:  # Pairs that appear together more than expected
                pair_nums.add(n1)
                pair_nums.add(n2)
        pair_nums = list(pair_nums)[:15]

        # Combine all factors with weights
        number_scores = np.zeros(WHITE_MAX + 1)

        # Score based on recency (highest weight)
        for i, (n, prob) in enumerate(whites_rank_recent[:30]):
            number_scores[n] += (30 - i) * 3.0

        # Score based on plain frequency
        for i, (n, prob) in enumerate(whites_rank_plain[:30]):
            number_scores[n] += (30 - i) * 2.0

        # Bonus for consistent patterns
        for n in consistent_nums:
            number_scores[n] += 15

        # Bonus for strong pair correlations
        for n in pair_nums:
            number_scores[n] += 10

        # Get top scored numbers
        top_scored = sorted(
            range(1, WHITE_MAX + 1), key=lambda x: number_scores[x], reverse=True
        )

        # Generate diverse games from top scored numbers
        rng = np.random.default_rng(seed_value)

        for game_idx in range(n_games):
            # Pick from top scored with some randomization
            pool_size = min(30 + game_idx * 3, 50)  # Expand pool for diversity
            pool = top_scored[:pool_size]

            # Ensure balanced coverage across ranges
            selected = []
            ranges = [
                [n for n in pool if 1 <= n <= 14],
                [n for n in pool if 15 <= n <= 28],
                [n for n in pool if 29 <= n <= 42],
                [n for n in pool if 43 <= n <= 56],
                [n for n in pool if 57 <= n <= 69],
            ]

            # Try to get one from each range
            for r in ranges:
                if r and len(selected) < 5:
                    selected.append(rng.choice(r))

            # Fill remaining with top scores
            while len(selected) < 5:
                candidates = [n for n in pool if n not in selected]
                if candidates:
                    selected.append(candidates[0])
                else:
                    break

            whites = tuple(sorted(selected[:5]))

            # Pick Powerball from top recency
            top_pb = [n for n, _ in mega_rank_recent[:8]]
            powerball = top_pb[game_idx % len(top_pb)]

            games.append((whites, powerball))

        return games

    # Display section for Top 10 Statistical Games
    st.markdown("---")
    st.markdown("### 📊 Top 10 Games - Advanced Statistical Analysis")
    st.markdown(
        "**Based on:** Recency weighting, frequency analysis, pair correlations, and streak patterns"
    )

    # Regenerate button to force update when seed changes
    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption(
            f"Current seed: **{seed}** - Change seed in sidebar and numbers will update automatically"
        )
    with col2:
        if st.button("🔄 Regenerate", use_container_width=True, key="regenerate_stats"):
            st.rerun()

    # Generate the games (using current seed value - pass explicitly for reactivity)
    statistical_games = generate_statistical_games(n_games=10, seed_value=seed)

    # Create DataFrame for clean display
    df_statistical = pd.DataFrame(
        [
            {
                "Game": i + 1,
                "White Balls": f"{w[0]:2d} - {w[1]:2d} - {w[2]:2d} - {w[3]:2d} - {w[4]:2d}",
                "Powerball": f"{m:02d}",
                "Raw": f"{w[0]}, {w[1]}, {w[2]}, {w[3]}, {w[4]}, PB:{m}",
            }
            for i, (w, m) in enumerate(statistical_games)
        ]
    )

    # Display the table (key includes seed to force update when seed changes)
    st.dataframe(
        df_statistical[["Game", "White Balls", "Powerball"]],
        use_container_width=True,
        hide_index=True,
        key=f"statistical_games_{seed}",
    )

    # Methodology explanation
    with st.expander("📖 How these games were selected"):
        st.markdown(
            """
        **Statistical Factors Used:**
        
        1. **Recency Weight (3x)**: Numbers that appeared recently get highest priority
           - Recent draws weighted exponentially more than old draws
        
        2. **Historical Frequency (2x)**: Numbers that appear most often overall
           - All-time frequency across entire dataset
        
        3. **Consistency Bonus (+15 points)**: Numbers with predictable gap patterns
           - Low standard deviation in gaps between appearances
        
        4. **Pair Correlation Bonus (+10 points)**: Numbers that appear together more than expected
           - Based on chi-square analysis of pair frequencies
        
        5. **Range Balancing**: Ensures coverage across all number ranges
           - One number from each: 1-14, 15-28, 29-42, 43-56, 57-69
        
        **Powerball Selection:** Rotates through top 8 most frequent Powerballs by recency
        
        **Scoring Formula:**
        ```
        Score = (Recency_Rank × 3) + (Frequency_Rank × 2) + Consistency_Bonus + Pair_Bonus
        ```
        
        ⚠️ **Critical Reminder**: These are statistically-informed selections for entertainment only. 
        In reality, every combination has exactly the same **1 in 292,201,338** probability.
        Past patterns cannot predict future random draws.
        """
        )

    # Download button
    st.download_button(
        "⬇️ Download Statistical Games (CSV)",
        data=df_to_csv_bytes(
            df_statistical[["Game", "Raw"]].rename(columns={"Raw": "Numbers"})
        ),
        file_name="powerball_statistical_games.csv",
        use_container_width=True,
    )

    st.info(
        "💡 **Tip**: Change the 'Random seed' in the sidebar to generate different game combinations using the same statistical criteria."
    )

    st.markdown("---")
    st.markdown("### 🎲 Other Ticket Generation Methods")

    # Generate tickets at tab level so they're accessible for exports
    balanced = generate_balanced_tickets(n=num_tix, seed=seed)
    recency = sample_tickets_from_probs(
        w_recent, m_recent, k_tickets=num_tix, seed=seed
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Balanced Coverage**")
        st.caption("Ensures numbers from all ranges (1-14, 15-28, etc.)")
        df_bal = pd.DataFrame(
            [
                {"whites": tuple(int(x) for x in w), "powerball": int(m)}
                for w, m in balanced
            ]
        )
        st.dataframe(df_bal, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Balanced tickets",
            data=df_to_csv_bytes(df_bal),
            file_name="tickets_balanced.csv",
        )

    with c2:
        st.markdown("**Recency-Weighted**")
        st.caption("Based on recent frequency patterns (entertainment)")
        df_rec = pd.DataFrame(
            [
                {"whites": tuple(int(x) for x in w), "powerball": int(m)}
                for w, m in recency
            ]
        )
        st.dataframe(df_rec, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Recency tickets",
            data=df_to_csv_bytes(df_rec),
            file_name="tickets_recency.csv",
        )

    with c3:
        st.markdown("**Quick Pick (Random)**")
        st.caption("Truly random selection")
        quick_picks = []
        rng = np.random.default_rng(seed + 999)
        for _ in range(num_tix):
            whites = tuple(
                sorted(rng.choice(range(1, WHITE_MAX + 1), 5, replace=False))
            )
            mega = int(rng.choice(range(1, MEGA_MAX + 1), 1)[0])
            quick_picks.append((whites, mega))
        df_qp = pd.DataFrame(
            [
                {"whites": tuple(int(x) for x in w), "powerball": int(m)}
                for w, m in quick_picks
            ]
        )
        st.dataframe(df_qp, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Quick picks",
            data=df_to_csv_bytes(df_qp),
            file_name="tickets_quickpick.csv",
        )

with tab6:
    st.subheader("📥 Export Reports")

    def build_pdf_bytes():
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.pagesizes import letter

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        story.append(
            Paragraph("<b>Powerball Statistical Analysis Report</b>", styles["Title"])
        )
        story.append(
            Paragraph(
                f"Draws: {len(df)} (from {df['date'].min().date()} to {df['date'].max().date()})",
                styles["Normal"],
            )
        )
        story.append(Spacer(1, 12))

        # Randomness test results
        story.append(
            Paragraph("<b>Statistical Randomness Tests</b>", styles["Heading2"])
        )
        story.append(
            Paragraph(
                f"White Balls: χ²={randomness_test['white_chi2']:.2f}, p={randomness_test['white_pvalue']:.4f}",
                styles["Normal"],
            )
        )
        story.append(
            Paragraph(
                f"Powerball: χ²={randomness_test['mega_chi2']:.2f}, p={randomness_test['mega_pvalue']:.4f}",
                styles["Normal"],
            )
        )
        story.append(Spacer(1, 10))

        # Ranked lists (top 10)
        story.append(Paragraph("<b>Top-10 White Balls (Plain)</b>", styles["Heading2"]))
        story.append(
            Paragraph(
                ", ".join(str(n) for n, _ in whites_rank_plain[:10]), styles["Normal"]
            )
        )
        story.append(Spacer(1, 6))
        story.append(
            Paragraph("<b>Top-10 White Balls (Recency)</b>", styles["Heading2"])
        )
        story.append(
            Paragraph(
                ", ".join(str(n) for n, _ in whites_rank_recent[:10]), styles["Normal"]
            )
        )
        story.append(Spacer(1, 6))
        story.append(Paragraph("<b>Top-5 Powerballs (Recency)</b>", styles["Heading2"]))
        story.append(
            Paragraph(
                ", ".join(str(n) for n, _ in mega_rank_recent[:5]), styles["Normal"]
            )
        )
        story.append(Spacer(1, 10))

        # Tickets
        story.append(
            Paragraph("<b>Suggested Tickets — Balanced</b>", styles["Heading2"])
        )
        for w, m in balanced:
            story.append(
                Paragraph(
                    f"{tuple(int(x) for x in w)}  PB:{int(m):02d}", styles["Normal"]
                )
            )
        story.append(Spacer(1, 6))
        story.append(
            Paragraph("<b>Suggested Tickets — Recency</b>", styles["Heading2"])
        )
        for w, m in recency:
            story.append(
                Paragraph(
                    f"{tuple(int(x) for x in w)}  PB:{int(m):02d}", styles["Normal"]
                )
            )
        story.append(Spacer(1, 10))

        # Charts
        story.append(Paragraph("<b>White Frequency (Plain)</b>", styles["Heading2"]))
        story.append(Image(io.BytesIO(img_plain), width=400, height=280))
        story.append(Spacer(1, 8))

        story.append(Paragraph("<b>White Frequency (Recency)</b>", styles["Heading2"]))
        story.append(Image(io.BytesIO(img_recent), width=400, height=280))
        story.append(Spacer(1, 8))

        story.append(Paragraph("<b>Powerball (Recency)</b>", styles["Heading2"]))
        story.append(Image(io.BytesIO(img_mega), width=400, height=280))
        story.append(Spacer(1, 8))

        if img_co:
            story.append(
                Paragraph("<b>White Pair Co-Occurrence</b>", styles["Heading2"])
            )
            story.append(Image(io.BytesIO(img_co), width=400, height=320))
            story.append(Spacer(1, 8))

        doc.build(story)
        return buf.getvalue()

    try:
        pdf_bytes = build_pdf_bytes()
        st.download_button(
            "📄 Download PDF report",
            data=pdf_bytes,
            file_name="Powerball_Report.pdf",
            mime="application/pdf",
        )
    except Exception as e:
        st.error(f"Error generating PDF: {e}")
        st.exception(e)

    # --- Build HTML bytes (inline images) ---
    def to_data_uri(png_bytes):
        return f"data:image/png;base64,{base64.b64encode(png_bytes).decode()}"

    def build_html_bytes():
        html = io.StringIO()
        html.write(
            "<html><head><meta charset='utf-8'><title>Powerball Report</title></head><body>"
        )
        html.write("<h1>Powerball Statistical Analysis Report</h1>")
        html.write(
            f"<p>Draws: {len(df)} (from {df['date'].min().date()} to {df['date'].max().date()})</p>"
        )

        # Randomness tests
        html.write("<h2>Statistical Randomness Tests</h2>")
        html.write(
            f"<p>White Balls: χ²={randomness_test['white_chi2']:.2f}, p={randomness_test['white_pvalue']:.4f}</p>"
        )
        html.write(
            f"<p>Powerball: χ²={randomness_test['mega_chi2']:.2f}, p={randomness_test['mega_pvalue']:.4f}</p>"
        )

        html.write("<h2>Top-10 White Balls (Plain)</h2>")
        html.write(
            "<p>" + ", ".join(str(n) for n, _ in whites_rank_plain[:10]) + "</p>"
        )
        html.write("<h2>Top-10 White Balls (Recency)</h2>")
        html.write(
            "<p>" + ", ".join(str(n) for n, _ in whites_rank_recent[:10]) + "</p>"
        )
        html.write("<h2>Top-5 Powerballs (Recency)</h2>")
        html.write("<p>" + ", ".join(str(n) for n, _ in mega_rank_recent[:5]) + "</p>")

        html.write("<h2>Suggested Tickets — Balanced</h2><ul>")
        for w, m in balanced:
            html.write(f"<li>{tuple(int(x) for x in w)} PB:{int(m):02d}</li>")
        html.write("</ul><h2>Suggested Tickets — Recency</h2><ul>")
        for w, m in recency:
            html.write(f"<li>{tuple(int(x) for x in w)} PB:{int(m):02d}</li>")
        html.write("</ul>")

        html.write("<h2>White Frequency (Plain)</h2>")
        html.write(f"<img src='{to_data_uri(img_plain)}' width='600' />")

        html.write("<h2>White Frequency (Recency)</h2>")
        html.write(f"<img src='{to_data_uri(img_recent)}' width='600' />")

        html.write("<h2>Powerball (Recency)</h2>")
        html.write(f"<img src='{to_data_uri(img_mega)}' width='600' />")

        if img_co:
            html.write("<h2>White Pair Co-Occurrence</h2>")
            html.write(f"<img src='{to_data_uri(img_co)}' width='600' />")

        html.write(
            "<hr><p><em>Disclaimer: Lottery draws are random; these visuals are descriptive only.</em></p>"
        )
        html.write("</body></html>")
        return html.getvalue().encode("utf-8")

    try:
        html_bytes = build_html_bytes()
        st.download_button(
            "🌐 Download HTML report",
            data=html_bytes,
            file_name="Powerball_Report.html",
            mime="text/html",
        )
    except Exception as e:
        st.error(f"Error generating HTML: {e}")
        st.exception(e)

if SEEDS_AVAILABLE:
    with tab7:
        render_seed_analysis(df, w_recent, m_recent, sample_tickets_from_probs)

st.markdown("---")
st.caption(
    "⚠️ Disclaimer: Powerball is truly random. These visuals highlight historical patterns for educational purposes only, not predictive guarantees."
)
