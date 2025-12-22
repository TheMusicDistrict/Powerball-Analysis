# Powerball.py
# Powerball patterns dashboard (Streamlit) with PDF & HTML export

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
    # Normalize/robust column access
    raw_cols = {c.lower(): c for c in raw.columns}
    date_col = raw_cols.get("draw date", "Draw Date")
    raw[date_col] = pd.to_datetime(raw[date_col])
    raw = raw.sort_values(date_col)
    cutoff = datetime.today() - timedelta(days=int(365.25 * years))
    raw = raw[raw[date_col] >= cutoff].copy()

    # Split winning numbers into 5 whites + 1 red (Powerball)
    win_col = raw_cols.get("winning numbers", "Winning Numbers")
    parts = raw[win_col].astype(str).str.split(" ", expand=True)
    whites = parts.iloc[:, :5].astype(int)
    whites.columns = ["w1", "w2", "w3", "w4", "w5"]
    # Prefer the 6th token as Powerball; fall back to a dedicated column if present
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
            "mega": red,  # keep internal name 'mega'
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


def gridify_1d(vec_1idx: np.ndarray, rows: int, cols: int) -> np.ndarray:
    arr = vec_1idx[1:].copy()
    if len(arr) != rows * cols:
        raise ValueError("Grid shape mismatch")
    return arr.reshape(rows, cols)


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


def fair_diverse_tickets(n=10, seed=123):
    rng = np.random.default_rng(seed)
    tickets = []
    megas_cycle = list(range(1, MEGA_MAX + 1))
    for i in range(n):
        whites = tuple(
            sorted(rng.choice(np.arange(1, WHITE_MAX + 1), size=5, replace=False))
        )
        mega = megas_cycle[i % MEGA_MAX]
        tickets.append((whites, mega))
    return tickets


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# Helpers: capture matplotlib figure to PNG bytes
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
seed = st.sidebar.number_input("Random seed", value=123, step=1)
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

st.title("Powerball • Patterns & Heat Maps")
st.caption(
    "Descriptive analysis only — lottery draws are random. This highlights patterns & recency emphasis, not guarantees."
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

# ---------- Layout ----------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "Heat Maps", "Co-Occurrence", "Tickets", "Exports"]
)

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

    coldl = st.columns(4)
    with coldl[0]:
        st.download_button(
            "⬇️ White ranks (plain)",
            data=df_to_csv_bytes(
                pd.DataFrame(whites_rank_plain, columns=["white", "plain_prob"])
            ),
            file_name="rank_whites_plain.csv",
        )
    with coldl[1]:
        st.download_button(
            "⬇️ White ranks (recency)",
            data=df_to_csv_bytes(
                pd.DataFrame(whites_rank_recent, columns=["white", "recency_prob"])
            ),
            file_name="rank_whites_recency.csv",
        )
    with coldl[2]:
        st.download_button(
            "⬇️ Powerball ranks (recency)",
            data=df_to_csv_bytes(
                pd.DataFrame(mega_rank_recent, columns=["powerball", "recency_prob"])
            ),
            file_name="rank_powerball_recency.csv",
        )
    with coldl[3]:
        st.download_button(
            "⬇️ Gaps (whites)", data=df_to_csv_bytes(cg), file_name="gap_whites.csv"
        )

with tab2:
    st.subheader("White Ball Frequency — Plain")
    grid_p = gridify_1d(w_plain, 3, 23)
    fig1, ax1 = plt.subplots()
    im1 = ax1.imshow(grid_p, aspect="auto")
    ax1.set_title("White Frequency (Plain)")
    fig1.colorbar(im1)
    st.pyplot(fig1, clear_figure=True)
    img_plain = fig_to_png_bytes(fig1)

    st.subheader("White Ball Frequency — Recency-Weighted")
    grid_r = gridify_1d(w_recent, 3, 23)
    fig2, ax2 = plt.subplots()
    im2 = ax2.imshow(grid_r, aspect="auto")
    ax2.set_title("White Frequency (Recency-Weighted)")
    fig2.colorbar(im2)
    st.pyplot(fig2, clear_figure=True)
    img_recent = fig_to_png_bytes(fig2)

    st.subheader("Powerball (Recency-Weighted) — Bar")
    fig3, ax3 = plt.subplots()
    xs = np.arange(1, MEGA_MAX + 1)
    ax3.bar(xs, m_recent[1:])
    ax3.set_xlabel("Powerball Number")
    ax3.set_ylabel("Weight")
    ax3.set_title("Powerball Recency-Weighted Frequency")
    st.pyplot(fig3, clear_figure=True)
    img_mega = fig_to_png_bytes(fig3)

with tab3:
    st.subheader("White Ball Pair Co-Occurrence (draw count)")
    img_co = None
    if show_cooccur:
        with st.spinner("Computing co-occurrence…"):
            C = cooccurrence_matrix(df)
        fig4, ax4 = plt.subplots(figsize=(7, 6))
        im4 = ax4.imshow(C[1:, 1:], aspect="auto")
        ax4.set_title("White Pair Co-Occurrence (1..69)")
        fig4.colorbar(im4)
        st.pyplot(fig4, clear_figure=True)
        img_co = fig_to_png_bytes(fig4)
    else:
        st.info(
            "Toggle the pair co-occurrence heatmap in the sidebar (heavy compute for large date ranges)."
        )

with tab4:
    st.subheader("Candidate Tickets")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Fair / Coverage-Oriented** (no historical bias)")
        fair = fair_diverse_tickets(n=num_tix, seed=seed)
        df_fair = pd.DataFrame(
            [{"whites": tuple(int(x) for x in w), "powerball": int(m)} for w, m in fair]
        )
        st.dataframe(df_fair, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Fair tickets (CSV)",
            data=df_to_csv_bytes(df_fair),
            file_name="tickets_fair.csv",
        )
    with c2:
        st.markdown("**Recency-Weighted Sample** (for fun)")
        recency = sample_tickets_from_probs(
            w_recent, m_recent, k_tickets=num_tix, seed=seed
        )
        df_rec = pd.DataFrame(
            [
                {"whites": tuple(int(x) for x in w), "powerball": int(m)}
                for w, m in recency
            ]
        )
        st.dataframe(df_rec, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Recency tickets (CSV)",
            data=df_to_csv_bytes(df_rec),
            file_name="tickets_recency.csv",
        )

with tab5:
    st.subheader("Exports")

    # --- Build PDF bytes (simple, embeds 3 charts + top lists) ---
    def build_pdf_bytes():
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.pagesizes import letter

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph("<b>Powerball Analysis Report</b>", styles["Title"]))
        story.append(
            Paragraph(
                f"Draws: {len(df)} (from {df['date'].min().date()} to {df['date'].max().date()})",
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
        story.append(Paragraph("<b>Suggested Tickets — Fair</b>", styles["Heading2"]))
        for w, m in fair:
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
        if "img_plain" in locals() and img_plain:
            story.append(
                Paragraph("<b>White Frequency (Plain)</b>", styles["Heading2"])
            )
            story.append(Image(io.BytesIO(img_plain), width=400, height=280))
            story.append(Spacer(1, 8))
        if "img_recent" in locals() and img_recent:
            story.append(
                Paragraph("<b>White Frequency (Recency)</b>", styles["Heading2"])
            )
            story.append(Image(io.BytesIO(img_recent), width=400, height=280))
            story.append(Spacer(1, 8))
        if "img_mega" in locals() and img_mega:
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

    pdf_bytes = build_pdf_bytes()
    st.download_button(
        "📄 Download PDF report",
        data=pdf_bytes,
        file_name="Powerball_Report.pdf",
        mime="application/pdf",
    )

    # --- Build HTML bytes (inline images) ---
    def to_data_uri(png_bytes):
        return f"data:image/png;base64,{base64.b64encode(png_bytes).decode()}"

    def build_html_bytes():
        html = io.StringIO()
        html.write(
            "<html><head><meta charset='utf-8'><title>Powerball Report</title></head><body>"
        )
        html.write("<h1>Powerball Analysis Report</h1>")
        html.write(
            f"<p>Draws: {len(df)} (from {df['date'].min().date()} to {df['date'].max().date()})</p>"
        )

        html.write("<h2>Top-10 White Balls (Plain)</h2>")
        html.write(", ".join(str(n) for n, _ in whites_rank_plain[:10]))
        html.write("<h2>Top-10 White Balls (Recency)</h2>")
        html.write(", ".join(str(n) for n, _ in whites_rank_recent[:10]))
        html.write("<h2>Top-5 Powerballs (Recency)</h2>")
        html.write(", ".join(str(n) for n, _ in mega_rank_recent[:5]))

        html.write("<h2>Suggested Tickets — Fair</h2><ul>")
        for w, m in fair:
            html.write(f"<li>{tuple(int(x) for x in w)} PB:{int(m):02d}</li>")
        html.write("</ul><h2>Suggested Tickets — Recency</h2><ul>")
        for w, m in recency:
            html.write(f"<li>{tuple(int(x) for x in w)} PB:{int(m):02d}</li>")
        html.write("</ul>")

        if "img_plain" in locals() and img_plain:
            html.write("<h2>White Frequency (Plain)</h2>")
            html.write(f"<img src='{to_data_uri(img_plain)}' width='600' />")
        if "img_recent" in locals() and img_recent:
            html.write("<h2>White Frequency (Recency)</h2>")
            html.write(f"<img src='{to_data_uri(img_recent)}' width='600' />")
        if "img_mega" in locals() and img_mega:
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

    html_bytes = build_html_bytes()
    st.download_button(
        "🌐 Download HTML report",
        data=html_bytes,
        file_name="Powerball_Report.html",
        mime="text/html",
    )

st.markdown("---")
st.caption(
    "Disclaimer: Powerball is random. These visuals highlight historical patterns and recency emphasis only, not predictive guarantees."
)
