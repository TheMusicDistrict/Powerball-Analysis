# ============================================================
# VIRTUAL SEED ASSIGNMENT - Pattern Detection in Historical Draws
# ============================================================

import os

# Fix KMeans memory leak warning on Windows
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
from scipy.stats import entropy as calc_entropy

try:
    from statsmodels.tsa.stattools import acf
except ImportError:
    # Fallback if statsmodels is not installed
    def acf(seeds, nlags=None, fft=True):
        """Simple autocorrelation fallback"""
        if nlags is None:
            nlags = min(10, len(seeds) // 2)
        n = len(seeds)
        result = [1.0]  # lag 0 is always 1
        mean = np.mean(seeds)
        var = np.var(seeds)
        if var == 0:
            return np.array([1.0] + [0.0] * nlags)
        for lag in range(1, min(nlags + 1, n)):
            cov = np.mean((seeds[:-lag] - mean) * (seeds[lag:] - mean))
            result.append(cov / var)
        return np.array(result)


def calculate_draw_signature(whites, powerball):
    """
    Create a unique numerical signature for each draw.
    This signature can be treated like a 'virtual seed'.
    """
    signatures = {}

    # Signature 1: Sum-based
    signatures["sum_signature"] = sum(whites) + powerball

    # Signature 2: Range spread
    signatures["spread_signature"] = max(whites) - min(whites)

    # Signature 3: Odd/Even pattern (binary encoding)
    odd_even = sum([2**i for i, n in enumerate(whites) if n % 2 == 1])
    signatures["odd_even_signature"] = odd_even + (32 if powerball % 2 == 1 else 0)

    # Signature 4: Low/High pattern (below/above 35)
    low_high = sum([2**i for i, n in enumerate(whites) if n > 35])
    signatures["low_high_signature"] = low_high + (32 if powerball > 13 else 0)

    # Signature 5: Prime number pattern
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    primes = sum([2**i for i, n in enumerate(whites) if is_prime(n)])
    signatures["prime_signature"] = primes + (32 if is_prime(powerball) else 0)

    # Signature 6: Digit sum pattern
    digit_sums = [sum(int(d) for d in str(n)) for n in whites]
    signatures["digit_sum_signature"] = sum(digit_sums) + sum(
        int(d) for d in str(powerball)
    )

    # Signature 7: Sequential pairs (consecutive numbers)
    sorted_w = sorted(whites)
    sequential_count = sum([1 for i in range(4) if sorted_w[i + 1] - sorted_w[i] == 1])
    signatures["sequential_signature"] = sequential_count * 20 + (powerball % 10)

    # Signature 8: Modulo pattern (mod 10)
    mod_pattern = sum([(n % 10) * (10**i) for i, n in enumerate(whites)])
    signatures["modulo_signature"] = (mod_pattern + powerball) % 10000

    # MASTER SIGNATURE: Weighted combination
    signatures["master_signature"] = (
        signatures["sum_signature"] * 3
        + signatures["spread_signature"] * 2
        + signatures["odd_even_signature"] * 5
        + signatures["low_high_signature"] * 4
        + signatures["prime_signature"] * 3
        + signatures["digit_sum_signature"] * 2
        + signatures["sequential_signature"] * 6
    ) % 10000

    return signatures


def assign_virtual_seeds_to_history(df, use_signature="master_signature"):
    """
    Assign virtual seeds to all historical draws based on their patterns.
    """
    draw_analysis = []

    for idx, row in df.iterrows():
        whites = tuple(
            sorted(
                [
                    int(row["w1"]),
                    int(row["w2"]),
                    int(row["w3"]),
                    int(row["w4"]),
                    int(row["w5"]),
                ]
            )
        )
        powerball = int(row["mega"])

        # Calculate all signatures
        sigs = calculate_draw_signature(whites, powerball)

        draw_analysis.append(
            {
                "date": row["date"],
                "draw_index": idx,
                "numbers": f"{whites[0]:2d},{whites[1]:2d},{whites[2]:2d},{whites[3]:2d},{whites[4]:2d} PB:{powerball:02d}",
                "virtual_seed": sigs[use_signature],
                "sum_sig": sigs["sum_signature"],
                "spread_sig": sigs["spread_signature"],
                "odd_even_sig": sigs["odd_even_signature"],
                "low_high_sig": sigs["low_high_signature"],
                "prime_sig": sigs["prime_signature"],
                "digit_sum_sig": sigs["digit_sum_signature"],
                "sequential_sig": sigs["sequential_signature"],
                "modulo_sig": sigs["modulo_signature"],
            }
        )

    return pd.DataFrame(draw_analysis)


def detect_seed_patterns(seed_df, lookback=50):
    """
    Advanced pattern detection in virtual seed sequences.
    """
    if len(seed_df) < lookback:
        lookback = len(seed_df)

    recent = seed_df.tail(lookback).copy()
    seeds = recent["virtual_seed"].values

    patterns = {}

    # 1. CYCLIC PATTERNS - Look for repeating cycles
    for cycle_length in [3, 5, 7, 10, 13]:
        if len(seeds) >= cycle_length * 2:
            segments = [seeds[i::cycle_length] for i in range(cycle_length)]
            correlations = []
            for seg in segments:
                if len(seg) > 1:
                    # Check if values are similar (within 10% range)
                    mean_val = np.mean(seg)
                    std_val = np.std(seg)
                    if mean_val > 0:
                        cv = std_val / mean_val  # Coefficient of variation
                        correlations.append(cv)

            if correlations:
                avg_cv = np.mean(correlations)
                patterns[f"cycle_{cycle_length}"] = {
                    "strength": max(0, 1 - avg_cv),  # Lower CV = stronger pattern
                    "confidence": (
                        "High" if avg_cv < 0.3 else "Medium" if avg_cv < 0.5 else "Low"
                    ),
                }

    # 2. TREND PATTERNS - Linear, exponential, oscillating
    X = np.arange(len(seeds)).reshape(-1, 1)
    y = seeds

    # Linear trend
    lr = LinearRegression()
    lr.fit(X, y)
    y_pred = lr.predict(X)
    r2_linear = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))

    patterns["linear_trend"] = {
        "r_squared": r2_linear,
        "slope": lr.coef_[0],
        "direction": "Increasing" if lr.coef_[0] > 0 else "Decreasing",
        "strength": abs(r2_linear),
    }

    # 3. OSCILLATION PATTERNS - Sine wave fitting
    def sine_wave(x, amplitude, frequency, phase, offset):
        return amplitude * np.sin(2 * np.pi * frequency * x + phase) + offset

    try:
        x_norm = np.arange(len(seeds)) / len(seeds)
        popt, _ = curve_fit(
            sine_wave,
            x_norm,
            seeds,
            p0=[np.std(seeds), 0.1, 0, np.mean(seeds)],
            maxfev=5000,
        )
        y_sine = sine_wave(x_norm, *popt)
        r2_sine = 1 - (
            np.sum((seeds - y_sine) ** 2) / np.sum((seeds - np.mean(seeds)) ** 2)
        )

        patterns["oscillation"] = {
            "r_squared": max(0, r2_sine),
            "frequency": popt[1],
            "amplitude": abs(popt[0]),
            "strength": max(0, r2_sine),
        }
    except Exception:
        patterns["oscillation"] = {"r_squared": 0, "strength": 0}

    # 4. CLUSTERING PATTERNS - Seeds tend to cluster in ranges
    n_clusters = min(5, len(seeds) // 10)
    if n_clusters >= 2:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(seeds.reshape(-1, 1))

        # Measure cluster stability (do consecutive draws stay in same cluster?)
        cluster_changes = sum(
            [1 for i in range(len(clusters) - 1) if clusters[i] != clusters[i + 1]]
        )
        stability = 1 - (cluster_changes / (len(clusters) - 1))

        patterns["clustering"] = {
            "n_clusters": n_clusters,
            "stability": stability,
            "centers": kmeans.cluster_centers_.flatten().tolist(),
            "strength": stability,
        }

    # 5. AUTOCORRELATION - Do seeds depend on previous seeds?
    try:
        autocorr = acf(seeds, nlags=min(10, len(seeds) // 2), fft=True)
        # Look for significant autocorrelation (lag 1-5)
        significant_lags = [
            i for i, val in enumerate(autocorr[1:6], 1) if abs(val) > 0.2
        ]

        patterns["autocorrelation"] = {
            "lag_1": autocorr[1] if len(autocorr) > 1 else 0,
            "significant_lags": significant_lags,
            "max_autocorr": max(abs(autocorr[1:6])) if len(autocorr) > 1 else 0,
            "strength": max(abs(autocorr[1:6])) if len(autocorr) > 1 else 0,
        }
    except Exception:
        patterns["autocorrelation"] = {"strength": 0}

    # 6. RANGE PATTERNS - Do seeds stay in certain ranges?
    ranges = {
        "0-2000": len([s for s in seeds if 0 <= s < 2000]),
        "2000-4000": len([s for s in seeds if 2000 <= s < 4000]),
        "4000-6000": len([s for s in seeds if 4000 <= s < 6000]),
        "6000-8000": len([s for s in seeds if 6000 <= s < 8000]),
        "8000-10000": len([s for s in seeds if 8000 <= s < 10000]),
    }

    total = sum(ranges.values())
    range_probs = {k: v / total for k, v in ranges.items()}

    # Entropy - lower entropy means more concentrated in certain ranges
    range_entropy = calc_entropy(list(range_probs.values()))
    max_entropy = np.log(5)  # Maximum entropy for 5 equal categories
    concentration = 1 - (range_entropy / max_entropy)

    patterns["range_preference"] = {
        "distribution": range_probs,
        "favorite_range": max(ranges, key=ranges.get),
        "concentration": concentration,
        "strength": concentration,
    }

    return patterns


def predict_next_seed_from_patterns(seed_df, patterns, n_predictions=5):
    """
    Use detected patterns to predict next virtual seeds.
    """
    seeds = seed_df["virtual_seed"].values
    predictions = {}

    # Method 1: Linear trend extrapolation
    if patterns.get("linear_trend", {}).get("strength", 0) > 0.3:
        X = np.arange(len(seeds)).reshape(-1, 1)
        lr = LinearRegression()
        lr.fit(X, seeds)
        next_x = np.array([[len(seeds)]])
        pred = int(lr.predict(next_x)[0]) % 10000
        predictions["Linear Trend"] = pred

    # Method 2: Oscillation pattern
    if patterns.get("oscillation", {}).get("strength", 0) > 0.3:
        osc = patterns["oscillation"]
        next_phase = (len(seeds) / len(seeds)) * 2 * np.pi * osc["frequency"]
        pred = (
            int(osc.get("amplitude", 0) * np.sin(next_phase) + np.mean(seeds)) % 10000
        )
        predictions["Oscillation"] = pred

    # Method 3: Cycle repetition
    strongest_cycle = None
    max_strength = 0
    for key, val in patterns.items():
        if key.startswith("cycle_") and val.get("strength", 0) > max_strength:
            max_strength = val["strength"]
            strongest_cycle = int(key.split("_")[1])

    if strongest_cycle and max_strength > 0.5:
        # Look back one cycle
        if len(seeds) >= strongest_cycle:
            pred = int(seeds[-strongest_cycle]) % 10000
            predictions[f"Cycle-{strongest_cycle}"] = pred

    # Method 4: Autocorrelation prediction
    if patterns.get("autocorrelation", {}).get("strength", 0) > 0.2:
        auto = patterns["autocorrelation"]
        significant_lags = auto.get("significant_lags", [])
        if significant_lags:
            lag = significant_lags[0]
            if len(seeds) >= lag:
                pred = int(seeds[-lag]) % 10000
                predictions[f"Autocorr-Lag{lag}"] = pred

    # Method 5: Range-based prediction
    if patterns.get("range_preference", {}).get("strength", 0) > 0.4:
        fav_range = patterns["range_preference"]["favorite_range"]
        range_map = {
            "0-2000": (0, 2000),
            "2000-4000": (2000, 4000),
            "4000-6000": (4000, 6000),
            "6000-8000": (6000, 8000),
            "8000-10000": (8000, 10000),
        }
        low, high = range_map[fav_range]
        # Use recent average within that range
        recent_in_range = [s for s in seeds[-20:] if low <= s < high]
        if recent_in_range:
            pred = int(np.mean(recent_in_range)) % 10000
            predictions["Range Pattern"] = pred

    # Method 6: Ensemble (weighted average of all methods)
    if len(predictions) > 1:
        weights = []
        values = []
        for method, pred in predictions.items():
            # Weight by pattern strength
            if "Linear" in method:
                weight = patterns.get("linear_trend", {}).get("strength", 0)
            elif "Oscillation" in method:
                weight = patterns.get("oscillation", {}).get("strength", 0)
            elif "Cycle" in method:
                weight = max_strength
            elif "Autocorr" in method:
                weight = patterns.get("autocorrelation", {}).get("strength", 0)
            elif "Range" in method:
                weight = patterns.get("range_preference", {}).get("strength", 0)
            else:
                weight = 0.5

            weights.append(weight)
            values.append(pred)

        if sum(weights) > 0:
            ensemble = int(np.average(values, weights=weights)) % 10000
            predictions["Ensemble"] = ensemble

    return predictions


def backtest_seed_methods(seed_df, min_lookback=50, step=1):
    """
    Walks forward through history and, for each draw, asks:
    'Based on ONLY the past seeds, which method would have come
    closest to the next seed?'

    Returns a DataFrame with one row per evaluated draw.
    """
    records = []

    # We need at least min_lookback draws before we start testing
    for t in range(min_lookback, len(seed_df), step):
        history_window = seed_df.iloc[:t].copy()

        # Detect patterns on past-only data
        patterns = detect_seed_patterns(history_window, lookback=min_lookback)

        # Predict next seed from those patterns (using only recent part)
        preds = predict_next_seed_from_patterns(
            history_window.tail(min_lookback), patterns
        )

        if not preds:
            continue  # nothing to evaluate

        # Actual next seed at time t
        actual_row = seed_df.iloc[t]
        actual_seed = int(actual_row["virtual_seed"])

        # Compute absolute error for each method
        method_errors = {
            method: abs(int(pred) - actual_seed) for method, pred in preds.items()
        }

        # Find "winner" for this draw
        best_method, best_error = min(method_errors.items(), key=lambda kv: kv[1])

        record = {
            "draw_index": int(actual_row["draw_index"]),
            "date": actual_row["date"],
            "virtual_seed": actual_seed,
            "best_method": best_method,
            "best_error": best_error,
        }

        # Optionally store each method's prediction/error too
        for method, pred in preds.items():
            record[f"pred_{method}"] = int(pred)
            record[f"err_{method}"] = method_errors[method]

        records.append(record)

    return pd.DataFrame(records)


# ============================================================
# COMPLETE VISUALIZATION SECTION
# ============================================================
# NOTE: This section requires variables from the main app:
# - df: DataFrame with historical draws
# - w_recent, m_recent: Recent probability distributions
# - sample_tickets_from_probs: Function to generate tickets


def render_seed_analysis(
    df, w_recent=None, m_recent=None, sample_tickets_from_probs=None
):
    """
    Render the virtual seed pattern analysis section.

    Args:
        df: DataFrame with historical Powerball draws
        w_recent: Recent white ball probability distribution (optional)
        m_recent: Recent Powerball probability distribution (optional)
        sample_tickets_from_probs: Function to generate tickets (optional)
    """
    st.markdown("---")
    st.markdown("### 🧬 Virtual Seed Pattern Analysis")
    st.markdown(
        """
**Concept:** Each historical draw is assigned a 'virtual seed' based on its numerical signature.
We then analyze if these virtual seeds show any patterns over time.
"""
    )

    # Choose signature type
    signature_type = st.selectbox(
        "Select Signature Method:",
        [
            "master_signature",
            "sum_signature",
            "odd_even_signature",
            "low_high_signature",
            "prime_signature",
            "modulo_signature",
        ],
        help="Different ways to convert draw numbers into a single signature value",
    )

    lookback_draws = st.slider("Analyze last N draws:", 20, 200, 100, step=10)

    with st.spinner("Assigning virtual seeds and detecting patterns..."):
        # Assign virtual seeds
        seed_df = assign_virtual_seeds_to_history(df, use_signature=signature_type)

        # Detect patterns
        patterns = detect_seed_patterns(seed_df, lookback=lookback_draws)

        # Predict next seeds
        predictions = predict_next_seed_from_patterns(
            seed_df.tail(lookback_draws), patterns
        )

        # Assign methods to past drawings (for display)
        # Run a quick backtest to assign methods to recent draws
        if len(seed_df) >= 50:
            method_assignments = backtest_seed_methods(
                seed_df,
                min_lookback=min(50, len(seed_df) // 2),
                step=max(1, len(seed_df) // 100),
            )
            # Merge method assignments back into seed_df
            if not method_assignments.empty:
                seed_df = seed_df.merge(
                    method_assignments[["draw_index", "best_method", "best_error"]],
                    on="draw_index",
                    how="left",
                )

    # Display virtual seed history
    st.markdown("#### Virtual Seed Timeline (Most Recent 20)")
    display_cols = ["date", "numbers", "virtual_seed"]
    if "best_method" in seed_df.columns:
        display_cols.append("best_method")
    recent_seeds = seed_df.tail(20)[display_cols]
    st.dataframe(recent_seeds, use_container_width=True, hide_index=True)

    # Visualize seed evolution
    st.markdown("#### Seed Evolution Over Time")
    fig_evolution, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Full history
    plot_data = seed_df.tail(lookback_draws)
    ax1.plot(
        range(len(plot_data)),
        plot_data["virtual_seed"],
        linewidth=2,
        color="#1f77b4",
        alpha=0.7,
    )
    ax1.fill_between(
        range(len(plot_data)), plot_data["virtual_seed"], alpha=0.3, color="#1f77b4"
    )
    ax1.set_title(
        f"Virtual Seed Pattern (Last {lookback_draws} Draws)",
        fontsize=14,
        fontweight="bold",
    )
    ax1.set_xlabel("Draw Number")
    ax1.set_ylabel("Virtual Seed Value")
    ax1.grid(True, alpha=0.3)

    # Recent zoom (last 30)
    recent_plot = plot_data.tail(30)
    ax2.plot(
        range(len(recent_plot)),
        recent_plot["virtual_seed"],
        marker="o",
        linewidth=2,
        markersize=6,
        color="#ff7f0e",
    )
    ax2.set_title("Recent Pattern (Last 30 Draws)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Draw Number (Recent)")
    ax2.set_ylabel("Virtual Seed Value")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig_evolution, clear_figure=True)

    # Pattern Detection Results
    st.markdown("#### 🔍 Detected Patterns")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Linear Trend**")
        lt = patterns.get("linear_trend", {})
        st.metric("R² Score", f"{lt.get('r_squared', 0):.3f}")
        st.metric("Direction", lt.get("direction", "N/A"))
        st.metric("Strength", f"{lt.get('strength', 0)*100:.1f}%")

        st.markdown("**Oscillation Pattern**")
        osc = patterns.get("oscillation", {})
        st.metric("R² Score", f"{osc.get('r_squared', 0):.3f}")
        st.metric("Frequency", f"{osc.get('frequency', 0):.4f}")
        st.metric("Strength", f"{osc.get('strength', 0)*100:.1f}%")

    with col2:
        st.markdown("**Autocorrelation**")
        auto = patterns.get("autocorrelation", {})
        st.metric("Lag-1 Correlation", f"{auto.get('lag_1', 0):.3f}")
        st.metric("Max Correlation", f"{auto.get('max_autocorr', 0):.3f}")
        st.metric("Significant Lags", str(auto.get("significant_lags", [])))

        st.markdown("**Range Preference**")
        rp = patterns.get("range_preference", {})
        st.metric("Favorite Range", rp.get("favorite_range", "N/A"))
        st.metric("Concentration", f"{rp.get('concentration', 0)*100:.1f}%")

    # Cyclic patterns
    st.markdown("**Cyclic Patterns Detected**")
    cycles = {k: v for k, v in patterns.items() if k.startswith("cycle_")}
    if cycles:
        cycle_df = pd.DataFrame(
            [
                {
                    "Cycle Length": k.split("_")[1],
                    "Strength": f"{v['strength']*100:.1f}%",
                    "Confidence": v["confidence"],
                }
                for k, v in cycles.items()
            ]
        ).sort_values("Strength", ascending=False)
        st.dataframe(cycle_df, use_container_width=True, hide_index=True)
    else:
        st.info("No significant cyclic patterns detected")

    # Next Seed Predictions
    st.markdown("---")
    st.markdown("### 🎯 Predicted Next Virtual Seeds")

    if predictions:
        pred_df = pd.DataFrame(
            [
                {"Method": method, "Predicted Seed": seed}
                for method, seed in predictions.items()
            ]
        )
        st.dataframe(pred_df, use_container_width=True, hide_index=True)

    # Generate numbers for each predicted seed
    st.markdown("#### Generated Numbers from Predicted Seeds")

    for method, pred_seed in list(predictions.items())[:3]:  # Top 3 predictions
        st.markdown(f"**{method} (Seed: {pred_seed})**")

        # Generate tickets using this seed (if function provided)
        if sample_tickets_from_probs and w_recent is not None and m_recent is not None:
            generated = sample_tickets_from_probs(
                w_recent, m_recent, k_tickets=5, seed=pred_seed
            )

            gen_df = pd.DataFrame(
                [
                    {
                        "Pick": i + 1,
                        "Numbers": f"{w[0]:2d} - {w[1]:2d} - {w[2]:2d} - {w[3]:2d} - {w[4]:2d}  PB:{m:02d}",
                    }
                    for i, (w, m) in enumerate(generated)
                ]
            )

            st.dataframe(gen_df, use_container_width=True, hide_index=True)
        else:
            st.info("Ticket generation function not available")

    else:
        st.warning("No strong patterns detected for prediction")

    # Method Backtest Section
    st.markdown("---")
    with st.expander(
        "📊 Method Backtest (Which method fit past draws best?)", expanded=False
    ):
        bt_lookback = st.slider(
            "Minimum lookback per test (draws used to detect patterns at each step):",
            30,
            150,
            60,
            step=10,
            help="For each historical test, we use this many past draws to detect patterns "
            "before predicting the next seed.",
        )

        bt_step = st.selectbox(
            "Evaluate every Nth draw (for speed):",
            [1, 2, 3, 5],
            index=1,
            help="Step of 1 = test every draw, step of 2 = every other draw, etc.",
        )

        with st.spinner("Running backtest of methods over seed history..."):
            bt_df = backtest_seed_methods(
                seed_df, min_lookback=bt_lookback, step=bt_step
            )

        if bt_df.empty:
            st.warning(
                "Not enough history to run this backtest with the current settings."
            )
        else:
            st.markdown("#### Best Method by Draw (Most Recent 50 tests)")
            st.dataframe(
                bt_df.tail(50)[["date", "virtual_seed", "best_method", "best_error"]],
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("#### Method Win Counts (across tested draws)")
            win_counts = bt_df["best_method"].value_counts().reset_index()
            win_counts.columns = ["Method", "Wins"]
            st.bar_chart(win_counts.set_index("Method")["Wins"])

            st.caption(
                "This backtest is purely exploratory. It shows which *pattern method* "
                "would have been closest to past seeds, but it **does not** create any "
                "predictive edge for future lottery draws."
            )

    # Pattern strength summary
    st.markdown("---")
    st.markdown("### 📊 Overall Pattern Strength")

    all_strengths = []
    for key, val in patterns.items():
        strength = val.get("strength", 0) if isinstance(val, dict) else 0
        if strength > 0:
            all_strengths.append((key, strength))

    if all_strengths:
        all_strengths.sort(key=lambda x: x[1], reverse=True)

        fig_strength, ax = plt.subplots(figsize=(10, 6))
        names = [s[0].replace("_", " ").title() for s in all_strengths]
        values = [s[1] * 100 for s in all_strengths]

        bars = ax.barh(names, values, color="#2ecc71")
        ax.set_xlabel("Strength (%)", fontsize=12)
        ax.set_title("Pattern Detection Strength", fontsize=14, fontweight="bold")
        ax.set_xlim(0, 100)
        ax.grid(axis="x", alpha=0.3)

        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(
                width + 2,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.1f}%",
                ha="left",
                va="center",
            )

        plt.tight_layout()
        st.pyplot(fig_strength, clear_figure=True)

    # Critical disclaimer
    st.error(
        """
⚠️ **CRITICAL UNDERSTANDING:**

These "virtual seeds" are mathematical constructs we've created to analyze patterns.
Real lottery drawings:
- Use physical balls and machines
- Have NO actual "seed" or computer generation
- Each draw is completely independent
- Past patterns CANNOT predict future outcomes

**This analysis shows whether PAST draws happened to have mathematical relationships.**
**It does NOT and CANNOT predict future draws.**

Think of it like finding patterns in cloud shapes - interesting to observe, but clouds 
don't follow those patterns intentionally.
"""
    )
