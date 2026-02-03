import os
import re
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from analysis import analyze_persona_response


# name normalization
CANONICAL_NAMES = {
    "utilitarian": "Utilitarian",
    "empath": "Empath",
    "egoist": "Egoist",
    "futurist": "Futurist",
    "hero": "Hero",
    "devilsadvocate": "DevilsAdvocate",
    "devils": "DevilsAdvocate",
    "devil": "DevilsAdvocate",
    "advocate": "DevilsAdvocate",
}

EXCLUDED_WINNERS = {"synthesizer", "adviser", "advisor", "judge"}


def normalize_winner_name(raw_name):
    if not raw_name:
        return "Unknown"

    # cleanup: no apostrophes, spaces, convert to lowercase
    cleaned = raw_name.lower().replace("'", "").replace(" ", "").replace("-", "")

    for excluded in EXCLUDED_WINNERS:
        if excluded in cleaned:
            return "Unknown"
    if cleaned in CANONICAL_NAMES:
        return CANONICAL_NAMES[cleaned]

    # partial matching for multi-word names ("Devil's Advocate")
    for key, canonical in CANONICAL_NAMES.items():
        if key in cleaned or cleaned in key:
            return canonical

    # fallback to title case of the raw name
    return raw_name.title()


def extract_winner(verdict_text, llm_ratings=None):
    """
    Extract winner persona name from Judge's verdict.

    Looks for patterns like:
    - "WINNER: Utilitarian"
    - "WINNER: Devil's Advocate"
    - "**Winner: Empath**"
    - "The Utilitarian wins"

    If no winner is found falls back to the persona with the highest
    LLM rating (if llm_ratings is provided).

    Returns:
        tuple: (winner_name: str, was_fallback: bool)
    """
    # WINNER: XXXXXX (capturing everything until newline or REASON)
    match = re.search(
        r"WINNER:\s*([A-Za-z'\s]+?)(?:\n|REASON|$)", verdict_text, re.IGNORECASE
    )
    if match:
        winner = normalize_winner_name(match.group(1).strip())
        if winner != "Unknown":
            return (winner, False)

    # **Winner: XXXXXX**
    match = re.search(r"\*\*Winner:\s*([A-Za-z'\s]+?)\*\*", verdict_text, re.IGNORECASE)
    if match:
        winner = normalize_winner_name(match.group(1).strip())
        if winner != "Unknown":
            return (winner, False)

    # The XXXXXX wins
    match = re.search(r"The\s+([A-Za-z'\s]+?)\s+wins", verdict_text, re.IGNORECASE)
    if match:
        winner = normalize_winner_name(match.group(1).strip())
        if winner != "Unknown":
            return (winner, False)

    # XXXXXX argument is strongest
    match = re.search(
        r"([A-Za-z'\s]+?)\s+argument\s+is\s+strongest", verdict_text, re.IGNORECASE
    )
    if match:
        winner = normalize_winner_name(match.group(1).strip())
        if winner != "Unknown":
            return (winner, False)

    # FALLBACK: extract winner from highest rating if jusge didn't state it individualtly
    if llm_ratings and isinstance(llm_ratings, dict) and len(llm_ratings) > 0:
        # gilter out Synthesizer since its not a comptetitor
        filtered_ratings = {
            k: v for k, v in llm_ratings.items() if k.lower() not in EXCLUDED_WINNERS
        }
        if filtered_ratings:
            best_persona = max(filtered_ratings, key=filtered_ratings.get)
            best_score = filtered_ratings[best_persona]
            # only use fallback if score is > 0
            if best_score > 0:
                return (normalize_winner_name(best_persona), True)

    return ("Unknown", False)


def plot_win_rates(all_results, output_dir):
    """
    Creates a bar chart showing how many times each persosna won.

    Args:
        all_results: List of result dicts from pipeline
        output_dir: Directory to save the chart
    """

    # counting wins
    win_counts = {}
    fallback_count = 0
    for result in all_results:
        winner, was_fallback = extract_winner(
            result.get("judge_verdict", ""), result.get("llm_ratings", {})
        )
        win_counts[winner] = win_counts.get(winner, 0) + 1

        if was_fallback:
            fallback_count += 1

    if not win_counts or (len(win_counts) == 1 and "Unknown" in win_counts):
        print(
            "  [!] Could not extract winners from verdicts, skipping win rates chart."
        )
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    personas = list(win_counts.keys())
    wins = list(win_counts.values())

    colors = sns.color_palette("husl", len(personas))

    bars = ax.bar(personas, wins, color=colors, edgecolor="black", linewidth=1.2)

    # value labels on bars
    for bar, win in zip(bars, wins):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            str(win),
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_xlabel("Persona", fontsize=12)
    ax.set_ylabel("Number of wins", fontsize=12)

    title = "Wins distribution across dilemmas"
    if fallback_count > 0:
        title += f"\n(Fallbacks = {fallback_count})"
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(wins) + 1)

    plt.tight_layout()
    filepath = os.path.join(output_dir, "win_rates.png")
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"  [+] Saved: {filepath}")


def plot_controllability_heatmap(all_results, output_dir):
    """
    Creates a heatmap showing controllability scores per persona per dilemma.

    Args:
        all_results: List of result dicts from pipeline
        output_dir: Directory to save the chart
    """
    # data matrix
    data = []
    dilemma_labels = []

    for result in all_results:
        dilemma_label = f"D{result['dilemma_id']}: {result['dilemma_title'][:15]}..."
        dilemma_labels.append(dilemma_label)

        row = {}
        for persona_name, opinion in result["opinions"].items():
            analysis = analyze_persona_response(persona_name, opinion)
            row[persona_name] = analysis["score"]
        data.append(row)

    df = pd.DataFrame(data, index=dilemma_labels)
    fig, ax = plt.subplots(figsize=(18, max(4, len(all_results) * 0.8)))

    sns.heatmap(
        df,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Controllability Score"},
    )

    ax.set_xlabel("Dilemma", fontsize=12)
    ax.set_ylabel("Persona", fontsize=12)
    ax.set_title(
        "Controllability Scores: How well each persona stayed in character",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    filepath = os.path.join(output_dir, "controllability_heatmap.png")
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"  [+] Saved: {filepath}")


def plot_metrics_comparison(all_results, output_dir):
    """
    Creates a grouped bar chart comparing avg controlability vs avg LLM rating.

    Args:
        all_results: List of result dicts from pipeline
        output_dir: Directory to save the chart
    """
    persona_ctrl_scores = {}
    persona_llm_scores = {}

    for result in all_results:
        for persona_name, opinion in result["opinions"].items():
            # skip Synthesizer - it's not supposed to be rated by the Judge
            if persona_name == "Synthesizer":
                continue
            analysis = analyze_persona_response(persona_name, opinion)
            if persona_name not in persona_ctrl_scores:
                persona_ctrl_scores[persona_name] = []
            persona_ctrl_scores[persona_name].append(analysis["score"])

        llm_ratings = result.get("llm_ratings", {})
        for persona_name, rating in llm_ratings.items():
            if persona_name not in persona_llm_scores:
                persona_llm_scores[persona_name] = []
            persona_llm_scores[persona_name].append(rating)

    # computing averages
    personas = list(persona_ctrl_scores.keys())
    avg_ctrl = [
        sum(persona_ctrl_scores[p]) / len(persona_ctrl_scores[p]) for p in personas
    ]

    # normalizing LLM scores to 0-1 range for comparison (they are 1-10 defaultly)
    avg_llm = []
    for p in personas:
        if p in persona_llm_scores and persona_llm_scores[p]:
            avg_llm.append(sum(persona_llm_scores[p]) / len(persona_llm_scores[p]) / 10)
        else:
            avg_llm.append(0)

    # creating grouped bar chart
    x = range(len(personas))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(
        [i - width / 2 for i in x],
        avg_ctrl,
        width,
        label="Controllability (Keyword-based)",
        color="steelblue",
        edgecolor="black",
    )
    bars2 = ax.bar(
        [i + width / 2 for i in x],
        avg_llm,
        width,
        label="LLM Rating (Judge, normalized)",
        color="coral",
        edgecolor="black",
    )

    # adding value labels
    for bar in bars1:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{bar.get_height():.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar in bars2:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{bar.get_height():.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xlabel("Persona", fontsize=12)
    ax.set_ylabel("Score (0-1)", fontsize=12)
    ax.set_title(
        "Controllability vs LLM affiliation rating", fontsize=14, fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(personas, rotation=45, ha="right")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Threshold (0.5)")

    plt.tight_layout()
    filepath = os.path.join(output_dir, "metrics_comparison.png")
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"  [+] Saved: {filepath}")


def plot_response_lengths(all_results, output_dir):
    """
    Creates a box plot showing response length distribution per persona.

    Args:
        all_results: List of result dicts from pipeline
        output_dir: Directory to save the chart

    """

    # collect word counts (skip Synthesizer - not a competitor)
    data = []
    for result in all_results:
        for persona_name, opinion in result["opinions"].items():
            if persona_name == "Synthesizer":
                continue
            word_count = len(opinion.split())
            data.append({"Persona": persona_name, "Word Count": word_count})

    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.boxplot(x="Persona", y="Word Count", data=df, palette="Set2", ax=ax)
    sns.stripplot(x="Persona", y="Word Count", data=df, color="black", alpha=0.5, ax=ax)

    ax.set_xlabel("Persona", fontsize=12)
    ax.set_ylabel("Response length (words)", fontsize=12)
    ax.set_title(
        "Response length distribution by Persona", fontsize=14, fontweight="bold"
    )
    ax.tick_params(axis="x", rotation=45)

    # we asked for 1-2 sentences, ~50 words if complex sentences
    ax.axhline(
        y=50, color="red", linestyle="--", alpha=0.7, label="Target max (~50 words)"
    )
    ax.legend()

    plt.tight_layout()
    filepath = os.path.join(output_dir, "response_lengths.png")
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"  [+] Saved: {filepath}")


def generate_visual_report(all_results, base_output_dir="results", model_key=None):
    """
    Generates all visualizations.

    Args:
        all_results: List of result dicts from the pipeline
        base_output_dir: Base directory for output (default: `results`)
        model_key: Optional model identifier (e.g., "3B") for folder naming

    Returns:
        str: Path to the output directory containing all files
    """
    # create timestamped output directory with optional model key
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if model_key:
        folder_name = f"run_{model_key}_{timestamp}"
    else:
        folder_name = f"run_{timestamp}"

    output_dir = os.path.join(base_output_dir, folder_name)
    os.makedirs(output_dir, exist_ok=True)

    model_info = f"(Model: {model_key})" if model_key else ""
    print("\n" + "=" * 60)
    print(f"  GENERATING VISUAL REPORT {model_info}")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}\n")

    # generate each visualization
    print("Generating charts...")

    try:
        plot_win_rates(all_results, output_dir)
    except Exception as e:
        print(f"  [!] Error generating win rates chart: {e}")

    try:
        plot_controllability_heatmap(all_results, output_dir)
    except Exception as e:
        print(f"  [!] Error generating heatmap: {e}")

    try:
        plot_metrics_comparison(all_results, output_dir)
    except Exception as e:
        print(f"  [!] Error generating metrics comparison: {e}")

    try:
        plot_response_lengths(all_results, output_dir)
    except Exception as e:
        print(f"  [!] Error generating response lengths chart: {e}")

    print(f"\n[✓] Visual report complete! See: {output_dir}")

    return output_dir
