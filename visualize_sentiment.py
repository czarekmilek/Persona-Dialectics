import matplotlib.pyplot as plt
import numpy as np

# data from report (3B model)
personas = [
    "Utilitarian",
    "Empath",
    "Egoist",
    "Futurist",
    "Hero",
    "DevilsAdvocate",
    "Synthesizer",
]

polarity = [0.31, 0.09, 0.22, 0.11, 0.20, 0.01, 0.21]
subjectivity = [0.60, 0.52, 0.63, 0.59, 0.62, 0.53, 0.58]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

colors = plt.cm.Set2(np.linspace(0, 1, len(personas)))

bars1 = ax1.barh(personas, polarity, color=colors, edgecolor="black", linewidth=1.2)
ax1.axvline(x=0, color="gray", linestyle="--", alpha=0.7)
ax1.set_xlim(-0.5, 0.5)
ax1.set_xlabel("Polarity (-1 = negative, +1 = positive)", fontsize=11)
ax1.set_title("Sentiment Polarity by Persona", fontsize=14, fontweight="bold")

for bar, val in zip(bars1, polarity):
    ax1.text(
        val + 0.02,
        bar.get_y() + bar.get_height() / 2,
        f"{val:+.2f}",
        va="center",
        fontsize=10,
        fontweight="bold",
    )

bars2 = ax2.barh(personas, subjectivity, color=colors, edgecolor="black", linewidth=1.2)
ax2.axvline(x=0.5, color="red", linestyle="--", alpha=0.7, label="Neutral (0.5)")
ax2.set_xlim(0, 1)
ax2.set_xlabel("Subjectivity (0 = objective, 1 = subjective)", fontsize=11)
ax2.set_title("Subjectivity by Persona", fontsize=14, fontweight="bold")
ax2.legend(loc="lower right")

for bar, val in zip(bars2, subjectivity):
    ax2.text(
        val + 0.02,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.2f}",
        va="center",
        fontsize=10,
        fontweight="bold",
    )

plt.tight_layout()
plt.savefig(
    "M:/Uni/LM/Project/Persona-Dialectics/results/run_0.5B_20260203_144258/sentiment_analysis.png",
    dpi=150,
)
plt.savefig(
    "M:/Uni/LM/Project/Persona-Dialectics/sentiment_visualization2.png", dpi=150
)
print("Saved: sentiment_visualization.png")
plt.show()
