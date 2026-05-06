import numpy as np
import matplotlib.pyplot as plt

def plot_shares(first_share, second_share, third_share, fourth_share, title, normalize=False):

    groups = ("Low job", "High job")

    before = [
        first_share,
        second_share,
    ]

    after = [
        third_share,
        fourth_share,
    ]

    x = np.arange(len(groups))
    width = 0.35

    fig, ax = plt.subplots()

    bars_before = ax.bar(x - width/2, before, width, label="Before shock", color="#008cff", edgecolor="#ffffff")
    bars_after = ax.bar(x + width/2, after, width, label="After shock", color="#ff2020", hatch="", edgecolor="#ffffff")

    ax.bar_label(bars_before, fmt="%.2f", padding=3)
    ax.bar_label(bars_after, fmt="%.2f", padding=3)


    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Share of young workers")
    if normalize:
        ax.set_ylim(0, 1)
    else:
        ax.set_ylim(0, max(max(before), max(after)) * 1.2)
    # ax.set_xlabel("Job type")
    ax.set_title(title)
    # ax.set_axisbelow(True)
    # ax.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    # ax.legend()

    plt.tight_layout()
    plt.show()



def plot_series(series, title, ylabel="Average wage of young workers"):

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        series,
        color="#008cff",
        linewidth=2.0,
    )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Time")

    # Match the clean style from your example
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)

    # Optional: small margins for aesthetics
    ax.margins(x=0.02)

    plt.tight_layout()
    plt.show()