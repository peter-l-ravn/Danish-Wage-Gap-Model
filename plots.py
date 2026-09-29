import numpy as np
import matplotlib.pyplot as plt
import graphs

def plot_shares(first_share, second_share, third_share, fourth_share, title, normalize=False, *, show_titles=None, show=True):

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

    bars_before = ax.bar(x - width/2, before, width, label="Before shock", color=graphs.BLUE, edgecolor="#ffffff")
    bars_after = ax.bar(x + width/2, after, width, label="After shock", color=graphs.RED, hatch="", edgecolor="#ffffff")

    ax.bar_label(bars_before, fmt="%.2f", padding=3)
    ax.bar_label(bars_after, fmt="%.2f", padding=3)


    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Share of young workers")
    if normalize:
        ax.set_ylim(0, 1)
    else:
        ax.set_ylim(0, max(max(before), max(after)) * 1.2)
    titles = graphs.SHOW_TITLES if show_titles is None else show_titles
    if titles:
        ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)

    ax.set_axisbelow(True)
    ax.grid(True, color="0.8", linewidth=0.8)
    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax



def plot_series(series, title, ylabel="Average wage of young workers", *, show_titles=None, show=True):

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(1991, 2024)

    ax.plot(
        x,
        series,
        color=graphs.BLUE,
        linewidth=2.0,
    )

    titles = graphs.SHOW_TITLES if show_titles is None else show_titles
    if titles:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Time")

    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)

    ax.margins(x=0.02)

    ax.set_axisbelow(True)
    ax.grid(True, color="0.8", linewidth=0.8)
    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax
