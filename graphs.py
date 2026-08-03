import numpy as np
import matplotlib.pyplot as plt


def plot_model_comparison(model_baseline, model_extension):
    plt.style.use("seaborn-v0_8-whitegrid")

    def last_period(model):
        age = model.sol.age[0, :]
        return age.shape[0] - np.count_nonzero(age) - 1

    def mean_by_age(model, variable, t, age_groups, skill=None):
        means = []
        for age in age_groups:
            mask = model.sol.age[:, t] == age
            if skill is not None:
                mask &= model.sol.l_h[:, t] == skill
            means.append(np.nanmean(variable[mask, t]))
        return np.array(means)

    def high_skill_mass_by_age(model, t, age_groups):
        high_skill = model.sol.l_h[:, t] == 1
        mass = [
            np.nansum(model.sol.mass[high_skill & (model.sol.age[:, t] == age), t])
            for age in age_groups
        ]
        return np.array(mass) / np.nansum(mass)

    t_baseline = last_period(model_baseline)
    t_extension = last_period(model_extension)
    age_groups = range(model_baseline.par.n)

    wage_age_baseline = mean_by_age(model_baseline, model_baseline.sol.wage, t_baseline, age_groups)
    wage_age_extension = mean_by_age(model_extension, model_extension.sol.wage, t_extension, age_groups)

    wage_high_baseline = mean_by_age(model_baseline, model_baseline.sol.wage, t_baseline, age_groups, skill=1)
    wage_high_extension = mean_by_age(model_extension, model_extension.sol.wage, t_extension, age_groups, skill=1)
    wage_low_baseline = mean_by_age(model_baseline, model_baseline.sol.wage, t_baseline, age_groups, skill=0)
    wage_low_extension = mean_by_age(model_extension, model_extension.sol.wage, t_extension, age_groups, skill=0)

    high_skill_mass_baseline = high_skill_mass_by_age(model_baseline, t_baseline, age_groups)
    high_skill_mass_extension = high_skill_mass_by_age(model_extension, t_extension, age_groups)

    high_skill_share_baseline = mean_by_age(model_baseline, model_baseline.sol.l_h, t_baseline, age_groups)
    high_skill_share_extension = mean_by_age(model_extension, model_extension.sol.l_h, t_extension, age_groups)

    n_graphs = 6
    n_cols = 2
    n_rows = int(np.ceil(n_graphs / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = axes.flatten()

    axes[0].plot(np.nanmean(model_baseline.sol.wage, axis=0), linewidth=2, label="Baseline")
    axes[0].plot(np.nanmean(model_extension.sol.wage, axis=0), linewidth=2, label="Extension")
    axes[0].set_title("Mean Wage over Time")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Mean wage")

    axes[1].plot(np.nanmean(model_baseline.sol.l_h, axis=0), linewidth=2, label="Baseline")
    axes[1].plot(np.nanmean(model_extension.sol.l_h, axis=0), linewidth=2, label="Extension")
    axes[1].set_title("Mean High-Skill Share over Time")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Mean high-skill share")

    axes[2].plot(age_groups, wage_age_baseline, marker="o", linewidth=2, label="Baseline")
    axes[2].plot(age_groups, wage_age_extension, marker="o", linewidth=2, label="Extension")
    axes[2].set_title("Mean Wage by Age")
    axes[2].set_xlabel("Age")
    axes[2].set_ylabel("Mean wage")

    axes[3].plot(age_groups, high_skill_mass_baseline, marker="o", linewidth=2, label="Baseline")
    axes[3].plot(age_groups, high_skill_mass_extension, marker="o", linewidth=2, label="Extension")
    axes[3].set_title("Age Distribution of High-Skilled Workers")
    axes[3].set_xlabel("Age")
    axes[3].set_ylabel("Share of high-skilled worker mass")

    axes[4].plot(age_groups, high_skill_share_baseline, marker="o", linewidth=2, label="Baseline")
    axes[4].plot(age_groups, high_skill_share_extension, marker="o", linewidth=2, label="Extension")
    axes[4].set_title("High-Skill Share within Each Age Group")
    axes[4].set_xlabel("Age")
    axes[4].set_ylabel("High-skill share")

    axes[5].plot(age_groups, wage_high_baseline, marker="o", linewidth=2, label="High-skilled, baseline")
    axes[5].plot(age_groups, wage_high_extension, marker="s", linewidth=2, linestyle="--", label="High-skilled, extension")
    axes[5].plot(age_groups, wage_low_baseline, marker="o", linewidth=2, label="Low-skilled, baseline")
    axes[5].plot(age_groups, wage_low_extension, marker="s", linewidth=2, linestyle="--", label="Low-skilled, extension")
    axes[5].set_title("Wages by Age and Skill Level")
    axes[5].set_xlabel("Age")
    axes[5].set_ylabel("Mean wage")

    for ax in axes:
        ax.legend()

    fig.tight_layout()
    plt.show()