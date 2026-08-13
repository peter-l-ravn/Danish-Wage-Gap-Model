import numpy as np
import matplotlib.pyplot as plt

def plot_model_comparison(model_baseline, model_extension):
    plt.style.use("seaborn-v0_8-whitegrid")

    def last_period(model):
        age = model.sol.age[0, :]
        return age.shape[0] - np.count_nonzero(age) - 1

    def weighted_mean(values, mass):
        valid = ~np.isnan(values) & ~np.isnan(mass)
        if not np.any(valid):
            return np.nan
        return np.sum(values[valid] * mass[valid]) / np.sum(mass[valid])

    def mean_by_age(model, variable, t, age_groups, skill=None):
        means = []
        for age in age_groups:
            mask = model.sol.age[:, t] == age
            if skill is not None:
                mask &= model.sol.l_h[:, t] == skill
            means.append(weighted_mean(variable[mask, t], model.sol.mass[mask, t]))
        return np.array(means)

    def mean_over_time(model, variable):
        return np.array([
            weighted_mean(variable[:, t], model.sol.mass[:, t])
            for t in range(variable.shape[1])
        ])

    def high_skill_mass_by_age(model, t, age_groups):
        high_skill = model.sol.l_h[:, t] == 1
        mass = np.array([
            np.nansum(model.sol.mass[high_skill & (model.sol.age[:, t] == age), t])
            for age in age_groups
        ])
        return mass / np.nansum(mass)

    t_baseline = last_period(model_baseline)
    t_extension = last_period(model_extension)
    age_groups = range(model_baseline.par.n)

    # Wage by age
    wage_age_baseline = mean_by_age(model_baseline, model_baseline.sol.wage, t_baseline, age_groups)
    wage_age_extension = mean_by_age(model_extension, model_extension.sol.wage, t_extension, age_groups)

    # Wage by age and skill
    wage_high_baseline = mean_by_age(model_baseline, model_baseline.sol.wage, t_baseline, age_groups, skill=1)
    wage_high_extension = mean_by_age(model_extension, model_extension.sol.wage, t_extension, age_groups, skill=1)
    wage_low_baseline = mean_by_age(model_baseline, model_baseline.sol.wage, t_baseline, age_groups, skill=0)
    wage_low_extension = mean_by_age(model_extension, model_extension.sol.wage, t_extension, age_groups, skill=0)

    # High-skill allocation
    high_skill_mass_baseline = high_skill_mass_by_age(model_baseline, t_baseline, age_groups)
    high_skill_mass_extension = high_skill_mass_by_age(model_extension, t_extension, age_groups)
    high_skill_share_baseline = mean_by_age(model_baseline, model_baseline.sol.l_h, t_baseline, age_groups)
    high_skill_share_extension = mean_by_age(model_extension, model_extension.sol.l_h, t_extension, age_groups)

    # Aggregate variables over time
    mean_wage_baseline = mean_over_time(model_baseline, model_baseline.sol.wage)
    mean_wage_extension = mean_over_time(model_extension, model_extension.sol.wage)
    high_skill_share_time_baseline = mean_over_time(model_baseline, model_baseline.sol.l_h)
    high_skill_share_time_extension = mean_over_time(model_extension, model_extension.sol.l_h)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()

    axes[0].plot(mean_wage_baseline, linewidth=2, label="Baseline")
    axes[0].plot(mean_wage_extension, linewidth=2, label="Extension")
    axes[0].set(title="Mean Wage over Time", xlabel="Time", ylabel="Mass-weighted mean wage")

    axes[1].plot(high_skill_share_time_baseline, linewidth=2, label="Baseline")
    axes[1].plot(high_skill_share_time_extension, linewidth=2, label="Extension")
    axes[1].set(title="Mean High-Skill Share over Time", xlabel="Time", ylabel="Mass-weighted high-skill share")

    axes[2].plot(age_groups, wage_age_baseline, marker="o", linewidth=2, label="Baseline")
    axes[2].plot(age_groups, wage_age_extension, marker="o", linewidth=2, label="Extension")
    axes[2].set(title="Mean Wage by Age", xlabel="Age", ylabel="Mass-weighted mean wage")

    axes[3].plot(age_groups, high_skill_mass_baseline, marker="o", linewidth=2, label="Baseline")
    axes[3].plot(age_groups, high_skill_mass_extension, marker="o", linewidth=2, label="Extension")
    axes[3].set(title="Age Distribution of High-Skilled Workers", xlabel="Age", ylabel="Share of high-skilled worker mass")

    axes[4].plot(age_groups, high_skill_share_baseline, marker="o", linewidth=2, label="Baseline")
    axes[4].plot(age_groups, high_skill_share_extension, marker="o", linewidth=2, label="Extension")
    axes[4].set(title="High-Skill Share within Each Age Group", xlabel="Age", ylabel="Mass-weighted high-skill share")

    axes[5].plot(age_groups, wage_high_baseline, marker="o", linewidth=2, label="High-skilled, baseline")
    axes[5].plot(age_groups, wage_high_extension, marker="s", linewidth=2, linestyle="--", label="High-skilled, extension")
    axes[5].plot(age_groups, wage_low_baseline, marker="o", linewidth=2, label="Low-skilled, baseline")
    axes[5].plot(age_groups, wage_low_extension, marker="s", linewidth=2, linestyle="--", label="Low-skilled, extension")
    axes[5].set(title="Wages by Age and Skill Level", xlabel="Age", ylabel="Mass-weighted mean wage")

    for ax in axes:
        ax.legend()

    fig.tight_layout()
    plt.show()