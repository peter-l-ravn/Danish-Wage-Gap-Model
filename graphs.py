import numpy as np
import matplotlib.pyplot as plt

def plot_model_comparison(model_baseline, model_extension):
    plt.style.use("seaborn-v0_8-whitegrid")

    def last_period(model):
        valid_periods = np.where(np.any(np.isfinite(model.sol.mass), axis=0))[0]
        return valid_periods[-1]

    def weighted_mean(values, weights):
        valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
        if not np.any(valid):
            return np.nan
        return np.sum(values[valid] * weights[valid]) / np.sum(weights[valid])

    def mean_by_age(model, variable, t, age_groups, skill=None):
        total_mass = model.sol.mass[:, t]
        high_share = np.clip(model.sol.l_h[:, t], 0.0, 1.0)

        if skill == "high":
            weights = high_share * total_mass
        elif skill == "low":
            weights = (1.0 - high_share) * total_mass
        else:
            weights = total_mass

        means = []
        for age in age_groups:
            age_mask = model.sol.age[:, t] == age
            means.append(weighted_mean(variable[age_mask, t], weights[age_mask]))

        return np.array(means)

    def mean_over_time(model, variable, t_end):
        return np.array([weighted_mean(variable[:, t], model.sol.mass[:, t]) for t in range(t_end + 1)])

    def high_skill_mass_by_age(model, t, age_groups):
        high_mass = np.clip(model.sol.l_h[:, t], 0.0, 1.0) * model.sol.mass[:, t]
        mass_by_age = np.array([np.nansum(high_mass[model.sol.age[:, t] == age]) for age in age_groups])
        total_high_mass = np.nansum(mass_by_age)
        return mass_by_age / total_high_mass if total_high_mass > 0.0 else np.full(len(age_groups), np.nan)

    t_baseline = last_period(model_baseline)
    t_extension = last_period(model_extension)
    age_groups = np.arange(model_baseline.par.n)

    # Wage by age
    wage_age_baseline = mean_by_age(model_baseline, model_baseline.sol.wage, t_baseline, age_groups)
    wage_age_extension = mean_by_age(model_extension, model_extension.sol.wage, t_extension, age_groups)

    # Wage by age and skill
    wage_high_baseline = mean_by_age(model_baseline, model_baseline.sol.wage_h, t_baseline, age_groups, skill="high")
    wage_high_extension = mean_by_age(model_extension, model_extension.sol.wage_h, t_extension, age_groups, skill="high")
    wage_low_baseline = mean_by_age(model_baseline, model_baseline.sol.wage_l, t_baseline, age_groups, skill="low")
    wage_low_extension = mean_by_age(model_extension, model_extension.sol.wage_l, t_extension, age_groups, skill="low")

    # High-skill allocation
    high_skill_mass_baseline = high_skill_mass_by_age(model_baseline, t_baseline, age_groups)
    high_skill_mass_extension = high_skill_mass_by_age(model_extension, t_extension, age_groups)
    high_skill_share_baseline = mean_by_age(model_baseline, model_baseline.sol.l_h, t_baseline, age_groups)
    high_skill_share_extension = mean_by_age(model_extension, model_extension.sol.l_h, t_extension, age_groups)

    # Aggregate variables over time
    mean_wage_baseline = mean_over_time(model_baseline, model_baseline.sol.wage, t_baseline)
    mean_wage_extension = mean_over_time(model_extension, model_extension.sol.wage, t_extension)
    high_skill_share_time_baseline = mean_over_time(model_baseline, model_baseline.sol.l_h, t_baseline)
    high_skill_share_time_extension = mean_over_time(model_extension, model_extension.sol.l_h, t_extension)

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

def plot_model(model_baseline):
    plt.style.use("seaborn-v0_8-whitegrid")

    def last_period(model):
        valid_periods = np.where(np.any(np.isfinite(model.sol.mass), axis=0))[0]
        return valid_periods[-1]

    def weighted_mean(values, weights):
        valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
        if not np.any(valid):
            return np.nan
        return np.sum(values[valid] * weights[valid]) / np.sum(weights[valid])

    def mean_by_age(model, variable, t, age_groups, skill=None):
        total_mass = model.sol.mass[:, t]
        high_share = np.clip(model.sol.l_h[:, t], 0.0, 1.0)

        if skill == "high":
            weights = high_share * total_mass
        elif skill == "low":
            weights = (1.0 - high_share) * total_mass
        else:
            weights = total_mass

        means = []
        for age in age_groups:
            age_mask = model.sol.age[:, t] == age
            means.append(weighted_mean(variable[age_mask, t], weights[age_mask]))

        return np.array(means)

    def mean_over_time(model, variable, t_end):
        return np.array([weighted_mean(variable[:, t], model.sol.mass[:, t]) for t in range(t_end + 1)])

    def high_skill_mass_by_age(model, t, age_groups):
        high_mass = np.clip(model.sol.l_h[:, t], 0.0, 1.0) * model.sol.mass[:, t]
        mass_by_age = np.array([np.nansum(high_mass[model.sol.age[:, t] == age]) for age in age_groups])
        total_high_mass = np.nansum(mass_by_age)
        return mass_by_age / total_high_mass if total_high_mass > 0.0 else np.full(len(age_groups), np.nan)

    t_baseline = last_period(model_baseline)
    age_groups = np.arange(model_baseline.par.n)

    wage_age_baseline = mean_by_age(model_baseline, model_baseline.sol.wage, t_baseline, age_groups)
    wage_high_baseline = mean_by_age(model_baseline, model_baseline.sol.wage_h, t_baseline, age_groups, skill="high")
    wage_low_baseline = mean_by_age(model_baseline, model_baseline.sol.wage_l, t_baseline, age_groups, skill="low")

    high_skill_mass_baseline = high_skill_mass_by_age(model_baseline, t_baseline, age_groups)
    high_skill_share_baseline = mean_by_age(model_baseline, model_baseline.sol.l_h, t_baseline, age_groups)

    mean_wage_baseline = mean_over_time(model_baseline, model_baseline.sol.wage, t_baseline)
    high_skill_share_time_baseline = mean_over_time(model_baseline, model_baseline.sol.l_h, t_baseline)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()

    axes[0].plot(mean_wage_baseline, linewidth=2)
    axes[0].set(title="Mean Wage over Time", xlabel="Time", ylabel="Mass-weighted mean wage")

    axes[1].plot(high_skill_share_time_baseline, linewidth=2)
    axes[1].set(title="Mean High-Skill Share over Time", xlabel="Time", ylabel="Mass-weighted high-skill share")

    axes[2].plot(age_groups, wage_age_baseline, marker="o", linewidth=2)
    axes[2].set(title="Mean Wage by Age", xlabel="Age", ylabel="Mass-weighted mean wage")

    axes[3].plot(age_groups, high_skill_mass_baseline, marker="o", linewidth=2)
    axes[3].set(title="Age Distribution of High-Skilled Workers", xlabel="Age", ylabel="Share of high-skilled worker mass")

    axes[4].plot(age_groups, high_skill_share_baseline, marker="o", linewidth=2)
    axes[4].set(title="High-Skill Share within Each Age Group", xlabel="Age", ylabel="Mass-weighted high-skill share")

    axes[5].plot(age_groups, wage_high_baseline, marker="o", linewidth=2, label="High-skilled")
    axes[5].plot(age_groups, wage_low_baseline, marker="s", linewidth=2, linestyle="--", label="Low-skilled")
    axes[5].set(title="Wages by Age and Skill Level", xlabel="Age", ylabel="Mass-weighted mean wage")
    axes[5].legend()

    fig.tight_layout()
    plt.show()


def plot_wage_gap(model, young_max, old_min, x_size=8, y_size=5):
    plt.style.use("seaborn-v0_8-whitegrid")

    def weighted_mean(values, weights):
        valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
        if not np.any(valid):
            return np.nan
        return np.sum(values[valid] * weights[valid]) / np.sum(weights[valid])

    valid_periods = np.where(np.any(np.isfinite(model.sol.mass), axis=0))[0]
    T = valid_periods[-1] + 1
    wage_gap = np.full(T, np.nan)

    for t in range(T):
        young = model.sol.age[:, t] <= young_max
        old = model.sol.age[:, t] >= old_min
        young_wage = weighted_mean(model.sol.wage[young, t], model.sol.mass[young, t])
        old_wage = weighted_mean(model.sol.wage[old, t], model.sol.mass[old, t])
        wage_gap[t] = old_wage - young_wage

    plt.figure(figsize=(x_size, y_size))
    plt.plot(np.arange(T), wage_gap, linewidth=2)
    plt.title(f"Wage Gap: Old (age >= {old_min}) - Young (age <= {young_max})")
    plt.xlabel("Time")
    plt.ylabel("Old wage - young wage")
    plt.tight_layout()
    plt.show()


def plot_mean_age_high_skill(model, x_size=8, y_size=5):
    plt.style.use("seaborn-v0_8-whitegrid")

    def weighted_mean(values, weights):
        valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
        if not np.any(valid):
            return np.nan
        return np.sum(values[valid] * weights[valid]) / np.sum(weights[valid])

    valid_periods = np.where(np.any(np.isfinite(model.sol.mass), axis=0))[0]
    T = valid_periods[-1] + 1
    mean_age = np.full(T, np.nan)

    for t in range(T):
        high_skill_share = np.clip(model.sol.l_h[:, t], 0.0, 1.0)
        high_skill_mass = high_skill_share * model.sol.mass[:, t]
        mean_age[t] = weighted_mean(model.sol.age[:, t], high_skill_mass)

    plt.figure(figsize=(x_size, y_size))
    plt.plot(np.arange(T), mean_age, linewidth=2)
    plt.title("Average Age of High-Skilled Workers")
    plt.xlabel("Time")
    plt.ylabel("Mass-weighted average age")
    plt.tight_layout()
    plt.show()
