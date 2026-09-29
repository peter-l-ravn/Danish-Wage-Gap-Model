"""Mass-weighted model plots.

Use configure_plots(show_titles=False, age_start=25) for LaTeX figures.
All plot functions return their figure and axes and accept show=False.
Legacy young_max/old_min arguments are age indices by default; pass
ages_in_years=True to supply actual ages. Legacy time arguments count periods
(time=10 selects index 9); new t_start/t_end arguments are zero-based indices.
"""
import numpy as np
import matplotlib.pyplot as plt

SHOW_TITLES = True
AGE_START = 25
BLUE = "#2166d1"
RED = "#e53935"


def configure_plots(*, show_titles=True, age_start=25):
    """Set defaults for all plots; individual calls can override either option."""
    global SHOW_TITLES, AGE_START
    SHOW_TITLES = bool(show_titles)
    AGE_START = age_start


def _age_start(value):
    return AGE_START if value is None else value


def _label(ax, title, xlabel, ylabel, show_titles, *, grid=True):
    titles = SHOW_TITLES if show_titles is None else show_titles
    ax.set(title=title if titles else "", xlabel=xlabel, ylabel=ylabel)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.set_axisbelow(True)
    ax.grid(grid, color="0.8", linewidth=0.8) if grid else ax.grid(False)


def _axis(ax, x_size, y_size):
    if ax is None:
        return plt.subplots(figsize=(x_size, y_size), layout="constrained")
    return ax.figure, ax


def _finish(fig, axes, show):
    if show:
        plt.show()
    return fig, axes


def _last_period(model):
    valid = np.flatnonzero(np.any(np.isfinite(model.sol.mass), axis=(1, 2)))
    if not valid.size:
        raise ValueError("The model has no populated periods")
    return int(valid[-1])


def _period(model, t=None):
    if t is None:
        return _last_period(model)
    n = model.sol.mass.shape[0]
    if not isinstance(t, (int, np.integer)) or not -n <= t < n:
        raise ValueError(f"Period must be an integer between {-n} and {n - 1}")
    t = t % n
    if not np.any(np.isfinite(model.sol.mass[t])):
        raise ValueError(f"Period {t} has not been simulated")
    return t


def _legacy_period(model, time):
    if time is not None and time < 1:
        raise ValueError("time counts periods starting at 1")
    return _period(model, None if time is None else time - 1)


def _weighted_mean(values, weights, axis=None):
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    weights = np.where(valid, weights, 0.0)
    numerator = np.sum(np.where(valid, values, 0.0) * weights, axis=axis)
    denominator = np.sum(weights, axis=axis)
    return np.divide(numerator, denominator,
                     out=np.full(np.shape(numerator), np.nan), where=denominator > 0)


def _weights(model, t, skill=None):
    mass = model.sol.mass[t]
    if skill is None:
        return mass
    high = np.clip(model.sol.l_h[t], 0, 1)
    return mass * (high if skill == "high" else 1 - high)


def _mean_by_age(model, t, skill=None):
    variable = "wage" if skill is None else "wage_h" if skill == "high" else "wage_l"
    return _weighted_mean(getattr(model.sol, variable)[t], _weights(model, t, skill), axis=1)


def _groups(model, young_max, old_min, age_start=None, ages_in_years=False):
    start = _age_start(age_start)
    ages = start + np.arange(model.par.n)
    young_age = young_max if ages_in_years else start + young_max
    old_age = old_min if ages_in_years else start + old_min
    young, old = ages <= young_age, ages >= old_age
    if not young.any() or not old.any():
        raise ValueError("Both age groups must contain at least one model age")
    labels = (f"Young ({start:g}-{young_age:g})", f"Old ({old_age:g}+)")
    return (young, old), labels


def _group_series(model, young_max, old_min, *, skill=None, measure="wage",
                  age_start=None, ages_in_years=False, fixed_weights=False):
    groups, labels = _groups(model, young_max, old_min, age_start, ages_in_years)
    result = np.full((2, _last_period(model) + 1), np.nan)
    for t in range(result.shape[1]):
        weight_t = 0 if fixed_weights else t
        if measure == "share":
            high = np.clip(model.sol.l_h[t], 0, 1)
            values = high if skill == "high" else 1 - high
            weights = model.sol.mass[weight_t]
        else:
            name = "wage" if skill is None else "wage_h" if skill == "high" else "wage_l"
            values = getattr(model.sol, name)[t]
            weights = _weights(model, weight_t, skill)
        for i, group in enumerate(groups):
            result[i, t] = _weighted_mean(values[group], weights[group])
    return result, labels


def _index(values):
    """Period-zero index; an empty or zero baseline remains undefined (NaN)."""
    initial = values[..., :1]
    return np.divide(100 * values, initial, out=np.full(values.shape, np.nan),
                     where=np.isfinite(initial) & (initial != 0))


def wage_gap_func(model, young_max, old_min, *, age_start=None,
                  ages_in_years=False, fixed_weights=False):
    """Percent change in the old/young mean wage ratio relative to period zero."""
    wages, _ = _group_series(model, young_max, old_min, age_start=age_start,
                             ages_in_years=ages_in_years, fixed_weights=fixed_weights)
    ratio = np.divide(wages[1], wages[0], out=np.full(wages.shape[1], np.nan),
                      where=np.isfinite(wages[0]) & (wages[0] != 0))
    if not np.isfinite(ratio[0]) or ratio[0] == 0:
        raise ValueError("The period-zero old/young wage ratio must be finite and nonzero")
    return 100 * (ratio / ratio[0] - 1), len(ratio)


def plot_wage_gap(model, young_max, old_min, x_size=8, y_size=5, *,
                  show_titles=None, age_start=None, ages_in_years=False,
                  fixed_weights=False, ax=None, show=True):
    values, _ = wage_gap_func(model, young_max, old_min, age_start=age_start,
                              ages_in_years=ages_in_years, fixed_weights=fixed_weights)
    _, labels = _groups(model, young_max, old_min, age_start, ages_in_years)
    fig, ax = _axis(ax, x_size, y_size)
    ax.plot(values, lw=2, color=BLUE, label="Initial population weights" if fixed_weights else "Current population weights")
    _label(ax, f"Old/young wage ratio: {labels[1]} / {labels[0]}", "Period",
           "Change in old/young wage ratio (%)", show_titles)
    return _finish(fig, ax, show)


def plot_wage_gap_single(model, young_age, old_age, x_size=8, y_size=5, *,
                         show_titles=None, age_start=None, ages_in_years=False,
                         ax=None, show=True):
    start = _age_start(age_start)
    indices = np.asarray([young_age, old_age]) - (start if ages_in_years else 0)
    if np.any(indices != indices.astype(int)) or np.any(indices < 0) or np.any(indices >= model.par.n):
        raise ValueError("Requested ages must be on the model age grid")
    young, old = indices.astype(int)
    values = np.array([_mean_by_age(model, t)[[young, old]]
                       for t in range(_last_period(model) + 1)])
    fig, ax = _axis(ax, x_size, y_size)
    ax.plot(values[:, 1] - values[:, 0], lw=2, color=BLUE)
    _label(ax, f"Wage difference: age {start + old:g} minus {start + young:g}",
           "Period", "Mean wage difference (old minus young)", show_titles)
    return _finish(fig, ax, show)


def plot_mean_age_high_skill(model, x_size=8, y_size=5, *, show_titles=None,
                            age_start=None, ax=None, show=True):
    ages = _age_start(age_start) + np.arange(model.par.n)
    values = [_weighted_mean(ages[:, None], _weights(model, t, "high"))
              for t in range(_last_period(model) + 1)]
    fig, ax = _axis(ax, x_size, y_size)
    ax.plot(values, lw=2, color=BLUE)
    _label(ax, "Mean age of high-skilled workers", "Period", "Mean age (years)", show_titles)
    return _finish(fig, ax, show)


def plot_wage_change_by_age(model, t_start=0, t_end=-1, age_start=None, *,
                            show_titles=None, ax=None, show=True):
    """Return (fig, ax, percent_change); compares ages rather than birth cohorts."""
    t_start, t_end = _period(model, t_start), _period(model, t_end)
    initial, final = _mean_by_age(model, t_start), _mean_by_age(model, t_end)
    change = np.divide(100 * (final - initial), initial, out=np.full(initial.shape, np.nan),
                       where=np.isfinite(initial) & (initial != 0))
    fig, ax = _axis(ax, 8, 5)
    ax.plot(_age_start(age_start) + np.arange(model.par.n), change, lw=2, color=BLUE)
    ax.axhline(0, color="gray", ls="--", lw=1)
    _label(ax, f"Wage change: period {t_start} to {t_end}", "Age",
           "Change in mean wage (%)", show_titles)
    _finish(fig, ax, show)
    return fig, ax, change


def plot_wage_high_low(model_baseline, time=None, *, show_titles=None,
                       age_start=None, ax=None, show=True):
    t = _legacy_period(model_baseline, time)
    ages = _age_start(age_start) + np.arange(model_baseline.par.n)
    fig, ax = _axis(ax, 8, 5)
    for skill, color in [("high", "#2166d1"), ("low", "#e53935")]:
        ax.plot(ages, _mean_by_age(model_baseline, t, skill), lw=2,
                color=color, label=f"{skill.capitalize()}-skilled")
    _label(ax, f"Wages by occupation: period {t}", "Age", "Mean wage", show_titles)
    ax.legend()
    return _finish(fig, ax, show)


def _plot_group_outcomes(model, young_max, old_min, x_size, y_size, *, skill,
                         measure, show_titles=None, age_start=None, ages_in_years=False,
                         normalize=True, ax=None, show=True):
    values, labels = _group_series(model, young_max, old_min, skill=skill, measure=measure,
                                   age_start=age_start, ages_in_years=ages_in_years)
    if normalize:
        values = _index(values)
    elif measure == "share":
        values = 100 * values
    occupation = f"{skill.capitalize()}-skilled"
    quantity = "wage" if measure == "wage" else "employment share"
    ylabel = f"{occupation} {quantity}"
    ylabel += " (period 0 = 100)" if normalize else " (%)" if measure == "share" else ""
    fig, ax = _axis(ax, x_size, y_size)
    for series, label, color in zip(values, labels, (BLUE, RED)):
        ax.plot(series, lw=2, label=label, color=color)
    _label(ax, f"{occupation} {quantity}s by age group", "Period", ylabel, show_titles)
    ax.legend()
    return _finish(fig, ax, show)


def plot_high_skill_wages_young_old(model, young_max, old_min, x_size=8, y_size=5, **kwargs):
    """High-job wages; accepts show_titles, age_start, ages_in_years, normalize, ax, show."""
    return _plot_group_outcomes(model, young_max, old_min, x_size, y_size,
                                skill="high", measure="wage", **kwargs)


def plot_low_skill_wages_young_old(model, young_max, old_min, x_size=8, y_size=5, **kwargs):
    """Low-job wages weighted by (1 - l_h) * mass; same options as high-job wages."""
    return _plot_group_outcomes(model, young_max, old_min, x_size, y_size,
                                skill="low", measure="wage", **kwargs)


def plot_high_skill_shares_young_old(model, young_max, old_min, x_size=8, y_size=5, **kwargs):
    """High-job shares; normalize=False shows percentages rather than indices."""
    return _plot_group_outcomes(model, young_max, old_min, x_size, y_size,
                                skill="high", measure="share", **kwargs)


def plot_low_skill_shares_young_old(model, young_max, old_min, x_size=8, y_size=5, **kwargs):
    """Low-job shares; accepts the same options as high-job shares."""
    return _plot_group_outcomes(model, young_max, old_min, x_size, y_size,
                                skill="low", measure="share", **kwargs)


def plot_skill_comparison(model, young_max, old_min, *, show_titles=None,
                          age_start=None, ages_in_years=False, normalize=True, show=True):
    """Four panels: high/low wages and employment shares for young/old workers."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    for row, skill in enumerate(("high", "low")):
        for col, measure in enumerate(("wage", "share")):
            _plot_group_outcomes(model, young_max, old_min, 8, 5, skill=skill,
                                 measure=measure, ax=axes[row, col], show=False,
                                 normalize=normalize, show_titles=show_titles,
                                 age_start=age_start, ages_in_years=ages_in_years)
    return _finish(fig, axes, show)


def plot_shock_summary(model, young_max, old_min, *, show_titles=None,
                       age_start=None, ages_in_years=False, show=True):
    """Four panels showing the wage ratio, mean age, and wage changes."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    opts = dict(show_titles=show_titles, age_start=age_start, show=False)
    plot_wage_gap(model, young_max, old_min, ages_in_years=ages_in_years,
                  ax=axes[0, 0], **opts)
    plot_mean_age_high_skill(model, ax=axes[0, 1], **opts)
    plot_wage_change_by_age(model, ax=axes[1, 0], **opts)
    wages, labels = _group_series(model, young_max, old_min, age_start=age_start,
                                  ages_in_years=ages_in_years)
    for series, label, color in zip(_index(wages) - 100, labels, (BLUE, RED)):
        axes[1, 1].plot(series, lw=2, label=label, color=color)
    _label(axes[1, 1], "Mean wage changes by age group", "Period",
           "Change in mean wage (%)", show_titles)
    axes[1, 1].legend()
    return _finish(fig, axes, show)


def _overview(models, periods, names, show_titles, age_start, show):
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), layout="constrained")
    flat = axes.ravel()
    for model, t, name, color in zip(models, periods, names, (BLUE, RED)):
        ages = _age_start(age_start) + np.arange(model.par.n)
        end = _last_period(model)
        mean_wages = [_weighted_mean(model.sol.wage[s], model.sol.mass[s]) for s in range(end + 1)]
        high_shares = [_weighted_mean(model.sol.l_h[s], model.sol.mass[s]) for s in range(end + 1)]
        high_mass = _weights(model, t, "high")
        high_mass = np.where(np.isfinite(high_mass) & (high_mass > 0), high_mass, 0)
        age_mass = high_mass.sum(axis=1)
        distribution = age_mass / age_mass.sum() if age_mass.sum() else np.full(model.par.n, np.nan)
        flat[0].plot(mean_wages, lw=2, label=name, color=color)
        flat[1].plot(np.asarray(high_shares) * 100, lw=2, label=name, color=color)
        snapshot = f"{name}, period {t}" if name else f"Period {t}"
        flat[2].plot(ages, _mean_by_age(model, t), lw=2, label=snapshot, color=color)
        flat[3].plot(ages, 100 * distribution, lw=2, label=snapshot, color=color)
        shares = _weighted_mean(model.sol.l_h[t], model.sol.mass[t], axis=1)
        flat[4].plot(ages, 100 * shares, lw=2, label=snapshot, color=color)
        for skill, style in [("high", "-"), ("low", "--")]:
            flat[5].plot(ages, _mean_by_age(model, t, skill), style, lw=2,
                         color=color if len(models) > 1 else BLUE if skill == "high" else RED,
                         label=f"{skill.capitalize()}-skilled, {snapshot}")
    titles = ["Mean wage over time", "High-skilled employment share over time",
              "Mean wage by age", "Age distribution of high-skilled workers",
              "High-skilled employment share by age", "Wages by occupation and age"]
    ylabels = ["Mean wage", "High-skilled share (%)", "Mean wage",
               "Share of high-skilled workers (%)", "High-skilled share (%)", "Mean wage"]
    for i, ax in enumerate(flat):
        _label(ax, titles[i], "Period" if i < 2 else "Age", ylabels[i], show_titles)
        if i >= 2 or len(models) > 1:
            ax.legend()
    return _finish(fig, axes, show)


def plot_model(model_baseline, time=None, *, show_titles=None, age_start=None, show=True):
    """Six-panel overview; time=10 selects period 9 for every age profile."""
    return _overview([model_baseline], [_legacy_period(model_baseline, time)], [None],
                      show_titles, age_start, show)


def plot_model_comparison(model_baseline, model_extension, *, show_titles=None,
                          age_start=None, show=True):
    """Compare two models; age profiles use each model's final populated period."""
    models = [model_baseline, model_extension]
    return _overview(models, [_last_period(m) for m in models], ["Baseline", "Extension"],
                      show_titles, age_start, show)


def plot_occupation_heatmaps(model, t_start=0, t_end=-1, age_start=None, *, show_titles=None, show=True):
    """Compare age-ability job allocations and their end-minus-start change.

    Red is low job (l_h=0), blue is high job (l_h=1); intermediate
    colors represent split bins. Gray indicates missing or zero-mass bins.
    The third panel shows the change in high-job share in percentage points,
    with red for decreases, white for no change, and blue for increases.
    These are cross-sectional snapshots, not individual transition histories.
    Requires the same distinct ability grid across ages and selected periods.
    Returns (fig, axes).
    """
    from matplotlib.colors import LinearSegmentedColormap

    age_start = _age_start(age_start)
    periods = (_period(model, t_start), _period(model, t_end))
    panels = []
    ability_grid = None
    for t in periods:
        ability = np.asarray(model.sol.ability[t])
        order = np.argsort(ability, axis=1)
        sorted_ability = np.take_along_axis(ability, order, axis=1)
        if ability_grid is None:
            ability_grid = sorted_ability[0]
        if (not np.all(np.isfinite(sorted_ability))
                or not np.allclose(sorted_ability, ability_grid)
                or np.any(np.diff(ability_grid) <= 0)):
            raise ValueError("Heatmaps require a common, distinct ability grid")
        share = np.take_along_axis(model.sol.l_h[t], order, axis=1)
        mass = np.take_along_axis(model.sol.mass[t], order, axis=1)
        valid = np.isfinite(share) & np.isfinite(mass) & (mass > 0)
        if not np.any(valid):
            raise ValueError(f"Period {t} has no job allocations with positive mass")
        panels.append(np.ma.array(np.clip(share, 0, 1), mask=~valid).T)

    # Use actual ability coordinates: the quantile grid is not equally spaced.
    if ability_grid.size == 1:
        half_width = max(abs(ability_grid[0]) * 0.01, 0.01)
        ability_edges = ability_grid[0] + np.array([-half_width, half_width])
    else:
        midpoints = (ability_grid[:-1] + ability_grid[1:]) / 2
        ability_edges = np.r_[ability_grid[0] - (midpoints[0] - ability_grid[0]),
                              midpoints,
                              ability_grid[-1] + (ability_grid[-1] - midpoints[-1])]
    age_edges = age_start + np.arange(panels[0].shape[1] + 1) - 0.5
    cmap = LinearSegmentedColormap.from_list("low_to_high_job", ["#e53935", "#2166d1"])
    cmap.set_bad("#dddddd")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True,
                             constrained_layout=True)
    for ax, t, panel in zip(axes, periods, panels):
        ax.grid(False)
        mesh = ax.pcolormesh(age_edges, ability_edges, panel, cmap=cmap,
                             vmin=0, vmax=1, shading="flat", rasterized=True)
        resolved_t = t if t >= 0 else model.sol.l_h.shape[0] + t
        _label(ax, f"Occupational allocation: period {resolved_t}", "Age", "", show_titles, grid=False)
    axes[0].set_ylabel("Ability")
    colorbar = fig.colorbar(mesh, ax=axes[:2], ticks=[0, 0.5, 1],
                           label="Share in high job")
    colorbar.ax.set_yticklabels(["0: low job", "0.5: split equally", "1: high job"])
    change = 100.0 * (panels[1] - panels[0])
    change_cmap = LinearSegmentedColormap.from_list(
        "job_share_change", ["#e53935", "#ffffff", "#2166d1"], N=257)
    change_cmap.set_bad("#dddddd")
    axes[2].grid(False)
    change_mesh = axes[2].pcolormesh(
        age_edges, ability_edges, change, cmap=change_cmap,
        vmin=-100, vmax=100, shading="flat", rasterized=True)
    _label(axes[2], "High-job share: final minus initial", "Age", "", show_titles, grid=False)
    fig.colorbar(change_mesh, ax=axes[2], ticks=[-100, -50, 0, 50, 100],
                 label="Change in high-job share (percentage points)")
    if show:
        plt.show()
    return fig, axes
