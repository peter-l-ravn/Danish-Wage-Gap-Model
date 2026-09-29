import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from types import SimpleNamespace
import graphs as g
import plots

mass = np.array([[[1., 3.], [2., 1.], [1., 2.]], [[2., 1.], [1., 3.], [2., 1.]]])
share = np.array([[[0., .5], [.5, 1.], [.25, .75]], [[.25, .75], [.5, 1.], [0., 1.]]])
low = np.array([[[2., 4.], [3., 6.], [4., 8.]], [[3., 5.], [4., 7.], [5., 9.]]])
high = low + 10
model = SimpleNamespace(par=SimpleNamespace(n=3), sol=SimpleNamespace(
    mass=mass, l_h=share, wage_l=low, wage_h=high,
    wage=share*high+(1-share)*low, ability=np.tile([.4,.8], (2,3,1))))
# Unequal mass and fractional job shares: low wages need (1-h)*mass.
fig, ax = g.plot_low_skill_wages_young_old(model, 0, 2, normalize=False, show=False)
np.testing.assert_allclose(ax.lines[0].get_ydata()[0], (2*1 + 4*1.5)/2.5)
plt.close(fig)
# Current vs fixed weights must agree at baseline and match explicit ratios.
for fixed in [False, True]:
    gap, T = g.wage_gap_func(model, 0, 2, fixed_weights=fixed)
    means = []
    for t in [0,1]:
        wt = 0 if fixed else t
        means.append([np.average(model.sol.wage[t,a], weights=mass[wt,a]) for a in [0,2]])
    expected = 100*((means[1][1]/means[1][0])/(means[0][1]/means[0][0])-1)
    np.testing.assert_allclose(gap,[0,expected])
np.testing.assert_allclose(g.wage_gap_func(model,0,2)[0],g.wage_gap_func(model,25,27,ages_in_years=True)[0])
# Every plotting entry point renders with titles on and off.
calls = [
    lambda **kw:g.plot_model(model, **kw),
    lambda **kw:g.plot_model(model, time=1, **kw),
    lambda **kw:g.plot_model_comparison(model, model, **kw),
    lambda **kw:g.plot_wage_high_low(model, **kw),
    lambda **kw:g.plot_wage_gap(model,0,2, **kw),
    lambda **kw:g.plot_wage_gap_single(model,0,2, **kw),
    lambda **kw:g.plot_mean_age_high_skill(model, **kw),
    lambda **kw:g.plot_wage_change_by_age(model, **kw),
    lambda **kw:g.plot_occupation_heatmaps(model, **kw),
    lambda **kw:g.plot_skill_comparison(model,0,2, **kw),
    lambda **kw:g.plot_shock_summary(model,0,2, **kw),
    lambda **kw:g.plot_high_skill_wages_young_old(model,0,2, **kw),
    lambda **kw:g.plot_low_skill_wages_young_old(model,0,2, **kw),
    lambda **kw:g.plot_high_skill_shares_young_old(model,0,2, **kw),
    lambda **kw:g.plot_low_skill_shares_young_old(model,0,2, **kw),
    lambda **kw:plots.plot_shares(.2,.8,.3,.7,'Shares', **kw),
    lambda **kw:plots.plot_series(np.ones(33),'Series', **kw),
]
for titles in [False, True]:
    g.configure_plots(show_titles=titles, age_start=25)
    for call in calls:
        result=call(show=False)
        fig=result[0]
        data_axes=np.asarray(result[1]).ravel()
        assert all(bool(ax.get_title()) == titles for ax in data_axes)
        fig.canvas.draw()
        plt.close(fig)
g.configure_plots(show_titles=False, age_start=40)
fig, ax=g.plot_wage_high_low(model,show=False,show_titles=True)
np.testing.assert_array_equal(ax.lines[0].get_xdata(), [40,41,42])
assert ax.get_title()
plt.close(fig)
fig,ax=g.plot_mean_age_high_skill(model,show=False)
assert np.all((ax.lines[0].get_ydata()>=40)&(ax.lines[0].get_ydata()<=42))
plt.close(fig)
# Zero baselines and empty cells remain undefined rather than infinite.
assert np.isnan(g._index(np.array([0.,1.]))).all()
assert np.isnan(g._weighted_mean(np.array([1.,np.nan]),np.array([0.,1.])))
print('Passed: all plotting APIs render, title defaults/overrides, age offsets, actual-age cutoffs, low-job weights, fixed/current ratios, empty data')
