import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
from scipy.signal import savgol_filter
import seaborn as sns
import matplotlib.patches as mpatches
import glob

from matplotlib import rc
rc('text', usetex=True)


color_list = sns.color_palette("muted")
sns.palplot(color_list)

log_dir = 'gcp3_logs'

def ema(x, decay=0.99, init=0.):
  res = []
  prev = init
  for val in x:
    prev = decay * prev + (1 - decay) * val
    res.append(prev)

  return res

# Read in results
results = {}
files = glob.glob(log_dir + '/*/*.json')
for log_file in files:
  if log_file.endswith('.json'):
    with open(log_file, 'r') as f:
      exp_name = log_file[log_file.find('/')+1:log_file.rfind('/')]
      env, optimizer, seed = exp_name.split('_')
      if optimizer == '':
        optimizer = 'none'
      key = (env, optimizer)
      print(key, seed)
      if key not in results:
        results[key] = []
      result = []
      print(log_file)
      for line in f:
        result.append(json.loads(line.strip()))
      if not (optimizer == 'none' and int(seed) > 1000):
        print(key, seed)
        results[key].append(result)

# Envs to plot
envs = ['halfcheetah', 'humanoid']
envs_titles = {
    'halfcheetah': 'HalfCheetah-v1',
    'walker2d': 'Walker2d',
    'humanoid': 'Humanoid-v1',
}
method_names = {
    'baseline': 'PPO',
    'none': 'Stein (biased)',
    'extra-sample': 'Stein (biased, no importance sampling)',
    'unbiased': 'Stein (unbiased)',
    'unbiased--state-only': 'Stein (unbiased, state-dependent baseline)'
}

plot_xlim = {
    'halfcheetah': 1000,
    'humanoid': 700,
}

savgol_window = 7
colors = {}
for i, method in enumerate(['none', 'baseline', 'extra-sample', 'unbiased', 'unbiased--state-only']):
  colors[method] = color_list[i]

fig, axes = plt.subplots(1, len(envs), figsize=(15, 6))
for envidx, env in enumerate(envs):
  ax = axes[envidx]
  print('Plotting results for env = %s' % env)
  ys = []
  for k, v in sorted(results.items()):
    if k[0] == env:
      y = [[row['_MeanReward'] for row in res] for res in v]
      print('%s has results with lengths: %s' % (k[1], str(list(map(len, y)))))

      min_len = min([len(x) for x in y])
      win_len = plot_xlim[env]
      assert(win_len <= min_len)

      # Cut the results down to the plotting length
      y = [y_i[:win_len] for y_i in y]
      y = np.stack(y)

      # Plot the data
      smooth = lambda y: savgol_filter(y, savgol_window, 5)

      y_max = smooth(y.max(0))
      y_min = smooth(y.min(0))
      y_plus_std = np.clip(smooth(y.mean(0) + y.std(0)),
                           y_min,
                           y_max)
      y_minus_std = np.clip(smooth(y.mean(0) - y.std(0)),
                            y_min,
                            y_max)
      y_mean = smooth(y.mean(0))

      ax.plot(np.arange(win_len) * 10, y_mean,
              color=colors[k[1]],
              label=method_names[k[1]])
      ax.fill_between(np.arange(win_len) * 10,
                      y_minus_std, y_plus_std,
                      color=colors[k[1]], alpha=0.2)

  # Plot options
  ax.legend(loc='upper left', prop={'size': 14})
  ax.set_title(envs_titles[env], fontsize=18)
  ax.tick_params(axis='both', which='major', labelsize=12)
  ax.tick_params(axis='both', which='minor', labelsize=12)
  ax.set_xlabel('Steps (thousands)', fontsize=14)
  ax.set_ylabel('Average Reward', fontsize=14)
  ax.grid(alpha=0.5)

# Uncomment this to generate the non-mini plot.
legend_handles = []
legend_order = ['none', 'baseline', 'extra-sample', 'unbiased', 'unbiased--state-only']
for method in legend_order:
  legend_handles.append(mpatches.Patch(
      color=colors[method],
      label=method_names[method]))
leg = plt.legend(handles=legend_handles, loc='lower center', ncol=3, prop={'size': 16})
# Move legend down.
bb = leg.get_bbox_to_anchor().inverse_transformed(ax.transAxes)
bb.y0 += -0.18
leg.set_bbox_to_anchor(bb, transform = ax.transAxes)

fig.savefig('stein_mean_stds_all2-mini.pdf', bbox_inches='tight', format='pdf')

