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
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
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

envs = ['halfcheetah', 'humanoid']
envs_titles = {'halfcheetah': 'HalfCheetah-v1', 'walker2d': 'Walker2d', 'humanoid': 'Humanoid-v1'}
thing_names = {'baseline': 'PPO', 'none': 'Stein (biased)', 'extra-sample': 'Stein (biased, no importance sampling)', 'unbiased': 'Stein (unbiased)', 'unbiased--state-only': 'Stein (unbiased, state-dependent baseline)'}
savgol_window = 7
colors = {}
for i, thing in enumerate(['none', 'baseline', 'extra-sample', 'unbiased', 'unbiased--state-only']):
  colors[thing] = color_list[i]

window_len = 1000
fig, axes = plt.subplots(1, len(envs), figsize=(15, 6))
for envidx, env in enumerate(envs):
  ax = axes[envidx]
  print('envidx', envidx, 'env', env)
  ys = []
  for k, v in sorted(results.items()):
    if k[0] == env:
      y = [[row['_MeanReward'] for row in res] for res in v]
      print(len(y))
      print(len(y[0]))
      print(len(y[1]))
      print(len(y[2]))
      print(len(y[3]))
      print(len(y[4]))
      min_len = min([len(x) for x in y])
      y = [[row['_MeanReward'] for row in res][:min_len] for res in v]

      y = np.stack(y)
      if env == "halfcheetah":
        win_len = 1000
      if env == "humanoid":
        win_len = 700
      y_z1 = savgol_filter(y.mean(0) + y.std(0), savgol_window, 5)[:win_len]
      y_z_1 = savgol_filter(y.mean(0) - y.std(0), savgol_window, 5)[:win_len]
      y_max = savgol_filter(y.max(0), savgol_window, 5)[:win_len]
      y_min = savgol_filter(y.min(0), savgol_window, 5)[:win_len]
      y_mean = savgol_filter(y.mean(0), savgol_window, 5)[:win_len]
      # ax.plot(np.arange(len(res)) * 5, ema(res, 0.95), color=colors[k[1]], label='-'.join(k) if j == 0 else None)

      ax.plot(np.arange(min_len)[:win_len] * 10, y_mean, color=colors[k[1]], label=thing_names[k[1]])
      ax.fill_between(np.arange(min_len)[:win_len] * 10, y_mean, np.where(y_z1 > y_max, y_max, y_z1), color=colors[k[1]], alpha=0.2)
      ax.fill_between(np.arange(min_len)[:win_len] * 10, np.where(y_z_1 < y_min, y_min, y_z_1), y_mean, color=colors[k[1]], alpha=0.2)
  ax.legend(loc='upper left', prop={'size': 14})

  # Sorting and plotting legend entries
  handles, labels = axes[envidx].get_legend_handles_labels()
  import operator
  hl = sorted(zip(handles, labels),
              key=operator.itemgetter(1))
  handles2, labels2 = zip(*hl)
  # ax.legend(handles2, labels2, loc='upper left')
  ax.set_title(envs_titles[env], fontsize=18)
  ax.tick_params(axis='both', which='major', labelsize=12)
  ax.tick_params(axis='both', which='minor', labelsize=12)
  ax.set_xlabel('Steps (thousands)', fontsize=14)
  ax.set_ylabel('Average Reward', fontsize=14)
  ax.grid(alpha=0.5)

# Uncomment this to generate the non-mini plot.
"""
h0 = mpatches.Patch(color=color_list[0], label='Stein (biased)')
h1 = mpatches.Patch(color=color_list[1], label='PPO')
h2 = mpatches.Patch(color=color_list[2], label='Stein (biased with no importance sampling)')
h3 = mpatches.Patch(color=color_list[3], label='Stein (unbiased)')
h4 = mpatches.Patch(color=color_list[4], label='Stein (unbiased with state-dependent baseline)')
leg = fig.legend(handles=[h0, h2, h3, h4, h1], loc='lower center', ncol=3, prop={'size': 16})
# Move legend down.
bb = leg.get_bbox_to_anchor().inverse_transformed(ax.transAxes)
bb.y0 += -0.18
leg.set_bbox_to_anchor(bb, transform = ax.transAxes)
"""

fig.savefig('stein_mean_stds_all2-mini.pdf', bbox_inches='tight', format='pdf')#dpi=200)

