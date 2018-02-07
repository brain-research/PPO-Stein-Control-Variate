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

log_dir = 'gcp_logs'

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
      # print(key, seed)
      if key not in results:
        results[key] = []
      result = []
      for line in f:
        result.append(json.loads(line.strip()))
      results[key].append(result)

envs = ['halfcheetah', 'humanoid']
envs_titles = {'halfcheetah': 'HalfCheetah-v1', 'walker2d': 'Walker2d', 'humanoid': 'Humanoid-v1'}
savgol_window = 7
colors = {}
for i, thing in enumerate(['none', 'extra-sample', 'unbiased', 'unbiased--state-only']):
  colors[thing] = color_list[i]

window_len = 1000
fig, axes = plt.subplots(1, len(envs), figsize=(15, 6))
for envidx, env in enumerate(envs):
  ax = axes[envidx]
  print('envidx', envidx, 'env', env)
  ys = []
  for k, v in results.items():
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
      y_z1 = savgol_filter(y.mean(0) + y.std(0), savgol_window, 5)
      y_z_1 = savgol_filter(y.mean(0) - y.std(0), savgol_window, 5)
      y_max = savgol_filter(y.max(0), savgol_window, 5)
      y_min = savgol_filter(y.min(0), savgol_window, 5)
      y_mean = savgol_filter(y.mean(0), savgol_window, 5)
      # ax.plot(np.arange(len(res)) * 5, ema(res, 0.95), color=colors[k[1]], label='-'.join(k) if j == 0 else None)

      ax.plot(np.arange(min_len) * 10, y_mean, color=colors[k[1]], label='-'.join(k))
      ax.fill_between(np.arange(min_len) * 10, y_mean, np.where(y_z1 > y_max, y_max, y_z1), color=colors[k[1]], alpha=0.2)
      ax.fill_between(np.arange(min_len) * 10, np.where(y_z_1 < y_min, y_min, y_z_1), y_mean, color=colors[k[1]], alpha=0.2)

  # Sorting and plotting legend entries
  handles, labels = axes[envidx].get_legend_handles_labels()
  import operator
  hl = sorted(zip(handles, labels),
              key=operator.itemgetter(1))
  handles2, labels2 = zip(*hl)
  # ax.legend(handles2, labels2, loc='upper left')
  ax.set_title(envs_titles[env])
  ax.tick_params(axis='both', which='major', labelsize=11)
  ax.tick_params(axis='both', which='minor', labelsize=11)

h1 = mpatches.Patch(color=color_list[0], label='Stein (biased)')
h2 = mpatches.Patch(color=color_list[1], label='Stein (no importance sampling)')
h3 = mpatches.Patch(color=color_list[2], label='Stein (unbiased)')
h4 = mpatches.Patch(color=color_list[3], label='Stein (unbiased with state-dependent control-variate)')
leg = fig.legend(handles=[h3, h4, h1, h2], loc='lower center', ncol=2, prop={'size': 14})
# Move legend down.
bb = leg.get_bbox_to_anchor().inverse_transformed(ax.transAxes)
bb.y0 += -0.12
leg.set_bbox_to_anchor(bb, transform = ax.transAxes)
fig.savefig('stein_mean_stds.png', bbox_inches='tight', dpi=200)

