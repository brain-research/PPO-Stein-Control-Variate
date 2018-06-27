import pickle
import os 
import errno
import os
import sys
import numpy as np
import json
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.rc('text', usetex=True)
import matplotlib.patches as mpatches

color_list = sns.color_palette("muted")
sns.palplot(color_list)

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def load_grad(file_path, fields=['mu_grad', 'sigma_grad']):
  with open(file_path, 'rb') as f:
    traj_data = pickle.load(f)
    sample_grads = np.concatenate(
        [traj_data[field] for field in fields], axis=1)

    return sample_grads

def load_sample_grads(batch_range, prefix_dir):
    file_dir = prefix_dir

    # load mc traj
    mc_grads = []
    for i in batch_range:
        file_path = os.path.join(file_dir, 'mc_num_episode=%d.pkl'%i)
        mc_grads.append(load_grad(file_path))


    ai_grads = []
    stein_grads = []
    for i in batch_range:
        file_path = os.path.join(file_dir, 'stein_num_episode=%d.pkl'%i)
        ai_grads.append(load_grad(file_path,
                                  ['mu_ai_grad', 'sigma_ai_grad']))
        stein_grads.append(load_grad(file_path))

    return mc_grads, ai_grads, stein_grads

legend_handles = {}

def plot(phi_obj, batch_range=range(10, 20, 10), max_timesteps=500, seeds=[13], env_name='Walker2d-v1'):
  k = 2000
  plot_stein_loss = []
  plot_mc_loss = []

  for seed in seeds:
      prefix_dir = 'max_timesteps=%s_eval_data/%s_%s_data_seed=%d_max-steps=%s'%(max_timesteps, env_name, phi_obj, seed, max_timesteps)
      print(prefix_dir)

      # This is gradient for each trajectory
      mc_x = []
      stein_x = []
      plot_stein_vars = []
      plot_mc_vars = []

      mc_grads, ai_grads, stein_grads = load_sample_grads(batch_range, prefix_dir)

      # Calculate variance
      grads = [mc_grads, ai_grads, stein_grads]
      variances = [[0]*len(grad) for grad in grads]

      x = []
      for i in range(len(mc_grads)):
        n_samples = len(mc_grads[i])  # all trajs are concatenated together
        x.append(n_samples)

        for _ in range(k):
          indices = np.random.choice(n_samples, int(n_samples/2), replace=False)
          total_indices = np.arange(0,  n_samples)
          mask = np.zeros(total_indices.shape, dtype=bool)
          mask[indices] = True

          for j, grad in enumerate(grads):
            g = np.array(grad[i])
            var = np.sum((np.mean(g[total_indices[mask], :], axis=0) -
                          np.mean(g[total_indices[~mask], :], axis=0)) ** 2)
            variances[j][i] += var/k

      print (seed)
      print(x)
      labels = ['Value', '%s-MLP (state only)' % phi_obj, '%s-MLP' % phi_obj]
      for plot_i, (variance, label) in enumerate(zip(variances, labels)):
        print(label)
        print(np.log(variance))
        if label not in legend_handles:
          legend_handles[label] = (mpatches.Patch(color=color_list[len(legend_handles)], label=label),
                                   color_list[len(legend_handles)])
        plt.plot(x, np.log(variance), color=legend_handles[label][1], label=label)
      plt.tick_params(labelsize=11)
      plt.ylabel('ln(Variance Proxy)', fontsize=16)
      plt.xlabel('Sample size', fontsize=16)
      plt.title('Walker2d-v1 %s' % phi_obj, fontsize=18)
      plt.grid(alpha=0.5)
      #plt.legend()

  return legend_handles

if __name__ == '__main__':
  plt.figure(figsize=(20,6))
  plt.subplot(1, 2, 1)
  plot('FitQ', batch_range=range(10, 40, 10))
  plt.subplot(1, 2, 2)
  legend_handles = plot('MinVar', batch_range=range(10, 80, 10))

  plt.legend(handles=[x for _, (x, _) in sorted(legend_handles.items(), key=lambda x: x[0])],
             loc='upper center', bbox_to_anchor=(-0.10, -0.13),
             ncol=6,
             prop={'size': 14})

  mkdir_p('results')
  plt.savefig('results/stein.pdf', bbox_inches='tight')

