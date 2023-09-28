import os
import numpy as np
import pandas as pd
from gower import *
from SACFACT.utils import *
from SACFACT.GeneralEnv import *
import torch as th
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
from scipy.spatial.distance import hamming
from stable_baselines3 import DDPG, TD3, SAC, A2C, PPO
from stable_baselines3.common import results_plotter
from stable_baselines3.common.noise import NormalActionNoise
from SACFACT.CustomCallbacks import *

from sb3_contrib import ARS

class Explainer:
  def __init__(self, data, point, target, class_model, anomaly_model, actor_critic = 'SAC', interval_width = 0.5, timesteps = 10000, model_type='sklearn'):
    # Create log dir
    self.log_dir = "./tmp/gym/"
    os.makedirs(self.log_dir, exist_ok=True)
    self.counterfactuals = []
    self.time_to_1st = []
    self.algorithm = actor_critic

    self.env = Monitor(GeneralEnv(point, data, class_model, anomaly_model, counterfactuals = self.counterfactuals, model_type = model_type, time_to_first_counterfactual = self.time_to_1st), self.log_dir)
    #Create the callback: check every 1000 steps
    self.callback = SaveOnBestTrainingRewardCallback(check_freq=10, log_dir=self.log_dir)
    self.env.initial_point = point
    self.initial_point = list(point.get_position())
    self.env.state = point
    self.env.classifier = class_model
    self.clf = class_model
    self.env.desired_label = target
    self.env.anomaly_detector = anomaly_model
    self.anomaly = anomaly_model
    self.env.interval_width = interval_width
    self.dataset = data
    self.point_to_be_explained = point
    self.state_size = self.env.observation_space.shape[0]
    self.action_size = self.env.action_space.shape[0]
    self.model = None
    self.timesteps = timesteps

    self.model_type = model_type
    


  def train(self):
    policy = 'MlpPolicy'
    policy_kwargs =   dict(net_arch=dict(pi=[32, 64, 64, 128], qf=[300, 200, 100])) #dict(net_arch=dict(pi=[32, 64, 128], qf=[100, 200, 300])) #
    action_noise = NormalActionNoise(mean=np.zeros(self.action_size), sigma=0.1 * np.ones(self.action_size))
    if self.algorithm == 'SAC':
       model = SAC(policy, self.env, action_noise=action_noise, tensorboard_log="./counterfactuals_/", policy_kwargs=policy_kwargs, batch_size = 256, buffer_size = 1000000, verbose=1) #, learning_rate = 0.01, gamma = 0.8, use_sde = False)
    elif self.algorithm == 'DDPG':
       model = DDPG(policy, self.env,  verbose=1)
    elif self.algorithm == 'TD3':
       model = TD3(policy, self.env, verbose= 1)
    elif self.algorithm == 'A2C':
       model = A2C(policy, self.env, tensorboard_log="./counterfactuals_/" , verbose=1) #, learning_rate = 0.2, gamma = 0.8, use_sde = True)
    elif self.algorithm == 'PPO':
       model = PPO(policy, self.env, verbose=1)
    elif self.algorithm == 'ARS':
       model = ARS("MlpPolicy", self.env, verbose=1)
       
       
    model.learn(total_timesteps=self.timesteps, log_interval=1, callback= self.callback)
    env = model.get_env()
    self.model = model
    self.env = env

  def predict(self):
    print("PREDICTING")
    obs = self.env.reset()
    dones = False
    while dones != True:
        action, _states = self.model.predict(obs)
        obs, rewards, dones, info = self.env.step(action)
        print(obs)


  def plot(self):     # Helper from the library
    results_plotter.plot_results([self.log_dir], self.timesteps, results_plotter.X_TIMESTEPS, "Explainer")
    return self.plot_results(self.log_dir)


  def moving_average(self, values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


  def plot_results(self, log_folder, title='Learning Curve'):
      """
      plot the results

      :param log_folder: (str) the save location of the results to plot
      :param title: (str) the title of the task to plot
      """
      x, y = ts2xy(load_results(log_folder), 'timesteps')
      y = self.moving_average(y, window=50)
      # Truncate x
      x = x[len(x) - len(y):]
      return (x, y)
      #fig = plt.figure(title)
      #plt.plot(x, y)
      #plt.xlabel('Number of Timesteps')
      #plt.ylabel('Rewards')
      #plt.title(title + " Smoothed")
      #plt.show()


  def report_counterfactuals(self):
      report = pd.DataFrame(columns = ['counterfactual', 'Number of changed features', 'gower distance', 'anomaly', 'new label'])
      for counterfactual in self.counterfactuals:
            hamming_distance = hamming(self.initial_point, counterfactual) * len(counterfactual)
            #df = pd.DataFrame(columns = self.dataset.columns)  
            #df.loc[0] = counterfactual
            #df.loc[1] = self.initial_point
            g_dist = gower_custom(self.dataset, counterfactual, self.initial_point)
            #g_dist = gower_matrix(df)[0, 1] #* 100
            anom = self.anomaly.predict([counterfactual])[0]
            if anom == 1:
              an = 'Not Anomaly'
            else:
              an = 'Anomaly'

            if self.model_type=='sklearn':
               lab = self.clf.predict([counterfactual])

            else:
               lab = np.argmax(self.clf.predict([counterfactual]))

            report.loc[len(report)] = [str(counterfactual), hamming_distance, g_dist, an, lab]
      return report.sort_values(by = 'Number of changed features')
  
  def detect_convergence(self, patience, threshold):
      """
      Detects when an increasing curve has converged based on the patience and threshold parameters.

      Parameters:
          patience (int): The number of iterations to wait for convergence.
          threshold (float): The minimum improvement required for convergence.

      Returns:
          (int): The iteration number when convergence was detected, or -1 if convergence was not detected.
      """
      x, y = self.plot_results(self.log_dir)
      best_value_x = -float("inf")
      best_value_y = -float('inf')
      wait_count = 0

      for i, value in enumerate(y):
          if value > best_value_y:
              best_value_y = value
              best_value_x = x[i]
              wait_count = 0
          else:
              wait_count += 1
              if wait_count >= patience and best_value_y - value < threshold:
                  return x[i - patience + 1]

      return -1

