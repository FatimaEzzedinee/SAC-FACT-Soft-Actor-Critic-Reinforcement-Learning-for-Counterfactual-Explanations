import os
import numpy as np
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import gym



class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}.zip")
                  self.model.save(self.save_path)

        return True

#https://github.com/DLR-RM/stable-baselines3/issues/341
class StopTrainingOnNoModelImprovement(BaseCallback):
    """
    Stop the training early if there is no new best model (new best mean reward) after more than N consecutive evaluations.
    
    It is possible to define a minimum number of evaluations before start to count evaluations without improvement.   
    
    It must be used with the ``ExtendedEvalCallback``.
    
    :param max_no_improvement_evals: Maximum number of consecutive evaluations without a new best model.
    :param min_evals: Number of evaluations before start to count evaluations without improvements.
    :param verbose:
    """

    def __init__(self, max_no_improvement_evals: int, min_evals: int = 0, verbose: int = 0):
        super(StopTrainingOnNoModelImprovement, self).__init__(verbose=verbose)
        self.max_no_improvement_evals = max_no_improvement_evals
        self.min_evals = min_evals
        self.last_best_mean_reward = -np.inf
        self.no_improvement_evals = 0

    def _on_step(self) -> bool:
        assert self.parent is not None, "``StopTrainingOnNoModelImprovement`` callback must be used " "with an ``ExtendedEvalCallback``"
        
        continue_training = True
        
        
        if self.n_calls > self.min_evals:
            if self.parent.best_mean_reward > self.last_best_mean_reward:
                self.no_improvement_evals = 0                
            else:
                self.no_improvement_evals += 1
                if self.no_improvement_evals > self.max_no_improvement_evals:
                    continue_training = False        
        
        self.last_best_mean_reward = self.parent.best_mean_reward
                
        if self.verbose > 0 and not continue_training:
            print(
                f"Stopping training on step {self.parent.num_timesteps} because there was no new best model in the last {self.no_improvement_evals:d} evaluations"
            )
        
        return continue_training

from typing import Union, Optional
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

class ExtendedEvalCallback(EvalCallback):
    """
    Extends Eval Callback by adding a new child callback called after each evaluation.
    """
    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: str = None,
        best_model_save_path: str = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
    ):
        super(ExtendedEvalCallback, self).__init__(
            eval_env,
            callback_on_new_best=callback_on_new_best,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            log_path=log_path,
            best_model_save_path=best_model_save_path,
            deterministic=deterministic,
            render=render,
            verbose=verbose,
            warn=warn)

        self.callback_after_eval = callback_after_eval
        # Give access to the parent
        if self.callback_after_eval is not None:
            self.callback_after_eval.parent = self
    
    def _init_callback(self) -> None:
        super(ExtendedEvalCallback, self)._init_callback()
        if self.callback_after_eval is not None:
            self.callback_after_eval.init_callback(self.model)
    
    def _on_step(self) -> bool:
        continue_training = super(ExtendedEvalCallback, self)._on_step()
        
        if continue_training:
            if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
                # Trigger callback if needed
                if self.callback_after_eval is not None:
                    return self.callback_after_eval.on_step()
        return continue_training
#####################################################################################################################################