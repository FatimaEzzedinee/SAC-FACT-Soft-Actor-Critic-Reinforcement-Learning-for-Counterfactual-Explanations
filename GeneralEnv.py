import numpy as np
from SACFACT.utils import *
from gower import *
import time
import copy
from gym import Env, spaces
from scipy.spatial.distance import hamming
from sklearn.metrics.pairwise import cosine_similarity

class GeneralEnv(Env):
    def __init__(self, initial_point, data, class_model, anomaly_model, counterfactuals, model_type, time_to_first_counterfactual, probability_threshold = 0.08):
        super(GeneralEnv, self).__init__()
        self.number_features = data.shape[1]
        self.dataset = data
        self.feature_names = self.dataset.columns
        min_max = minMax(self.dataset)
        self.mins = np.array(min_max['min'].values)
        self.maxs = np.array(min_max['max'].values)
        self.interval_width = 2

        self.model_type = model_type
        

        self.action_space = spaces.Box(low=np.array([0, -1]), high=np.array([self.number_features, 1]), dtype=np.float16)
        self.observation_space = spaces.Box(low = self.mins, high = self.maxs, dtype=np.float16)

        self.elements = []
        self.initial_point = initial_point
        self.state = initial_point
        self.classifier = class_model
        if self.model_type == 'sklearn':
          self.initial_label = self.classifier.predict([self.initial_point.get_position()])[0]   #self.initial_point.label
        else:
          self.initial_label = np.argmax(self.classifier.predict([self.initial_point.get_position()])[0])#, axis=1)  #self.initial_point.label

        self.desired_label = None
        self.anomaly_detector = anomaly_model
        self.counterfactuals = counterfactuals
        self.time_to_first_counterfactual = time_to_first_counterfactual
               

        self.current_step = 0
        self.achievment_reward = 1000 # reward for achieving the goal
        self.delta = -50 # reward for minimal number of changes
        self.anomaly_penalty = -500
        self.probability_threshold = probability_threshold

       

    # Gower Distance is a distance measure that can be used to calculate distance between two entity whose attribute has a mixed of categorical and numerical values.
    # default is gower
    # It uses the concept of Manhattan distance for continuous variables and dice distance for measuring similarity between Binary variables.
    def similarity(self, new_state, amount, sim ='hamming'):
      initial = self.initial_point.get_position()
      new = new_state.get_position()

      if sim == 'cosine':
        print("cosine")
        A = np.array([initial])
        B = np.array([new])
        print("Similarity")
        print(A)
        print(B)
        cosine = cosine_similarity(np.array(A) , np.array(B))[0][0] # np.dot(A,B)/(norm(A)*norm(B))
        print("Cosine similarity: " + str(cosine))
        return cosine * 100

      if sim == 'hamming':
        hamming_distance = hamming(initial, new) * len(new)
        print("Hamming Distance: " + str(hamming_distance))
        g_dist = gower_custom(self.dataset, initial, new)
        print("Gower Distance: "+ str(g_dist))
        if hamming_distance == 1:
          return  0 #g_dist * 10
          #return g_dist  * 100 * 2
        return -1 * hamming_distance * 10 - g_dist * 1000

  
    def distribution(self, new_state, action, method = 'anomaly'):   #default method is probability, can be changed to anomaly 
      if method == 'anomaly':
          #predict outlier
          point = new_state.get_position()
          pred = self.anomaly_detector.predict([point])[0]
          if pred == 1:
            print("Not anomaly -> No penalty")
            return 0
          else:
            print("Anomaly -> Penalty of: " + str(self.anomaly_penalty))
            return self.anomaly_penalty

      if method =='probability':
        np_intervals = prepare_intervals(20, self.interval_width)
        value = new_state.features[action]
        values = self.dataset.iloc[:, action].values
        (a, b), index = find_interval(np_intervals, value)
        prob = probability(values, a, b)
        print("Probability of value: "+ str(value) + " is: " + str(prob) + " in interval: ["+ str(a) + ", " + str(b) + "]" )
        if prob < self.probability_threshold:   #(index, np_intervals, values, threshold):
          _, dist = closest_interval_count(index, np_intervals, values, self.probability_threshold)
          print("Dist: " + str(dist))
          #########################################
          sl, sw, pl, pw = new_state.get_position()
          point = [sl, sw, pl, pw]
          pred = self.anomaly_detector.predict([point])[0]
          if pred != 1:
            dist += self.anomaly_penalty 
          return dist
        else:
          return 0 

    def reset(self): # Reset the state of the environment to an initial state
        self.state = self.initial_point
        return np.array([self.state.get_position()])

    def _custom_reward(self, new_state, amount, a):
      reward = 0
      done = False
      reward += self.delta
      reward += self.similarity(new_state, amount)
      anomaly = self.distribution(new_state, a)
      reward += anomaly
      if new_state.label != self.initial_label: # == self.desired_label: # and anomaly == 0 : #
        reward += self.achievment_reward
        done = True
        self.counterfactuals.append(new_state.get_position())         
      return reward, done
    
    def return_counterfactuals(self):
      return self.counterfactuals

   # Execute one time step within the environment
    def _take_action(self, action):
      action_type = action[0] # feature to be changed
      amount = action[1]      # amount to be changed (+ or -)
      a = int(action_type)
      print("Action: ", a)
      if a == self.number_features:
        a = a - 1
      values = self.state.get_position()

      new_state = GenPoint(self.state.name, self.state.label)
      new_state.set_position(copy.deepcopy(values))

      info = 'Changing feature ' + self.feature_names[a] + ' by: '+ str(amount) + " %"
      new_state.move(a, amount)

      #if new_state.features[a] == 0.0:
      #  new_state.features[a] = 0.1

      # Keep the counterfactual bounded 
      if new_state.features[a] > self.maxs[a]:
        new_state.features[a] = self.maxs[a]

      if new_state.features[a] < self.mins[a]:
        new_state.features[a] = self.mins[a]
      
      if self.model_type == 'sklearn':
        new_state.label = self.classifier.predict([new_state.get_position()])[0]

      else:
        new_state.label = np.argmax(self.classifier.predict([new_state.get_position()])[0])#, axis=1)
        
      print("action type: " + str(action_type))
      print("Feature : "+ str(a))
      print(info)
      print("New State Label : " + str(new_state.label))
      print('##$#$#$#')
      print("State Position "+ str (self.state.get_position()))
      print("New State Position " + str(new_state.get_position()))
      print('##$#$#$#')
      #new_state.name = label_list[new_state.label]

      reward, done = self._custom_reward(new_state, amount, a)
      print("Reward: " + str(reward))
      print("Done: "+ str(done))
      print("############################")
      return new_state, reward, done, info


    def step(self, action):
      # Execute one time step within the environment
      if self.current_step == 0:
        self.start_time = time.time()
        self.time_to_first_counterfactual.append(self.start_time)
      new_state, reward, done, info = self._take_action(action)
      if len(self.time_to_first_counterfactual) == 1 and done == True:
        self.time_to_first_counterfactual.append(time.time() - self.start_time)
      self.current_step += 1
      self.state = new_state
      return np.array(new_state.get_position()), reward, done, {}