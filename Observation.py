import math

class Observation:
  def __init__(self, classification, features):
    self._classification = classification
    self._features = features


  def __lt__(self, observation):
    return self._classification < observation.get_classification()


  def get_classification(self):
    return self._classification


  def get_features(self):
    return self._features


  def compute_distance(self, observation):
    paired_features = zip(self.get_features(), observation.get_features())
    return math.sqrt(sum([(a - b) ** 2 for a, b in paired_features]))
