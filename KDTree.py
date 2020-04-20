import bisect

class KDTree:
  def __init__(self, observation=None, index=None, left_tree=None, right_tree=None):
    self._observation = observation
    self._index = index
    self._left_tree = left_tree
    self._right_tree = right_tree


  @staticmethod
  def build(observations, depth=0):
    if len(observations) <= 0:
      return None

    index_to_split = depth % len(observations[0].get_features())
    observations.sort(key=lambda observation: observation.get_features()[index_to_split])

    median_index = len(observations) // 2
    left_tree = KDTree.build(observations[:median_index], depth + 1)
    right_tree = KDTree.build(observations[median_index + 1:], depth + 1)

    return KDTree(observations[median_index], index_to_split, left_tree, right_tree)


  # Find the majority vote class given the observation using a maximum distance
  # and maximum number of observations to consider in the consensus.
  def predict_class(self, observation, max_distance, max_observations_in_grouping):
    nearest_observations = self.create_grouping(observation, max_distance, max_observations_in_grouping)
    classes = [observation.get_classification() for observation in nearest_observations]
    return max(set(classes), key=classes.count)


  # Creates a group of the closest observations surrounding the observation using
  # a maximum distance and maximum number of observations to consider.
  def create_grouping(self, observation, max_distance, max_observations_in_grouping):
    if max_distance <= 0 or max_observations_in_grouping < 1:
      return []

    nearest_observations = self._create_grouping(observation, max_distance, max_observations_in_grouping)
    return list(map(lambda nearest_observation: nearest_observation[1], nearest_observations))


  def _create_grouping(self, observation, max_distance, max_observations_in_grouping, nearest_observations=[]):
    distance = self._observation.compute_distance(observation)
    if distance <= max_distance:
      self._add_nearest_observation(distance, nearest_observations, max_observations_in_grouping)

    chosen_branch, unchosen_branch = self._get_chosen_and_unchosen_branches(observation)

    if chosen_branch is not None:
      nearest_observations = chosen_branch._create_grouping(
        observation,
        max_distance,
        max_observations_in_grouping,
        nearest_observations
      )

    if self._should_check_unchosen_branch(observation, nearest_observations, max_observations_in_grouping):
      if unchosen_branch is not None:
        nearest_observations = unchosen_branch._create_grouping(
          observation,
          max_distance,
          max_observations_in_grouping,
          nearest_observations
        )

    return nearest_observations


  # Insert the distance and current pivot observation into nearest_observations
  # When the distance is farther than the farthest nearest observation,
  # replace it at the front of the list for future comparisons.
  def _add_nearest_observation(self, distance, nearest_observations, max_observations_in_grouping):
    nearest_observation = (-distance, self._observation)
    if len(nearest_observations) < max_observations_in_grouping:
      bisect.insort(nearest_observations, nearest_observation)
    elif nearest_observations[0][0] < -distance:
      nearest_observations[0] = nearest_observation


  # Choose the left branch when the observation has a smaller
  # value than the current pivot observation value and the
  # right branch when the value is greater than or equal.
  def _get_chosen_and_unchosen_branches(self, observation):
    comparision_value = self._observation.get_features()[self._index]
    observation_value = observation.get_features()[self._index]

    if observation_value < comparision_value:
      chosen_branch = self._left_tree
      unchosen_branch = self._right_tree
    else:
      chosen_branch = self._right_tree
      unchosen_branch = self._left_tree

    return chosen_branch, unchosen_branch


  # Determine if the unchosen region is intersected by the radius
  # from the farthest nearest observation so far. Distance to the region
  # is simply the absolute value between the pivot axis and the other
  # branch axis value. If there are less than max_observations_in_grouping
  # observations found, check both branches.
  def _should_check_unchosen_branch(self, observation, nearest_observations, max_observations_in_grouping):
    if len(nearest_observations) < max_observations_in_grouping:
      return True

    comparision_value = self._observation.get_features()[self._index]
    observation_value = observation.get_features()[self._index]

    distance_from_observation_to_farthest_node = observation.compute_distance(nearest_observations[0][1])
    distance_from_observation_to_unchosen_region = abs(observation_value - comparision_value)

    return distance_from_observation_to_unchosen_region <= distance_from_observation_to_farthest_node