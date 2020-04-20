import csv
from Observation import Observation
from KDTree import KDTree

if __name__ == '__main__':
  observations = []
  with open('example/observations.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    headers = next(csv_reader)

    for row in csv_reader:
      classification = int(row[0])
      features = [float(feature) for feature in row[1:]]
      observations.append(Observation(classification, features))

  kd_tree = KDTree.build(observations)

  observation = Observation(None, [
    -1.4096938683781,
    0,
    -0.570643869378607,
    -0.163979687631107,
    -0.78891028462005,
    -0.998607403228397,
    -0.08425584140619,
    0.173225008691556,
    0.063331825211717
  ])

  predicted_class = kd_tree.predict_class(observation, 1.5, 20)
  closest_neighbors = kd_tree.create_grouping(observation, 1.5, 20)