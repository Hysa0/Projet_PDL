# Ce code a été initialement généré par ChatGPT (OpenAI, avril 2025),
# puis modifié manuellement pour les besoins de ce projet.
import numpy as np
import pandas as pd
import pickle as pk
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.datasets import load_iris
import pandas as pd

import utils
import distribution
from FACE import *
from kernel import *
from dataLoader import *

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def plot_recourse(data, face_recourse, plot_idx=0):
	#all_pos = data[data.y == 1]
	all_x1 = data['sepal length (cm)']
	all_x2 = data['sepal width (cm)']

	#all_neg = data[data.y == 0]
	#all_neg_x1 = all_neg.x1.values
	#all_neg_x2 = all_neg.x2.values

	#plt.plot(all_x1, all_x2, '*')
	#plt.plot(all_neg_x1, all_neg_x2, '*')
	plt.scatter(all_x1, all_x2, c=data['target'], cmap='viridis', label='Data points')
	plt.xlabel('sepal length (cm)')
	plt.ylabel('sepal width (cm)')
	plt.title('FACE Iris data set')

	assert(plot_idx < len(data))

	plot_pt = face_recourse[plot_idx]['factual_instance']
	plot_cfpt = face_recourse[plot_idx]['counterfactual_target']
	points_x1 = []
	points_x2 = []
	points_x1 = [data.iloc[x]['sepal length (cm)'] for x in face_recourse[plot_idx]['path']]
	points_x2 = [data.iloc[x]['sepal width (cm)'] for x in face_recourse[plot_idx]['path']]
	plt.plot(points_x1, points_x2, color='green')
	plt.plot(plot_pt['sepal length (cm)'], plot_pt['sepal width (cm)'], 'o',color='red')
	plt.plot(plot_cfpt['sepal length (cm)'], plot_cfpt['sepal width (cm)'], 'o', color='red')
	plt.show()
	#plt.savefig('./tmp/recourse_path_{}.jpg'.format(plot_idx))

# Charger les données Iris
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)

# Ajouter la colonne cible
df['target'] = data.target

# Afficher les premières lignes pour vérifier
#print(df.head())

# Supposons que tu aies la classe FACE et la méthode make_graph déjà définie
# On charge les données Iris
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Création de l'objet FACE
density_method = distribution.distribution(df)  # Remplacer par ta méthode de densité
distance_object = distance_obj()  # Remplacer par l'objet de distance (ex. distance Euclidienne)
kernel_obj = Kernel_obj(density_method, Num_points=len(df), knnK=5) # Remplacer par l'objet de noyau
feature_columns = data.feature_names
kernel_obj.fitKernel(df[feature_columns])
target_column = 'target'
epsilon = 8  # Un paramètre de distance ou d'epsilon

clf = LogisticRegression(random_state=utils.random_seed)
clf.fit(df[feature_columns], df[target_column])
print("Training accuracy:", clf.score(df[feature_columns], df[target_column]))

face = FACE(df, density_method, distance_object, kernel_obj, feature_columns, target_column, epsilon, clf)

# Construction du graphe
datasetName = "no_constraint"
feasibilitySet = utils.getFeasibilityConstraints(feature_columns, dataset_name=datasetName)  # Définir ou générer un ensemble de faisabilité
face.make_graph(feasibilitySet, epsilon)
print(f"Nombre de nœuds dans le graphe : {len(face._Graph)}")
print(f"Exemple d’arêtes pour un nœud : {next(iter(face._Graph.items()))}")
# Sélectionner un point de départ et de destination
source_id = 0  # Exemple, on prend le premier point de la base de données
target_id = 30  # Exemple, on prend un autre point pour tester

# Calculer le recourse
tp = 0.6  # Seuil de probabilité pour la prédiction du classificateur
td = 0.001  # Seuil de densité (pdf)
recourse_points = {}
path_lengths = []
negative_points = utils.get_negatively_classified(df, clf, feature_columns)
print("# negative points:", len(negative_points) , "/", len(df))

#path={}
#path_points = []
for n_id, n in enumerate(negative_points):
	#print("Computing recourse for: {}/{}".format(n_id, len(negative_points)))
	recourse_point, cost, recourse_path = face.compute_recourse(n, tp, td)
	#print(f"recurse path: {recourse_path}")
	#print(f"recurse point: {recourse_point}")
	recourse_points[n_id] = {}
	recourse_points[n_id]['name'] = n
	recourse_points[n_id]['factual_instance'] = negative_points[n]
	recourse_points[n_id]['counterfactual_target'] = recourse_point
	recourse_points[n_id]['cost'] = cost
	recourse_points[n_id]['path'] = recourse_path
	#path[n_id] = recourse_path
	#path_points.append(df.iloc[recourse_path])
	if (recourse_path is not None):
		path_lengths.append(len(recourse_path))
print(n)
print(recourse_points[2]['path'])

#all_x1 = df['sepal length (cm)']
#all_x2 = df['sepal width (cm)']
#plt.scatter(all_x1, all_x2, c=data['target'], cmap='viridis', label='Data points', alpha=0.5)
#plt.xlabel('sepal length (cm)')
#plt.ylabel('sepal width (cm)')
#plt.title('FACE Iris data set')

#plot_idx = 10
#plot_pt = recourse_points[plot_idx]['factual_instance']
#plot_cfpt = recourse_points[plot_idx]['counterfactual_target']
#plt.scatter(plot_pt['sepal length (cm)'], plot_pt['sepal width (cm)'], marker='X',color='red', s=100)
#plt.plot(plot_cfpt['sepal length (cm)'], plot_cfpt['sepal width (cm)'], 'o', color='red')
#plt.show()
plot_recourse(df, recourse_points, 10)
#target, path_cost, path = face.compute_recourse(source_id, tp, td)

# Visualisation du chemin trouvé
#source_point = df.iloc[source_id]
#target_point = df.iloc[target_id]

# Extraire les coordonnées des points du chemin
#path_points = df.iloc[path]

# Visualiser les points sur un graphique
#plt.figure(figsize=(8, 6))
#plt.scatter(df['sepal length (cm)'], df['sepal width (cm)'], c=df['target'], cmap='viridis', label='Data points')

# Mettre en évidence le point de départ et le point d'arrivée
#plt.scatter(source_point['sepal length (cm)'], source_point['sepal width (cm)'], color='red', s=100, label='Source')
#plt.scatter(target_point['sepal length (cm)'], target_point['sepal width (cm)'], color='blue', s=100, label='Target')

# Tracer le chemin
#for i in range(1, len(path)):
#    plt.plot([path_points.iloc[i-1]['sepal length (cm)'], path_points.iloc[i]['sepal length (cm)']],
#             [path_points.iloc[i-1]['sepal width (cm)'], path_points.iloc[i]['sepal width (cm)']], color='gray')

#plt.title('Path from Source to Target')
#plt.xlabel('Sepal Length (cm)')
#plt.ylabel('Sepal Width (cm)')
#plt.legend()
#plt.show()