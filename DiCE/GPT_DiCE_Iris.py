# Ce code a été initialement généré par ChatGPT (OpenAI, avril 2025),
# puis modifié manuellement pour les besoins de ce projet.
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import dice_ml

# Charger les données Iris
iris = load_iris(as_frame=True)
df = iris.frame.copy()
df['target'] = iris.target  # au cas où

# On garde deux classes pour un exemple binaire (DiCE est conçu pour binaire)
df = df[df['target'].isin([0, 1])]

# Définir les colonnes
outcome_name = 'target'
continuous_features = list(df.columns[:-1])  # les 4 premières colonnes

# Split
X = df.drop(columns=[outcome_name])
y = df[outcome_name]

# Entraîner un modèle simple
model = LogisticRegression()
model.fit(X, y)

# Créer les objets DiCE
data_dice = dice_ml.Data(dataframe=pd.concat([X, y], axis=1),
                         continuous_features=continuous_features,
                         outcome_name=outcome_name)

model_dice = dice_ml.Model(model=model, backend="sklearn")

dice_exp = dice_ml.Dice(data_dice, model_dice, method="genetic") # methode kdtree ou genetic plus intéressant que random

# Choisir une instance
query_instance = X.iloc[[10]]
print("Instance :\n", query_instance)

# Générer des contrefactuels (target = 1 si l'instance est de classe 0, ou inversement)
cf = dice_exp.generate_counterfactuals(query_instance,
                                       total_CFs=3,
                                       desired_class="opposite")

# Visualiser les résultats
cf.visualize_as_dataframe()

# Extraire les contrefactuels sous forme DataFrame
cf_df = cf.cf_examples_list[0].final_cfs_df[continuous_features]
#print("Contrefactuels :\n", cf_df)

# --- VISUALISATION ---

# Réduire en 2D avec PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
query_pca = pca.transform(query_instance)
cf_pca = pca.transform(cf_df)

# Afficher tout
plt.figure(figsize=(8, 6))

# Tous les points du dataset
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, alpha=1, label="Dataset original")

# Point d'origine
plt.scatter(query_pca[0, 0], query_pca[0, 1], color='blue', label='Point original', marker='X', s=100)

# Contrefactuels
plt.scatter(cf_pca[:, 0], cf_pca[:, 1], color='red', marker='*', label='Contrefactuels', s=80)

# Lignes entre l’original et les CFs
for i in range(len(cf_pca)):
    plt.plot([query_pca[0, 0], cf_pca[i, 0]], [query_pca[0, 1], cf_pca[i, 1]], 
             color='gray', linestyle='--', linewidth=1)

plt.title("Visualisation des contrefactuels (Iris + PCA)")
plt.legend()
plt.grid(True)
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.tight_layout()
plt.show()