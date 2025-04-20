# Ce code a été initialement généré par ChatGPT (OpenAI, avril 2025),
# puis modifié manuellement pour les besoins de ce projet.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import dice_ml

# 1. Charger les données
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
column_names = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income"
]
df = pd.read_csv(url, names=column_names, header=None, na_values=" ?", skipinitialspace=True)
df.dropna(inplace=True)
df["income"] = df["income"].apply(lambda x: 1 if x == ">50K" else 0)

# 2. Définir les colonnes
outcome_name = "income"
continuous_features = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
categorical_features = ["workclass", "education", "marital-status", "occupation",
                        "relationship", "race", "sex", "native-country"]
features = continuous_features + categorical_features

# 3. Séparer X/y
X = df[features]
y = df[outcome_name]

# 4. Encodage pour sklearn
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), continuous_features),
    ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# 5. Pipeline modèle
pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("clf", LogisticRegression(max_iter=1000))
])

pipeline.fit(X, y)

# 6. Créer les objets DiCE
data_dice = dice_ml.Data(dataframe=df,
                         continuous_features=continuous_features,
                         categorical_features=categorical_features,
                         outcome_name=outcome_name)

model_dice = dice_ml.Model(model=pipeline, backend="sklearn")

exp = dice_ml.Dice(data_dice, model_dice, method="random")

# 7. Sélectionner une instance à expliquer
query_instance = X.iloc[[0]]
print("Instance :\n", query_instance)

# 8. Générer des contrefactuels
cf = exp.generate_counterfactuals(query_instance,
                                  total_CFs=3,
                                  desired_class="opposite",
                                  features_to_vary=["age", "hours-per-week", "education-num"])

# 9. Affichage
cf.visualize_as_dataframe()