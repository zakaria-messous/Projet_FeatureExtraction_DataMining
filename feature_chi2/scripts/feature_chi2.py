import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_selection import SelectKBest, chi2
from imblearn.over_sampling import SMOTE
import os

# Assurer que les répertoires de résultats existent
os.makedirs('../results', exist_ok=True)

# Étape 1 : Charger les données TF-IDF avec G3
try:
    data = pd.read_csv('../data/tfidf_output.csv')  # Ajuster le chemin si nécessaire
    print("Les données ont été chargées avec succès !")
except FileNotFoundError:
    print("Erreur : Fichier introuvable. Assurez-vous que 'tfidf_output.csv' existe dans le dossier 'data'.")
    exit()

# Nettoyer et valider les données de G3
data = data[pd.to_numeric(data['G3'], errors='coerce').notnull()]  # Supprimer les valeurs non valides de G3
data['G3'] = data['G3'].astype(float)  # Assurer que G3 est numérique

# Simplifier G3 en catégories (Low, Medium, High)
bins = [0, 10, 15, 20]
labels = ['Low', 'Medium', 'High']
data['G3_category'] = pd.cut(data['G3'], bins=bins, labels=labels)

# Supprimer les lignes avec des catégories manquantes ou invalides
data = data[data['G3_category'].notna()]  # Assurer qu'il n'y a pas de catégories manquantes
data['G3_category'] = data['G3_category'].astype(str)  # S'assurer que G3_category est de type string

# Étape 2 : Séparer les caractéristiques et la cible
X = data.drop(columns=['G3', 'G3_category'])
y = data['G3_category']

# Étape 3 : Sélection de caractéristiques avec Chi2
print("Effectuer la sélection de caractéristiques avec Chi2...")
k = 10  # Nombre de meilleures caractéristiques à sélectionner
selector = SelectKBest(score_func=chi2, k=k)
X_selected = selector.fit_transform(X, y)

# Obtenir les noms des caractéristiques sélectionnées
selected_features = X.columns[selector.get_support()]
print(f"Les {k} meilleures caractéristiques sélectionnées : {list(selected_features)}")

# Sauvegarder les scores Chi2
chi2_scores = selector.scores_
chi2_scores_df = pd.DataFrame({"Feature": X.columns, "Chi2_Score": chi2_scores})
chi2_scores_df.to_csv('../results/chi2_feature_scores.csv', index=False)
print("Les scores Chi2 ont été sauvegardés avec succès !")

# Visualiser les scores Chi2
chi2_scores_df = chi2_scores_df.sort_values(by="Chi2_Score", ascending=False)
chi2_fig = px.bar(
    chi2_scores_df,
    x="Feature",
    y="Chi2_Score",
    title="Scores Chi2 des Caractéristiques",
    labels={"Chi2_Score": "Score Chi2", "Feature": "Caractéristique"},
    template="plotly_white",
)
chi2_fig.write_html('../results/chi2_scores_chart.html')
print("Le graphique des scores Chi2 a été sauvegardé.")

# Étape 4 : Équilibrer les classes avec SMOTE
print("Équilibrer les classes avec SMOTE...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_selected, y)
print("Les classes ont été équilibrées avec succès !")

# Étape 5 : Division Entraînement/Test
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Étape 6 : Entraîner et évaluer les modèles
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(kernel='linear', random_state=42),
    "Multinomial NB": MultinomialNB()
}

results = []
metrics_data = []

for model_name, model in models.items():
    print(f"Entraînement du modèle {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    metrics = classification_report(y_test, y_pred, output_dict=True)
    accuracy = accuracy_score(y_test, y_pred)
    results.append({"Model": model_name, "Metrics": metrics})
    
    print(f"Résultats pour le modèle {model_name} :\n")
    print(classification_report(y_test, y_pred))
    
    metrics_data.append({"Model": model_name, "Accuracy": accuracy})
    
    # Visualiser les métriques par classe
    class_metrics = []
    for label, values in metrics.items():
        if label in ['Low', 'Medium', 'High']:  # Focus sur les classes valides
            class_metrics.append({
                "Class": label,
                "Precision": values["precision"],
                "Recall": values["recall"],
                "F1-Score": values["f1-score"]
            })
    class_metrics_df = pd.DataFrame(class_metrics)

    fig = px.bar(
        class_metrics_df,
        x="Class",
        y=["Precision", "Recall", "F1-Score"],
        barmode="group",
        title=f"{model_name} : Métriques par classe",
        template="plotly_white",
    )
    fig.write_html(f"../results/{model_name.lower().replace(' ', '_')}_chart.html")
    print(f"Le graphique des métriques pour {model_name} a été sauvegardé.")

# Sauvegarder les résultats des modèles
with open('../results/model_metrics.txt', 'w') as f:
    for result in results:
        f.write(f"Modèle : {result['Model']}\n")
        f.write(f"Métriques :\n{result['Metrics']}\n\n")
print("Les résultats des évaluations des modèles ont été sauvegardés dans 'model_metrics.txt'.")

# Étape 7 : Créer un graphique de comparaison
metrics_df = pd.DataFrame(metrics_data)
comparison_fig = px.bar(
    metrics_df,
    x="Model",
    y="Accuracy",
    title="Comparaison des Précisions des Modèles",
    labels={"Accuracy": "Précision", "Model": "Nom du Modèle"},
    template="plotly_white",
)
comparison_fig.write_html('../results/comparison_chart.html')
print("Le graphique de comparaison a été sauvegardé.")
