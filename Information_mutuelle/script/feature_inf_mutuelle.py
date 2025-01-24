import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_selection import mutual_info_classif
from imblearn.over_sampling import SMOTE
import os

os.makedirs('../results', exist_ok=True)


try:
    data = pd.read_csv('../data/tfidf_output_with_G3.csv')  
    print("Data loaded successfully!")
except FileNotFoundError:
    print("Error: File not found. Ensure 'tfidf_output_with_G3.csv' exists in the 'data' folder.")
    exit()

#Valider les données du dataset 
data = data[pd.to_numeric(data['G3'], errors='coerce').notnull()]  # supprimer les valeurs invalides de G3
data['G3'] = data['G3'].astype(float)  # vérifier que les données sont numériques

# simplification de la colonne G3 en des catégories LOW, Medium et HIGH
bins = [0, 10, 15, 20]
labels = ['Low', 'Medium', 'High']
data['G3_category'] = pd.cut(data['G3'], bins=bins, labels=labels)

#Supprimer les lignes avec des catégories manquantes ou invalides. 
data = data[data['G3_category'].notna()] 
data['G3_category'] = data['G3_category'].astype(str) 

# Séparation des caractéristiques et la cible.
X = data.drop(columns=['G3', 'G3_category'])
y = data['G3_category']

# Application de l'information mutuelle
print("Calculating Mutual Information for feature selection...")
mi_scores = mutual_info_classif(X, y, discrete_features=True, random_state=42)

# selectionner un K nombres des features
k = 10 
top_k_features = X.columns[np.argsort(mi_scores)[-k:]]
X_selected = X[top_k_features]

# Enregistrer les scores d'information mutuelle.
mi_scores_df = pd.DataFrame({"Feature": X.columns, "MI_Score": mi_scores})
mi_scores_df.to_csv('../results/feature_scores.csv', index=False)
print(f"Top {k} features selected: {list(top_k_features)}")

# Visualisation des scores d'information mutuelle
mi_scores_df = mi_scores_df.sort_values(by="MI_Score", ascending=False)
mi_fig = px.bar(
    mi_scores_df,
    x="Feature",
    y="MI_Score",
    title="Mutual Information Scores for Features",
    labels={"MI_Score": "MI Score", "Feature": "Feature"},
    template="plotly_white",
)
mi_fig.write_html('../results/mi_scores_chart.html')
print("Mutual Information Scores chart saved.")

#  Équilibrer les classes en utilisant SMOTE
print("Balancing classes with SMOTE...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_selected, y)
print("Classes balanced successfully!")

# division des données en des données d'entraînement et de teste
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Entraînement et évaluation des modèles
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(kernel='linear', random_state=42),
    "Multinomial NB": MultinomialNB()
}

results = []
metrics_data = []  # pour le diagramme de comparaison

for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Metrics
    metrics = classification_report(y_test, y_pred, output_dict=True)
    accuracy = accuracy_score(y_test, y_pred)
    results.append({"Model": model_name, "Metrics": metrics})
    
    print(f"Results for {model_name}:\n")
    print(classification_report(y_test, y_pred))
    
    # Collecter la précision pour la comparaison
    metrics_data.append({"Model": model_name, "Accuracy": accuracy})
    
    # Visualiser la précision, le rappel et le score F1 pour chaque modèle
    class_metrics = []
    for label, values in metrics.items():
        if label in ['Low', 'Medium', 'High']:  # se focalizer sur les classes valides
            class_metrics.append({
                "Class": label,
                "Precision": values["precision"],
                "Recall": values["recall"],
                "F1-Score": values["f1-score"]
            })
    class_metrics_df = pd.DataFrame(class_metrics)

    # Créer et enregistrer le graphique
    fig = px.bar(
        class_metrics_df,
        x="Class",
        y=["Precision", "Recall", "F1-Score"],
        barmode="group",
        title=f"{model_name}: Metrics by Class",
        template="plotly_white",
    )
    fig.write_html(f"../results/{model_name.lower().replace(' ', '_')}_chart.html")
    print(f"{model_name} chart saved.")

# Enregistrer les résultats
with open('../results/model_metrics.txt', 'w') as f:
    for result in results:
        f.write(f"Model: {result['Model']}\n")
        f.write(f"Metrics:\n{result['Metrics']}\n\n")
print("Model evaluation results saved in 'model_metrics.txt'.")

# création du graphique de comparaison
metrics_df = pd.DataFrame(metrics_data)
comparison_fig = px.bar(
    metrics_df,
    x="Model",
    y="Accuracy",
    title="Comparison of Model Accuracies",
    labels={"Accuracy": "Accuracy", "Model": "Model Name"},
    template="plotly_white",
)
comparison_fig.write_html('../results/comparison_chart.html')
print("Comparison chart saved.")

print("Feature selection and model evaluation complete with visualizations!")
