import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from scipy.stats import f_oneway
from imblearn.over_sampling import SMOTE
import os

# Ensure directories exist
os.makedirs('../results', exist_ok=True)

# Step 1: Load TF-IDF data with G3
try:
    data = pd.read_csv('../data/tfidf_output.csv')  # Adjust the path as needed
    print("Data loaded successfully!")
except FileNotFoundError:
    print("Error: File not found. Ensure 'tfidf_output.csv' exists in the 'data' folder.")
    exit()

# Clean and validate G3 data
data = data[pd.to_numeric(data['G3'], errors='coerce').notnull()]  # Remove invalid G3 values
data['G3'] = data['G3'].astype(float)  # Ensure G3 is numerical

# Simplify G3 into categories (Low, Medium, High)
bins = [0, 10, 15, 20]
labels = ['Low', 'Medium', 'High']
data['G3_category'] = pd.cut(data['G3'], bins=bins, labels=labels)

# Drop rows with missing or invalid categories
data = data[data['G3_category'].notna()]  # Ensure no missing categories
data['G3_category'] = data['G3_category'].astype(str)  # Ensure G3_category is string type

# Step 2: Separate features and target
X = data.drop(columns=['G3', 'G3_category'])
y = data['G3_category']

# Step 3: Fisher's Score for Feature Selection
def fisher_score(X, y):
    scores = []
    for column in X.columns:
        classes = [X[y == cls][column] for cls in y.unique()]
        if any(len(cls) < 2 for cls in classes):  # Check for small sample sizes
            scores.append(0)  # Assign 0 if sample size is too small
        else:
            f_score, _ = f_oneway(*classes)
            scores.append(f_score)
    return np.array(scores)

print("Calculating Fisher's Scores...")
f_scores = fisher_score(X, y)

# Select top k features
k = 10  # Adjust the number of top features
top_k_features = X.columns[np.argsort(f_scores)[-k:]]
X_selected = X[top_k_features]

# Save Fisher's Scores
fisher_scores_df = pd.DataFrame({"Feature": X.columns, "Fisher_Score": f_scores})
fisher_scores_df.to_csv('../results/feature_scores.csv', index=False)
print(f"Top {k} features selected: {list(top_k_features)}")

# Visualize Fisher's Scores
fisher_scores_df = fisher_scores_df.sort_values(by="Fisher_Score", ascending=False)
fisher_fig = px.bar(
    fisher_scores_df,
    x="Feature",
    y="Fisher_Score",
    title="Fisher's Scores for Features",
    labels={"Fisher_Score": "Fisher's Score", "Feature": "Feature"},
    template="plotly_white",
)
fisher_fig.write_html('../results/fisher_scores_chart.html')
print("Fisher's Scores chart saved.")

# Step 4: Balance Classes Using SMOTE
print("Balancing classes with SMOTE...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_selected, y)
print("Classes balanced successfully!")

# Step 5: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Step 6: Train and Evaluate Models
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(kernel='linear', random_state=42),
    "Multinomial NB": MultinomialNB()
}

results = []
metrics_data = []  # For comparison chart

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
    
    # Collect accuracy for comparison
    metrics_data.append({"Model": model_name, "Accuracy": accuracy})
    
    # Visualize Precision, Recall, and F1-Score for each model
    class_metrics = []
    for label, values in metrics.items():
        if label in ['Low', 'Medium', 'High']:  # Focus on valid classes
            class_metrics.append({
                "Class": label,
                "Precision": values["precision"],
                "Recall": values["recall"],
                "F1-Score": values["f1-score"]
            })
    class_metrics_df = pd.DataFrame(class_metrics)

    # Create and save the chart
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

# Save Model Results
with open('../results/model_metrics.txt', 'w') as f:
    for result in results:
        f.write(f"Model: {result['Model']}\n")
        f.write(f"Metrics:\n{result['Metrics']}\n\n")
print("Model evaluation results saved in 'model_metrics.txt'.")

# Step 7: Create Comparison Chart
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
