import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the dataset
file_path = 'C:/Users/Hamza/Desktop/CC/datamining/student_data.csv'  # Replace with your actual file path
data = pd.read_csv(file_path)

# Select relevant text columns for school performance
text_columns = ['Mjob', 'Fjob', 'reason', 'guardian', 'activities']

# Combine text columns into a single column for TF-IDF
data['combined_text'] = data[text_columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

# Apply TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(data['combined_text'])

# Convert the TF-IDF matrix to a DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Combine G3 with the TF-IDF DataFrame
tfidf_df['G3'] = data['G3']

# Save or inspect the TF-IDF DataFrame
tfidf_df.to_csv('tfidf_output_with_G3.csv', index=False)
print("TF-IDF matrix with G3 column saved as 'tfidf_output_with_G3.csv'.")
