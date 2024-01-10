import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_documents(directory):
    documents = []
    file_names = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                documents.append(file.read())
                file_names.append(filename)
    return documents, file_names

def query_relevance(query, documents):
    # Combine documents and query for TF-IDF vectorization
    documents.append(query)

    # Calculate TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Calculate cosine similarity between query and all documents
    cosine_similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])

    # Flatten to 1D array for easier handling
    cosine_similarities = cosine_similarities.flatten()

    return cosine_similarities

def main():
    directory = 'C:/users/petro/tuc_db'  # Change this to your documents directory
    query = "Apostolos Dollas"  # Change this to your query

    documents, file_names = load_documents(directory)
    similarities = query_relevance(query, documents)

    # Sort the documents by their similarity scores
    sorted_docs = sorted(zip(file_names, similarities), key=lambda x: x[1], reverse=True)[:10]

    for doc, score in sorted_docs:
        print(f"Document: {doc}, Relevance Score: {score}")

if __name__ == "__main__":
    main()
