import os
import json
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

def query_relevance(query, documents, file_names):
    # Combine documents and query for TF-IDF vectorization
    all_docs = documents + [query]

    # Calculate TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_docs)

    # Calculate cosine similarity between query and all documents
    cosine_similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1]).flatten()

    # Sort the documents by their similarity scores and get top 10
    sorted_docs = sorted(zip(file_names, cosine_similarities), key=lambda x: x[1], reverse=True)[:16]

    return [doc for doc, _ in sorted_docs]

def save_results(results, filename):
    with open(filename, 'w') as file:
        json.dump(results, file)

def load_results(filename):
    with open(filename, 'r') as file:
        return json.load(file)

def main():
    directory = 'C:/users/petro/tuc_db'  # Change this to your documents directory
    queries = ["Applied Research and Innovation","Cloud Computing","Laboratory Staff","Air Polution","Data Analysis & Forecasting"]

    documents, file_names = load_documents(directory)
    results = {}

    for query in queries:
        results[query] = query_relevance(query, documents, file_names)

    save_results(results, 'query_relevance.json')

if __name__ == "__main__":
    main()
