import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_recall_curve, auc


def load_inverted_index(file_path):
    with open(file_path, "r") as f:
        inverted_index = json.load(f)
    return inverted_index


def calculate_cosine_similarity(query_vector, document_vectors):
    similarities = cosine_similarity(query_vector, document_vectors)
    return similarities.flatten()


def preprocess_documents(documents_folder):
    document_files = os.listdir(documents_folder)
    document_texts = []

    for filename in document_files:
        with open(
            os.path.join(documents_folder, filename), "r", encoding="utf-8"
        ) as file:
            document_texts.append(file.read())

    vectorizer = TfidfVectorizer(stop_words="english")
    document_vectors = vectorizer.fit_transform(document_texts)

    return document_files, vectorizer, document_vectors, document_texts


def feedback_loop(relevant_indices, irrelevant_indices, feedback_weights):
    feedback_weights[relevant_indices] += 0.1
    feedback_weights[irrelevant_indices] -= 0.1


def main():
    documents_folder = "documents"
    inverted_index_file = "inverted_index.json"

    inverted_index = load_inverted_index(inverted_index_file)

    document_files, vectorizer, document_vectors, document_texts = preprocess_documents(
        documents_folder
    )

    feedback_weights = np.ones(len(document_files))

    # Simulate that only 10 documents are relevant
    num_relevant_documents = 10
    relevant_indices = np.random.choice(
        len(document_files), num_relevant_documents, replace=False
    )

    # Initialize lists to store precision and recall values
    precision_list = []
    recall_list = []

    # 11-point interpolated PR curve
    interpolated_precision_list = []
    interpolated_recall_list = []

    while True:
        query = input("Enter your query (type 'exit' to quit): ")
        if query.lower() == "exit":
            break

        query_vector = vectorizer.transform([query])
        feedback_vectors = document_vectors.multiply(feedback_weights[:, np.newaxis])
        similarities = calculate_cosine_similarity(query_vector, feedback_vectors)

        top_indices = np.argsort(similarities)[::-1][:10]
        num_documents_to_display = len(top_indices)

        print("Relevant Documents:")
        for idx in top_indices:
            similarity_score = similarities[idx]
            if similarity_score > 0:
                print(
                    f"\nDocument: {document_files[idx]} - Similarity: {similarity_score}\n"
                )
                print(document_texts[idx])

                feedback = input("Is this document relevant? (yes/no): ").lower()
                if feedback == "yes":
                    feedback_loop([idx], [], feedback_weights)
                elif feedback == "no":
                    feedback_loop([], [idx], feedback_weights)

                # Calculate precision and recall for each document
                relevant_docs = set(relevant_indices)
                true_positives = relevant_docs.intersection(set(top_indices))
                false_positives = set(top_indices).difference(true_positives)
                false_negatives = relevant_docs.difference(true_positives)

                precision = (
                    len(true_positives) / (len(true_positives) + len(false_positives))
                    if len(true_positives) + len(false_positives) > 0
                    else 0
                )
                recall = (
                    len(true_positives) / (len(true_positives) + len(false_negatives))
                    if len(true_positives) + len(false_negatives) > 0
                    else 0
                )

                precision_list.append(precision)
                recall_list.append(recall)

        # Plot precision-recall curve after each query
        plt.plot(recall_list, precision_list, label="Precision-Recall curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend(loc="best")
        plt.show()

        # Calculate 11-point interpolated precision-recall curve
        interpolated_recall = np.linspace(0, 1, 11)
        interpolated_precision = np.interp(
            interpolated_recall, np.array(recall_list), np.array(precision_list)
        )

        # Plot 11-point interpolated precision-recall curve
        plt.plot(
            interpolated_recall,
            interpolated_precision,
            marker="o",
            linestyle="--",
            color="r",
            label="11-point Interpolated PR curve",
        )
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("11-point Interpolated Precision-Recall Curve")
        plt.legend(loc="best")
        plt.show()

        # Clear lists for the next query
        precision_list = []
        recall_list = []


if __name__ == "__main__":
    main()
