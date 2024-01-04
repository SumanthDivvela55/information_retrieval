Topic : Wikipedia - Countries and territories search

1. open vscode or any other terminal

      navigate to : "cd P38-MiniProject-SUMAN" and then "cd codes":

Installation:

Ensure you have Python installed on your machine.

Install the necessary libraries using pip:
   
      'pip install nltk scikit-learn textblob tkinter matplotlib'
      
Download NLTK data by running the Python interpreter and executing the following:

        import nltk
        nltk.download('punkt')
        nltk.download('stopwords')


Download any additional pakages require to run code.

Run the Python script app.py to launch the document searching in terminal.

Usage:

    Run the code.py script.
    Enter your query(eg:india,pakistan) in the provided text field.
    Click enter to start the search process.
    Review the top 10 relevant documents displayed in the Terminal.
    Provide feedback on document relevance.
    Then the top 10 relevant documents displayed
    Then we get PR-curve It provide the PR-curve based on the Relevence feedback

-->app.py: The main Python script containing the search functionality.

----------------------------------------------------------------------------------------------------------

-->inverted_index.json: JSON file storing the inverted index required for document retrieval.

Link  : https://drive.google.com/file/d/1pZV_JtkMDCobpmcmrgKmLL3FX2BsIz2T/view?usp=sharing

-->Use above link and extract the inverted_index.json file from google drive and replace the path of the file with  inverted_index.json in app.py


-->documents : folder contain the documents.

Link : https://drive.google.com/file/d/1x7hFLMlDa2iUIzK-kk-RSFXjpXOrQgsi/view?usp=sharing

-->use above link download the zip folder and extract the folder and copy the folder path then replace the documents path present in app.py 

----------------------------------------------------------------------------------------------------------


Please use above two links the dataset and inverted index present in above links only.

Requirements:

Python 3.x
Libraries: NLTK, scikit-learn, TextBlob

After doing above all things :

                   just run "python app.py"

----------------------------------------------------------------------------------------------------------


Simple discription:

A simple information retrieval system using a terminal to retrieve documents. It allows users to enter a query, performs a search within a collection of documents, and uses relevance feedback to refine the search results.



Brief discription of functionalities :

load_inverted_index():

*This function takes a file path as an argument (file_path).It opens the specified file in read mode and uses the json.load function to load the content, assuming it contains a serialized JSON object.
The loaded content is assigned to the variable inverted_index.
The function then returns the inverted_index, which is essentially a data structure commonly used in information retrieval systems.


calculate_cosine_similarity():

This function computes cosine similarities between a query vector and a set of document vectors.
It takes two arguments: query_vector (representing the vector for a query) and document_vectors (representing a collection of document vectors).
The cosine_similarity function is applied to calculate the cosine similarities, and the result is stored in the variable similarities.
Finally, the result is flattened using the flatten method and returned. The flattened result typically represents the cosine similarities between the query vector and each document vector in a one-dimensional array.

preprocess_documents(text)

Description: Converts the input text to lowercase, tokenizes it, removes punctuation and stopwords from the text, and returns the processed text.

feedback_loop()

Description: Processes relevance feedback for a query entered by the user. It utilizes spell correction, retrieves relevant documents, calculates TF-IDF scores, displays top documents, and gathers feedback on document relevance.


calculate_precision_recall:

Description: Calculates precision and recall values based on the top 10 documents retrieved and stores these values in global variables for later analysis or visualization.


generate_PR_curve:

Description: Generates and displays the precision-recall curve for the top 10 documents retrieved based on stored precision and recall value
