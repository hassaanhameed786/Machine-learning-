

import os
import nltk
from nltk.corpus import stopwords
from pdfminer.high_level import extract_text
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict

# Download nltk resources (one-time download)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

def extract_keywords(text):
    # Preprocess text
    text = text.lower()  # Convert to lowercase
    text = nltk.word_tokenize(text)  # Tokenize words
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = [word for word in text if word not in stop_words]
    
    # Identify potential keywords: nouns, verbs, and adjectives
    keywords = [word for (word, pos) in nltk.pos_tag(text) 
                if pos in ['NN', 'NNS', 'VB', 'VBD', 'VBG', 'JJ']]
    return keywords

def compute_cosine_similarity(jd_keywords, cv_keywords):
    unique_keywords = set(cv_keywords + jd_keywords)
    cv_vector = np.array([1 if keyword in cv_keywords else 0 for keyword in unique_keywords]).reshape(1, -1)
    jd_vector = np.array([1 if keyword in jd_keywords else 0 for keyword in unique_keywords]).reshape(1, -1)
    cos_sim = cosine_similarity(cv_vector, jd_vector)
    return cos_sim[0][0]

def process_cv(cv_file_name, jd_keywords):
    cv_file_path = os.path.join("/home/muhammadhassan/hr/cvs", cv_file_name)
    cv_extracted_text = extract_text(cv_file_path)
    cv_keywords = extract_keywords(cv_extracted_text)
    similarity = compute_cosine_similarity(jd_keywords, cv_keywords)
    return similarity

def main():
    # Read all job descriptions
    jd_folder = "/home/muhammadhassan/hr/jds"
    jd_files = os.listdir(jd_folder)
    
    for jd_file_name in jd_files:
        with open(os.path.join(jd_folder, jd_file_name), "r") as file:
            job_description = file.read()
        
        # Extract keywords from job description
        jd_keywords = extract_keywords(job_description)
        print(f"Processing JD: {jd_file_name}")
        # print("Extracted Keywords from JD:", jd_keywords)
        
        # Process CVs
        cv_files = os.listdir("/home/muhammadhassan/hr/cvs")
        cv_similarities = defaultdict(list)
        for cv_file in cv_files:
            similarity = process_cv(cv_file, jd_keywords)
            cv_similarities[jd_file_name].append((cv_file, similarity))
        
        # Rank CVs based on similarity
        top_3_cv_for_jd = sorted(cv_similarities[jd_file_name], key=lambda x: x[1], reverse=True)[:3]
        print("Top 3 CVs for JD:", jd_file_name)
        print()
        print()
        for i, (cv_file, similarity) in enumerate(top_3_cv_for_jd, start=1):
           
            print(f"{i}. {cv_file}: {similarity}")
        print()

if __name__ == "__main__":
    main()

