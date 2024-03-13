import nltk
from nltk.corpus import stopwords
from pdfminer.high_level import extract_text
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download stopwords (one-time download)
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

# Read the job description if multiple files 
with open("/home/muhammadhassan/hr/jd.csv", "r") as file:
  job_description = file.read()

# Extract keywords
keywords = extract_keywords(job_description)

jd_ls = [keyword for keyword in keywords]
print("Extracted Keywords:", jd_ls)
print()
print()
print()

file_path = '/home/muhammadhassan/hr/cvs/resume_juanjosecarin.pdf'
cv_extracted_text = extract_text(file_path)

cv_key_words = extract_keywords(cv_extracted_text)
print()
print()
cv_ls = [keyword for keyword in cv_key_words]
print("cv extracted text:", cv_ls)
print()





unique_keywords = set(cv_ls + jd_ls)
# Create vectors representing the presence of keywords in each list
cv_vector = [1 if keyword in cv_ls else 0 for keyword in unique_keywords]
jd_vector = [1 if keyword in jd_ls else 0 for keyword in unique_keywords]

# Convert vectors to numpy arrays
cv_vector = np.array(cv_vector).reshape(1, -1)
jd_vector = np.array(jd_vector).reshape(1, -1)

# Compute cosine similarity
cos_sim = cosine_similarity(cv_vector, jd_vector)

print("Cosine Similarity:", cos_sim[0][0])


