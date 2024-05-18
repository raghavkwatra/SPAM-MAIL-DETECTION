import tkinter as tk
from tkinter import messagebox
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load dataset from Excel
df = pd.read_csv('mail_data.csv')

# Convert dataset to list of tuples
emails = [(row['text'], row['label']) for index, row in df.iterrows()]

# Preprocessing function
def preprocess(email):
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    words = word_tokenize(email.lower())
    filtered = [ps.stem(w) for w in words if w.isalnum() and w not in stop_words]
    return ' '.join(filtered)

# Preprocess the emails
preprocessed_emails = [(preprocess(email), label) for email, label in emails]

# Vectorize the emails
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform([email[0] for email in preprocessed_emails])
y_train = [email[1] for email in preprocessed_emails]

# Train Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_counts, y_train)

# Function to predict email type
def predict_email_type():
    input_email = text_entry.get("1.0", "end-1c")
    preprocessed_input = preprocess(input_email)
    input_counts = vectorizer.transform([preprocessed_input])
    predicted_label = clf.predict(input_counts)
    messagebox.showinfo("Prediction", f"The email is predicted to be: {predicted_label[0]}")

# Create GUI window
window = tk.Tk()
window.title("Email Spam Detection")

# Create text entry
text_entry = tk.Text(window, height=10, width=50)
text_entry.pack()

# Create button to predict email type
predict_button = tk.Button(window, text="Predict", command=predict_email_type)
predict_button.pack()

# Run GUI
window.mainloop()