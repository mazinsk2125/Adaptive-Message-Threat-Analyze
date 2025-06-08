import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import pickle

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def transform_text(text):
    text = text.lower()
    words = nltk.word_tokenize(text)
    words = [word for word in words if word.isalnum()]
    words = [word for word in words if word not in stop_words]
    return " ".join(words)


with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)


message = "Congratulations! You have won a free pizza party."


transformed_msg = transform_text(message)
print("Transformed Text:", transformed_msg)

vector = vectorizer.transform([transformed_msg])


prediction = model.predict(vector)[0]

print("Message:", message)
print("Prediction:", "Spam" if prediction == 1 else "Not Spam")
