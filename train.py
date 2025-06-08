import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

nltk.download('punkt')
nltk.download('stopwords')

df = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'])


df['label'] = df['label'].map({'ham': 0, 'spam': 1})

stop_words = set(stopwords.words('english'))

def transform_text(text):
    text = text.lower()
    words = nltk.word_tokenize(text)
    words = [word for word in words if word.isalnum()]  
    words = [word for word in words if word not in stop_words]
    return " ".join(words)


df['transformed_text'] = df['message'].apply(transform_text)


vectorizer = CountVectorizer(max_features=3000)
X = vectorizer.fit_transform(df['transformed_text'])
y = df['label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = MultinomialNB()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("✅ Training done with real dataset. Model and vectorizer saved.")
