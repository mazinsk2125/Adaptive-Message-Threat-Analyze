import pickle


with open('vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)


print("Number of features:", len(tfidf.get_feature_names_out()))
print("Sample features:", tfidf.get_feature_names_out()[:20])


sample_message = ["This is a test message"]
sample_vector = tfidf.transform(sample_message)

print("Vector shape:", sample_vector.shape)
print("Vector data:", sample_vector.toarray())
