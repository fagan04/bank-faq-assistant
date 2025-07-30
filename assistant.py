from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


with open("faqs.txt", "r", encoding="utf-8") as f:
    raw_data = f.read()

entries = raw_data.strip().split("\n\n")
questions = []
answers = []

for entry in entries:
    lines = entry.split("\n")
    q = lines[0].replace("Q: ", "").strip()
    a = lines[1].replace("A: ", "").strip()
    questions.append(q)
    answers.append(a)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

while True:
    user_q = input("\nAsk me a question (or type 'exit'): ")
    if user_q.lower() == "exit":
        break

    user_vec = vectorizer.transform([user_q])
    similarities = cosine_similarity(user_vec, X)
    best_match = similarities[0].argmax()
    print("Answer:", answers[best_match])
