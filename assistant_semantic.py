from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')

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

question_embeddings = model.encode(questions)

print("📘 Bank FAQ Assistant (semantic version)")
print("Type 'exit' to quit.")

while True:
    user_q = input("\nYou: ")
    if user_q.lower() == "exit":
        break

    user_embedding = model.encode([user_q])
    similarities = cosine_similarity(user_embedding, question_embeddings)
    best_match = similarities[0].argmax()

    print("🤖 Answer:", answers[best_match])
