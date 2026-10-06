# Exercise 4.19: sentiment analysis dengan kalimat positif dan negatif
from transformers import pipeline

classifier = pipeline('sentiment-analysis')

sentences = [
    # Positif
    "I love this smartphone, the camera is amazing.",
    "The teacher explained the lesson clearly and I enjoyed the class.",
    "This restaurant serves delicious food and the staff are very friendly.",
    # Negatif
    "This is the worst movie I have ever seen.",
    "The internet connection is terrible and keeps disconnecting.",
    "I am very disappointed with the slow service.",
    # Sulit / ambigu
    "The movie was not bad at all.",
    "The food was okay, but the service was awful.",
]

for s in sentences:
    result = classifier(s)[0]
    print(f"{result['label']:8s} {result['score']:.4f}  |  {s}")