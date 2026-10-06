# Example 4.23
from transformers import pipeline

classifier = pipeline('sentiment-analysis')
print(classifier('This is a good movie.'))