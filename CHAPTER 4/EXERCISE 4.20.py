import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

from transformers import pipeline

question_answerer = pipeline('question-answering', framework='pt')

context = (
    "Politeknik Elektronika Negeri Surabaya (PENS) is a leading technological "
    "institution located in Surabaya, Indonesia. Founded in 1988, PENS focuses "
    "on engineering and multimedia applied sciences."
)

questions = [
    "Where is PENS located?",
    "When was PENS founded?",
    "What does PENS focus on?",
    "Who is the director of PENS?",
]

for q in questions:
    result = question_answerer({'question': q, 'context': context})
    print(f"Q: {q}")
    print(f"A: {result['answer']}  (score={result['score']:.4f}, "
          f"start={result['start']}, end={result['end']})\n")