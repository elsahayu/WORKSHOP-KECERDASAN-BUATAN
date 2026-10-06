import nltk

# Daftar lengkap dependensi untuk NLTK versi terbaru
nltk.download('maxent_ne_chunker')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng') # Tambahan untuk mengatasi error pos_tag

sentence = """
Artificial intelligence (AI), is intelligence demonstrated by 
machines,
unlike the natural intelligence displayed by humans and animals.
Leading AI textbooks define the field as the study of "intelligent 
agents":
any device that perceives its environment and takes actions that 
maximize its
chance of successfully achieving its goals.[3] Colloquially, the 
term
"artificial intelligence" is often used to describe machines (or 
computers)
that mimic "cognitive" functions that humans associate with the 
human mind,
such as "learning" and "problem solving".[4]
"""

# 1. Tokenisasi
tokens = nltk.word_tokenize(sentence)
print("--- TOKENS ---")
print(tokens)

# 2. Part-of-Speech (POS) Tagging
tagged = nltk.pos_tag(tokens)
print("\n--- POS TAGS ---")
print(tagged)

# 3. Named Entity Recognition (NER) / Chunking
entities = nltk.chunk.ne_chunk(tagged)
print("\n--- ENTITIES ---")
print(entities)