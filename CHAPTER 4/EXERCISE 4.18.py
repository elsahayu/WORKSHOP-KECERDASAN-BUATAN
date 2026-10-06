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
Politeknik Elektronika Negeri Surabaya (PENS) is a leading technological 
institution located in Surabaya, Indonesia. Founded in 1988, PENS focuses 
on engineering and multimedia applied sciences.
"""

tokens = nltk.word_tokenize(sentence)
print("--- TOKENS ---")
print(tokens)

tagged = nltk.pos_tag(tokens)
print("\n--- POS TAGS ---")
print(tagged)

entities = nltk.chunk.ne_chunk(tagged)
print("\n--- ENTITIES ---")
print(entities)