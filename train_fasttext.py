import json
from gensim.models import FastText

with open('large_tokenized_2.txt.txt', 'r') as f:
    tokenized_texts = json.loads(f.read())

# print(tokenized_texts[0:5])

model = FastText(vector_size=100, window=5, min_count=10, workers=8)  # Skip-gram model

# Train the model on the tokenized data
model.build_vocab(corpus_iterable=tokenized_texts)
model.train(corpus_iterable=tokenized_texts, total_examples=len(tokenized_texts), epochs=10)
print()
print("Final training loss:", model.get_latest_training_loss())
print()
print(f"Vocabulary size: {len(model.wv)}")

print(f"Corpus size (number of sentences/documents): {model.corpus_count}")

similarity = model.wv.similarity("anxiety", "stress")
print(f"Similarity between 'anxiety' and 'stress': {similarity}\n")

similar_words = model.wv.most_similar("depression", topn=20)
print(f"Most similar words to 'depression': {similar_words}\n")

similar_words = model.wv.most_similar("stress", topn=20)
print(f"Most similar words to 'stress': {similar_words}\n")

similar_words = model.wv.most_similar("mental", topn=20)
print(f"Most similar words to 'mental': {similar_words}\n")

similar_words = model.wv.most_similar("human", topn=20)
print(f"Most similar words to 'human': {similar_words}\n")

model.save("models/FT models/fasttext_large5.model")
