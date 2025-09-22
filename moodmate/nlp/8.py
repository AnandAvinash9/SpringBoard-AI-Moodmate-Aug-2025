# file: 8_topic_modeling.py
from gensim import corpora, models

# Example dataset
docs = [
    "I love deep learning and natural language processing",
    "Artificial intelligence is the future",
    "Cooking and baking are my hobbies",
    "I enjoy trying new recipes in the kitchen",
    "Machine learning and AI are closely related"
]

# Tokenize
texts = [doc.lower().split() for doc in docs]

# Create dictionary & corpus
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# Train LDA model (2 topics)
lda_model = models.LdaModel(corpus, num_topics=2, id2word=dictionary, passes=10)

# Show topics
for idx, topic in lda_model.print_topics(-1):
    print(f"Topic {idx}: {topic}")