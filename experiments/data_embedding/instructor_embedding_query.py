from InstructorEmbedding import INSTRUCTOR
import pandas as pd
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
import matplotlib.pyplot as plt
import random

curriculum = [
    ("Leetcode-style Algorithmic Problems", "Includes sorting, searching, dynamic programming, graph algorithms, trees, hashing, recursion, and greedy algorithms."),
    ("Mathematical and Computational Problems", "Covers number theory, combinatorics, probability and statistics, computational geometry, and game theory."),
    ("Database and SQL Problems", "Involves query optimization, schema design, transaction management, concurrency control, indexing, and search."),
    ("System Design and Architecture Problems", "Encompasses scalability, load balancing, system integration, microservices architecture, and distributed systems."),
    ("Security and Cryptography Problems", "Deals with encryption, decryption, authentication, authorization, network security, and secure coding practices."),
    ("Performance Optimization Problems", "Focuses on memory management, code optimization, and parallel/concurrent programming."),
    ("User Interface and Experience Problems", "Includes responsive design, cross-browser compatibility, and accessibility."),
    ("Software Engineering Practices", "Covers debugging, troubleshooting, code readability, maintainability, version control, and testing and quality assurance."),
    ("Domain-Specific Problems", "Specific to areas such as machine learning and AI, bioinformatics, financial modeling, gaming, and graphics.")
]

model_name = "hkunlp/instructor-large"
model = INSTRUCTOR(model_name)

def get_text_embeddings(text_snippets, model):
    embeddings = []
    for text in text_snippets:
        instruction = "Represent the topic of the programming problem."
        embedding = model.encode([[instruction, text]])
        embeddings.append(embedding[0]) 
    return np.array(embeddings)


with open('/home/zhe/zhe/data_embedding/data/data.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

df = pd.DataFrame(data)
df['text'] = df['solution']  
embeddings = get_text_embeddings(df['text'].tolist(), model)


queries = [f"{c[0]}: {c[1]}" for c in curriculum]
query_embeddings = get_text_embeddings(queries, model)

def get_label(embedding, query_embeddings, queries):
    similarities = cos_sim(embedding.reshape(1, -1), query_embeddings)
    return queries[np.argmax(similarities[0])]

labels = np.array([get_label(embedding, query_embeddings, queries) for embedding in embeddings])

df['topic'] = labels

df.to_json('/home/zhe/zhe/data_embedding/data/data_topic.json', orient='records', lines=True)


categories = list(set(labels))
values = [np.sum(labels == category) for category in categories]

colors = [(random.random(), random.random(), random.random()) for _ in categories]


plt.figure(figsize=(18, 18))
bars = plt.bar([category.split(":")[0] for category in categories], values, color=colors)
plt.xlabel("Categories")
plt.ylabel("Number of Problems")
plt.title("Problem Distribution Across Different Categories")
plt.xticks([]) 
plt.legend(bars, [category.split(":")[0] for category in categories])
plt.savefig("/home/zhe/zhe/data_embedding/result/clusters.png")
plt.show()
