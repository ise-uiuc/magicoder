import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import numpy as np
from InstructorEmbedding import INSTRUCTOR

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
mode = "solution"
rule = "topic"
model_name = "hkunlp/instructor-large"
model = INSTRUCTOR(model_name)

def get_text_embeddings(text_snippets):
    embeddings = []
    for text in text_snippets:
        if rule == "topic":
            instruction = "Represent the code files for classifying different topics as Leetcode-style Algorithmic Problems, Mathematical and Computational Problems, Database and SQL Problems, System Design and Architecture Problems, Security and Cryptography Problems, Performance Optimization Problems, Performance Optimization Problems, Software Engineering Practices and Domain-Specific Problems."
        else:
            instruction = "Represent the code files for classifying difficulty levels as simple, medium and complex."
        embedding = model.encode([[instruction, text]])
        embeddings.append(embedding)
    return embeddings

with open('/home/zhe/zhe/data_embedding/data/data.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

df = pd.DataFrame(data)
df['text'] = df['problem'] if mode == "problem" else df['solution']
df['embedding'] = get_text_embeddings(df['text'].tolist())
matrix = np.vstack(df['embedding'])

tsne = TSNE(n_components=2, perplexity=100, random_state=42, init='random', learning_rate=200)
vis_dims = tsne.fit_transform(matrix)

kmeans = KMeans(n_clusters=9, random_state=42)
df['cluster'] = kmeans.fit_predict(matrix)
cluster_indices = []
for cluster_id in range(9):
    cluster_indices.append(df[df['cluster'] == cluster_id]['index'].tolist())

for cluster_id, indices in enumerate(cluster_indices):
    print(f"Cluster {cluster_id + 1} indices: {indices}")

# cluster_to_difficulty = {
#     0: "simple",
#     1: "medium",
#     2: "complex"
# }

# df['difficulty_level'] = df['cluster'].map(cluster_to_difficulty)


plt.figure(figsize=(8, 6))
# colors = ['r', 'g', 'b']  
colors = [
    'r',  # 红色
    'g',  # 绿色
    'b',  # 蓝色
    'c',  # 青色
    'm',  # 品红
    'y',  # 黄色
    'orange',  # 橙色
    'purple',  # 紫色
    'brown'  # 棕色
]

for cluster_id in range(9):
    cluster_data = vis_dims[df['cluster'] == cluster_id]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], c=colors[cluster_id], label=f'Cluster {cluster_id + 1}')

plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend()
plt.title('TSNE Visualization of Clusters')
plt.show()

# selected_columns = ['index', 'text', 'difficulty_level']

# df_selected = df[selected_columns]

# output_file_path = '/home/zhe/zhe/data_embedding/data/data_2.json'
# df_selected.to_json(output_file_path, orient='records', lines=True, force_ascii=False)



x = [x for x, y in vis_dims]
y = [y for x, y in vis_dims]

# category_colors = {
#     'classification': 'blue',
#     'brainstorming': 'orange',
#     'closed_qa': 'red',
#     'open_qa': 'purple',
#     'summarization': 'green',
#     'general_qa': 'brown'
# }

highlighted_x = []
highlighted_y = []
non_highlighted_x = []
non_highlighted_y = []

highlight_range = range(5000, 5011)

# for i, (x_coord, y_coord) in enumerate(vis_dims):
#     index = df['index'].iloc[i]  
#     if index in highlight_range:
#         highlighted_x.append(x_coord)
#         highlighted_y.append(y_coord)
#         plt.annotate(str(index), (x_coord, y_coord), textcoords="offset points", xytext=(0, 10), ha='center')
#     else:
#         non_highlighted_x.append(x_coord)
#         non_highlighted_y.append(y_coord)

# plt.scatter(non_highlighted_x, non_highlighted_y, color='blue', alpha=0.5)
# plt.scatter(highlighted_x, highlighted_y, color='red', alpha=0.7)

plt.title("t-SNE visualization of instructions")
plt.savefig(f'/home/zhe/zhe/data_embedding/result/tsne_visualization_{mode}_instructor_{rule}.png')
plt.show()
