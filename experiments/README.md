### 📚Description
Here are some descriptions for the `experiments/data_embedding` directory:
- `length.py`: provides the token length distribution for data file problems and solutions.
- `cosine_similarity.py`: computes the cosine similarity between the TF-IDF embeddings of data file and HumanEval.
- `instruction_embedding.py`:  classifies and calculates the percentage composition of data within the data file based on the instruction you provide.

### 🔍Data Analysis
1. To depict the length distribution for either problems or solutions of the data file, you can run the command:
    ```bash
    python experiments/data_embedding/length.py 
    ```
    The result will be shown in `Length.png`

2. To see the similarity between the data file and HumanEval, you can run the command:
    ```bash
    python experiments/data_embedding/cosine_similarity.py
    ```
    The result will be shown in `HE_similarity_comparison.png`

3. To study the categories of the data file, there are two different modes:
    - In the **instruction** mode, the model will generate the corresponding embeddings according to the instructions and number of clusters you give, and then generate clusters based on these embeddings.
      
      You can change the clustering criteria by adjusting the `--instruction`.
      
      For example, if you want to cluster the data file according to the programming languages, you can run the command:
      
      ```bash
      python experiments/data_embedding/instructor_embedding.py \
      --data_files data-clean-decontaminated.jsonl \
      --model_key  instructor-base \
      --embedding_mode solution \
      --instruction "Represent the programming language used" \
      --n_clusters 2
      ```
      The clustering result will be shown in  `Clusters.png`.
      
    - In the **query** mode,  the model will generate the corresponding embeddings according to the instructions and queries you give,  then classifies them by calculating the cosine similarity between the embeddings of the data file and the embeddings of queries.
      
      You can change the clustering criteria by adjusting the `--query_instruction` and `--queries`.
      
      For example, if you want to classify the data file according to the topic of the content, you can run the command:
      
      ```bash
      python experiments/data_embedding/instructor_embedding.py \
      --data_files data-clean-decontaminated.jsonl \
      --model_key  instructor-base \
      --embedding_mode solution \
      --instruction "Represent the code for retrieving" \
      --query_instruction "Represent the comment for retrieving the corresponding code" \
      --queries "Algorithmic and Data Structure Problems" "Mathematical and Computational Problems" "Database and SQL Problems" "System Design and Architecture Problems" "Security and Cryptography Problems" "Performance Optimization Problems" "Web Problems" "Domain Specific Problems" "User Interface and Application Design Problems" "Data Science and Machine Learning Problems" 
      ```
       The classification result will be shown in  `Pie_Chart.png`.
    - You can find more information about how to generate data embeddings by using specific instructions and queries [here](https://arxiv.org/pdf/2212.09741.pdf) 
