import os
import json
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset

# plt.rcParams['text.usetex'] = True

def load_scores(file_path):
    with open(file_path, 'r') as file:
        scores = json.load(file)
    return scores

def plot_scores(oss_scores, evol_scores, codealpaca_scores):
    num_samples_oss = len(oss_scores)
    num_samples_evol = len(evol_scores)
    num_samples_codealpaca = len(codealpaca_scores)

    score_counts_oss = Counter([round(score, 2) for score in oss_scores])
    score_counts_evol = Counter([round(score, 2) for score in evol_scores])
    score_counts_codealpaca = Counter([round(score, 2) for score in codealpaca_scores])

    avg_score_oss = np.mean(oss_scores)
    avg_score_evol = np.mean(evol_scores)
    avg_score_codealpaca = np.mean(codealpaca_scores)

    percentage_counts_oss = {score: count / num_samples_oss for score, count in score_counts_oss.items()}
    percentage_counts_evol = {score: count / num_samples_evol for score, count in score_counts_evol.items()}
    percentage_counts_codealpaca = {score: count / num_samples_codealpaca for score, count in score_counts_codealpaca.items()}

    plt.figure(figsize=(8, 6))

    plt.bar(percentage_counts_evol.keys(), percentage_counts_evol.values(), color='Royalblue', width=0.01, alpha=0.6, label=r'\textsc{Evol-Instruct}')
    plt.axvline(x=avg_score_evol, color='royalblue', linestyle='-.', label=f'Avg Evol-Instruct: {avg_score_evol:.3f}') 
    plt.bar(percentage_counts_codealpaca.keys(), percentage_counts_codealpaca.values(), color='Tomato', width=0.01, alpha=0.6, label=r'\textsc{CodeAlpaca}')
    plt.bar(percentage_counts_oss.keys(), percentage_counts_oss.values(), color='Gold', width=0.01, alpha=0.6, label=r'\textsc{OSS-Instruct}')

    plt.xlabel("Cosine Similarity Score", fontsize=22)
    plt.ylabel("Percentage", fontsize=22)
    plt.xlim(0, 0.5)
    plt.xticks(np.arange(0, 0.55, 0.1))
    plt.tick_params(axis='both', labelsize=14)

    plt.axvline(x=avg_score_oss, color='darkgoldenrod', linestyle='--', label=f'Avg OSS-Instruct: {avg_score_oss:.3f}')
    plt.axvline(x=avg_score_codealpaca, color='orangered', linestyle='--', label=f'Avg CodeAlpaca: {avg_score_codealpaca:.3f}')

    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(f'/home/zhe/similarity_data/similarity.png')

def load_data_oss(file_paths):
    all_problem_solutions = []
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                problem = data.get('problem', '')
                solution = data.get('solution', '')
                combined_text = problem + " " + solution
                all_problem_solutions.append(combined_text)
    return all_problem_solutions

def load_data_evol(file_paths):
    all_problem_solutions = []
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                problem = data.get('instruction', '')
                solution = data.get('output', '')
                combined_text = problem + " " + solution
                all_problem_solutions.append(combined_text)
    return all_problem_solutions

def load_data_codealpaca(file_paths):
    all_problem_solutions = []
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                problem = data.get('instruction', '')
                solution = data.get('response', '')
                combined_text = problem + " " + solution
                all_problem_solutions.append(combined_text)
    return all_problem_solutions

def cosine_similarity_gpu(batch_tensor1, batch_tensor2):
    batch_tensor1_norm = F.normalize(batch_tensor1, p=2, dim=-1)
    batch_tensor2_norm = F.normalize(batch_tensor2, p=2, dim=-1)
    return torch.mm(batch_tensor1_norm, batch_tensor2_norm.transpose(0, 1))

def prepare_data_for_gpu(vectorizer, texts):
    vectors = vectorizer.transform(texts).toarray()
    return torch.tensor(vectors, device='cuda')

def calculate_cosine_scores_gpu(data1, data2_tensors, vectorizer, batch_size=100, save_path_1=None, save_path_2=None):
    max_cosine_scores = []
    most_similar_indices = []
    for i in range(0, len(data1), batch_size):
        batch_text = data1[i:i + batch_size]
        batch_vector = prepare_data_for_gpu(vectorizer, batch_text)
        max_scores_batch = []
        indices_batch = []
        for data2_tensor in data2_tensors:
            scores = cosine_similarity_gpu(batch_vector, data2_tensor)
            max_scores, indices = torch.max(scores, dim=1)
            max_scores_batch.append(max_scores)
            indices_batch.append(indices)

        max_scores_batch_tensor = torch.stack(max_scores_batch)
        max_scores, max_indices = torch.max(max_scores_batch_tensor, dim=0)
        max_cosine_scores.extend(max_scores.cpu().numpy())
        most_similar_indices.extend(max_indices.cpu().numpy())

    if save_path_1:
        most_similar_indices_native = [int(index) for index in most_similar_indices]
        with open(save_path_1, "w") as file:
            json.dump(most_similar_indices_native, file)

    if save_path_2:
        most_similar_scores_native = [float(score) for score in max_cosine_scores]
        with open(save_path_2, "w") as file:
            json.dump(most_similar_scores_native, file)

    return max_cosine_scores, most_similar_indices

def main():
    oss_score_file = '/home/zhe/similarity_data/oss_score.json'
    evol_score_file = '/home/zhe/similarity_data/evol_score.json'
    codealpaca_score_file = '/home/zhe/similarity_data/codealpaca_score.json'

    if os.path.exists(oss_score_file) and os.path.exists(evol_score_file) and os.path.exists(codealpaca_score_file):
        oss_scores = load_scores(oss_score_file)
        evol_scores = load_scores(evol_score_file)
        codealpaca_scores = load_scores(codealpaca_score_file)
    else:
        oss_data = load_data_oss(['/home/zhe/data-clean-decontaminated.jsonl'])
        evol_data = load_data_evol(['/home/zhe/data-evol_instruct-decontaminated.jsonl'])
        codealpaca_data = load_data_codealpaca(['/home/zhe/data-codealpaca-decontaminated.jsonl'])

        dataset_name = 'openai_humaneval'
        dataset = load_dataset(dataset_name, split='test')
        data2 = [item['prompt'] + " " + item['canonical_solution'] for item in dataset]

        all_texts = oss_data + evol_data + codealpaca_data + data2
        vectorizer = TfidfVectorizer()
        vectorizer.fit(all_texts)
        data2_tensors = [prepare_data_for_gpu(vectorizer, [text]) for text in data2]

        num_samples_oss = len(oss_data)
        num_samples_evol = len(evol_data)
        num_samples_codealpaca = len(codealpaca_data)

        oss_scores, oss_indices = calculate_cosine_scores_gpu(oss_data[:num_samples_oss], data2_tensors, vectorizer, batch_size=100, save_path_1='/home/zhe/similarity_data/oss_index.json', save_path_2='/home/zhe/similarity_data/oss_score.json')
        evol_scores, evol_indices = calculate_cosine_scores_gpu(evol_data[:num_samples_evol], data2_tensors, vectorizer, batch_size=100, save_path_1='/home/zhe/similarity_data/evol_index.json', save_path_2='/home/zhe/similarity_data/evol_score.json')
        codealpaca_scores, codealpaca_indices = calculate_cosine_scores_gpu(codealpaca_data[:num_samples_codealpaca], data2_tensors, vectorizer, batch_size=100, save_path_1='/home/zhe/similarity_data/codealpaca_index.json', save_path_2='/home/zhe/similarity_data/codealpaca_score.json')
    
    plot_scores(oss_scores, evol_scores, codealpaca_scores)

if __name__ == "__main__":
    main()
