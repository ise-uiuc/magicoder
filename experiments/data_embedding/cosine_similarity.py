import os
import json
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import load_dataset

class CosineSimilarityAnalysis:
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def load_scores(self, file_path):
        with open(file_path, 'r') as file:
            scores = json.load(file)
        return scores

    def calculate_cosine_scores_gpu(self, data1, data2_tensors, vectorizer, batch_size=100, save_path_1=None, save_path_2=None):
        max_cosine_scores = []
        most_similar_indices = []
        for i in range(0, len(data1), batch_size):
            batch_text = data1[i:i + batch_size]
            batch_vector = self.prepare_data_for_gpu(vectorizer, batch_text)
            max_scores_batch = []
            indices_batch = []
            for data2_tensor in data2_tensors:
                scores = self.cosine_similarity_gpu(batch_vector, data2_tensor)
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

    def cosine_similarity_gpu(self, batch_tensor1, batch_tensor2):
        batch_tensor1_norm = F.normalize(batch_tensor1, p=2, dim=-1)
        batch_tensor2_norm = F.normalize(batch_tensor2, p=2, dim=-1)
        return torch.mm(batch_tensor1_norm, batch_tensor2_norm.transpose(0, 1))

    def prepare_data_for_gpu(self, vectorizer, texts):
        vectors = vectorizer.transform(texts).toarray()
        return torch.tensor(vectors, device='cuda')

    def plot_scores(self, oss_scores, evol_scores, codealpaca_scores):
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

        plt.figure(figsize=(4.8, 3))

        sorted_keys_oss = sorted(percentage_counts_oss.keys())
        sorted_keys_evol = sorted(percentage_counts_evol.keys())
        sorted_keys_codealpaca = sorted(percentage_counts_codealpaca.keys())

        color1 = (59/256, 117/256, 175/256)  
        color2 = (82/256, 159/256, 64/256)  
        color3 = (239/256, 139/256, 54/256) 

        color4 = (0/256, 60/256, 146/256)  
        color5 = (0/256, 146/256, 10/256) 
        color6 = (230/256, 120/256, 0/37)   

        alpha=0.33

        plt.plot(sorted_keys_codealpaca, [percentage_counts_codealpaca[k] for k in sorted_keys_codealpaca], color=color2, alpha=1, label=r'Self-Instruct; Avg Score: ' + f'{avg_score_codealpaca:.3f}', zorder=3)
        plt.fill_between(sorted_keys_codealpaca, [percentage_counts_codealpaca[k] for k in sorted_keys_codealpaca], color=color5, alpha=alpha, zorder=2)

        plt.plot(sorted_keys_evol, [percentage_counts_evol[k] for k in sorted_keys_evol], color=color1, alpha=1, label=r'Evol-Instruct; Avg Score: ' + f'{avg_score_evol:.3f}', zorder=3)
        plt.fill_between(sorted_keys_evol, [percentage_counts_evol[k] for k in sorted_keys_evol], color=color4, alpha=alpha, zorder=2)

        plt.plot(sorted_keys_oss, [percentage_counts_oss[k] for k in sorted_keys_oss], color=color3, alpha=1, label=r'OSS-Instruct; Avg Score: ' + f'{avg_score_oss:.3f}', zorder=3)
        plt.fill_between(sorted_keys_oss, [percentage_counts_oss[k] for k in sorted_keys_oss], color=color6, alpha=alpha, zorder=2)


        plt.xlabel("Cosine Similarity Score", fontsize=15)
        plt.ylabel("Percentage", fontsize=15)
        plt.xlim(0, 0.5)
        plt.ylim(bottom=0)  
        plt.xticks(np.arange(0, 0.55, 0.1))
        plt.yticks(np.arange(0, 0.16, 0.02), [f'{i:.2f}' for i in np.arange(0, 0.16, 0.02)]) 
        plt.tick_params(axis='both', labelsize=14)

        plt.axvline(x=avg_score_codealpaca, color='Forestgreen', linestyle='dotted')  
        plt.axvline(x=avg_score_evol, color='royalblue', linestyle='dotted') 
        plt.axvline(x=avg_score_oss, color='tomato', linestyle='dotted')  
        plt.legend(prop={'size': 10})
        plt.tight_layout()
        plt.savefig('HE_similarity_comparison.png')

class DataLoader:
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

def main():
    file_paths = {
        'oss': 'oss_score.json',
        'evol': 'evol_score.json',
        'codealpaca': 'codealpaca_score.json'
    }
    analysis = CosineSimilarityAnalysis(file_paths)

    if all(os.path.exists(file) for file in file_paths.values()):
        oss_scores = analysis.load_scores(file_paths['oss'])
        evol_scores = analysis.load_scores(file_paths['evol'])
        codealpaca_scores = analysis.load_scores(file_paths['codealpaca'])
    else:
        oss_data = DataLoader.load_data_oss(['data-clean-decontaminated.jsonl'])
        evol_data = DataLoader.load_data_evol(['data-evol_instruct-decontaminated.jsonl'])
        codealpaca_data = DataLoader.load_data_codealpaca(['data-codealpaca-decontaminated.jsonl'])

        dataset_name = 'openai_humaneval'
        dataset = load_dataset(dataset_name, split='test')
        data2 = [item['prompt'] + " " + item['canonical_solution'] for item in dataset]

        all_texts = oss_data + evol_data + codealpaca_data + data2
        vectorizer = TfidfVectorizer()
        vectorizer.fit(all_texts)
        data2_tensors = [analysis.prepare_data_for_gpu(vectorizer, [text]) for text in data2]

        oss_scores, oss_indices = analysis.calculate_cosine_scores_gpu(oss_data, data2_tensors, vectorizer, batch_size=100, save_path_1='oss_index.json', save_path_2='oss_score.json')
        evol_scores, evol_indices = analysis.calculate_cosine_scores_gpu(evol_data, data2_tensors, vectorizer, batch_size=100, save_path_1='evol_index.json', save_path_2='evol_score.json')
        codealpaca_scores, codealpaca_indices = analysis.calculate_cosine_scores_gpu(codealpaca_data, data2_tensors, vectorizer, batch_size=100, save_path_1='codealpaca_index.json', save_path_2='codealpaca_score.json')
    
    analysis.plot_scores(oss_scores, evol_scores, codealpaca_scores)

if __name__ == "__main__":
    main()
