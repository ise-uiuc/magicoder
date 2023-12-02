import json
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import tiktoken

class TokenLengthAnalysis:
    def __init__(self, file_paths, types, encoding_name='cl100k_base'):
        self.file_paths = file_paths
        self.types = types
        self.encoding_name = encoding_name

    def num_tokens_from_string(self, string):
        encoding = tiktoken.get_encoding(self.encoding_name)
        return len(encoding.encode(string))

    def load_data(self, type):
        all_problem_solutions = []
        for file_path in self.file_paths:
            with open(file_path, 'r') as file:
                for line in file:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        key = type if type in data else None
                        text = data[key]
                        num_tokens = self.num_tokens_from_string(text)
                        all_problem_solutions.append(num_tokens)
        return all_problem_solutions

    def plot_data(self):
        fig, ax = plt.subplots(figsize=(4.8, 3))
        for type in self.types:
            data = self.load_data(type)
            width = 20
            token_length = [(length // width) * width for length in data]
            token_length_counts = Counter(token_length)
            x_values = sorted(list(token_length_counts.keys()))
            sorted_indices = np.argsort(x_values)
            y_values = [token_length_counts[x_values[i]] / 1000 for i in sorted_indices]
            fill_color = (0/256, 90/256, 146/256) if type == 'problem' else (230/256, 120/256, 0/37)
            ax.fill_between(x_values, 0, y_values, alpha=0.4, color=fill_color)
            ax.plot(x_values, y_values, linestyle='-', label=f'{type}')

        ax.set_xlim(left=0, right=700)
        ax.set_ylim(bottom=0)
        ax.set_xticks(np.arange(0, 700, 100))
        ax.set_yticks(np.arange(0, 8, 1))
        ax.set_xlabel("Number of Tokens", fontsize=14)
        ax.set_ylabel("#Count (Thousand)", fontsize=14)
        ax.legend(prop={'size': 10})
        plt.tight_layout()
        plt.savefig('Length.png')
        plt.show()

def main():
    file_paths = ['data-clean-decontaminated.jsonl']
    types = ['problem', 'solution']
    analysis = TokenLengthAnalysis(file_paths, types)
    analysis.plot_data()

if __name__ == "__main__":
    main()
