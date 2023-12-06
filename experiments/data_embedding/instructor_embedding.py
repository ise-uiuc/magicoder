from dataclasses import dataclass, field
from typing import Literal, cast
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb  
import matplotlib.pyplot as plt
import numpy as np
from appdirs import user_cache_dir
from datasets import Dataset, concatenate_datasets, load_dataset
from InstructorEmbedding import INSTRUCTOR
from joblib import Memory  # type: ignore # fmt: off
from sentence_transformers.util import cos_sim
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from transformers import HfArgumentParser
from matplotlib.font_manager import FontProperties
import random
import sys
sys.path.append("src/magicoder")  
from prompt_template import SRC_INSTRUCT_ILLUSTRATION_PROMPT
from collections import Counter
from adjustText import adjust_text
import matplotlib.colors as mcolors
from matplotlib.colors import to_rgba, to_hex

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'sans-serif'

MEM = Memory(location=user_cache_dir("magicoder-experiments"))

ModelKey = Literal["instructor-large", "instructor-base", "instructor-xl"]
EmbeddingMode = Literal["seed", "problem", "solution", "problem-solution"]


@dataclass(frozen=True)
class Args:
    data_files: list[str]
    instruction: str
    model_key: ModelKey
    embedding_mode: EmbeddingMode

    queries: list[str] = field(default_factory=list)
    query_instruction: str | None = field(default=None)
    batch_size: int = field(default=32)
    n_clusters: int | None = field(default=None)



@MEM.cache(ignore=["model", "batch_size"])
def get_dataset_embedding(
    model: INSTRUCTOR,
    # for hashing only, must be consistent with `model`
    _model_name: str,
    embedding_mode: EmbeddingMode,
    dataset: Dataset,
    instruction: str,
    batch_size: int,
) -> np.ndarray:
    def map_fn(example: dict) -> dict:
        if embedding_mode == "seed":
            text = example["seed"]
        elif embedding_mode == "problem":
            text = example["problem"]
        elif embedding_mode == "solution":
            text = example["solution"]
        elif embedding_mode == "problem-solution":
            text = SRC_INSTRUCT_ILLUSTRATION_PROMPT.format(
                problem=example["problem"], solution=example["solution"]
            )
        else:
            assert False
        return {"pair": (instruction, text)}

    dataset = dataset.map(map_fn)
    sentences = dataset.to_dict()["pair"]
    embeddings = model.encode(sentences, batch_size=batch_size, show_progress_bar=True)
    return embeddings


def get_dataset_embeddings(
    args: Args, instruction: str, model: INSTRUCTOR
) -> tuple[Dataset, np.ndarray]:
    all_datasets: list[Dataset] = []
    all_embeddings: list[np.ndarray] = []
    for data_file in args.data_files:
        raw_dataset = load_dataset("json", data_files=[data_file], split="train")
        all_datasets.append(raw_dataset)

        embeddings = get_dataset_embedding(
            model,
            args.model_key,
            args.embedding_mode,
            raw_dataset,
            instruction,
            args.batch_size,
        )
        all_embeddings.append(embeddings)

    raw_dataset = concatenate_datasets(all_datasets)
    embeddings = np.concatenate(all_embeddings, axis=0)
    return raw_dataset, embeddings


def main():
    args = cast(Args, HfArgumentParser(Args).parse_args_into_dataclasses()[0])
    assert len(args.queries) or args.n_clusters is not None
    method = "cluster" if len(args.queries) == 0 else "query"
    if method == "query":
        assert args.query_instruction is not None
    else:
        assert method == "cluster"
        assert args.n_clusters is not None

    model = INSTRUCTOR(f"hkunlp/{args.model_key}")
    dataset, embeddings = get_dataset_embeddings(args, args.instruction, model=model)
    if method == "cluster":
        assert args.n_clusters is not None
        kmeans = KMeans(n_clusters=args.n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        assert labels.max() == args.n_clusters - 1
        assert labels.min() == 0
        n_clusters = args.n_clusters
        indices = {
        label: np.where(labels == label)[0].tolist() for label in range(n_clusters)
    }
        tsne = TSNE(
            n_components=2,
            perplexity=100,
            random_state=42,
            init="random",
            learning_rate=200,
        )
        vis_dims = tsne.fit_transform(embeddings)
        xs = np.array([x for x, y in vis_dims])
        ys = np.array([y for x, y in vis_dims])
        for label in range(n_clusters):
            indices_for_label = indices[label]
            x = xs[indices_for_label]
            y = ys[indices_for_label]
            legend_label = (
                f"Cluster {label + 1}" if method == "cluster" else args.queries[label]
            )
            plt.scatter(x, y, label=legend_label)
            avg_x = xs.mean()
            avg_y = ys.mean()
            plt.scatter(avg_x, avg_y, marker="x", s=100)
        plt.legend()
        plt.title("Visualization of Clusters")
        plt.savefig("Clusters.png")
    else:
        assert method == "query"
        queries = [[args.query_instruction, query] for query in args.queries]
        print(queries)
        query_embeddings = model.encode(queries, batch_size=args.batch_size)

        def get_label(embedding: np.ndarray) -> int:
            similarities = cos_sim(embedding.reshape(1, -1), query_embeddings)
            assert similarities.shape == (1, len(queries))
            similarities = similarities[0]
            return np.argmax(similarities).item()

        labels = np.array([get_label(embedding) for embedding in embeddings])
        print(labels)
        assert labels.max() == len(queries) - 1
        assert labels.min() == 0
        n_clusters = len(queries)
        # query_labels = [query[1].split()[0] for query in queries]
        query_labels = [query[1] for query in queries]
        label_counts = Counter(labels)
        categories = [query_labels[label] for label in label_counts.keys()]
        counts = list(label_counts.values())     
        colors = plt.cm.tab10(np.random.permutation(np.linspace(0, 1, len(categories))))
        colors = [
            "Tomato", "Coral", "Orange", "Gold", "YellowGreen", "MediumSeaGreen", "MediumTurquoise","LightSeaGreen",
             "RoyalBlue", "MediumOrchid", 
        ]
        # selected_colors = np.random.choice(color_names, len(categories), replace=False)
        selected_colors = np.array(colors[:len(categories)])
        def adjust_saturation_and_brightness(color, saturation_factor, brightness_factor):
            """
            Adjust the saturation and brightness of a color.
            color: Hex or RGB color
            saturation_factor: float (less than 1 to desaturate, more than 1 to increase saturation)
            brightness_factor: float (less than 1 to darken, more than 1 to lighten)
            """
            rgba = to_rgba(color)
            hsv = mcolors.rgb_to_hsv(rgba[:3])
            hsv[1] *= saturation_factor
            hsv[2] *= brightness_factor
            new_rgb = mcolors.hsv_to_rgb(hsv)
            return to_hex(new_rgb)

        explode = [0.02] * len(counts)
        fig, ax = plt.subplots(figsize=(13.5, 3))
        
        def calc_angle(percent):
            return 140 - (percent * 3.6) / 2
        saturation_factor = 0.55 
        brightness_factor = 0.85  
        modified_colors = [adjust_saturation_and_brightness(color, saturation_factor, brightness_factor) for color in selected_colors]
       
        wedges, texts = ax.pie(counts, colors=modified_colors, explode=explode,startangle=140, wedgeprops={'width': 0.65, 'edgecolor': 'w'})
        for i, p in enumerate(wedges):
            angle = (p.theta2 + p.theta1) / 2
            x = np.cos(np.deg2rad(angle))
            y = np.sin(np.deg2rad(angle))
            horizontalalignment = {1: "left", -1: "right"}[int(np.sign(x))]
            label_words = categories[i].split()
            label_words = [word for word in label_words if "Problems" not in word]
            label_words = [word.replace("and", "\&") for word in label_words]
            font_bold = FontProperties(weight='bold')

            label = " ".join(label_words)
            value = "{:.1f}\%".format(counts[i] / sum(counts) * 100) 

            if x>0:
                label_and_value = f" ({value}) "+label 
            else:
                label_and_value = label + f" ({value})"
            angle_B=-120*np.sign(x)*np.sign(y)
            xytext = (1.1 * np.sign(x), 1.3*y)
            ax.annotate(label_and_value, xy=(x, y), xytext=xytext, 
                horizontalalignment=horizontalalignment, fontsize=18,
                arrowprops=dict(arrowstyle="-", connectionstyle=f"angle,angleA=0,angleB={angle_B},rad=0"),
                )
        plt.axis('equal')
        plt.savefig("Pie_Chart.png")
      
    


if __name__ == "__main__":
    main()
