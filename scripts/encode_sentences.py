"""Encode sentences from PG essays into embeddings
"""

from pathlib import Path
from tqdm import tqdm

import numpy as np
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer, util
from sklearn.manifold import TSNE


# Parameters

# Choose model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# MODEL_NAME = "sentence-transformers/multi-qa-mpnet-base-dot-v1"


# Extract embeddings
model = SentenceTransformer(MODEL_NAME)

essay_dir = Path("pg_essays/")
essay_paths = list(sorted(essay_dir.glob("*.txt")))
embedding_dir = Path(f"pg_embeddings/{MODEL_NAME.rsplit('/', maxsplit=1)[-1]}")
embedding_dir.mkdir(exist_ok=True, parents=True)

all_embeddings = []
for essay_path in tqdm(essay_paths):
    embeddings_path = embedding_dir / f"{essay_path.stem}.npy"

    if embeddings_path.is_file():
        embeddings = np.load(embeddings_path)
    else:
        text = essay_path.read_text()
        # TODO: for paragraphs longer than 256 tokens, split and average embedding
        #       for the big model, it can take up to 512 tokens but was trained on 250
        paragraphs = text.split("\n")

        # Compute paragraph embeddings
        embeddings = model.encode(paragraphs)

        # Save to disk
        np.save(embeddings_path, embeddings)

    embedding = np.mean(embeddings, axis=0)
    all_embeddings.append(embedding)

all_embeddings = np.stack(all_embeddings, axis=0)

# Query
query = [
    # "How to iterate initially on your startup and find a good idea that solves a real problem",
    # "How to talk to users and iterate to find product market fit?",
    "What makes a good team of cofounders, what are the attributes of a good cofounder?",
]

embedding_query = model.encode(query)
if MODEL_NAME == "sentence-transformers/all-MiniLM-L6-v2":
    scores = util.pytorch_cos_sim(embedding_query, all_embeddings)
elif MODEL_NAME == "sentence-transformers/multi-qa-mpnet-base-dot-v1":
    scores = util.dot_score(embedding_query, all_embeddings)
else:
    raise ValueError(MODEL_NAME)

print(scores.shape)

top_scores, top_ids = scores.topk(10, largest=True, sorted=True)
top_scores = top_scores.flatten()
top_ids = top_ids.flatten()

for top_score, top_id in zip(top_scores, top_ids):
    print(top_score, essay_paths[top_id])


# Try t-sne projection of all essays
tsne = TSNE(n_components=2, random_state=0)
projections = tsne.fit_transform(all_embeddings)

fig = go.Figure(
    data=go.Scatter(
        x=projections[:, 0],
        y=projections[:, 1],
        mode="markers",
        text=[essay_path.stem for essay_path in essay_paths],
    )
)

fig.update_layout(title="Paul Graham essays")
fig.show()
