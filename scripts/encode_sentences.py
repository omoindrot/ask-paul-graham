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
# MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_NAME = "sentence-transformers/multi-qa-mpnet-base-dot-v1"

DISPLAY_TSNE = False


# Extract embeddings
model = SentenceTransformer(MODEL_NAME)

essay_dir = Path("pg_essays/")
essay_paths = list(sorted(essay_dir.glob("*.txt")))
embedding_dir = Path(f"pg_embeddings/{MODEL_NAME.rsplit('/', maxsplit=1)[-1]}")
embedding_dir.mkdir(exist_ok=True, parents=True)

paragraph_embeddings = []
essay_embeddings = []
all_paragraphs = []
all_essays = []
for essay_path in tqdm(essay_paths):
    text = essay_path.read_text()
    # TODO: for paragraphs longer than 256 tokens, split and average embedding
    #       for the big model, it can take up to 512 tokens but was trained on 250
    paragraphs = text.split("\n")
    all_paragraphs.extend(paragraphs)
    all_essays.extend([essay_path.stem] * len(paragraphs))

    embeddings_path = embedding_dir / f"{essay_path.stem}.npy"
    if embeddings_path.is_file():
        embeddings = np.load(embeddings_path)
    else:
        # Compute paragraph embeddings
        embeddings = model.encode(paragraphs)

        # Save to disk
        np.save(embeddings_path, embeddings)

    paragraph_embeddings.append(embeddings)
    essay_embeddings.append(np.mean(embeddings, axis=0, keepdims=True))

paragraph_embeddings = np.concatenate(paragraph_embeddings, axis=0)
essay_embeddings = np.concatenate(essay_embeddings, axis=0)
assert len(paragraph_embeddings) == len(all_paragraphs)
assert len(essay_embeddings) == len(essay_paths)

# Query
query = [
    # "How to iterate initially on your startup and find a good idea that solves a real problem",
    # "How to talk to users and iterate to find product market fit?",
    # "What makes a good team of cofounders, what are the attributes of a good cofounder?",
    # "Should I start a startup globally or in my country first?",
    # "How hard do I need to work to start a startup?",
    # "How to know if there is a good fit between me and my idea and stay motivated?",
    "Why should I hire slowly in my startup?",
]

embedding_query = model.encode(query)
if MODEL_NAME == "sentence-transformers/all-MiniLM-L6-v2":
    scores = util.pytorch_cos_sim(embedding_query, paragraph_embeddings)
elif MODEL_NAME == "sentence-transformers/multi-qa-mpnet-base-dot-v1":
    scores = util.dot_score(embedding_query, paragraph_embeddings)
else:
    raise ValueError(MODEL_NAME)

print(scores.shape)

top_scores, top_ids = scores.topk(10, largest=True, sorted=True)
top_scores = top_scores.flatten()
top_ids = top_ids.flatten()

for top_score, top_id in zip(top_scores, top_ids):
    print(
        float(top_score),
        all_essays[top_id],
        " ".join(all_paragraphs[top_id].split()),
    )


if DISPLAY_TSNE:
    # Try t-sne projection of all essays
    tsne = TSNE(n_components=2, random_state=0)
    projections = tsne.fit_transform(essay_embeddings)

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
