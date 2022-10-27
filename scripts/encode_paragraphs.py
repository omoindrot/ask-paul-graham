"""Encode paragraphs from PG essays into embeddings
"""

from pathlib import Path
from tqdm import tqdm

import numpy as np
from sentence_transformers import SentenceTransformer


# Choose model to use
# MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_NAME = "sentence-transformers/multi-qa-mpnet-base-dot-v1"


if __name__ == "__main__":
    # Extract embeddings
    model = SentenceTransformer(MODEL_NAME)

    essay_dir = Path("pg_essays/")
    essay_paths = list(sorted(essay_dir.glob("*.txt")))
    embedding_dir = Path(f"pg_embeddings/{MODEL_NAME.rsplit('/', maxsplit=1)[-1]}")
    embedding_dir.mkdir(exist_ok=True, parents=True)

    for essay_path in tqdm(essay_paths):
        text = essay_path.read_text()
        # TODO: for paragraphs longer than 256 tokens, split and average embedding
        #       for the big model, it can take up to 512 tokens but was trained on 250
        paragraphs = text.split("\n")

        embeddings_path = embedding_dir / f"{essay_path.stem}.npy"
        if embeddings_path.is_file():
            embeddings = np.load(embeddings_path)
        else:
            # Compute paragraph embeddings
            embeddings = model.encode(paragraphs)

            # Save to disk
            np.save(embeddings_path, embeddings)
