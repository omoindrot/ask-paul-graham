"""Streamlit demo app
"""

from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
import plotly.express as px
from sentence_transformers import SentenceTransformer, util
from sklearn.manifold import TSNE
import streamlit as st


MODEL_NAME = "sentence-transformers/multi-qa-mpnet-base-dot-v1"


@st.cache
def load_data(
    essay_paths: list[Path],
    model: SentenceTransformer,
    cache_dir: Path,
):
    """Load essays and embeddings, cache it."""

    paragraph_embeddings = []
    essay_embeddings = []
    all_paragraphs = []
    all_essays = []
    for essay_path in essay_paths:
        text = essay_path.read_text()
        # TODO: for paragraphs longer than 256 tokens, split and average embedding
        #       for the big model, it can take up to 512 tokens but was trained on 250
        paragraphs = text.split("\n")
        all_paragraphs.extend(paragraphs)
        all_essays.extend([essay_path.stem] * len(paragraphs))

        embeddings_path = cache_dir / f"{essay_path.stem}.npy"
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

    return all_essays, all_paragraphs, essay_embeddings, paragraph_embeddings


@st.cache(suppress_st_warning=True)
def display_query_matches(
    query, model, all_essays, all_paragraphs, paragraph_embeddings, topk
):
    """Display top matches for the query."""
    embedding_query = model.encode(query)
    if MODEL_NAME == "sentence-transformers/all-MiniLM-L6-v2":
        scores = util.pytorch_cos_sim(embedding_query, paragraph_embeddings)
    elif MODEL_NAME == "sentence-transformers/multi-qa-mpnet-base-dot-v1":
        scores = util.dot_score(embedding_query, paragraph_embeddings)
    else:
        raise ValueError(MODEL_NAME)

    top_scores, top_ids = scores.topk(topk, largest=True, sorted=True)
    top_scores = top_scores.flatten()
    top_ids = top_ids.flatten()

    for score, top_id in zip(top_scores, top_ids):
        essay = all_essays[top_id]
        st.subheader(f"{essay.split('_', maxsplit=1)[1]} ({score:.2f})")
        st.markdown(all_paragraphs[top_id - 1])
        st.markdown(f"**{all_paragraphs[top_id]}**")
        st.markdown(all_paragraphs[top_id + 1])


@st.cache
def tsne_fit(paragraph_embeddings):
    """Fit TSNE on paragraph embeddings."""
    tsne = TSNE(n_components=2, random_state=0, learning_rate="auto", init="pca")
    projections = tsne.fit_transform(paragraph_embeddings)
    return projections


def main():  # pylint: disable=too-many-locals,missing-function-docstring
    st.set_page_config(layout="wide")
    st.title("Ask Paul Graham")

    # Extract embeddings
    model = SentenceTransformer(MODEL_NAME)

    essay_dir = Path("pg_essays/")
    essay_paths = list(sorted(essay_dir.glob("*.txt")))
    embedding_dir = Path(f"pg_embeddings/{MODEL_NAME.rsplit('/', maxsplit=1)[-1]}")
    embedding_dir.mkdir(exist_ok=True, parents=True)

    all_essays, all_paragraphs, _, paragraph_embeddings = load_data(
        essay_paths=essay_paths,
        model=model,
        cache_dir=embedding_dir,
    )

    projections = tsne_fit(paragraph_embeddings)

    df = pd.DataFrame(
        dict(
            x=projections[:, 0],
            y=projections[:, 1],
            essay=all_essays,
            text=["<br>".join(textwrap.wrap(p)) for p in all_paragraphs],
        )
    )
    fig = px.scatter(
        df, x="x", y="y", hover_name="text", hover_data=["essay"], color="essay"
    )
    fig.update_layout(
        autosize=False,
        width=1000,
        height=700,
    )
    tab_tsne, tab_query = st.tabs(["TSNE", "Query"])

    with tab_tsne:
        st.plotly_chart(fig, use_container_width=True)

    # Example queries:
    # "How to iterate on your startup and find a good idea that solves a real problem"
    # "How to talk to users and iterate to find product market fit?"
    # "What makes a good team of cofounders, what are the attributes of a good cofounder?"
    # "Should I start a startup globally or in my country first?"
    # "How hard do I need to work to start a startup?"
    # "How to know if there is a good fit between me and my idea and stay motivated?"
    with tab_query:
        _, middle, _ = st.columns((2, 5, 2))

        with middle:
            example_query = "What are the attributes of a great cofounder?"
            query = st.text_input("Query", example_query)

            display_query_matches(
                query,
                model,
                all_essays,
                all_paragraphs,
                paragraph_embeddings,
                topk=5,
            )


if __name__ == "__main__":
    main()
