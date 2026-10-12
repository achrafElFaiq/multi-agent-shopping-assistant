"""Embedder running a small multilingual model locally with fastembed: no API key, no PyTorch."""

from fastembed import TextEmbedding

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class FastEmbedEmbedder:
    dimensions = 384

    def __init__(self) -> None:
        self._model = TextEmbedding(MODEL)  # downloaded on first use (about 220 MB), then read from the cache

    def embed(self, text: str) -> list[float]:
        [vector] = self._model.embed([text])
        return [float(value) for value in vector]
