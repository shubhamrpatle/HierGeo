"""Shard-scoped cosine retrieval over pre-computed image embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .routing import GeoIndex, PlaceParser, Route

# Routing policies evaluated in the paper (Tab. 7):
#   global       - no routing, search all images
#   country      - restrict to the named country
#   country_city - restrict to the named city when one is given, else the country
ROUTING_MODES = ("global", "country", "country_city")


@dataclass
class SearchResult:
    route: Route
    image_ids: List[int]
    scores: List[float]
    num_candidates: int


class ShardedImageIndex:
    """Image embeddings plus a country/city geo index.

    Scores are computed as ``fp16`` dot products on the chosen device, exactly as
    in the evaluator used for the paper, so that rankings (including ties) are
    reproducible.
    """

    def __init__(self, embeddings: np.ndarray, geo: GeoIndex, device: Optional[str] = None) -> None:
        if embeddings.shape[0] != geo.num_images:
            raise ValueError(
                f"{embeddings.shape[0]} embeddings but {geo.num_images} metadata rows; "
                "embeddings must follow the metadata row order"
            )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        emb = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.where(norms == 0, 1.0, norms)
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.embeddings = torch.from_numpy(emb).to(self.device).to(dtype)
        self.geo = geo
        self.parser = PlaceParser(geo)
        self._scopes: Dict[Tuple[str, ...], torch.Tensor] = {}

    @classmethod
    def load(cls, embeddings_path: str | Path, metadata_csv: str | Path, device: Optional[str] = None) -> "ShardedImageIndex":
        return cls(np.load(embeddings_path, mmap_mode="r"), GeoIndex.from_metadata_csv(metadata_csv), device)

    def _scope_tensor(self, route: Route) -> Optional[torch.Tensor]:
        if route.scope == "global":
            return None
        key = (route.scope, route.country or "", route.city or "")
        if key not in self._scopes:
            ids = self.parser.candidates(route)
            self._scopes[key] = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        return self._scopes[key]

    def route(self, query: str, mode: str = "country_city") -> Route:
        if mode not in ROUTING_MODES:
            raise ValueError(f"mode must be one of {ROUTING_MODES}")
        if mode == "global":
            return Route(scope="global")
        return self.parser.parse(query, use_city=(mode == "country_city"))

    def search_vector(self, query_vec: np.ndarray, route: Route, top_k: int = 10) -> SearchResult:
        q = torch.from_numpy(np.asarray(query_vec, dtype=np.float32)).to(self.device).to(self.embeddings.dtype)
        idx = self._scope_tensor(route)
        if idx is None or idx.numel() == 0:
            scores = self.embeddings @ q
            k = min(top_k, scores.numel())
            vals, top = torch.topk(scores, k=k, largest=True, sorted=True)
            ids = top
            n = self.embeddings.shape[0]
        else:
            scores = self.embeddings.index_select(0, idx) @ q
            k = min(top_k, scores.numel())
            vals, local = torch.topk(scores, k=k, largest=True, sorted=True)
            ids = idx[local]
            n = idx.numel()
        return SearchResult(route, ids.cpu().tolist(), vals.float().cpu().tolist(), int(n))


class HierGeoRetriever:
    """End-to-end search: parse the place, pick the shard, rank with a dual encoder."""

    def __init__(self, index: ShardedImageIndex, encoder, text_padding: str = "max_length") -> None:
        self.index = index
        self.encoder = encoder
        self.text_padding = text_padding

    def search(self, query: str, top_k: int = 10, mode: str = "country_city", semantic_text: Optional[str] = None) -> SearchResult:
        """Search with a natural-language query such as ``"Residential in Luanda, Angola"``.

        By default the place phrase after the last " in " is removed before the
        text is embedded (the benchmark embeds only the category text); pass
        ``semantic_text`` to embed something else.
        """
        from .routing import split_query

        route = self.index.route(query, mode)
        text = semantic_text if semantic_text is not None else split_query(query)[0]
        vec = self.encoder.encode_text([text], padding=self.text_padding)[0]
        return self.index.search_vector(vec, route, top_k)


def encode_unique_texts(encoder, texts: Sequence[str], padding: str, batch_size: int = 4096, autocast_dtype=None) -> Dict[str, np.ndarray]:
    """Encode each distinct text once (in first-seen order)."""
    unique = list(dict.fromkeys(t.strip() for t in texts))
    vecs = encoder.encode_text(unique, batch_size=batch_size, padding=padding, autocast_dtype=autocast_dtype)
    return dict(zip(unique, vecs))
