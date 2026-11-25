import numpy as np

from hiergeo.baselines import MetadataCorpus
from hiergeo.retrieval import ShardedImageIndex
from hiergeo.routing import GeoIndex


def _write_metadata(path):
    rows = [
        ("a.png", "Luanda", "Angola", "Residential", "Residential, apartment building"),
        ("b.png", "Luanda", "Angola", "Industrial", "Industrial, warehouse"),
        ("c.png", "Lobito", "Angola", "Residential", "Residential"),
        ("d.png", "Nairobi", "Kenya", "Residential", "Residential"),
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("image_path,caption,city,country,poi_type,multi_labels\n")
        for r in rows:
            handle.write(",".join(f'"{x}"' for x in r[:1]) + ',"",' + ",".join(f'"{x}"' for x in r[1:]) + "\n")


def test_sharded_search_respects_routes(tmp_path):
    meta = tmp_path / "meta.csv"
    _write_metadata(meta)
    emb = np.eye(4, dtype=np.float32)
    index = ShardedImageIndex(emb, GeoIndex.from_metadata_csv(meta), device="cpu")
    query = np.array([0.1, 0.2, 0.3, 1.0], dtype=np.float32)  # closest to the Kenyan image

    r = index.search_vector(query, index.route("Residential in Luanda, Angola"), top_k=10)
    assert r.route.scope == "city" and r.image_ids == [1, 0] and r.num_candidates == 2
    r = index.search_vector(query, index.route("Residential in Angola"), top_k=10)
    assert r.image_ids == [2, 1, 0]
    r = index.search_vector(query, index.route("Residential in Luanda, Angola", mode="global"), top_k=1)
    assert r.image_ids == [3] and r.num_candidates == 4


def test_metadata_baseline_prefers_exact_category(tmp_path):
    meta = tmp_path / "meta.csv"
    _write_metadata(meta)
    corpus = MetadataCorpus(meta)
    q = {"query_id": "q1", "country": "Angola", "city": "Luanda", "query_category_generic": "Industrial"}
    assert corpus.rank(q, "geo_poi_lexical", seed=0)[0] == 1
    assert corpus.rank(q, "geo_poi_exact", seed=0)[0] == 1
    assert sorted(corpus.rank(q, "geo_random", seed=0)) == [0, 1]
