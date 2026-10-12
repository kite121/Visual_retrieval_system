import numpy as np
import pytest
import torch
from PIL import Image

from visual_retrieval import data, models
from visual_retrieval.embeddings import l2_normalize
from visual_retrieval.retrieval import GalleryIndex, query


def _index(n=50, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    emb = l2_normalize(torch.randn(n, d, generator=g))
    ids = [f"img{i}" for i in range(n)]
    return emb, GalleryIndex(emb, ids, np.arange(n) % 5, [f"b{i % 5}" for i in range(n)])


def test_search_matches_bruteforce_and_is_sorted():
    emb, idx = _index()
    q = l2_normalize(torch.randn(7, 16, generator=torch.Generator().manual_seed(1)))
    scores, ind = idx.search(q, 10)
    ref = (q @ emb.T).topk(10, dim=1)
    assert torch.equal(ind, ref.indices) and torch.allclose(scores, ref.values)
    assert (scores[:, :-1] >= scores[:, 1:]).all()


def test_self_is_top1_and_exclusion():
    emb, idx = _index()
    s, i = idx.search(emb[3], 3)
    assert i[0, 0] == 3 and s[0, 0] == pytest.approx(1.0, abs=1e-5)
    s, i = idx.search(emb[3], 3, exclude_ids=["img3"])
    assert 3 not in i[0].tolist()


def test_rejects_unnormalized_and_big_k():
    with pytest.raises(ValueError):
        GalleryIndex(torch.ones(4, 8) * 3, list("abcd"), np.zeros(4))
    _, idx = _index(n=5)
    with pytest.raises(ValueError):
        idx.search(torch.randn(1, 16), 6)


def test_query_end_to_end_with_random_encoder(synthetic_dataset):
    """query(image, model, ...) returns the image itself first when it is part of the gallery."""
    from visual_retrieval.embeddings import build_transform, extract_embeddings

    m = data.load_manifest(synthetic_dataset["manifest_path"])
    gal = data.PetDataset(data.subset(m, "gallery"), synthetic_dataset["root"], build_transform())
    enc = models.load_encoder("deit_vit_s16_supervised", "cpu", pretrained=False)
    emb = extract_embeddings(enc, gal, "cpu", batch_size=8, num_workers=0, progress=False)
    index = GalleryIndex(emb, gal.image_ids, gal.labels, list(gal.frame.breed))
    hits = query(gal.paths[2], enc, index, k=5)
    assert len(hits) == 5 and hits[0].image_id == gal.image_ids[2] and hits[0].score == pytest.approx(1.0, abs=1e-4)
    assert [h.rank for h in hits] == [1, 2, 3, 4, 5]
    assert query(Image.open(gal.paths[2]), enc, index, k=5, exclude_ids=[gal.image_ids[2]])[0].image_id != gal.image_ids[2]


def test_registry_shapes():
    for name in ("dino_vit_s16", "deit_vit_s16_supervised"):
        enc = models.load_encoder(name, "cpu", pretrained=False)
        out = enc(torch.zeros(2, 3, 224, 224))
        assert tuple(out.shape) == (2, models.get_spec(name).embed_dim)
        assert not any(p.requires_grad for p in enc.parameters()) and not enc.training
