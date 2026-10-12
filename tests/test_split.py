import numpy as np
import pandas as pd

from visual_retrieval import data


def _frame(shuffle_seed):
    rows = [(f"B{b}_{i}", f"B{b}", b) for b in range(4) for i in range(1, 21)]
    df = pd.DataFrame(rows, columns=["image_id", "breed", "label"])
    return df.sample(frac=1, random_state=shuffle_seed)


def test_split_is_stratified_disjoint_and_order_independent():
    a = data.split_trainval(_frame(1)).set_index("image_id").split
    b = data.split_trainval(_frame(2)).set_index("image_id").split
    assert a.sort_index().equals(b.sort_index())  # input order does not matter (sorted by id)
    tv = data.split_trainval(_frame(1))
    counts = tv.groupby(["breed", "split"]).size().unstack()
    assert (counts["gallery"] == 10).all() and (counts["val"] == 10).all()  # 50/50 within every breed
    assert not tv["image_id"].duplicated().any()


def test_manifest_roundtrip_and_validation(synthetic_dataset, tmp_path):
    m = data.load_manifest(synthetic_dataset["manifest_path"])
    for s in data.SPLITS:
        assert len(data.subset(m, s)) > 0
    g = set(data.subset(m, "gallery").image_id)
    assert g.isdisjoint(data.subset(m, "val").image_id) and g.isdisjoint(data.subset(m, "test").image_id)
    bad = m.copy()
    bad.loc[0, "split"] = "oops"
    try:
        data.validate_manifest(bad)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_dataset_items(synthetic_dataset):
    from visual_retrieval.embeddings import build_transform

    m = data.load_manifest(synthetic_dataset["manifest_path"])
    ds = data.PetDataset(data.subset(m, "gallery"), synthetic_dataset["root"], build_transform())
    x, y = ds[0]
    assert tuple(x.shape) == (3, 224, 224) and y == ds.labels[0]
