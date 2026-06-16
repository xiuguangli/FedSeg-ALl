from __future__ import annotations

from pathlib import Path

from segmentation.myseg.datasplit import cityscapes_iid, cityscapes_noniid_extend


class DummyDataset:
    def __init__(self, size: int):
        self._size = size

    def __len__(self):
        return self._size


def test_cityscapes_iid_is_deterministic_and_disjoint():
    dataset = DummyDataset(24)
    split_a = cityscapes_iid(dataset, num_users=4, seed=7)
    split_b = cityscapes_iid(dataset, num_users=4, seed=7)

    assert split_a == split_b
    assert sum(len(v) for v in split_a.values()) == 24

    union = set()
    for indices in split_a.values():
        assert union.isdisjoint(indices)
        union |= indices


def test_cityscapes_noniid_extend_is_deterministic(tmp_path: Path):
    root = tmp_path / "mock_split"
    train_folder = "images/train"
    for cls_name, count in {"a": 5, "b": 7, "c": 6}.items():
        class_dir = root / train_folder / cls_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(count):
            (class_dir / f"{idx}.png").write_bytes(b"x")

    split_a = cityscapes_noniid_extend(str(root), train_folder, num_users=6, seed=3)
    split_b = cityscapes_noniid_extend(str(root), train_folder, num_users=6, seed=3)

    assert split_a == split_b
    assert len(split_a) == 6
    assert sum(len(v) for v in split_a.values()) == 18
