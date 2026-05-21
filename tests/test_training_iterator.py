from __future__ import annotations

import pytest

from set_switch.training.train import infinite_dataloader


class ReiterableLoader:
    def __init__(self) -> None:
        self.epochs = 0

    def __iter__(self):
        self.epochs += 1
        yield f"{self.epochs}:a"
        yield f"{self.epochs}:b"


class EmptyLoader:
    def __iter__(self):
        return iter(())


def test_infinite_dataloader_reiterates_without_replaying_cached_batches():
    loader = ReiterableLoader()
    iterator = infinite_dataloader(loader)

    assert [next(iterator) for _ in range(4)] == ["1:a", "1:b", "2:a", "2:b"]
    assert loader.epochs == 2


def test_infinite_dataloader_rejects_empty_loader():
    iterator = infinite_dataloader(EmptyLoader())

    with pytest.raises(ValueError, match="empty dataloader"):
        next(iterator)
