import numpy as np
import pytest

from face.embedding import cosine_similarity, normalize_embedding, representative_embedding


def test_cosine_similarity() -> None:
    assert cosine_similarity(np.array([1, 0]), np.array([2, 0])) == pytest.approx(1.0)
    assert cosine_similarity(np.array([1, 0]), np.array([0, 1])) == pytest.approx(0.0)


def test_representative_embedding_is_normalized() -> None:
    result = representative_embedding([np.array([1.0, 0.1]), np.array([0.9, 0.0])])
    assert np.linalg.norm(result) == pytest.approx(1.0)


def test_zero_embedding_rejected() -> None:
    with pytest.raises(ValueError):
        normalize_embedding(np.zeros(3))
