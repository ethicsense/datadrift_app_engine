"""Unit checks for CIEDE2000 and embedding controls."""

from __future__ import annotations

import numpy as np

from color_space import delta_e00, rgb_u8_to_lab
from controls import fingerprint, l2_normalize, mix, project, remove_subspace, subspace_basis


def test_delta_e00_known_pair() -> None:
    lab1 = np.array([50.0, 2.6772, -79.7751])
    lab2 = np.array([50.0, 0.0, -82.7485])
    value = float(delta_e00(lab1, lab2))
    assert abs(value - 2.0425) < 1e-3, value


def test_delta_e00_identity_and_order() -> None:
    red = rgb_u8_to_lab(np.array([188, 32, 42]))
    burgundy = rgb_u8_to_lab(np.array([112, 24, 40]))
    navy = rgb_u8_to_lab(np.array([18, 32, 68]))
    same = float(delta_e00(red, red))
    near = float(delta_e00(red, burgundy))
    far = float(delta_e00(red, navy))
    assert same < 1e-6
    assert near < far


def test_mix_reproduces_weighted_cosine() -> None:
    rng = np.random.default_rng(0)
    z = l2_normalize(rng.normal(size=(4, 6)).astype(np.float32))
    a = l2_normalize(rng.normal(size=(4, 3)).astype(np.float32))
    alpha = 0.25
    mixed = mix(z, a, alpha)
    got = float(mixed[0] @ mixed[1])
    expected = (1 - alpha) * float(z[0] @ z[1]) + alpha * float(a[0] @ a[1])
    assert abs(got - expected) < 1e-5


def test_projection_and_removal() -> None:
    prompts = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.2, 0.0],
            [0.0, 1.0, 0.0],
            [0.1, 0.9, 0.0],
        ],
        dtype=np.float32,
    )
    basis = subspace_basis(prompts, rank=2)
    images = np.array([[0.2, 0.3, 0.9], [0.4, -0.2, 0.5]], dtype=np.float32)
    removed = remove_subspace(images, basis)
    # Residual is orthogonal to every basis direction.
    assert np.allclose(removed @ basis, 0.0, atol=1e-5)
    projected = project(images, basis)
    assert projected.shape == (2, 2)


def test_fingerprint_shape() -> None:
    images = l2_normalize(np.array(
        [[1.0, 0.2, 0.0], [0.1, 1.0, 0.2], [0.0, 0.2, 1.0]],
        dtype=np.float32,
    ))
    groups = [np.eye(3, dtype=np.float32)[i : i + 1] for i in range(3)]
    scores = fingerprint(images, groups)
    assert scores.shape == (3, 3)
    assert np.allclose(np.linalg.norm(scores, axis=1), 1.0, atol=1e-5)


if __name__ == "__main__":
    test_delta_e00_known_pair()
    test_delta_e00_identity_and_order()
    test_mix_reproduces_weighted_cosine()
    test_projection_and_removal()
    test_fingerprint_shape()
    print("ok")
