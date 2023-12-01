from __future__ import annotations

from typing import Any

import pytest
import torch

from ludwig.api import LudwigModel
from ludwig.constants import ENCODER_OUTPUT
from ludwig.datasets import mnist
from ludwig.models.ecd import ECD


@pytest.mark.slow
def test_ludwig_model_encoder(tmpdir: str):
    df = mnist.load()[::1000]
    config: dict[str, Any] = {
        "input_features": [
            {
                "name": "image_path",
                "type": "image",
            }
        ],
        "output_features": [{"name": "label", "type": "category"}],
        "training": {"train_steps": 1, "batch_size": 2},
    }

    model = LudwigModel(config)
    model.train(df)

    model.save(tmpdir)

    transfer_config = config.copy()
    transfer_config["input_features"][0]["encoder"] = {
        "type": "ludwig-model",
        "path": str(tmpdir),
        "input_feature": "image_path",
        "trainable": False,
    }

    transfer_model = LudwigModel(transfer_config)
    transfer_model.train(df)  # train to initialize model

    assert isinstance(model.model, ECD)
    assert isinstance(transfer_model.model, ECD)
    data = {"image_path": torch.ones(1, 1, 28, 28, dtype=torch.float32)}
    encoded_normal = model.model.encode(data)["image_path"][ENCODER_OUTPUT]
    encoded_transfer = transfer_model.model.encode(data)["image_path"][ENCODER_OUTPUT]
    assert torch.equal(encoded_normal, encoded_transfer)
