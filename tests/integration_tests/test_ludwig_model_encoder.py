from __future__ import annotations

import pathlib
import shutil
from typing import Any

import pytest
import torch

from ludwig.api import LudwigModel
from ludwig.constants import ENCODER_OUTPUT
from ludwig.datasets import mnist
from ludwig.models.ecd import ECD


@pytest.mark.slow
def test_ludwig_model_encoder(tmpdir: pathlib.Path):
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

    save_path = str(tmpdir / "model")

    model.save(str(save_path))

    transfer_config = config.copy()
    transfer_config["input_features"][0]["encoder"] = {
        "type": "ludwig-model",
        "path": save_path,
        "input_feature": "image_path",
        "trainable": False,
    }

    transfer_model = LudwigModel(transfer_config)
    transfer_model.train(df)  # train to initialize model

    assert isinstance(model.model, ECD)
    assert isinstance(transfer_model.model, ECD)
    tensor = torch.ones(1, 1, 28, 28, dtype=torch.float32, device=model.model.device)
    data = {"image_path": tensor}
    encoded_normal = model.model.encode(data)["image_path"][ENCODER_OUTPUT]
    encoded_transfer = transfer_model.model.encode(data)["image_path"][ENCODER_OUTPUT]
    assert torch.equal(encoded_normal, encoded_transfer)

    # check loading of a ludwig model encoder
    shutil.rmtree(save_path)
    transfer_save_path = str(tmpdir / "transfer")
    transfer_model.save(transfer_save_path)
    loaded_model = LudwigModel.load(transfer_save_path)
    assert isinstance(loaded_model.model, ECD)
    tensor = tensor.to(loaded_model.model.device)
    data["image_path"] = tensor
    encoded_loaded = loaded_model.model.encode(data)["image_path"][ENCODER_OUTPUT]
    encoded_normal = encoded_normal.to(loaded_model.model.device)
    assert torch.allclose(encoded_normal, encoded_loaded, atol=1e-3)
