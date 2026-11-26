from apps.gui.lib import ModelHandler


def test_custom_model_scale_is_parsed_from_filename(tmp_path, monkeypatch):
    monkeypatch.setattr(ModelHandler, "CUSTOM_MODELS_PATH", str(tmp_path))
    for name in ("2x_anime.onnx", "4x_photo.onnx", "modelx8.onnx", "plain.onnx"):
        (tmp_path / name).write_bytes(b"")

    models = ModelHandler._custom_onnx_models()

    assert models["2x_anime.onnx"][2] == 2
    assert models["4x_photo.onnx"][2] == 4
    assert models["modelx8.onnx"][2] == 8
    assert models["plain.onnx"][2] == 4  # ambiguous names keep the old 4x default
    assert all(entry[3] == "custom" for entry in models.values())
