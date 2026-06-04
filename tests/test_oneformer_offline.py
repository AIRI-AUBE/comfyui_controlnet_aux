import json
from pathlib import Path

import pytest

from custom_controlnet_aux.oneformer.transformers import (
    _build_oneformer_class_info,
    _prepare_oneformer_processor_kwargs,
)


def _write_preprocessor_config(model_dir: Path, config: dict) -> None:
    (model_dir / "preprocessor_config.json").write_text(
        json.dumps(config),
        encoding="utf-8",
    )


def test_build_oneformer_class_info_from_embedded_metadata():
    metadata = {
        "0": "background",
        "1": "person",
        "class_names": ["background", "person"],
        "thing_ids": [1],
    }

    assert _build_oneformer_class_info(metadata) == {
        "0": {"name": "background", "isthing": False},
        "1": {"name": "person", "isthing": True},
    }


def test_prepare_oneformer_processor_kwargs_prefers_local_metadata_file(tmp_path):
    class_info_file = "coco_panoptic.json"
    _write_preprocessor_config(
        tmp_path,
        {
            "class_info_file": class_info_file,
            "repo_path": "shi-labs/oneformer_demo",
            "metadata": {
                "0": "background",
                "class_names": ["background"],
                "thing_ids": [],
            },
        },
    )
    (tmp_path / class_info_file).write_text("{}", encoding="utf-8")

    with _prepare_oneformer_processor_kwargs(str(tmp_path)) as kwargs:
        assert kwargs == {
            "local_files_only": True,
            "class_info_file": class_info_file,
            "repo_path": str(tmp_path),
        }


def test_prepare_oneformer_processor_kwargs_reconstructs_missing_metadata_file(tmp_path):
    class_info_file = "coco_panoptic.json"
    _write_preprocessor_config(
        tmp_path,
        {
            "class_info_file": class_info_file,
            "repo_path": "shi-labs/oneformer_demo",
            "metadata": {
                "0": "background",
                "1": "person",
                "class_names": ["background", "person"],
                "thing_ids": [1],
            },
        },
    )

    with _prepare_oneformer_processor_kwargs(str(tmp_path)) as kwargs:
        metadata_dir = Path(kwargs["repo_path"])
        reconstructed_metadata = json.loads((metadata_dir / class_info_file).read_text(encoding="utf-8"))

        assert kwargs["local_files_only"] is True
        assert kwargs["class_info_file"] == class_info_file
        assert reconstructed_metadata == {
            "0": {"name": "background", "isthing": False},
            "1": {"name": "person", "isthing": True},
        }

    assert not metadata_dir.exists()


def test_prepare_oneformer_processor_kwargs_uses_local_ckpts_repo_id(monkeypatch, tmp_path):
    class_info_file = "coco_panoptic.json"
    demo_dir = tmp_path / "demo"
    demo_dir.mkdir()
    (demo_dir / class_info_file).write_text("{}", encoding="utf-8")
    _write_preprocessor_config(
        tmp_path,
        {
            "class_info_file": class_info_file,
            "repo_path": "shi-labs/oneformer_demo",
            "metadata": {
                "0": "background",
                "class_names": ["background"],
                "thing_ids": [],
            },
        },
    )

    monkeypatch.setattr(
        "custom_controlnet_aux.oneformer.transformers.resolve_local_hf_repo_path",
        lambda repo_path: str(demo_dir),
    )

    with _prepare_oneformer_processor_kwargs(str(tmp_path)) as kwargs:
        assert kwargs == {
            "local_files_only": True,
            "class_info_file": class_info_file,
            "repo_path": str(demo_dir),
        }


def test_prepare_oneformer_processor_kwargs_errors_without_any_local_metadata(tmp_path):
    _write_preprocessor_config(
        tmp_path,
        {
            "class_info_file": "coco_panoptic.json",
            "repo_path": "shi-labs/oneformer_demo",
        },
    )

    with pytest.raises(FileNotFoundError, match="embedded preprocessor metadata could not reconstruct it"):
        with _prepare_oneformer_processor_kwargs(str(tmp_path)):
            pass