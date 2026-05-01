import pytest
from rl_recsys.data import registry as reg
from rl_recsys.data.pipelines.movielens import MovieLensPipeline


def test_register_and_get_pipeline(tmp_path):
    reg.register(
        "_test_reg_a",
        MovieLensPipeline,
        schema="interactions",
        tags=["CF"],
        variant="100k",
        raw_dir=str(tmp_path / "raw"),
        processed_dir=str(tmp_path / "proc"),
    )
    pipeline = reg.get_pipeline("_test_reg_a")
    assert isinstance(pipeline, MovieLensPipeline)


def test_list_datasets_is_sorted(tmp_path):
    reg.register("_test_z", MovieLensPipeline, schema="interactions", tags=[],
                 variant="100k", raw_dir=str(tmp_path), processed_dir=str(tmp_path))
    reg.register("_test_a", MovieLensPipeline, schema="interactions", tags=[],
                 variant="1m", raw_dir=str(tmp_path), processed_dir=str(tmp_path))
    datasets = reg.list_datasets()
    assert datasets == sorted(datasets)


def test_get_pipeline_unknown_raises():
    with pytest.raises(ValueError, match="Unknown dataset"):
        reg.get_pipeline("_nonexistent_xyz_abc")


def test_get_dataset_info_returns_metadata(tmp_path):
    reg.register(
        "_test_info",
        MovieLensPipeline,
        schema="interactions",
        tags=["CF"],
        variant="100k",
        raw_dir=str(tmp_path / "raw"),
        processed_dir=str(tmp_path / "proc"),
    )
    info = reg.get_dataset_info("_test_info")
    assert info.name == "_test_info"
    assert info.schema == "interactions"


def test_get_pipeline_overrides_raw_dir(tmp_path):
    reg.register(
        "_test_override",
        MovieLensPipeline,
        schema="interactions",
        tags=["CF"],
        variant="100k",
        raw_dir="data/raw/movielens",
        processed_dir="data/processed/movielens",
    )
    pipeline = reg.get_pipeline(
        "_test_override",
        raw_dir=str(tmp_path / "custom_raw"),
        processed_dir=str(tmp_path / "custom_proc"),
    )
    assert str(pipeline.raw_dir) == str(tmp_path / "custom_raw")
