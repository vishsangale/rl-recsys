from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type

from rl_recsys.data.pipelines.base import BasePipeline


@dataclass
class DatasetInfo:
    name: str
    pipeline_cls: Type[BasePipeline]
    schema: str
    tags: list[str]
    default_kwargs: dict = field(default_factory=dict)


_REGISTRY: dict[str, DatasetInfo] = {}


def register(
    name: str,
    pipeline_cls: Type[BasePipeline],
    schema: str,
    tags: list[str],
    **default_kwargs,
) -> None:
    _REGISTRY[name] = DatasetInfo(
        name=name,
        pipeline_cls=pipeline_cls,
        schema=schema,
        tags=tags,
        default_kwargs=default_kwargs,
    )


def get_pipeline(
    name: str,
    raw_dir: str | None = None,
    processed_dir: str | None = None,
    **kwargs,
) -> BasePipeline:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown dataset {name!r}. Available: {sorted(_REGISTRY)}")
    info = _REGISTRY[name]
    merged = {**info.default_kwargs, **kwargs}
    if raw_dir is not None:
        merged["raw_dir"] = raw_dir
    if processed_dir is not None:
        merged["processed_dir"] = processed_dir
    return info.pipeline_cls(**merged)


def get_dataset_info(name: str) -> DatasetInfo:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown dataset {name!r}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_datasets() -> list[str]:
    return sorted(_REGISTRY)
