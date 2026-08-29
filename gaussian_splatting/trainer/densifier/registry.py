from functools import partial
from typing import Callable, Concatenate, Dict

from gaussian_splatting import GaussianModel
from gaussian_splatting.dataset import CameraDataset

from ..registry import WRAPPER_SEP, ROOT_KEY_SEP, ROOT_VALUE_SEP, validate_wrap_signature
from .abc import AbstractDensifier, NoopDensifier

DensifierConstructor = Callable[..., AbstractDensifier]
DensifierWrapFn = Callable[Concatenate[DensifierConstructor, GaussianModel, CameraDataset, ...], AbstractDensifier]


class DensifierEntry:
    def __init__(self, fn: DensifierWrapFn):
        validate_wrap_signature(fn)
        self.fn = fn


DENSIFIERS: Dict[str, DensifierEntry] = {}


def register(key: str, entry: DensifierEntry):
    if not isinstance(key, str):
        raise TypeError("densifier key must be a string")
    if WRAPPER_SEP in key or ROOT_KEY_SEP in key or ROOT_VALUE_SEP in key:
        raise ValueError(
            f"densifier key {key!r} must not contain {WRAPPER_SEP!r}, {ROOT_KEY_SEP!r} or {ROOT_VALUE_SEP!r}"
        )
    if key in DENSIFIERS:
        raise ValueError(f"densifier {key!r} is already registered: {DENSIFIERS[key]}")
    DENSIFIERS[key] = entry


def densifier(key: str):
    def decorator(fn: DensifierWrapFn) -> DensifierWrapFn:
        register(key, DensifierEntry(fn))
        return fn
    return decorator


def build_constructor(values: list[str]) -> DensifierConstructor:
    constructor: DensifierConstructor = NoopDensifier
    for wrap_name in values:
        constructor = partial(DENSIFIERS[wrap_name].fn, constructor)
    return constructor
