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
ALIASES: Dict[str, list[str]] = {}


def validate_key(key: str):
    if not isinstance(key, str):
        raise TypeError("densifier key must be a string")
    if WRAPPER_SEP in key or ROOT_KEY_SEP in key or ROOT_VALUE_SEP in key:
        raise ValueError(
            f"densifier key {key!r} must not contain {WRAPPER_SEP!r}, {ROOT_KEY_SEP!r} or {ROOT_VALUE_SEP!r}"
        )
    if key in DENSIFIERS:
        raise ValueError(f"densifier {key!r} is already registered: {DENSIFIERS[key]}")
    if key in ALIASES:
        raise ValueError(f"densifier {key!r} is already registered as alias: {ALIASES[key]}")


def register(key: str, entry: DensifierEntry):
    validate_key(key)
    DENSIFIERS[key] = entry


def register_alias(key: str, keys: list[str]):
    validate_key(key)
    if not keys:
        raise ValueError("alias names must be a non-empty list")
    for k in keys:
        if k not in DENSIFIERS and k not in ALIASES:
            raise KeyError(f"{k!r} is not a registered densifier or alias")
    ALIASES[key] = list(keys)


def expand_alias(key: str) -> list[str]:
    if key not in ALIASES:
        return [key]
    return [n for part in ALIASES[key] for n in expand_alias(part)]


def parse_names(names: list[str]) -> list[str]:
    return [n for name in names for n in expand_alias(name)]


def densifier(key: str):
    def decorator(fn: DensifierWrapFn) -> DensifierWrapFn:
        register(key, DensifierEntry(fn))
        return fn
    return decorator


def build_constructor(values: list[str]) -> DensifierConstructor:
    constructor: DensifierConstructor = NoopDensifier
    for wrap_name in parse_names(values):
        constructor = partial(DENSIFIERS[wrap_name].fn, constructor)
    return constructor
