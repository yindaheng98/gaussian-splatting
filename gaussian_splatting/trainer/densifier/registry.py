from collections.abc import Sequence
from functools import partial
from typing import Callable, Concatenate, Dict

from gaussian_splatting import GaussianModel
from gaussian_splatting.dataset import CameraDataset

from ..dsl import NAME_PATTERN, NAME_RE
from ..registry import validate_wrap_signature
from .abc import AbstractDensifier, NoopDensifier

DensifierConstructor = Callable[..., AbstractDensifier]
DensifierWrapFn = Callable[Concatenate[DensifierConstructor, GaussianModel, CameraDataset, ...], AbstractDensifier]


class DensifierEntry:
    def __init__(self, fn: DensifierWrapFn):
        validate_wrap_signature(fn)
        self.fn = fn


DENSIFIERS: Dict[str, DensifierEntry] = {}


def register(name: str, entry: DensifierEntry):
    if not isinstance(name, str):
        raise TypeError("densifier name must be a string")
    if NAME_RE.fullmatch(name) is None:
        raise ValueError(f"densifier name {name!r} must match {NAME_PATTERN}")
    if name in DENSIFIERS:
        raise ValueError(f"densifier {name!r} is already registered: {DENSIFIERS[name]}")
    DENSIFIERS[name] = entry


def densifier(name: str):
    def decorator(fn: DensifierWrapFn) -> DensifierWrapFn:
        register(name, DensifierEntry(fn))
        return fn
    return decorator


def build_constructor(names: Sequence[str]) -> DensifierConstructor:
    if isinstance(names, (str, bytes)) or not isinstance(names, Sequence):
        raise TypeError("densifier names must be a sequence of strings")

    seen_names = set()
    for index, name in enumerate(names):
        if not isinstance(name, str):
            raise TypeError(f"densifier name at index {index} must be a string")
        if NAME_RE.fullmatch(name) is None:
            raise ValueError(
                f"densifier name at index {index} {name!r} must match {NAME_PATTERN}"
            )
        if name in seen_names:
            raise ValueError(f"duplicate densifier {name!r}")
        seen_names.add(name)

    constructor: DensifierConstructor = NoopDensifier
    for name in names:
        entry = DENSIFIERS.get(name)
        if entry is None:
            raise KeyError(
                f"unknown densifier {name!r}; "
                f"available densifiers: {sorted(DENSIFIERS)}"
            )
        constructor = partial(entry.fn, constructor)
    return constructor
