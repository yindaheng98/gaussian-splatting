import inspect
from abc import ABC
from collections.abc import Callable as AbcCallable
from functools import partial
from typing import Callable, Concatenate, Dict, Type, get_origin

from gaussian_splatting import GaussianModel
from gaussian_splatting.dataset import CameraDataset

from .abc import AbstractTrainer
from .dsl import NAME_PATTERN, NAME_RE, TrainerSpec

TrainerConstructor = Callable[..., AbstractTrainer]
TrainerWrapFn = Callable[Concatenate[TrainerConstructor, GaussianModel, CameraDataset, ...], AbstractTrainer]


class TrainerEntry(ABC):
    pass


class TrainerRootEntry(TrainerEntry):
    def __init__(self, cls: Type[AbstractTrainer]):
        if not issubclass(cls, AbstractTrainer):
            raise TypeError(f"{cls.__name__} must be a subclass of AbstractTrainer")
        params = list(inspect.signature(cls.__init__).parameters.values())
        p = params[1]
        if p.default is not inspect.Parameter.empty or not issubclass(p.annotation, GaussianModel):
            raise TypeError(
                f"{cls.__name__}.__init__ first parameter must be a GaussianModel subclass "
                f"without a default, got {p.name}: {p.annotation}"
            )
        p = params[2]
        if p.default is not inspect.Parameter.empty or not issubclass(p.annotation, CameraDataset):
            raise TypeError(
                f"{cls.__name__}.__init__ second parameter must be a CameraDataset subclass "
                f"without a default, got {p.name}: {p.annotation}"
            )
        self.cls = cls

    def construct(
        self,
        components: tuple[str, ...],
        model: GaussianModel,
        dataset: CameraDataset,
        **configs,
    ) -> AbstractTrainer:
        if components:
            raise ValueError(
                f"trainer root {self.cls.__name__!r} does not accept components, "
                f"got {components}"
            )
        return self.cls(model, dataset, **configs)


def validate_wrap_signature(fn: Callable):
    params = list(inspect.signature(fn).parameters.values())
    name = getattr(fn, "__name__", repr(fn))
    if len(params) < 5:
        raise TypeError(
            f"{name} must be (constructor, model, dataset, *args, ..., **configs), got {len(params)} parameters"
        )

    ctor, model, dataset, varargs, *rest = params
    if ctor.kind not in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD) \
            or ctor.default is not inspect.Parameter.empty \
            or get_origin(ctor.annotation) is not AbcCallable:
        raise TypeError(
            f"{name} first parameter must be a Callable without a default, got {ctor.name}: {ctor.annotation}"
        )
    if model.kind not in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD) \
            or model.default is not inspect.Parameter.empty \
            or not (isinstance(model.annotation, type) and issubclass(model.annotation, GaussianModel)):
        raise TypeError(
            f"{name} second parameter must be a GaussianModel subclass without a default, "
            f"got {model.name}: {model.annotation}"
        )
    if dataset.kind not in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD) \
            or dataset.default is not inspect.Parameter.empty \
            or not (isinstance(dataset.annotation, type) and issubclass(dataset.annotation, CameraDataset)):
        raise TypeError(
            f"{name} third parameter must be a CameraDataset subclass without a default, "
            f"got {dataset.name}: {dataset.annotation}"
        )
    if varargs.kind is not inspect.Parameter.VAR_POSITIONAL:
        raise TypeError(f"{name} fourth parameter must be *args, got {varargs.name}")
    for p in rest[:-1]:
        if p.kind is not inspect.Parameter.KEYWORD_ONLY:
            raise TypeError(f"{name} parameters after *args must be keyword-only, got {p.name}")
        if p.default is inspect.Parameter.empty:
            raise TypeError(f"{name} parameter {p.name!r} after *args must have a default")
    if rest[-1].kind is not inspect.Parameter.VAR_KEYWORD:
        raise TypeError(f"{name} last parameter must be **configs, got {rest[-1].name}")


class TrainerWrapEntry(TrainerEntry):
    def __init__(self, fn: TrainerWrapFn):
        validate_wrap_signature(fn)
        self.fn = fn


TRAINERS: Dict[str, TrainerEntry] = {}


def register(name: str, entry: TrainerEntry):
    if not isinstance(name, str):
        raise TypeError("name must be a string")
    if NAME_RE.fullmatch(name) is None:
        raise ValueError(f"name {name!r} must match {NAME_PATTERN}")
    if name in TRAINERS:
        raise ValueError(f"{name!r} is already registered as trainer: {TRAINERS[name]}")
    TRAINERS[name] = entry


def trainer_root(name: str, entry_cls: Type[TrainerRootEntry] = TrainerRootEntry):
    if not issubclass(entry_cls, TrainerRootEntry):
        raise TypeError("entry_cls must be a TrainerRootEntry subclass")

    def decorator(cls: Type[AbstractTrainer]) -> Type[AbstractTrainer]:
        register(name, entry_cls(cls))
        return cls
    return decorator


def trainer_wrap(name: str):
    def decorator(fn: TrainerWrapFn) -> TrainerWrapFn:
        register(name, TrainerWrapEntry(fn))
        return fn
    return decorator


def build_trainer(
    spec: TrainerSpec,
    model: GaussianModel,
    dataset: CameraDataset,
    **configs,
) -> AbstractTrainer:
    """Build a trainer from a fully resolved :class:`TrainerSpec`."""
    if not isinstance(spec, TrainerSpec):
        raise TypeError("spec must be a TrainerSpec; parse strings with parse_trainer_spec()")
    seen_wrappers = set()
    for name in spec.wrappers:
        if name in seen_wrappers:
            raise ValueError(f"duplicate trainer wrapper {name!r}")
        seen_wrappers.add(name)

    root = TRAINERS.get(spec.root)
    if root is None:
        raise KeyError(
            f"unknown trainer root {spec.root!r}; "
            f"available roots: {sorted(n for n, e in TRAINERS.items() if isinstance(e, TrainerRootEntry))}"
        )
    if not isinstance(root, TrainerRootEntry):
        raise KeyError(
            f"first component {spec.root!r} must be a trainer root; "
            f"available roots: {sorted(n for n, e in TRAINERS.items() if isinstance(e, TrainerRootEntry))}"
        )

    def constructor(model, dataset, **configs):
        return root.construct(spec.root_components, model, dataset, **configs)

    for wrap_name in spec.wrappers:
        entry = TRAINERS.get(wrap_name)
        if entry is None:
            raise KeyError(
                f"unknown trainer wrapper {wrap_name!r}; "
                f"available wrappers: {sorted(n for n, e in TRAINERS.items() if isinstance(e, TrainerWrapEntry))}"
            )
        if not isinstance(entry, TrainerWrapEntry):
            raise KeyError(
                f"{wrap_name!r} must be a trainer wrapper; "
                f"available wrappers: {sorted(n for n, e in TRAINERS.items() if isinstance(e, TrainerWrapEntry))}"
            )
        constructor = partial(entry.fn, constructor)
    return constructor(model, dataset, **configs)
