import re
from dataclasses import dataclass


NAME_PATTERN = r"[A-Za-z0-9_]+"
NAME_RE = re.compile(NAME_PATTERN)
ROOT_RE = re.compile(rf"(?P<name>{NAME_PATTERN})(?:\s*\((?P<components>[^()]*)\))?")


@dataclass(frozen=True)
class TrainerSpec:
    root: str
    root_components: tuple[str, ...] = ()
    wrappers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.root, str):
            raise TypeError("root name must be a string")
        if NAME_RE.fullmatch(self.root) is None:
            raise ValueError(f"root name {self.root!r} must match {NAME_PATTERN}")
        if not isinstance(self.root_components, tuple):
            raise TypeError("root components must be a tuple")
        for component in self.root_components:
            if not isinstance(component, str):
                raise TypeError("root component must be a string")
            if NAME_RE.fullmatch(component) is None:
                raise ValueError(f"root component {component!r} must match {NAME_PATTERN}")
        if not isinstance(self.wrappers, tuple):
            raise TypeError("wrappers must be a tuple")
        for wrapper in self.wrappers:
            if not isinstance(wrapper, str):
                raise TypeError("wrapper name must be a string")
            if NAME_RE.fullmatch(wrapper) is None:
                raise ValueError(f"wrapper name {wrapper!r} must match {NAME_PATTERN}")


def parse_trainer_spec(source: str) -> TrainerSpec:
    """Parse ``root(components,...) | wrapper | wrapper`` into a :class:`TrainerSpec`."""
    if not isinstance(source, str):
        raise TypeError("trainer specification must be a string")
    if not source.strip():
        raise ValueError("trainer specification must not be empty")

    components = source.split("|")
    for index, component in enumerate(components, start=1):
        if not component.strip():
            raise ValueError(
                f"component {index} is empty in trainer specification {source!r}"
            )

    root_source, *wrapper_sources = (component.strip() for component in components)
    root_match = ROOT_RE.fullmatch(root_source)
    if root_match is None:
        raise ValueError(
            f"invalid root {root_source!r}; expected NAME or NAME(arg1, arg2)"
        )
    components_source = root_match.group("components")
    if components_source is None or not components_source.strip():
        root_components = ()
    else:
        raw_components = components_source.split(",")
        if any(not component.strip() for component in raw_components):
            raise ValueError(
                f"invalid components in root {root_source!r}; expected comma-separated names"
            )
        root_components = tuple(component.strip() for component in raw_components)

    wrappers = []
    for wrapper_source in wrapper_sources:
        if NAME_RE.fullmatch(wrapper_source) is None:
            raise ValueError(
                f"invalid wrapper {wrapper_source!r}; expected {NAME_PATTERN}"
            )
        wrappers.append(wrapper_source)

    return TrainerSpec(
        root=root_match.group("name"),
        root_components=root_components,
        wrappers=tuple(wrappers),
    )
