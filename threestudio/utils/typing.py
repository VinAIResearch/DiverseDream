# Basic types
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    NamedTuple,
    NewType,
    Optional,
    Sized,
    Tuple,
    Type,
    TypeVar,
    Union,
)

# for jaxtyping usage, see https://github.com/google/jaxtyping/blob/main/API.md
from jaxtyping import Bool, Complex, Float, Inexact, Int, Integer, Num, Shaped, UInt

# Config type
from omegaconf import DictConfig
from torch import Tensor

# Runtime type checking decorator
from typeguard import typechecked as typechecker


dummy = (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    NamedTuple,
    NewType,
    Optional,
    Sized,
    Tuple,
    Type,
    TypeVar,
    Union,
    DictConfig,
    Tensor,
    typechecker,
    Bool,
    Complex,
    Float,
    Inexact,
    Int,
    Integer,
    Num,
    Shaped,
    UInt,
)
