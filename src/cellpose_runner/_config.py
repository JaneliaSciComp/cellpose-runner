from typing import Any

from pydantic import BaseModel, ConfigDict

# Fields that configure CellposeModel(...) rather than CellposeModel.eval(...).
_MODEL_FIELDS = frozenset({"pretrained_model", "gpu"})

# Fields that select which outputs are written and never reach Cellpose.
_OUTPUT_FIELDS = frozenset({"save_flows", "save_styles"})


class CellposeConfig(BaseModel):
    """Parameters for one Cellpose segmentation run.

    Fields map onto `CellposeModel` constructor and its `eval()` method, except for
    `save_flows` and `save_styles`, which select what gets written to the run
    directory.
    """

    model_config = ConfigDict(extra="forbid")

    # CellposeModel(...)
    pretrained_model: str = "cpsam"
    gpu: bool = True

    # CellposeModel.eval(...)
    diameter: float | None = None
    do_3D: bool = False
    stitch_threshold: float = 0.0
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    # Gaussian sigma smoothing the 3D flow field before masks are followed
    # from it. 0 (cellpose's default) is no smoothing.
    flow3D_smooth: float = 0.0
    normalize: bool = True
    anisotropy: float | None = None
    min_size: int = 15
    batch_size: int = 8

    # Output selection
    save_flows: bool = False
    save_styles: bool = False

    def model_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for the `CellposeModel` constructor."""
        return {name: getattr(self, name) for name in sorted(_MODEL_FIELDS)}

    def eval_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for `CellposeModel.eval()`.

        Derived from the model fields so that a newly added eval parameter is
        forwarded without also having to be listed here.
        """
        excluded = _MODEL_FIELDS | _OUTPUT_FIELDS
        return {
            name: getattr(self, name) for name in type(self).model_fields if name not in excluded
        }
