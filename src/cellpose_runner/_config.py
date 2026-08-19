from typing import Any

from pydantic import BaseModel, ConfigDict

# Fields that select which outputs are written and never reach Cellpose.
_OUTPUT_FIELDS = frozenset({"save_flows", "save_styles"})


class ModelConfig(BaseModel):
    """Parameters for the `CellposeModel(...)` constructor."""

    model_config = ConfigDict(extra="forbid")

    pretrained_model: str = "cpsam"
    gpu: bool = True
    # Cellpose takes a torch.device; a string form is used here so the config
    # stays plain data (serializable to TOML, comparable, hashable). Overrides
    # `gpu` when set, matching cellpose's own precedence.
    device: str | None = None
    # bfloat16 halves the model's memory footprint against float32, at some
    # precision cost. Matches cellpose's own default of on.
    use_bfloat16: bool = True

    def to_init_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for the `CellposeModel` constructor.

        `device` converts from this config's plain string to the `torch.device`
        cellpose's constructor actually takes.
        """
        kwargs = self.model_dump()
        if kwargs["device"] is not None:
            import torch

            kwargs["device"] = torch.device(kwargs["device"])
        return kwargs


class NormalizeConfig(BaseModel):
    """Image normalization parameters, forming `eval()`'s `normalize` dict.

    Field names and defaults match cellpose's own `normalize_default`, so this
    is passed through as-is rather than translated.
    """

    model_config = ConfigDict(extra="forbid")

    # False skips normalization entirely, and makes every field below moot.
    normalize: bool = True
    # Gaussian sigma, in native XY pixels, smoothing the image. Applied per Z
    # slice within the XY plane only -- never across Z -- and before cellpose
    # resamples for `diameter`/`anisotropy`, so this sigma is in the input
    # volume's own pixels. Cellpose recommends 1/10-1/4 of the cell diameter.
    smooth_radius: float = 0.0
    # High-pass surround subtraction, sharpening blurry images. Cellpose
    # recommends 2-3x `smooth_radius` when using both.
    sharpen_radius: float = 0.0
    # Explicit [low, high] intensity bounds, replacing percentile scaling.
    # Cellpose ignores smoothing and sharpening when this is set.
    lowhigh: tuple[float, float] | None = None
    # [low, high] percentiles to scale to 0-1. None means cellpose's (1, 99).
    percentile: tuple[float, float] | None = None
    # Normalize over the whole stack rather than per Z slice. Cellpose forces
    # this True when do_3D is set.
    norm3D: bool = True
    # Window size in pixels for tile-wise normalization, brightening dark
    # regions. 0 is off.
    tile_norm_blocksize: int = 0
    tile_norm_smooth3D: int = 1
    # For cells darker than their background.
    invert: bool = False

    def to_eval_arg(self) -> bool | dict[str, Any]:
        """The value for `eval()`'s `normalize` argument.

        Returns a plain `False` when normalization is off, since the dict form
        always implies it is on. Otherwise passes every field, including those
        left at their default, so the call is fully determined by this config: a
        changed cellpose default shows up as a test failure rather than as
        quietly different segmentations in a run that pins its environment.
        """
        if not self.normalize:
            return False
        return {name: value for name, value in self.model_dump().items() if name != "normalize"}


class PreprocessConfig(BaseModel):
    """Parameters consumed before `CellposeModel`'s network forward pass."""

    model_config = ConfigDict(extra="forbid")

    # Rescales the image to a 30px cell diameter before the forward pass.
    diameter: float | None = None
    normalize: NormalizeConfig = NormalizeConfig()
    # Axis of the input array holding channels.
    channel_axis: int | None = None
    # Axis of the input array holding Z, for volumes that have one.
    z_axis: int | None = None

    def to_eval_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for `CellposeModel.eval()`.

        `normalize` converts from this config's nested `NormalizeConfig` to
        the bool-or-dict shape `eval()` itself takes.
        """
        kwargs = self.model_dump()
        kwargs["normalize"] = self.normalize.to_eval_arg()
        return kwargs


class CellposeConfig(BaseModel):
    """Parameters for one Cellpose segmentation run.

    Fields map onto `CellposeModel` constructor and its `eval()` method, except for
    `save_flows` and `save_styles`, which select what gets written to the run
    directory.
    """

    model_config = ConfigDict(extra="forbid")

    model: ModelConfig = ModelConfig()
    preprocess: PreprocessConfig = PreprocessConfig()

    # CellposeModel.eval(...)
    do_3D: bool = False
    stitch_threshold: float = 0.0
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    # Gaussian sigma smoothing the 3D flow field before masks are followed
    # from it. 0 (cellpose's default) is no smoothing.
    flow3D_smooth: float = 0.0
    anisotropy: float | None = None
    min_size: int = 15
    batch_size: int = 8

    # Output selection
    save_flows: bool = False
    save_styles: bool = False

    def model_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for the `CellposeModel` constructor."""
        return self.model.to_init_kwargs()

    def eval_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for `CellposeModel.eval()`.

        Derived from the model fields so that a newly added eval parameter is
        forwarded without also having to be listed here.
        """
        excluded = {"model", "preprocess"} | _OUTPUT_FIELDS
        kwargs = {
            name: getattr(self, name) for name in type(self).model_fields if name not in excluded
        }
        kwargs.update(self.preprocess.to_eval_kwargs())
        return kwargs
