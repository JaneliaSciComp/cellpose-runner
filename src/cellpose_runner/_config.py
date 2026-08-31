from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator


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


class InferenceConfig(BaseModel):
    """Parameters for `CellposeModel.eval()`'s network forward pass (`_run_net`).

    Common to every segmentation mode. `anisotropy` (3D-flows only) is on
    `ThreeDFlowsInferenceConfig`, not here, since it only resizes the volume
    when `do_3D` is set.
    """

    model_config = ConfigDict(extra="forbid")

    batch_size: int = 8
    resample: bool = True
    augment: bool = False
    tile_overlap: float = 0.1
    # None lets cellpose pick its own default tile size.
    bsize: int | None = None


class ThreeDFlowsInferenceConfig(InferenceConfig):
    """`InferenceConfig` for the 3D-flows mode, adding its one inference-stage field.

    `anisotropy` resizes the volume before the forward pass so a
    coarser-sampled Z axis is treated at the same physical scale as XY --
    only meaningful when the network is run in 3D-flows mode (`do_3D=True`).
    """

    # e.g. 2.0 when Z is sampled half as densely as X or Y. None (cellpose's
    # default) applies no rescaling.
    anisotropy: float | None = None


class PostprocessConfig(BaseModel):
    """Parameters for mask computation after the network forward pass (`_compute_masks`).

    Common to every segmentation mode. `stitch_threshold` (stitch mode) and
    `flow3D_smooth` (3D-flows mode) are on their own subclasses below, since
    each only does anything in its one mode.
    """

    model_config = ConfigDict(extra="forbid")

    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    min_size: int = 15
    # Masks larger than this fraction of the image are discarded as
    # (likely) merged/background artifacts.
    max_size_fraction: float = 0.4
    # None lets cellpose pick its own default number of dynamics iterations.
    niter: int | None = None


class StitchPostprocessConfig(PostprocessConfig):
    """`PostprocessConfig` for the stitch mode, adding its one postprocessing field.

    `stitch_threshold` stitches per-plane 2D masks into 3D (`utils.stitch3D`)
    by IoU overlap between adjacent planes -- the network itself never sees a
    3D flow field in this mode.
    """

    stitch_threshold: float = 0.0


class ThreeDFlowsPostprocessConfig(PostprocessConfig):
    """`PostprocessConfig` for the 3D-flows mode, adding its one postprocessing field.

    `flow3D_smooth` gaussian-smooths the 3D flow field before masks are
    followed from it, before `_compute_masks` runs. 0 (cellpose's default) is
    no smoothing.
    """

    flow3D_smooth: float = 0.0


_MODE_INFERENCE: dict[str, type[InferenceConfig]] = {
    "two_d": InferenceConfig,
    "stitch": InferenceConfig,
    "three_d_flows": ThreeDFlowsInferenceConfig,
}
_MODE_POSTPROCESS: dict[str, type[PostprocessConfig]] = {
    "two_d": PostprocessConfig,
    "stitch": StitchPostprocessConfig,
    "three_d_flows": ThreeDFlowsPostprocessConfig,
}
_MODE_TO_DO_3D: dict[str, bool] = {
    "two_d": False,
    "stitch": False,
    "three_d_flows": True,
}


class CellposeConfig(BaseModel):
    """Parameters for one Cellpose segmentation run.

    Fields map onto `CellposeModel`'s constructor and its `eval()` method,
    except for `save_flows` and `save_styles`, which select what gets written
    to the run directory. `mode` selects one of cellpose's three genuinely
    different segmentation algorithms (2D, stitch, 3D-flows -- see
    `scratch/config-reorg-plan.md`) and constrains which `inference`/
    `postprocess` subclass pairs with it, so a mode-specific field (e.g.
    `anisotropy`) can't be set for a mode it does nothing in.
    """

    model_config = ConfigDict(extra="forbid")

    mode: Literal["two_d", "stitch", "three_d_flows"] = "two_d"
    model: ModelConfig = ModelConfig()
    preprocess: PreprocessConfig = PreprocessConfig()
    inference: InferenceConfig | ThreeDFlowsInferenceConfig = InferenceConfig()
    postprocess: PostprocessConfig | StitchPostprocessConfig | ThreeDFlowsPostprocessConfig = (
        PostprocessConfig()
    )

    # Output selection
    save_flows: bool = False
    save_styles: bool = False

    @model_validator(mode="before")
    @classmethod
    def _parse_stage_configs_for_mode(cls, data: Any) -> Any:
        """Parse `inference`/`postprocess` dicts against `mode`'s own subclass.

        A dict lacking a mode-specific field (e.g. `anisotropy` omitted
        because `_without_nones` stripped it before writing TOML) still
        structurally validates against the plain base config, so pydantic's
        smart-union would silently pick the base class over the mode's own
        subclass -- the wrong type, but not a type error. Parsing explicitly
        against `mode`'s subclass here, before that union resolution runs,
        makes the round trip exact regardless of which fields a dict omits.
        """
        if not isinstance(data, dict) or "mode" not in data:
            return data
        mode = data["mode"]
        if isinstance(data.get("inference"), dict) and mode in _MODE_INFERENCE:
            data = {**data, "inference": _MODE_INFERENCE[mode](**data["inference"])}
        if isinstance(data.get("postprocess"), dict) and mode in _MODE_POSTPROCESS:
            data = {**data, "postprocess": _MODE_POSTPROCESS[mode](**data["postprocess"])}
        return data

    @model_validator(mode="after")
    def _check_mode_matches_stage_configs(self) -> "CellposeConfig":
        expected_inference = _MODE_INFERENCE[self.mode]
        expected_postprocess = _MODE_POSTPROCESS[self.mode]
        if type(self.inference) is not expected_inference:
            raise ValueError(
                f"mode={self.mode!r} requires inference={expected_inference.__name__}, "
                f"got {type(self.inference).__name__}."
            )
        if type(self.postprocess) is not expected_postprocess:
            raise ValueError(
                f"mode={self.mode!r} requires postprocess={expected_postprocess.__name__}, "
                f"got {type(self.postprocess).__name__}."
            )
        return self

    def model_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for the `CellposeModel` constructor."""
        return self.model.to_init_kwargs()

    def eval_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for `CellposeModel.eval()`.

        Merges every stage config's fields (preprocess, inference,
        postprocess) unconditionally, then translates `mode` into cellpose's
        own `do_3D` flag at this boundary -- `mode` doesn't exist in
        cellpose's own API, so it is never passed through itself.
        """
        kwargs = self.preprocess.to_eval_kwargs()
        kwargs.update(self.inference.model_dump())
        kwargs.update(self.postprocess.model_dump())
        kwargs["do_3D"] = _MODE_TO_DO_3D[self.mode]
        # stitch_threshold only exists as a field on StitchPostprocessConfig;
        # every other mode's postprocess config has none, so cellpose's own
        # default (0.0, meaning "off") is correct without an explicit else.
        kwargs.setdefault("stitch_threshold", 0.0)
        return kwargs
