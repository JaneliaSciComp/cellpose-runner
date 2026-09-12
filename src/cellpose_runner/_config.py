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


class CPDinoModelConfig(BaseModel):
    """Parameters for building and loading a `CPDINO_3D` net (`cellpose3d.utils3d`).

    `three_d_dino` mode's own model config, standing in for `ModelConfig`,
    since `CPDINO_3D` is not `CellposeModel` -- it wraps a DINO ViT patched
    with a 3D-conv input stem. Loading is two separate calls, matching
    `cellpose3d`'s own usage (see its `example_eval3d.ipynb`):
    `CPDINO_3D(...)`'s own constructor loads the 2D backbone (`base_model_path`,
    while its patch-embedding stem is still 2D); the 3D conv stem's own
    trained weights load afterward, via a second `net.load_model(checkpoint_path,
    device)` call once that stem has become the 3D module its checkpoint's
    keys actually match.
    """

    model_config = ConfigDict(extra="forbid")

    # The retrained 3D checkpoint (backbone + 3D conv stem), loaded via a
    # second net.load_model() call after construction. No cache-resolved
    # name to fall back on -- this is always an explicit file path.
    checkpoint_path: str
    # The 2D backbone, loaded by CPDINO_3D's own constructor while its
    # patch-embedding stem is still 2D. None resolves to cellpose's own
    # cached `cpdino` base model, matching CPDINO_3D's own default.
    base_model_path: str | None = None
    gpu: bool = True
    device: str | None = None
    # Architecture knobs, matching CPDINO_3D's own constructor defaults.
    model_name: str = "vitl"
    nout: int = 3
    ps: int = 8
    conv_multi: bool = True
    bsize: int = 256
    rdrop: float = 0.4
    # Z-window depth: how many consecutive slices the network sees per
    # forward pass along whichever axis is "Z" in a given orthogonal view.
    wsize: int = 25

    def to_init_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for the `CPDINO_3D` constructor.

        Excludes `checkpoint_path`, which isn't a constructor argument -- see
        this class's own docstring. `device` resolves the same
        string-to-`torch.device` conversion as `ModelConfig.to_init_kwargs`,
        falling back to `gpu` when unset.
        """
        import torch

        kwargs = self.model_dump(exclude={"gpu", "checkpoint_path"})
        kwargs["model_path"] = kwargs.pop("base_model_path")
        if kwargs["device"] is not None:
            kwargs["device"] = torch.device(kwargs["device"])
        else:
            kwargs["device"] = torch.device("cuda" if self.gpu else "cpu")
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


class ThreeDDinoInferenceConfig(BaseModel):
    """Parameters for `CPDINO_3D`'s per-view forward pass (`run_net_3d`).

    Deliberately not a `PostprocessConfig`/`InferenceConfig` subclass: those
    model `CellposeModel.eval()`'s parameters, none of which `eval_3d` takes.
    `bsize`/`tile_overlap` here are `run_net_3d`'s own XY tiling, independent
    of `CPDinoModelConfig.bsize` (that one sizes `CPDINO_3D`'s patch stem).
    """

    model_config = ConfigDict(extra="forbid")

    batch_size: int = 8
    bsize: int = 256
    tile_overlap: float = 0.1
    # e.g. 2.0 when Z is sampled half as densely as X or Y. None applies no
    # rescaling. Unlike upstream cellpose3d's eval_3d, which resizes Z by the
    # same diameter-derived scale as XY with no anisotropy correction at all,
    # this field lets Z be rescaled independently before the forward pass --
    # same semantics and formula as ThreeDFlowsInferenceConfig.anisotropy.
    anisotropy: float | None = None


class ThreeDDinoPostprocessConfig(PostprocessConfig):
    """`PostprocessConfig` for the `three_d_dino` mode, adding its one postprocessing field.

    `min_size`/`max_size_fraction`/`niter` are inherited from `PostprocessConfig`
    unchanged -- upstream `eval_3d` calls `dynamics.compute_masks` with those
    hardcoded (no `min_size`/`max_size_fraction` at all, `niter=1000`), but
    nothing about `compute_masks` itself makes them DINO-specific (see
    `_view_consolidate.consolidate`'s `dino=True` path), so `niter`'s default
    here is `1000` to match `eval_3d`'s own hardcoded value rather than
    `PostprocessConfig`'s `None`. `flow_threshold` is inherited too but is a
    no-op in 3D regardless of mode -- `dynamics.compute_masks` only applies it
    when `do_3D=False`.
    """

    niter: int | None = 1000
    # Gaussian sigma smoothing the fused 3D flow field before masks are
    # computed. 0 is no smoothing. Named to match ThreeDFlowsPostprocessConfig's
    # field of the same meaning, though the two modes compute their flow
    # fields independently.
    flow3D_smooth: float = 1.0


_MODE_MODEL: dict[str, type[BaseModel]] = {
    "two_d": ModelConfig,
    "stitch": ModelConfig,
    "three_d_flows": ModelConfig,
    "three_d_dino": CPDinoModelConfig,
}
_MODE_INFERENCE: dict[str, type[BaseModel]] = {
    "two_d": InferenceConfig,
    "stitch": InferenceConfig,
    "three_d_flows": ThreeDFlowsInferenceConfig,
    "three_d_dino": ThreeDDinoInferenceConfig,
}
_MODE_POSTPROCESS: dict[str, type[BaseModel]] = {
    "two_d": PostprocessConfig,
    "stitch": StitchPostprocessConfig,
    "three_d_flows": ThreeDFlowsPostprocessConfig,
    "three_d_dino": ThreeDDinoPostprocessConfig,
}
_MODE_TO_DO_3D: dict[str, bool] = {
    "two_d": False,
    "stitch": False,
    "three_d_flows": True,
    "three_d_dino": True,
}
# Modes not built on CellposeModel at all -- model_kwargs()/eval_kwargs()
# raise for these rather than returning kwargs for a call that never happens.
_NON_CELLPOSE_MODEL_MODES = frozenset({"three_d_dino"})


class CellposeConfig(BaseModel):
    """Parameters for one Cellpose segmentation run.

    Fields map onto `CellposeModel`'s constructor and its `eval()` method,
    except for `save_flows` and `save_styles`, which select what gets written
    to the run directory, and except for `mode="three_d_dino"`, which maps
    onto `CPDINO_3D`/`eval_3d` (`cellpose3d`) instead -- a structurally
    different model and call path, not a `CellposeModel.eval()` variant. Every
    mode's `model`/`inference`/`postprocess` subclass triple is constrained
    together, so a mode-specific field (e.g. `anisotropy`, or `three_d_dino`'s
    `checkpoint_path`) can't be set for a mode it does nothing in.
    """

    model_config = ConfigDict(extra="forbid")

    mode: Literal["two_d", "stitch", "three_d_flows", "three_d_dino"] = "two_d"
    model: ModelConfig | CPDinoModelConfig = ModelConfig()
    preprocess: PreprocessConfig = PreprocessConfig()
    inference: InferenceConfig | ThreeDFlowsInferenceConfig | ThreeDDinoInferenceConfig = (
        InferenceConfig()
    )
    postprocess: (
        PostprocessConfig
        | StitchPostprocessConfig
        | ThreeDFlowsPostprocessConfig
        | ThreeDDinoPostprocessConfig
    ) = PostprocessConfig()

    # Output selection
    save_flows: bool = False
    save_styles: bool = False

    @model_validator(mode="before")
    @classmethod
    def _parse_stage_configs_for_mode(cls, data: Any) -> Any:
        """Parse `model`/`inference`/`postprocess` dicts against `mode`'s own subclass.

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
        if mode in _MODE_MODEL and isinstance(data.get("model", {}), dict):
            data = {**data, "model": _MODE_MODEL[mode](**data.get("model", {}))}
        if mode in _MODE_INFERENCE and isinstance(data.get("inference", {}), dict):
            data = {**data, "inference": _MODE_INFERENCE[mode](**data.get("inference", {}))}
        if mode in _MODE_POSTPROCESS and isinstance(data.get("postprocess", {}), dict):
            data = {**data, "postprocess": _MODE_POSTPROCESS[mode](**data.get("postprocess", {}))}
        return data

    @model_validator(mode="after")
    def _check_mode_matches_stage_configs(self) -> "CellposeConfig":
        expected_model = _MODE_MODEL[self.mode]
        expected_inference = _MODE_INFERENCE[self.mode]
        expected_postprocess = _MODE_POSTPROCESS[self.mode]
        if type(self.model) is not expected_model:
            raise ValueError(
                f"mode={self.mode!r} requires model={expected_model.__name__}, "
                f"got {type(self.model).__name__}."
            )
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
        """Keyword arguments for the `CellposeModel` constructor.

        Raises:
            TypeError: If `mode` doesn't build a `CellposeModel` at all (e.g.
                `three_d_dino`, which builds `CPDINO_3D` instead -- see
                `cellpose3d`'s own model-building code for that mode).
        """
        if self.mode in _NON_CELLPOSE_MODEL_MODES:
            raise TypeError(
                f"mode={self.mode!r} does not build a CellposeModel; "
                "there is no model_kwargs() for it."
            )
        return self.model.to_init_kwargs()

    def eval_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for `CellposeModel.eval()`.

        Merges every stage config's fields (preprocess, inference,
        postprocess) unconditionally, then translates `mode` into cellpose's
        own `do_3D` flag at this boundary -- `mode` doesn't exist in
        cellpose's own API, so it is never passed through itself.

        Raises:
            TypeError: If `mode` doesn't call `CellposeModel.eval()` at all
                (e.g. `three_d_dino`) -- see `model_kwargs()`.
        """
        if self.mode in _NON_CELLPOSE_MODEL_MODES:
            raise TypeError(
                f"mode={self.mode!r} does not call CellposeModel.eval(); "
                "there is no eval_kwargs() for it."
            )
        kwargs = self.preprocess.to_eval_kwargs()
        kwargs.update(self.inference.model_dump())
        kwargs.update(self.postprocess.model_dump())
        kwargs["do_3D"] = _MODE_TO_DO_3D[self.mode]
        # stitch_threshold only exists as a field on StitchPostprocessConfig;
        # every other mode's postprocess config has none, so cellpose's own
        # default (0.0, meaning "off") is correct without an explicit else.
        kwargs.setdefault("stitch_threshold", 0.0)
        return kwargs
