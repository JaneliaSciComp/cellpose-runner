import inspect

import pytest
import torch
from cellpose.models import CellposeModel, normalize_default
from pydantic import ValidationError

from cellpose_runner import (
    CellposeConfig,
    CPDinoModelConfig,
    ModelConfig,
    NormalizeConfig,
    StitchPostprocessConfig,
    ThreeDDinoInferenceConfig,
    ThreeDDinoPostprocessConfig,
    ThreeDFlowsInferenceConfig,
    ThreeDFlowsPostprocessConfig,
)
from cellpose_runner._config import PreprocessConfig

# Modes whose eval_kwargs()/model_kwargs() call CellposeModel -- three_d_dino
# builds CPDINO_3D instead, so it's excluded from every test below that pins
# against CellposeModel's own signature or eval_kwargs()'s CellposeModel contract.
_MODE_CONFIGS = {
    "two_d": CellposeConfig(),
    "stitch": CellposeConfig(
        mode="stitch", postprocess=StitchPostprocessConfig(stitch_threshold=0.3)
    ),
    "three_d_flows": CellposeConfig(
        mode="three_d_flows",
        inference=ThreeDFlowsInferenceConfig(anisotropy=2.0),
        postprocess=ThreeDFlowsPostprocessConfig(flow3D_smooth=1.0),
    ),
}

_three_d_dino_config = CellposeConfig(
    mode="three_d_dino",
    model=CPDinoModelConfig(model_path="/fake/cpdino3d.pt"),
    inference=ThreeDDinoInferenceConfig(),
    postprocess=ThreeDDinoPostprocessConfig(),
)


@pytest.mark.parametrize("config", _MODE_CONFIGS.values(), ids=_MODE_CONFIGS.keys())
def test_eval_kwargs_covers_every_remaining_field(config):
    # Asserted against the field list, not a literal, so a field added later is
    # either forwarded or deliberately excluded -- never silently dropped.
    expected = set(PreprocessConfig.model_fields)
    expected |= set(type(config.inference).model_fields)
    expected |= set(type(config.postprocess).model_fields)
    expected |= {"do_3D", "stitch_threshold"}
    assert config.eval_kwargs().keys() == expected


@pytest.mark.parametrize("config", _MODE_CONFIGS.values(), ids=_MODE_CONFIGS.keys())
def test_eval_kwargs_are_accepted_by_cellpose(config):
    accepted = set(inspect.signature(CellposeModel.eval).parameters)
    assert config.eval_kwargs().keys() <= accepted


def test_normalize_off_passes_false_not_a_dict():
    # The dict form always implies normalization is on, so "off" has to be the
    # bool -- a dict with normalize=False stripped would silently turn it on.
    config = CellposeConfig(preprocess=PreprocessConfig(normalize=NormalizeConfig(normalize=False)))
    assert config.eval_kwargs()["normalize"] is False


def test_normalize_passes_every_field_including_defaults():
    # Runs pin their environment, so the eval call is fully determined by the
    # config rather than partly by whatever cellpose currently defaults to.
    config = CellposeConfig(
        preprocess=PreprocessConfig(normalize=NormalizeConfig(smooth_radius=3.0))
    )
    normalize = config.eval_kwargs()["normalize"]
    assert normalize["smooth_radius"] == 3.0
    # `normalize` itself is the on/off switch, expressed by the dict's presence.
    assert normalize.keys() == set(NormalizeConfig.model_fields) - {"normalize"}


def test_normalize_matches_cellpose_defaults():
    # NormalizeConfig restates cellpose's normalize_default so it can be passed
    # through untranslated. Names AND values are compared, because passing every
    # field means a changed upstream default silently changes what runs do --
    # this test is what turns that into a visible failure.
    ours = {name: field.default for name, field in NormalizeConfig.model_fields.items()}
    theirs = dict(normalize_default)
    # cellpose spells these as lists; tuples are the immutable equivalent.
    ours = {k: list(v) if isinstance(v, tuple) else v for k, v in ours.items()}
    assert ours == theirs


def test_model_kwargs_are_accepted_by_cellpose():
    accepted = set(inspect.signature(CellposeModel.__init__).parameters)
    assert CellposeConfig().model_kwargs().keys() <= accepted


def test_model_kwargs_come_from_nested_model_config():
    config = CellposeConfig(model=ModelConfig(pretrained_model="livecell", gpu=False))
    kwargs = config.model_kwargs()
    assert kwargs["pretrained_model"] == "livecell"
    assert kwargs["gpu"] is False


def test_model_kwargs_converts_device_string_to_torch_device():
    config = CellposeConfig(model=ModelConfig(device="cpu"))
    assert config.model_kwargs()["device"] == torch.device("cpu")


def test_model_kwargs_device_defaults_to_none():
    assert CellposeConfig().model_kwargs()["device"] is None


def test_switching_mode_uses_a_distinct_inference_postprocess_pair():
    # Modes aren't one shared config with a flipped bool any more -- each
    # mode's extra fields live on that mode's own subclass pair.
    three_d_flows = _MODE_CONFIGS["three_d_flows"]
    stitched = _MODE_CONFIGS["stitch"]

    assert three_d_flows.eval_kwargs()["do_3D"] is True
    assert three_d_flows.eval_kwargs()["anisotropy"] == 2.0
    assert three_d_flows.eval_kwargs()["flow3D_smooth"] == 1.0
    # 3D-flows mode's postprocess has no stitch_threshold field at all.
    assert stitched.eval_kwargs()["do_3D"] is False
    assert stitched.eval_kwargs()["stitch_threshold"] == 0.3


def test_mismatched_mode_and_stage_config_is_rejected():
    # The mode/stage-pair validator is what prevents constructing the
    # nonsensical states do_3D-gating used to allow implicitly, e.g.
    # 3D-flows inference paired with stitch postprocessing.
    with pytest.raises(ValidationError, match="mode"):
        CellposeConfig(
            mode="three_d_flows",
            inference=ThreeDFlowsInferenceConfig(anisotropy=2.0),
            postprocess=StitchPostprocessConfig(stitch_threshold=0.3),
        )


def test_mismatched_mode_and_model_config_is_rejected():
    # three_d_dino requires CPDinoModelConfig, not CellposeModel's ModelConfig
    # -- the two aren't structurally compatible (no pretrained_model, but a
    # required model_path).
    with pytest.raises(ValidationError, match="mode"):
        CellposeConfig(
            mode="three_d_dino",
            model=ModelConfig(),
            inference=ThreeDDinoInferenceConfig(),
            postprocess=ThreeDDinoPostprocessConfig(),
        )


def test_three_d_dino_requires_model_path():
    # CPDinoModelConfig.model_path has no default -- unlike ModelConfig's
    # pretrained_model, there is no cache-resolved name to fall back on.
    with pytest.raises(ValidationError):
        CPDinoModelConfig()


def test_three_d_dino_eval_kwargs_not_supported():
    # three_d_dino never calls CellposeModel.eval() -- eval_kwargs() must say
    # so rather than silently returning kwargs for a call that never happens.
    with pytest.raises(TypeError, match="three_d_dino"):
        _three_d_dino_config.eval_kwargs()


def test_three_d_dino_model_kwargs_not_supported():
    with pytest.raises(TypeError, match="three_d_dino"):
        _three_d_dino_config.model_kwargs()


def test_cpdino_model_config_to_init_kwargs_converts_device_string():
    config = CPDinoModelConfig(model_path="/fake/cpdino3d.pt", device="cpu")
    assert config.to_init_kwargs()["device"] == torch.device("cpu")


def test_cpdino_model_config_to_init_kwargs_falls_back_to_gpu():
    config = CPDinoModelConfig(model_path="/fake/cpdino3d.pt", gpu=False)
    assert config.to_init_kwargs()["device"] == torch.device("cpu")
    config = CPDinoModelConfig(model_path="/fake/cpdino3d.pt", gpu=True)
    assert config.to_init_kwargs()["device"] == torch.device("cuda")
