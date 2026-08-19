import inspect

import torch
from cellpose.models import CellposeModel, normalize_default

from cellpose_runner import CellposeConfig, ModelConfig, NormalizeConfig
from cellpose_runner._config import _OUTPUT_FIELDS


def test_eval_kwargs_covers_every_remaining_field():
    # Asserted against the field list, not a literal, so a field added later is
    # either forwarded or deliberately excluded -- never silently dropped.
    expected = set(CellposeConfig.model_fields) - {"model"} - _OUTPUT_FIELDS
    assert CellposeConfig().eval_kwargs().keys() == expected


def test_eval_kwargs_are_accepted_by_cellpose():
    accepted = set(inspect.signature(CellposeModel.eval).parameters)
    assert CellposeConfig().eval_kwargs().keys() <= accepted


def test_normalize_off_passes_false_not_a_dict():
    # The dict form always implies normalization is on, so "off" has to be the
    # bool -- a dict with normalize=False stripped would silently turn it on.
    config = CellposeConfig(normalize=NormalizeConfig(normalize=False))
    assert config.eval_kwargs()["normalize"] is False


def test_normalize_passes_every_field_including_defaults():
    # Runs pin their environment, so the eval call is fully determined by the
    # config rather than partly by whatever cellpose currently defaults to.
    normalize = CellposeConfig(normalize=NormalizeConfig(smooth_radius=3.0)).eval_kwargs()[
        "normalize"
    ]
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


def test_do_3d_can_be_flipped_on_one_config():
    # One base config carries both modes' parameters; flipping do_3D needs no
    # bookkeeping about which fields to strip.
    base = CellposeConfig(do_3D=True, anisotropy=2.0, stitch_threshold=0.3)
    flipped = base.model_copy(update={"do_3D": False})
    assert flipped.do_3D is False
    assert flipped.anisotropy == 2.0
    assert flipped.stitch_threshold == 0.3
