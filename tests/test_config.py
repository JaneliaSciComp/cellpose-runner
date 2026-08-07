import inspect

from cellpose.models import CellposeModel

from cellpose_runner import CellposeConfig
from cellpose_runner._config import _MODEL_FIELDS, _OUTPUT_FIELDS


def test_eval_kwargs_covers_every_remaining_field():
    # Asserted against the field list, not a literal, so a field added later is
    # either forwarded or deliberately excluded -- never silently dropped.
    expected = set(CellposeConfig.model_fields) - _MODEL_FIELDS - _OUTPUT_FIELDS
    assert CellposeConfig().eval_kwargs().keys() == expected


def test_eval_kwargs_are_accepted_by_cellpose():
    accepted = set(inspect.signature(CellposeModel.eval).parameters)
    assert CellposeConfig().eval_kwargs().keys() <= accepted


def test_model_kwargs_are_accepted_by_cellpose():
    accepted = set(inspect.signature(CellposeModel.__init__).parameters)
    assert CellposeConfig().model_kwargs().keys() <= accepted


def test_do_3d_can_be_flipped_on_one_config():
    # One base config carries both modes' parameters; flipping do_3D needs no
    # bookkeeping about which fields to strip.
    base = CellposeConfig(do_3D=True, anisotropy=2.0, stitch_threshold=0.3)
    flipped = base.model_copy(update={"do_3D": False})
    assert flipped.do_3D is False
    assert flipped.anisotropy == 2.0
    assert flipped.stitch_threshold == 0.3
