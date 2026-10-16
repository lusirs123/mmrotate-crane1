"""Regression checks for the amended evaluation protocol, not model training."""
from copy import deepcopy
import pytest
from crane_project.tools import eval_port_geometry_size_distribution_val_v2 as tool


def results():
    return [dict(epoch=e,selection_key=[abs(e-7),.1,e],train_pass_diagnostic_only=False,
        legacy_val_only=dict(val_pass=e in (2,9))) for e in range(1,25)]


def test_all_train_failures_still_participate_and_adoption_is_separate():
    chosen=tool.choose(results())
    assert chosen['representative']['epoch']==7
    assert chosen['legacy_val_only_selected']['epoch']==9
    assert chosen['legacy_val_only_passing_epochs']==[2,9]
    values=results()
    for r in values: r['legacy_val_only']['val_pass']=False
    assert tool.choose(values)['representative']['epoch']==7
    assert tool.choose(values)['legacy_val_only_selected'] is None


@pytest.mark.parametrize('values',[results()[:-1],results()[:-1]+[results()[0]]])
def test_incomplete_or_duplicate_epochs_cannot_be_called_complete(values):
    with pytest.raises(ValueError,match='all 24'): tool.choose(values)


@pytest.mark.parametrize('field',['arm','epoch','proof','head_digest'])
def test_wrong_checkpoint_binding_rejected(field):
    payload=dict(protocol=tool.old.VERSION,arm='distribution',epoch=7,updates=7*639,
        proof={'sha':'original'},head_digest={'parameters':'sealed'})
    metadata=dict(checkpoint=dict(head_digest=deepcopy(payload['head_digest'])))
    bad=deepcopy(payload);bad[field]='wrong'
    with pytest.raises(ValueError,match='differs'):
        tool.check_checkpoint_metadata(bad,metadata,'distribution',7,{'sha':'original'})


def test_replay_uses_numeric_tolerance_but_rejects_denominator_and_nan():
    tool.compare_numeric({'n':887,'x':.5},{'n':887,'x':.5+1e-14})
    for bad in ({'n':886,'x':.5},{'n':887,'x':float('nan')}):
        with pytest.raises(ValueError): tool.compare_numeric({'n':887,'x':.5},bad)
