from pathlib import Path
import importlib.util
import pytest

spec = importlib.util.spec_from_file_location('audit_sim_val', Path(__file__).resolve().parents[1] / 'crane_project/tools/audit_sim_val_epochs_v1.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_gt_and_actual_export_format(tmp_path):
    gt = tmp_path / 'gt.txt'
    pred = tmp_path / 'pred.txt'
    gt.write_text('0 0 20 0 20 10 0 10 grab 0\n')
    pred.write_text('0.00 0.00 20.00 0.00 20.00 10.00 0.00 10.00 0.9500\n')
    g = audit.read_boxes(gt)[0]
    p = audit.read_boxes(pred, prediction=True)[0]
    assert audit.compute_riou(g, p) == pytest.approx(1)
    with pytest.raises(ValueError):
        audit.read_boxes(pred)
    with pytest.raises(ValueError):
        audit.read_boxes(gt, prediction=True)


@pytest.mark.parametrize('tail', ['nan', '1.1', '-0.1', 'grab', '0.9 extra'])
def test_invalid_score(tmp_path, tail):
    p = tmp_path / 'pred.txt'
    p.write_text('0 0 20 0 20 10 0 10 ' + tail)
    with pytest.raises(ValueError):
        audit.read_boxes(p, prediction=True)


def test_empty_prediction(tmp_path):
    p = tmp_path / 'pred.txt'
    p.write_text('')
    assert audit.read_boxes(p, prediction=True) == []
    assert audit.summarize([1, None])['mean_riou_all_frames'] == .5
