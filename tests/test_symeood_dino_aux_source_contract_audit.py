import json
from pathlib import Path

from crane_project.tools import symeood_dino_aux_source_contract_audit as audit


def _write_frame(root, frame):
    stem = 'real_seq11_{:05d}'.format(frame)
    (root / 'images' / (stem + '.jpg')).write_bytes(b'jpg')
    (root / 'annfiles' / (stem + '.txt')).write_text(
        '8 18 12 18 12 22 8 22 grab 0\n')
    return stem


def test_aux_contract_ignores_appledouble_and_preserves_true_adjacency(
        tmp_path):
    split = tmp_path / 'extra_source_real_seq11_pilot_k1p9'
    (split / 'images').mkdir(parents=True)
    (split / 'annfiles').mkdir()
    stems = [_write_frame(split, frame) for frame in (10, 11)]
    (split / 'images' / ('._' + stems[0] + '.jpg')).write_bytes(b'sidecar')
    (split / 'annfiles' / ('._' + stems[0] + '.txt')).write_bytes(b'sidecar')
    records = [dict(
        filename=str(split / 'images' / (stem + '.jpg')),
        sequence='real_seq11', frame=frame, dino_invoked=True,
        dino_native_box=[10., 20., 4., 4., 0., 0.9],
        sym_eood_box=[10., 20., 4., 4., 0., 0.8])
        for stem, frame in zip(stems, (10, 11))]
    all_lane = tmp_path / 'audit.json'
    all_lane.write_text(json.dumps(dict(
        protocol='source_owned_geometry_union_v2', records=records)))

    result = audit.audit_payload(
        all_lane, tmp_path, split.name, expected_frame_count=2)

    assert result['eligible_for_auxiliary_source_training'] is True
    assert result['visible_image_count'] == 2
    assert result['visible_annotation_count'] == 2
    assert result['ignored_appledouble_image_count'] == 1
    assert result['ignored_appledouble_annotation_count'] == 1
    assert result['dino_summary'] == {'present': 2, 'hit': 2}
    assert result['adjacent_pair_count'] == 1
    assert result['cached_sym_identity'] == (
        'unverified_not_formal_k1_evidence')
    assert result['eligible_for_router_claim'] is False
    assert result['eligible_for_independent_sequence_claim'] is False


def test_crane_dataset_source_filters_appledouble_annotations():
    path = Path(__file__).resolve().parents[1] / (
        'mmrotate/datasets/crane_custom_dota.py')
    text = path.read_text()
    assert "os.path.basename(path).startswith('._')" in text

