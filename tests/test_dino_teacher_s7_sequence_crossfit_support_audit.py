import json

from crane_project.tools import dino_teacher_s7_sequence_crossfit_support_audit as audit


def _row(split, seq, domain, frame, native_correct, s7_correct):
    return dict(
        split=split, seq=seq, domain=domain, frame=frame,
        frame_key='{}|{}|{}'.format(split, seq, frame),
        views=dict(clean=dict(
            eligible=True, native_correct=native_correct,
            s7_correct_count=int(s7_correct))))


def _protocol31_payload():
    train = [
        _row('train', 'real_seq01', 'real', 0, True, 0),
        _row('train_sim', 'sim_seq08', 'sim', 0, False, 1),
        _row('train_sim', 'sim_seq08', 'sim', 1, False, 1),
        _row('train_sim', 'sim_seq08', 'sim', 2, False, 1),
        _row('train_sim', 'sim_seq08', 'sim', 3, False, 1),
        _row('train_sim', 'sim_seq08', 'sim', 4, False, 1),
        _row('train_sim', 'sim_seq08', 'sim', 5, False, 1),
        _row('train_sim', 'sim_seq08', 'sim', 6, False, 1),
    ]
    val = [
        _row('val', 'real_seq07', 'real', index, False, 1)
        for index in range(9)] + [
        _row('val', 'sim_seq10', 'sim', index, False, 1)
        for index in range(16)]
    return dict(
        protocol_version=31, target_dev=None, parameter_update_count=0,
        protocol=dict(target_read=False,
                      full_dino_rpn_roi_forward_per_view=True,
                      no_feature_tensor_augmentation=True),
        isolation=dict(parameter_updates_performed=False),
        source=dict(clean_native_reproduction_gate=dict(passed=True)),
        train_frame_rows=train, validation_clean_frame_rows=val)


def _args(tmp_path):
    class Args:
        min_train_gain_frames = 16
        min_train_gain_sequences = 2
        min_heldout_gain_frames = 7
        min_valid_folds = 2
        min_heldout_real_hard_sequences = 1
        min_heldout_sim_hard_sequences = 1
    return Args()


def test_load_rejects_non_protocol31(tmp_path):
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps({'protocol_version': 30}), encoding='utf-8')
    try:
        audit.load_protocol31(str(path))
    except ValueError as error:
        assert 'protocol_version' in str(error)
    else:
        raise AssertionError('Expected protocol-31 validation failure')


def test_crossfit_detects_absent_real_heldout_support():
    payload = _protocol31_payload()
    result = audit.build_audit(payload, _args(None), {'path': 'fixture'})
    gate = result['source']['support_gate']
    assert result['decision'].endswith('INSUFFICIENT_TARGET_NOT_READ')
    assert result['candidate_forward_count'] == 0
    assert gate['viable_heldout_real_sequences'] == []
    assert gate['viable_heldout_sim_sequences'] == ['sim_seq08', 'sim_seq10']
    assert gate['checks']['heldout_real_hard_sequence_coverage'] is False


def test_crossfit_keeps_complete_sequences_isolated():
    payload = _protocol31_payload()
    result = audit.build_audit(payload, _args(None), {'path': 'fixture'})
    for fold in result['source']['crossfit_folds']:
        assert fold['heldout_sequence'] not in fold['train_sequences']
        assert len(fold['train_sequences']) == 3


def test_main_writes_json_without_model_or_target_access(tmp_path, monkeypatch):
    source = tmp_path / 'protocol31.json'
    source.write_text(json.dumps(_protocol31_payload()), encoding='utf-8')
    output = tmp_path / 'result.json'
    monkeypatch.setattr('sys.argv', [
        'audit.py', '--paired-view-result-json', str(source),
        '--out-json', str(output)])
    audit.main()
    result = json.loads(output.read_text(encoding='utf-8'))
    assert result['protocol_version'] == 32
    assert result['parameter_update_count'] == 0
    assert result['target_dev'] is None
