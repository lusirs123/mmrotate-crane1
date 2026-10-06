"""Provenance recovery must not silently accept another model or run inference."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from tools.depth import recover_webots_native_s14_v1 as r


def fixture(root):
    root = Path(root)
    folder = root/'work_dirs/webots_train03_old'; folder.mkdir(parents=True)
    weight = root/'work_dirs/source_safe_interpolated_head.pth'
    weight.write_bytes(b'not a pickle: hash without deserialization')
    expected = dict(sequence_id='calibration_train_03_tilt_obb', head_sha256=r.sha(weight))
    rows = [dict(frame_id='frame_%05d'%i, image_name='frame_%05d.jpg'%i,
                 detected=False, num_detections=0, top1=None) for i in range(981)]
    prediction = folder/'predictions_full.jsonl'
    prediction.write_text(''.join(json.dumps(v)+'\n' for v in rows))
    contract = dict(component='frozen_dino_native_s14_alpha05', all_frames=True,
        **{k:False for k in ('symeood_enabled','brightaug_enabled','s7_enabled','target_scope',
                            'sequence_identity_routing','temporal_takeover','box_stabilizer')})
    receipt = dict(sequence_id=expected['sequence_id'], checkpoint_sha256=expected['head_sha256'],
        model_type='FrozenDinoNativeS14Detector', discovered_frame_count=981,
        processed_frame_count=981, detected_frame_count=0, limit=None,
        model_selection_performed=False, threshold_tuning_performed=False,
        metric_truth_read=False, detector_outputs_modified=False, formal_detection_contract=contract)
    manifest = Path(str(prediction)+'.manifest.json')
    manifest.write_text(json.dumps(receipt))
    return expected, rows, receipt, prediction, manifest


class RecoveryTests(unittest.TestCase):
    def test_consistent_receipt_is_candidate_not_approved_depth_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            e, _, _, _, m = fixture(tmp)
            x = r.inspect_manifest(m,e)
            self.assertTrue(x['metadata_and_rows_consistent'])
            self.assertFalse(x['original_image_bytes_verified'])
            self.assertFalse(x['evaluation_reuse_approved'])

    def test_other_weight_fusion_and_partial_run_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            e, _, receipt, _, m = fixture(tmp)
            for key,value in [('checkpoint_sha256','another'),('model_type','SymEOOD'),
                              ('processed_frame_count',980),('limit',50),('metric_truth_read',True)]:
                wrong=deepcopy(receipt);wrong[key]=value;m.write_text(json.dumps(wrong))
                self.assertFalse(r.inspect_manifest(m,e)['metadata_and_rows_consistent'])
            wrong=deepcopy(receipt);wrong['formal_detection_contract']['symeood_enabled']=True
            m.write_text(json.dumps(wrong))
            self.assertFalse(r.inspect_manifest(m,e)['metadata_and_rows_consistent'])

    def test_frame_identity_and_invalid_box_rejected_without_modification(self):
        with tempfile.TemporaryDirectory() as tmp:
            e,rows,_,p,m=fixture(tmp)
            rows.reverse();p.write_text(''.join(json.dumps(v)+'\n' for v in rows))
            before=p.read_bytes()
            self.assertFalse(r.inspect_manifest(m,e)['metadata_and_rows_consistent'])
            self.assertEqual(before,p.read_bytes())
            rows.reverse();rows[0].update(detected=True,num_detections=1,top1=dict(
                class_id=0,cx_px=0,cy_px=0,width_px=-1,height_px=1,angle_rad=0,score=.9))
            p.write_text(''.join(json.dumps(v)+'\n' for v in rows))
            self.assertFalse(r.inspect_manifest(m,e)['metadata_and_rows_consistent'])

    def test_unknown_fixeddev_and_symlink_results_are_not_traversed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ('unknown_test_02','fixed_dev_01','safe_train03'):
                p=root/name;p.mkdir();(p/'receipt.json').write_text('{}')
            (root/'linked').symlink_to(root/'safe_train03',target_is_directory=True)
            found=list(r.files_under(root))
            self.assertEqual(found,[root/'safe_train03/receipt.json'])

    def test_legacy_or_wrong_model_report_not_recovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            e,_,receipt,_,_=fixture(tmp);p=Path(tmp)/'report.json'
            doc=dict(coordinate_contract='raw_opt_v1',sequence_id=e['sequence_id'],
                     current=dict(prediction_manifest=receipt,prediction_sha256='historical'))
            p.write_text(json.dumps(doc));self.assertIsNotNone(r.inspect_report(p,e))
            doc['coordinate_contract']='legacy_plumb_opt_v1';p.write_text(json.dumps(doc))
            self.assertIsNone(r.inspect_report(p,e))

    def test_cpu_inventory_packages_only_receipt_and_candidate_never_weight(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);e,_,_,p,m=fixture(root)
            identity=root/'identity.json'
            identity.write_text(json.dumps(dict(protocol=r.VERSION,sources={},
                prediction_contract=e,weights=dict(head=dict(filename='source_safe_interpolated_head.pth',
                sha256=e['head_sha256'])))))
            before={p:r.sha(p),m:r.sha(m)}
            args=r.parser().parse_args(['--project-root',str(root),'--out-dir',str(root/'review')])
            with patch.object(r,'IDENTITY',identity):r.run(args)
            with tarfile.open(root/'review.tar.gz') as t:
                names=t.getnames();self.assertEqual(len(names),4)
                self.assertFalse(any(n.endswith('.pth') for n in names))
                member=next(n for n in names if n.endswith('predictions_full.jsonl'))
                self.assertEqual(hashlib.sha256(t.extractfile(member).read()).hexdigest(),before[p])
            self.assertEqual(before,{path:r.sha(path) for path in before})
            with patch.object(r,'IDENTITY',identity),self.assertRaises(FileExistsError):r.run(args)


if __name__=='__main__':
    unittest.main()
