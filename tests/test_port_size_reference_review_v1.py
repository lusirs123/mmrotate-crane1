import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('reference_review', ROOT/'tools/data/prepare_port_size_reference_review_v1.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


class ReviewContracts(unittest.TestCase):
    def test_geometry_uses_independent_four_corners_not_fixed_ratio(self):
        text = '\n'.join('DEF '+name+' Pose { translation '+v for name, v in {
            'GRAB_PT_TL':'0 0 0', 'GRAB_PT_TR':'3 0 0', 'GRAB_PT_BL':'0 0 2',
            'GRAB_PT_BR':'3 0 2', 'GRAB_REFERENCE':'1.5 0 1'}.items())
        result = m.geometry(text)
        self.assertEqual(result['long_edge_mean_m'], 3)
        self.assertEqual(result['short_edge_mean_m'], 2)
        self.assertEqual(result['reference_center_error_m'], 0)
        with self.assertRaises(ValueError): m.geometry(text+text)

    def test_world_and_manifest_must_have_frozen_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp)/'untrusted'; f.write_text('{}')
            with self.assertRaises(ValueError): m.reference(f, f)

    def test_video_role_cannot_accept_renamed_val_or_unreviewed_source(self):
        used = {'1 (7).mp4':'train_sha', '1 (10).mp4':'val_sha'}
        m.validate_video('1 (7).mp4', 'train_sha', used)
        m.validate_video('1 (8).mp4', m.CANDIDATE_SHA['1 (8).mp4'], used)
        for name, digest in (('1 (7).mp4','val_sha'), ('1 (8).mp4','new_sha'), ('1 (10).mp4','val_sha')):
            with self.assertRaises(ValueError): m.validate_video(name, digest, used)

    def test_train_review_never_globs_val_or_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for group in m.TRAIN_GROUPS:
                split = 'train_sim' if group.startswith('sim_') else 'train'
                for sub in ('images', 'annfiles'):(root/split/sub).mkdir(parents=True, exist_ok=True)
                for i in range(6):
                    (root/split/'images'/f'{group}_{i:05d}.jpg').touch()
                    (root/split/'annfiles'/f'{group}_{i:05d}.txt').touch()
            result = m.train_samples(root)
            self.assertEqual(len(result), 24)
            self.assertTrue(all('val' not in x[1].parts and 'test' not in x[1].parts for x in result))

    def test_output_cannot_overwrite_or_write_dataset(self):
        with self.assertRaises(ValueError):m.output_path(ROOT/'crane_project/data/port_size_reference_review_v1_test')
        with tempfile.TemporaryDirectory(prefix=m.VERSION+'_', dir=ROOT/'work_dirs') as tmp:
            with self.assertRaises(FileExistsError):m.output_path(tmp)


if __name__ == '__main__':unittest.main()
