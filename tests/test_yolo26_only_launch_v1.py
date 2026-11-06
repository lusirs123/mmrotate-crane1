"""A new authorization must not rerun GauCho or reopen the cancelled queue."""
from copy import deepcopy
import unittest
from types import SimpleNamespace
from crane_project.tools.start_yolo26_only_v1 import yolo_only_protocol
from crane_project.tools.run_external_obb_training_v1 import yolo_epoch_for_status


class YOLOOnlyQueue(unittest.TestCase):
    def test_status_before_first_epoch_does_not_require_or_mutate_loop_state(self):
        trainer=SimpleNamespace(start_epoch=0)
        self.assertEqual(yolo_epoch_for_status(trainer),0)
        self.assertFalse(hasattr(trainer,'epoch'))
        trainer.epoch=0;self.assertEqual(yolo_epoch_for_status(trainer),1)
        trainer.epoch=23;self.assertEqual(yolo_epoch_for_status(trainer),24)

    def test_only_yolo_changes_queue_without_changing_recipe_or_input(self):
        old=dict(yolo26=dict(epochs=24,batch=2,nbs=4),physical_gpus=[0,1],
                 queue=['gaucho','yolo26'],auto_TEST=False)
        before=deepcopy(old);new=yolo_only_protocol(old)
        self.assertEqual(old,before)
        self.assertEqual(new['queue'],['yolo26'])
        self.assertEqual(new['yolo26'],old['yolo26'])
        self.assertFalse(new['auto_TEST'])
        self.assertTrue(new['result_requirements']['joint_accuracy_efficiency_report_required'])
        self.assertEqual(new['allocator_mib'],6144)
        self.assertEqual(new['driver_mib'],6656)
        self.assertEqual(new['minimum_free_mib'],1536)
        self.assertEqual(new['allocator_config'],'max_split_size_mb:128')
        self.assertNotIn('garbage_collection_threshold',new['allocator_config'])

    def test_different_budget_or_gpu_pair_is_rejected(self):
        for epochs,gpus in ((100,[0,1]),(24,[2,3])):
            with self.assertRaises(ValueError):
                yolo_only_protocol(dict(yolo26=dict(epochs=epochs),physical_gpus=gpus))


if __name__=='__main__':unittest.main()
