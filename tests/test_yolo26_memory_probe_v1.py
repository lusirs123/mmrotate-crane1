"""Resource diagnosis is bounded and cannot become a formal/TEST run."""
from copy import deepcopy
import unittest
from crane_project.tools.probe_yolo26_memory_v1 import check_plan,summarize_records
from crane_project.tools import probe_yolo26_memory_v1 as probe


class MemoryProbe(unittest.TestCase):
    def test_plan_forbids_formal_validation_test_and_unbounded_steps(self):
        plan=probe.native.read(probe.ROOT/probe.PLAN);check_plan(plan)
        for key in ('formal_training','VAL_inference','TEST_access','save_checkpoints'):
            bad=deepcopy(plan);bad[key]=True
            with self.assertRaises(ValueError):check_plan(bad)
        bad=deepcopy(plan);bad['cases'][0]['steps']=1000
        with self.assertRaises(ValueError):check_plan(bad)

    def test_summary_separates_initial_growth_from_repeated_allocator_reclaims(self):
        rows=[dict(reserved_mib=3000 if i<8 else 4000,
            allocation_retries=0,OOMs=0,step_peak_allocated_mib=3100,
            step_peak_reserved_mib=4000) for i in range(16)]
        s=summarize_records(rows)
        self.assertEqual(s['tail_reserved_range_mib'],0)
        self.assertEqual(s['tail_allocation_retries'],0)
        rows[-1]['allocation_retries']=3;rows[-1]['reserved_mib']=3500
        s=summarize_records(rows)
        self.assertEqual(s['tail_allocation_retries'],3)
        self.assertEqual(s['tail_reserved_range_mib'],500)


if __name__=='__main__':unittest.main()
