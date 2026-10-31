"""Regression checks for strict single-device diagnostic resource guards."""
import unittest
from unittest.mock import patch
from crane_project.tools.diagnose_eood_AB_frozen_v2 import parse_memory, telemetry, enforce_resources


class ResourceTests(unittest.TestCase):
    def test_query_is_limited_to_selected_gpu(self):
        with patch('subprocess.check_output',return_value='0, 7000, 1192, 45, 3\n') as query:
            row=telemetry(0)
            self.assertEqual(row['free_mib'],7000)
            args=query.call_args.args[0]
            self.assertEqual(args[args.index('-i')+1],'0')

    def test_missing_faulty_or_wrong_card_fails_closed(self):
        for text in ('','1,7000,1192,45,3','0,N/A,N/A,N/A,N/A',
                     '0,7000,1192,45,3\n2,7000,1192,45,3','0,-1,1192,45,3'):
            with self.subTest(text=text),self.assertRaises(ValueError):parse_memory(text,0)

    def test_memory_limits_and_default_driver_thermal_policy(self):
        row=dict(free_mib=6656,temperature_C=44)
        enforce_resources(row,1000,before=True)
        for bad,own,before in ((dict(row,free_mib=6655),0,True),
            (dict(row,free_mib=511),0,False),
            (row,6657,False)):
            with self.subTest(row=bad,own=own),self.assertRaises(RuntimeError):
                enforce_resources(bad,own,before)
        enforce_resources(dict(row,temperature_C=85),1000)


if __name__=='__main__':unittest.main()
