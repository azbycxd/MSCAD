"""Tests orchestration guards using mocks; no dataset or model execution."""
import hashlib
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/thesis/06_validity/run_validity_p1.py'

def load_runner():
    spec=importlib.util.spec_from_file_location('validity_guard_under_test',SCRIPT)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class StopGuardTests(unittest.TestCase):
    def test_failed_swat_gate_does_not_enter_hai_or_new_controls(self):
        m=load_runner();calls=[]
        def prep(dataset,log):
            return {'target_check':{'status':'PASS'},'threshold_check':{'status':'PASS'},'gate_pass':True}
        def replay(state,log):
            calls.append('S3_Test');return {'passed':False,'gates':{'status':'FAIL'}}
        real=types.SimpleNamespace(prepare=prep,replay_test=replay)
        ctrl=types.SimpleNamespace(prepare=lambda root:{'donor_audit':{},'input_paths':[]},replay_s3=lambda state:{'passed':True})
        comut=types.SimpleNamespace(prepare=lambda root:{'input_paths':[]})
        with tempfile.TemporaryDirectory() as temp:
            m.OUT=Path(temp)/'new_audit'
            with patch.dict(sys.modules,{'real_helper':real,'controlled_helper':ctrl,'comut_helper':comut}):
                with self.assertRaises(m.ReplayGateStop):m.main()
            self.assertEqual(calls,['S3_Test'])
            self.assertNotIn('S3_REPLAY_HAI',m.manifest['steps'])
            self.assertNotIn('NORMAL_STATS_HASH_FROZEN',m.manifest['steps'])
            self.assertFalse((m.OUT/'metrics_main.csv').exists())

    def test_existing_audit_is_refused_without_mutation(self):
        m=load_runner()
        with tempfile.TemporaryDirectory() as temp:
            m.OUT=Path(temp)
            sentinel=m.OUT/'REPORT.md';sentinel.write_text('KEEP_EXISTING_EVIDENCE',encoding='utf-8')
            before=hashlib.sha256(sentinel.read_bytes()).hexdigest()
            with self.assertRaisesRegex(RuntimeError,'Refusing to overwrite'):m.main()
            self.assertFalse(m.output_created)
            self.assertEqual(before,hashlib.sha256(sentinel.read_bytes()).hexdigest())
            self.assertEqual([p.name for p in m.OUT.iterdir()],['REPORT.md'])

if __name__=='__main__':unittest.main()
