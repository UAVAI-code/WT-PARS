import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

class DemoTests(unittest.TestCase):
    def test_release_checksums(self):
        for line in (ROOT / 'MANIFEST.sha256').read_text().splitlines():
            expected, name = line.split('  ', 1)
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(), expected, name)

    def test_data_regeneration(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([sys.executable, str(ROOT/'generate_demo.py'), '--output', tmp], check=True)
            for name in ['demo_input.npz','demo_reference.npz']:
                with np.load(ROOT/'data'/name, allow_pickle=False) as a, np.load(Path(tmp)/name, allow_pickle=False) as b:
                    for key in a.files: np.testing.assert_array_equal(a[key], b[key])

    def test_both_modes_repeatability_and_accounting(self):
        with tempfile.TemporaryDirectory() as tmp:
            for repeat in ['first','repeat']:
                subprocess.run([sys.executable, str(ROOT/'run_demo.py'), '--mode','both',
                    '--reference',str(ROOT/'data/demo_reference.npz'), '--output',str(Path(tmp)/repeat)],
                    check=True, cwd=tmp, env={**os.environ, 'PYTHONPATH':''})
            for mode in ['targeted','untargeted']:
                first = Path(tmp)/'first'/mode
                s = json.loads((first/'summary.json').read_text())
                q = s['query_accounting']
                self.assertEqual(q['total_service_calls'], q['reference_service_calls']+q['search_service_calls']+q['selection_service_calls'])
                self.assertEqual(q['reference_service_calls'], 4)
                self.assertEqual(q['search_service_calls'], 4*s['candidate_queries'])
                self.assertEqual(q['candidate_requests'], q['unique_candidate_evaluations']+q['candidate_cache_hits']+q['budget_fallback_returns'])
                self.assertGreater(s['optimizer_evaluations'], 0)
                self.assertFalse(s['ground_truth_used_during_search'])
                with np.load(first/'arrays.npz') as a, np.load(Path(tmp)/'repeat'/mode/'arrays.npz') as b:
                    for key in a.files: np.testing.assert_array_equal(a[key], b[key])
            # Ground-truth evaluation must not change the generated cube.
            subprocess.run([sys.executable, str(ROOT/'run_demo.py'), '--output',str(Path(tmp)/'no_truth')], check=True, cwd=tmp)
            for mode in ['targeted','untargeted']:
                with np.load(Path(tmp)/'first'/mode/'arrays.npz') as a, np.load(Path(tmp)/'no_truth'/mode/'arrays.npz') as b:
                    np.testing.assert_array_equal(a['cube'], b['cube'])

if __name__ == '__main__':
    unittest.main()
