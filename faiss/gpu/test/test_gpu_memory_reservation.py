import subprocess
import sys
import textwrap
import unittest

import numpy as np
import faiss


class TestGpuMemoryReservation(unittest.TestCase):
    @unittest.skipIf(faiss.get_num_gpus() < 1, "gpu only test")
    def test_prealloc_allows_small_add(self):
        reserved_bytes = 32 * 1024 * 1024
        dims = 128
        num_add = 10000

        res = faiss.StandardGpuResources()
        res.setTempMemory(0)
        res.setDeviceMemoryReservation(reserved_bytes)

        config = faiss.GpuIndexFlatConfig()
        config.device = 0
        if hasattr(config, "use_cuvs"):
            config.use_cuvs = False

        index = faiss.GpuIndexFlatL2(res, dims, config)
        xb = np.random.RandomState(123).rand(num_add, dims).astype("float32")
        index.add(xb)
        self.assertEqual(index.ntotal, num_add)

    @unittest.skipIf(faiss.get_num_gpus() < 1, "gpu only test")
    def test_prealloc_exhausts_in_subprocess(self):
        code = textwrap.dedent(
            """
            import numpy as np
            import faiss

            reserved_bytes = 32 * 1024 * 1024
            dims = 128
            num_add = 120000

            res = faiss.StandardGpuResources()
            res.setTempMemory(0)
            res.setDeviceMemoryReservation(reserved_bytes)

            config = faiss.GpuIndexFlatConfig()
            config.device = 0
            if hasattr(config, "use_cuvs"):
                config.use_cuvs = False

            index = faiss.GpuIndexFlatL2(res, dims, config)
            xb = np.random.RandomState(456).rand(num_add, dims).astype("float32")
            index.add(xb)
            """
        )

        proc = subprocess.run(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertNotEqual(proc.returncode, 0)
        self.assertIn(
            b"preallocated pool exhausted",
            proc.stdout + proc.stderr,
        )
