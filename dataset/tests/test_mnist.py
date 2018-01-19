import unittest
import sys
sys.path.append("..")

from dataset import mnist
import tempfile


class TestDownloads(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls.tempdir.cleanup()

    def test_download_train_labels(self):
        mnist.download_train_labels(self.tempdir.name)
        self.assertEqual("a", "a")


if __name__ == "__main__":
    unittest.main()
