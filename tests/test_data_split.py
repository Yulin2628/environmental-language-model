from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data_split import purged_window_split


class SplitChecks(unittest.TestCase):
    def test_no_raw_token_position_is_shared(self):
        for context, stride in ((512, 256), (15, 1), (32, 32), (32, 64)):
            train, validation = purged_window_split(1000, context, stride)
            train_last_position = train[-1] * stride + context
            validation_first_position = validation[0] * stride
            self.assertGreater(validation_first_position, train_last_position)

    def test_existing_corpus_requires_two_purged_windows(self):
        train, validation = purged_window_split(868, 512, 256)
        self.assertEqual((len(train), len(validation), validation[0]), (781, 85, 783))

    def test_empty_partition_is_rejected(self):
        for args in ((2, 512, 1), (100, 32, 0), (100, 32, 32, 1)):
            with self.assertRaises(ValueError):
                purged_window_split(*args)


if __name__ == "__main__":
    unittest.main()
