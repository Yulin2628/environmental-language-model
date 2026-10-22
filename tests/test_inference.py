import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from config import ModelConfig
from inference import Decoder, greedy_generate
from model import EnvironmentalLanguageModel


class InferenceChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)
        self.config = ModelConfig(vocab_size=79, context_length=32, d_model=32,
                                  n_layers=2, n_heads=4, d_ff=64, dropout=0)
        self.model = EnvironmentalLanguageModel(self.config).eval()
        self.ids = torch.randint(0, 79, (2, 17))

    def test_all_logits_match_original_forward(self):
        reference = self.model(self.ids)
        for backend in ("manual", "sdpa"):
            decoder = Decoder(self.model, backend)
            full, _ = decoder(self.ids)
            last, _ = decoder(self.ids, last_only=True)
            torch.testing.assert_close(full, reference, atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(last, reference[:, -1:], atol=2e-6, rtol=2e-5)

    def test_cached_chunks_have_correct_causal_offset(self):
        reference = self.model(self.ids)
        for backend in ("manual", "sdpa"):
            decoder = Decoder(self.model, backend)
            cache = None
            chunks = []
            start = 0
            for end in (5, 11, 12, 17):
                logits, cache = decoder(self.ids[:, start:end], cache, use_cache=True)
                chunks.append(logits)
                start = end
            torch.testing.assert_close(torch.cat(chunks, dim=1), reference, atol=2e-6, rtol=2e-5)
            self.assertEqual(cache[0][0].shape, (2, 4, 17, 8))

    def test_future_tokens_cannot_change_prefix_logits(self):
        altered = self.ids.clone()
        altered[:, 9:] = (altered[:, 9:] + 1) % 79
        for backend in ("manual", "sdpa"):
            decoder = Decoder(self.model, backend)
            a, _ = decoder(self.ids)
            b, _ = decoder(altered)
            torch.testing.assert_close(a[:, :9], b[:, :9], atol=0, rtol=0)

    def test_sdpa_preserves_loss_and_all_parameter_gradients(self):
        alternative = copy.deepcopy(self.model)
        target = torch.randint(0, 79, self.ids.shape)
        original_loss = torch.nn.functional.cross_entropy(self.model(self.ids).flatten(0, 1), target.flatten())
        logits, _ = Decoder(alternative, "sdpa")(self.ids)
        new_loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target.flatten())
        original_loss.backward()
        new_loss.backward()
        torch.testing.assert_close(original_loss, new_loss)
        for (name, original), (other_name, changed) in zip(self.model.named_parameters(), alternative.named_parameters()):
            self.assertEqual(name, other_name)
            torch.testing.assert_close(original.grad, changed.grad, atol=2e-6, rtol=2e-4)

    def test_greedy_matches_original_across_context_rollover(self):
        prefix = torch.randint(0, 79, (2, 30))
        reference = prefix.clone()
        with torch.inference_mode():
            for _ in range(8):
                prediction = self.model(reference[:, -32:])[:, -1].argmax(-1)
                reference = torch.cat((reference, prediction[:, None]), dim=1)
        for backend in ("manual", "sdpa"):
            for cached in (False, True):
                actual = greedy_generate(Decoder(self.model, backend), prefix, 8, cached)
                self.assertTrue(torch.equal(actual, reference))

    def test_interleaved_requests_keep_separate_cache(self):
        decoder = Decoder(self.model)
        _, cache_a = decoder(self.ids[:, :4], use_cache=True)
        _, cache_b = decoder(self.ids[:, 4:8], use_cache=True)
        a, _ = decoder(self.ids[:, 4:5], cache_a, use_cache=True)
        b, _ = decoder(self.ids[:, 8:9], cache_b, use_cache=True)
        torch.testing.assert_close(a, self.model(self.ids[:, :5])[:, -1:], atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(b, self.model(self.ids[:, 4:9])[:, -1:], atol=2e-6, rtol=2e-5)

    def test_invalid_cache_and_training_use_are_rejected(self):
        decoder = Decoder(self.model)
        _, cache = decoder(self.ids, use_cache=True)
        with self.assertRaises(ValueError):
            decoder(self.ids[:, :1], cache)
        with self.assertRaises(ValueError):
            decoder(self.ids[:1, :1], cache, use_cache=True)
        with self.assertRaises(ValueError):
            decoder(self.ids, cache, use_cache=True)
        self.model.train()
        with self.assertRaises(ValueError):
            decoder(self.ids, use_cache=True)


if __name__ == "__main__":
    unittest.main()
