"""Validate published measurements without requiring the private checkpoints."""
import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def main():
    audit = read("results/audit.json")
    for split in ("train", "validation"):
        path = ROOT / f"data/finetune/{split}.jsonl"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == audit["data"][split]["sha256"]
    assert audit["models"]["base"]["unique_parameters"] == 25_309_184
    for item in audit["models"].values():
        metrics = item["validation"]
        rows = metrics["cases"]
        assert metrics["examples"] == len(rows) == audit["data"]["validation"]["examples"]
        total = sum(row["scored_tokens_including_eos"] for row in rows)
        assert total == metrics["tokens_including_eos"]
        assert math.isclose(sum(row["nll_sum"] for row in rows) / total, metrics["token_weighted_nll"])
        assert math.isclose(sum(row["correct_tokens_including_eos"] for row in rows) / total,
                            metrics["teacher_forced_token_match_including_eos"])
    checks = read("results/optimization_checks.json")
    assert checks["all_passed"] and all(row["passed"] for row in checks["checks"])
    for precision in ("fp32", "fp16"):
        benchmark = read(f"results/benchmark_{precision}.json")
        assert benchmark["checkpoint_sha256"] == audit["models"]["finetuned"]["checkpoint_sha256"]
        assert len(benchmark["scenarios"]) == 4
        for scenario in benchmark["scenarios"]:
            assert scenario["batch"] == len(scenario["prefix_ids"])
            reference = scenario["modes"]["original"]["generated_ids"]
            for name, item in scenario["modes"].items():
                assert item["greedy_tokens_equal_original"] == (reference == item["generated_ids"])
                assert len(item["samples"]) == benchmark["repeats"]
                for key, median in item["median"].items():
                    assert math.isclose(statistics.median(row[key] for row in item["samples"]), median)
                for sample in item["samples"]:
                    assert sample["request_ms"] > 0
                    assert math.isclose(sample["request_ms"], sample["decode_ms"] + sample["time_to_first_token_ms"])
                    expected = scenario["batch"] * scenario["new_tokens"] * 1000 / sample["request_ms"]
                    assert math.isclose(expected, sample["generated_tokens_per_second"])
                    assert sample["greedy_tokens_equal_original"] == item["greedy_tokens_equal_original"]
    print("Verified dataset identities, audit denominators, logit checks and all raw benchmark medians.")


if __name__ == "__main__":
    main()
