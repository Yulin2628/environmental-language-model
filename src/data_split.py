"""Purge overlapping token windows at the train/validation boundary."""
import math


def purged_window_split(total, context_length, stride, train_fraction=.9):
    if total < 2 or context_length < 1 or stride < 1 or not 0 < train_fraction < 1:
        raise ValueError("Invalid dataset length, window, stride or training fraction")
    train_count = int(total * train_fraction)
    if train_count < 1:
        raise ValueError("Training partition is empty")
    train_end_exclusive = (train_count - 1) * stride + context_length + 1
    validation_start = max(train_count, math.ceil(train_end_exclusive / stride))
    if validation_start >= total:
        raise ValueError("No validation windows remain after purging")
    return range(train_count), range(validation_start, total)
