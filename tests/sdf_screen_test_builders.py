"""Builders shared by the SDF screen tests: screen replies that remove no hands or chosen hands, and the
config fixture for a 25-hand screen in batches of ten."""

import json
import re

import pytest


def numbers_in(user_message):
    """(number, text) for every <hand> block the screen prompt shows."""
    return [(int(n), body) for n, body in re.findall(
        r'<hand number="(\d+)">\n(.*?)\n</hand>', user_message, re.DOTALL)]


def screen_answer(count, remove=(), reasons=None, **overrides):
    """A schema-shaped answer for a batch of ``count`` hands."""
    entries = [
        {"number": n, "reasoning": f"Considered hand {n}.", "remove": n in remove,
         "reason": (reasons or {}).get(n, "No realistic document could satisfy this hand.") if n in remove else ""}
        for n in range(1, count + 1)
    ]
    return json.dumps({"specifications": entries, **overrides})


def no_removals(user_message, **kw):
    return screen_answer(len(numbers_in(user_message)))


def remove_first_hand_of_round_zero_batches(user_message, **kw):
    """Removes hand 1 of every round-0 batch; keeps everything in later rounds."""
    count = len(numbers_in(user_message))
    return screen_answer(count, remove={1} if "round0" in kw["item_id"] else ())


def remove_the_first_hand_of_every_batch(user_message, **kw):
    return screen_answer(len(numbers_in(user_message)), remove={1})


def remove_the_first_three_hands_of_every_batch(user_message, **kw):
    return screen_answer(len(numbers_in(user_message)), remove={1, 2, 3})


@pytest.fixture
def screen_config(tiny_config):
    tiny_config["sdf"]["n_prompts"] = 25
    tiny_config["sdf"]["screen_batch_size"] = 10
    return tiny_config
