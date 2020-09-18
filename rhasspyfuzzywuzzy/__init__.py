"""Rhasspy intent recognition with rapidfuzz"""
import json
import logging
import sqlite3
import time
import typing

import networkx as nx
import rapidfuzz.fuzz as fuzzy_fuzz
import rapidfuzz.utils as fuzz_utils
import rhasspynlu
from rhasspynlu.intent import Recognition

from .train import train

_LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


def extract_one_sqlite(query: str, examples_path: str):
    """Finds the best text/path for a query"""
    conn = sqlite3.connect(examples_path)
    c = conn.cursor()
    c.execute("SELECT sentence, path FROM intents")

    score_cutoff = 0
    result_score = None
    best_path = None
    best_text = None

    for choice in c:
        score = fuzzy_fuzz.WRatio(
            query, choice[0], processor=None, score_cutoff=score_cutoff
        )

        if score >= score_cutoff:
            score_cutoff = score + 0.00001
            if score_cutoff > 100:
                return (choice[0], json.loads(choice[1]), score)
            result_score = score
            best_path = choice[1]
            best_text = choice[0]

    conn.close()

    if (result_score is None) or (best_path is None):
        return None

    return (best_text, json.loads(best_path), result_score)


def recognize(
    input_text: str,
    intent_graph: nx.DiGraph,
    examples_path: str,
    intent_filter: typing.Optional[typing.Callable[[str], bool]] = None,
    extra_converters: typing.Optional[
        typing.Dict[str, typing.Callable[..., typing.Any]]
    ] = None,
) -> typing.List[Recognition]:
    """Find the closest matching intent(s)."""
    start_time = time.perf_counter()
    intent_filter = intent_filter or (lambda i: True)

    # Find closest match
    # pylint: disable=unpacking-non-sequence
    best_text, best_path, best_score = extract_one_sqlite(
        fuzz_utils.default_process(input_text), examples_path
    )
    _LOGGER.debug("input=%s, match=%s, score=%s", input_text, best_text, best_score)

    end_time = time.perf_counter()
    _, recognition = rhasspynlu.fsticuffs.path_to_recognition(
        best_path, intent_graph, extra_converters=extra_converters
    )

    assert recognition and recognition.intent, "Failed to find a match"
    recognition.intent.confidence = best_score / 100.0
    recognition.recognize_seconds = end_time - start_time
    recognition.raw_text = input_text
    recognition.raw_tokens = input_text.split()

    return [recognition]
