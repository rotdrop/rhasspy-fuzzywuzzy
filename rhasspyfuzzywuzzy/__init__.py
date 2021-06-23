"""Rhasspy intent recognition with rapidfuzz"""
import json
import logging
import sqlite3
import time
import typing

import networkx as nx
import rapidfuzz
import rhasspynlu
from rhasspynlu.intent import Recognition

from .train import train

_LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


def extract_one_sqlite(query: str, examples_path: str):
    """Finds the best text/path for a query"""
    conn = sqlite3.connect(examples_path)
    c = conn.cursor()
    c.execute("SELECT sentence FROM intents ORDER BY rowid")

    result = rapidfuzz.process.extractOne(
        [query], c, processor=lambda s: s[0], scorer=rapidfuzz.fuzz.ratio
    )

    if not result:
        conn.close()
        return result

    c.execute("SELECT path FROM intents ORDER BY rowid LIMIT 1 OFFSET ?", (result[2],))
    best_path = c.fetchone()[0]

    conn.close()

    return (result[0][0], json.loads(best_path), result[1])


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
        rapidfuzz.utils.default_process(input_text), examples_path
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
