"""Training methods for rhasspyfuzzywuzzy"""
import io
import logging
import typing
from collections import defaultdict
from pathlib import Path

import networkx as nx
import rhasspynlu
from rhasspynlu.jsgf import Expression, Word

from .const import ExamplesType

_LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


def train(
    sentences_dict: typing.Dict[str, str],
    slots_dirs: typing.Optional[typing.List[Path]] = None,
    slot_programs_dirs: typing.Optional[typing.List[Path]] = None,
    replace_numbers: bool = True,
    language: str = "en",
    word_transform: typing.Optional[typing.Callable[[str], str]] = None,
) -> typing.Tuple[nx.DiGraph, ExamplesType]:
    """Transform sentences to an intent graph and examples"""
    slots_dirs = slots_dirs or []
    slot_programs_dirs = slot_programs_dirs or []

    # Parse sentences and convert to graph
    with io.StringIO() as ini_file:
        # Join as single ini file
        for lines in sentences_dict.values():
            print(lines, file=ini_file)
            print("", file=ini_file)

        # Parse JSGF sentences
        intents = rhasspynlu.parse_ini(ini_file.getvalue())

    # Split into sentences and rule/slot replacements
    sentences, replacements = rhasspynlu.ini_jsgf.split_rules(intents)

    word_visitor: typing.Optional[
        typing.Callable[[Expression], typing.Union[bool, Expression]]
    ] = None

    if word_transform:
        # Apply transformation to words

        def transform_visitor(word: Expression):
            if isinstance(word, Word):
                assert word_transform
                word.text = word_transform(word.text)

            return word

        word_visitor = transform_visitor

    # Apply case/number transforms
    if word_visitor or replace_numbers:
        for intent_sentences in sentences.values():
            for sentence in intent_sentences:
                if replace_numbers:
                    # Replace number ranges with slot references
                    # type: ignore
                    rhasspynlu.jsgf.walk_expression(
                        sentence, rhasspynlu.number_range_transform, replacements
                    )

                if word_visitor:
                    # Do case transformation
                    # type: ignore
                    rhasspynlu.jsgf.walk_expression(
                        sentence, word_visitor, replacements
                    )

    # Load slot values
    slot_replacements = rhasspynlu.get_slot_replacements(
        sentences,
        slots_dirs=slots_dirs,
        slot_programs_dirs=slot_programs_dirs,
        slot_visitor=word_visitor,
    )

    # Merge with existing replacements
    for slot_key, slot_values in slot_replacements.items():
        replacements[slot_key] = slot_values

    if replace_numbers:
        # Do single number transformations
        for intent_sentences in sentences.values():
            for sentence in intent_sentences:
                rhasspynlu.jsgf.walk_expression(
                    sentence,
                    lambda w: rhasspynlu.number_transform(w, language),
                    replacements,
                )

    # Convert to directed graph
    intent_graph = rhasspynlu.sentences_to_graph(sentences, replacements=replacements)

    # Generate all possible intents
    _LOGGER.debug("Generating examples")
    examples: typing.Dict[str, typing.Dict[str, typing.List[int]]] = defaultdict(dict)
    for intent_name, words, path in generate_examples(intent_graph):
        word_str = " ".join(words)
        examples[intent_name][word_str] = path

    _LOGGER.debug("Examples generated")

    return (intent_graph, examples)


# -----------------------------------------------------------------------------


def generate_examples(
    intent_graph: nx.DiGraph
) -> typing.Iterable[typing.Tuple[str, typing.List[str], typing.List[int]]]:
    """Generate all possible sentences/paths from an intent graph."""
    n_data = intent_graph.nodes(data=True)

    # Get start/end nodes for graph
    start_node, end_node = rhasspynlu.jsgf_graph.get_start_end_nodes(intent_graph)
    assert (start_node is not None) and (
        end_node is not None
    ), "Missing start/end node(s)"

    # Generate all sentences/paths
    paths = nx.all_simple_paths(intent_graph, start_node, end_node)
    for path in paths:
        assert len(path) > 2

        # First edge has intent name (__label__INTENT)
        olabel = intent_graph.edges[(path[0], path[1])]["olabel"]
        assert olabel.startswith("__label__")
        intent_name = olabel[9:]

        sentence = []
        for node in path:
            word = n_data[node].get("word")
            if word:
                sentence.append(word)

        yield (intent_name, sentence, path)
