"""Command-line interface for rhasspy fuzzywuzzy"""
import argparse
import json
import logging
import os
import sys
import typing
from pathlib import Path

import rhasspynlu
from rhasspynlu.intent import Recognition

from . import train as fuzzywuzzy_train, recognize as fuzzywuzzy_recognize

_LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


def main():
    """Main method."""
    args = get_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    _LOGGER.debug(args)

    # Dispatch to appropriate sub-command
    args.func(args)


# -----------------------------------------------------------------------------


def get_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog="rhasspy-fuzzywuzzy")
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to the console"
    )

    # Create subparsers for each sub-command
    sub_parsers = parser.add_subparsers()
    sub_parsers.required = True
    sub_parsers.dest = "command"

    # Run settings
    recognize_parser = sub_parsers.add_parser("recognize", help="Do intent recognition")
    recognize_parser.set_defaults(func=recognize)

    recognize_parser.add_argument(
        "--examples", required=True, help="Path to examples JSON file"
    )
    recognize_parser.add_argument(
        "--intent-graph", required=True, help="Path to intent graph JSON file"
    )
    recognize_parser.add_argument(
        "--replace-numbers",
        action="store_true",
        help="Automatically replace numbers in query text",
    )
    recognize_parser.add_argument(
        "--language", default="en", help="Language used for number replacement"
    )
    recognize_parser.add_argument(
        "--word-casing",
        choices=["upper", "lower", "ignore"],
        default="ignore",
        help="Case transformation applied to query text",
    )
    recognize_parser.add_argument("query", nargs="*", help="Query input sentences")

    # -------------------------------------------------------------------------

    # Train settings
    train_parser = sub_parsers.add_parser(
        "train", help="Generate intent examples from sentences and slots"
    )
    train_parser.set_defaults(func=train)

    train_parser.add_argument("--examples", help="Path to write examples JSON file")
    train_parser.add_argument(
        "--intent-graph", help="Path to write intent graph JSON file"
    )
    train_parser.add_argument(
        "--sentences", action="append", help="Paths to sentences ini files"
    )
    train_parser.add_argument(
        "--slots", action="append", help="Directories with static slot text files"
    )
    train_parser.add_argument(
        "--slot-programs", action="append", help="Directories with slot programs"
    )
    train_parser.add_argument(
        "--replace-numbers",
        action="store_true",
        help="Automatically replace numbers and number ranges in sentences/slots",
    )
    train_parser.add_argument(
        "--language", default="en", help="Language used for number replacement"
    )
    train_parser.add_argument(
        "--word-casing",
        choices=["upper", "lower", "ignore"],
        default="ignore",
        help="Case transformation applied to words",
    )

    return parser.parse_args()


# -----------------------------------------------------------------------------


def recognize(args: argparse.Namespace):
    """Do intent recognition from query text."""
    try:
        # Convert to Paths
        args.examples = Path(args.examples)
        args.intent_graph = Path(args.intent_graph)

        # Load graph/examples
        _LOGGER.debug("Loading intent graph from %s", str(args.intent_graph))
        with open(args.intent_graph, "r") as intent_graph_file:
            graph_dict = json.load(intent_graph_file)
            intent_graph = rhasspynlu.json_to_graph(graph_dict)

        _LOGGER.debug("Loading examples from %s", str(args.examples))
        with open(args.examples, "r") as examples_file:
            examples = json.load(examples_file)

        _LOGGER.debug("Processing sentences")
        word_transform = get_word_transform(args.word_casing) or (lambda s: s)

        # Process queries
        if args.query:
            sentences = args.query
        else:
            if os.isatty(sys.stdin.fileno()):
                print("Reading queries from stdin...", file=sys.stderr)

            sentences = sys.stdin

        for sentence in sentences:
            # Handle casing
            sentence = sentence.strip()
            sentence = word_transform(sentence)

            # Do recognition
            recognitions = fuzzywuzzy_recognize(
                sentence,
                intent_graph,
                examples,
                replace_numbers=args.replace_numbers,
                language=args.language,
            )

            if recognitions:
                # Intent recognized
                recognition = recognitions[0]
            else:
                # Intent not recognized
                recognition = Recognition.empty()

            # Print as a line of JSON
            json.dump(recognition.asdict(), sys.stdout)
            print("")
            sys.stdout.flush()

    except KeyboardInterrupt:
        pass


# -----------------------------------------------------------------------------


def train(args: argparse.Namespace):
    """Generate intent examples from sentences and slots."""
    # Convert to Paths
    if args.examples:
        args.examples = Path(args.examples)
    else:
        assert (
            args.intent_graph
        ), "--intent-graph is required when examples are printed to stdout"

    if args.intent_graph:
        args.intent_graph = Path(args.intent_graph)

    if args.slots:
        args.slots = [Path(p) for p in args.slots]

    if args.slot_programs:
        args.slot_programs = [Path(p) for p in args.slot_programs]

    if args.sentences:
        # Load sentences from text files
        sentences = {p: Path(p).read_text() for p in args.sentences}
    else:
        # Load sentences from stdin
        if os.isatty(sys.stdin.fileno()):
            print("Reading sentences from stdin...", file=sys.stderr)

        sentences = {"<stdin>": sys.stdin.read()}

    # -------------------------------------------------------------------------

    # Do training
    intent_graph, examples = fuzzywuzzy_train(
        sentences,
        slots_dirs=args.slots,
        slot_programs_dirs=args.slot_programs,
        replace_numbers=args.replace_numbers,
        language=args.language,
        word_transform=get_word_transform(args.word_casing),
    )

    if args.examples:
        # Write examples to JSON file
        with open(args.examples, "w") as examples_file:
            json.dump(examples, examples_file)

        _LOGGER.debug("Wrote %s", str(args.examples))

    if args.intent_graph:
        # Write graph to JSON file
        with open(args.intent_graph, "w") as graph_file:
            graph_dict = rhasspynlu.graph_to_json(intent_graph)
            json.dump(graph_dict, graph_file)

        _LOGGER.debug("Wrote %s", str(args.intent_graph))

    # Write results to stdout
    json.dump(examples, sys.stdout)
    print("")
    sys.stdout.flush()


# -----------------------------------------------------------------------------


def get_word_transform(name: str) -> typing.Optional[typing.Callable[[str], str]]:
    """Gets a word transformation function by name."""
    if name == "upper":
        return str.upper

    if name == "lower":
        return str.lower

    return None


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
