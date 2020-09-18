"""Command-line interface for rhasspy fuzzywuzzy"""
import argparse
import json
import logging
import os
import sqlite3
import sys
import typing
from pathlib import Path

import rhasspynlu
from rhasspynlu.intent import Recognition

from . import recognize as fuzzywuzzy_recognize
from . import train as fuzzywuzzy_train

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
        "--examples", required=True, help="Path to examples SQLite Database"
    )
    recognize_parser.add_argument(
        "--intent-graph", required=True, help="Path to intent graph JSON file"
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

    train_parser.add_argument(
        "--examples", help="Path to write examples SQLite Database"
    )
    train_parser.add_argument(
        "--intent-graph", help="Path to write intent graph JSON file"
    )
    train_parser.add_argument(
        "--sentences", action="append", help="Paths to sentences ini files"
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

        _LOGGER.debug("Processing sentences")
        word_transform = get_word_transform(args.word_casing)

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
                sentence, intent_graph, str(args.examples)
            )

            if recognitions:
                # Intent recognized
                recognition = recognitions[0]
            else:
                # Intent not recognized
                recognition = Recognition.empty()

            # Print as a line of JSON
            json.dump(recognition.asdict(), sys.stdout, ensure_ascii=False)
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

    if args.intent_graph:
        # Load intent graph from file
        args.intent_graph = Path(args.intent_graph)

        with open(args.intent_graph, "r") as graph_file:
            graph_dict = json.load(graph_file)
    else:
        # Load intent graph from stdin
        if os.isatty(sys.stdin.fileno()):
            print("Reading intent graph JSON from stdin...", file=sys.stderr)

        graph_dict = json.load(sys.stdin)

    if args.slots:
        args.slots = [Path(p) for p in args.slots]

    if args.slot_programs:
        args.slot_programs = [Path(p) for p in args.slot_programs]

    # -------------------------------------------------------------------------

    # Do training
    examples = fuzzywuzzy_train(graph_dict)

    if args.examples:
        # Write examples to SQLite database
        conn = sqlite3.connect(str(args.examples))
        c = conn.cursor()
        c.execute("""DROP TABLE IF EXISTS intents""")
        c.execute("""CREATE TABLE intents (sentence text, path text)""")

        for _, sentences in examples.items():
            for sentence, path in sentences.items():
                c.execute(
                    "INSERT INTO intents VALUES (?, ?)",
                    (sentence, json.dumps(path, ensure_ascii=False)),
                )

        conn.commit()
        conn.close()

        _LOGGER.debug("Wrote %s", str(args.examples))
    else:
        # Write results to stdout
        json.dump(examples, sys.stdout, ensure_ascii=False)
        print("")
        sys.stdout.flush()


# -----------------------------------------------------------------------------


def get_word_transform(name: str) -> typing.Callable[[str], str]:
    """Gets a word transformation function by name."""
    if name == "upper":
        return str.upper

    if name == "lower":
        return str.lower

    return lambda s: s


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
