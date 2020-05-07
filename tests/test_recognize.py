"""Test cases for recognition functions."""
import unittest

from rhasspynlu.ini_jsgf import parse_ini
from rhasspynlu.intent import Intent, Recognition
from rhasspynlu.jsgf import Sentence
from rhasspynlu.jsgf_graph import intents_to_graph

from rhasspyfuzzywuzzy import recognize, train


class RecognizeTestCase(unittest.TestCase):
    """Recognition test cases."""

    def test_single_sentence(self):
        """Single intent, single sentence."""
        intents = parse_ini(
            """
        [TestIntent]
        this is a test?
        """
        )

        graph = intents_to_graph(intents)
        examples = train(graph)

        # Exact
        recognitions = zero_times(recognize("this is a test", graph, examples))

        self.assertEqual(
            recognitions,
            [
                Recognition(
                    intent=Intent(name="TestIntent", confidence=1),
                    text="this is a test?",
                    raw_text="this is a test",
                    tokens=["this", "is", "a", "test?"],
                    raw_tokens=["this", "is", "a", "test"],
                )
            ],
        )

        # Mispellings, too many tokens (lower confidence)
        for sentence in ["this is a bad test", "this iz b tst"]:
            recognitions = zero_times(recognize(sentence, graph, examples))
            self.assertEqual(len(recognitions), 1)

            intent = recognitions[0].intent
            self.assertIsNotNone(intent)
            self.assertLess(intent.confidence, 1.0)

    def test_converters(self):
        """Check sentence with converters."""
        intents = parse_ini(
            """
        [TestIntent]
        this is a test!upper ten:10!int!square
        """
        )

        graph = intents_to_graph(intents)
        examples = train(graph)

        # Should upper-case "test" and convert "ten" -> 10 -> 100
        recognitions = zero_times(
            recognize(
                "this is a test ten",
                graph,
                examples,
                extra_converters={"square": lambda *args: [x ** 2 for x in args]},
            )
        )
        self.assertEqual(
            recognitions,
            [
                Recognition(
                    intent=Intent(name="TestIntent", confidence=1),
                    text="this is a TEST 100",
                    raw_text="this is a test ten",
                    tokens=["this", "is", "a", "TEST", 100],
                    raw_tokens=["this", "is", "a", "test", "ten"],
                )
            ],
        )

    def test_converter_args(self):
        """Check converter with arguments."""
        intents = parse_ini(
            """
        [TestIntent]
        this is a test ten:10!int!pow,3
        """
        )

        graph = intents_to_graph(intents)
        examples = train(graph)

        def pow_converter(*args, converter_args=None):
            exponent = int(converter_args[0]) if converter_args else 1
            return [x ** exponent for x in args]

        # Should convert "ten" -> 10 -> 1000
        recognitions = zero_times(
            recognize(
                "this is a test ten",
                graph,
                examples,
                extra_converters={"pow": pow_converter},
            )
        )
        self.assertEqual(
            recognitions,
            [
                Recognition(
                    intent=Intent(name="TestIntent", confidence=1),
                    text="this is a test 1000",
                    raw_text="this is a test ten",
                    tokens=["this", "is", "a", "test", 1000],
                    raw_tokens=["this", "is", "a", "test", "ten"],
                )
            ],
        )

    def test_intent_filter(self):
        """Identical sentences from two different intents with filter."""
        intents = parse_ini(
            """
        [TestIntent1]
        this is a test

        [TestIntent2]
        this is a test
        """
        )

        graph = intents_to_graph(intents)
        examples = train(graph)

        def intent_filter(name):
            return name == "TestIntent1"

        # Should produce a recognition for first intent only
        recognitions = zero_times(
            recognize("this is a test", graph, examples, intent_filter=intent_filter)
        )
        self.assertEqual(
            recognitions,
            [
                Recognition(
                    intent=Intent(name="TestIntent1", confidence=1),
                    text="this is a test",
                    raw_text="this is a test",
                    tokens=["this", "is", "a", "test"],
                    raw_tokens=["this", "is", "a", "test"],
                )
            ],
        )

    def test_rules(self):
        """Make sure local and remote rules work."""
        intents = parse_ini(
            """
        [Intent1]
        rule = test
        this is a <rule>

        [Intent2]
        rule = this is
        <rule> another <Intent1.rule>
        """
        )

        graph = intents_to_graph(intents)
        examples = train(graph)

        # Lower confidence with no stop words
        recognitions = zero_times(recognize("this is another test", graph, examples))
        self.assertEqual(
            recognitions,
            [
                Recognition(
                    intent=Intent(name="Intent2", confidence=1),
                    text="this is another test",
                    raw_text="this is another test",
                    tokens=["this", "is", "another", "test"],
                    raw_tokens=["this", "is", "another", "test"],
                )
            ],
        )

    def test_optional_entity(self):
        """Ensure entity inside optional is recognized."""
        ini_text = """
        [playBook]
        read me ($audio-book-name){book} in [the] [($assistant-zones){zone}]
        """

        replacements = {
            "$audio-book-name": [Sentence.parse("the hound of the baskervilles")],
            "$assistant-zones": [Sentence.parse("bedroom")],
        }

        graph = intents_to_graph(parse_ini(ini_text), replacements)
        examples = train(graph)

        recognitions = zero_times(
            recognize(
                "read me the hound of the baskervilles in the bedroom", graph, examples
            )
        )
        self.assertEqual(len(recognitions), 1)
        recognition = recognitions[0]
        self.assertTrue(recognition.intent)

        entities = {e.entity: e for e in recognition.entities}
        self.assertIn("book", entities)
        book = entities["book"]
        self.assertEqual(book.value, "the hound of the baskervilles")

        self.assertIn("zone", entities)
        zone = entities["zone"]
        self.assertEqual(zone.value, "bedroom")

    def test_converters_in_entities(self):
        """Check sentence with converters inside an entity."""
        intents = parse_ini(
            """
        [TestIntent]
        this is a test (ten:10!int){number}
        """
        )

        graph = intents_to_graph(intents)
        examples = train(graph)

        # ten -> 10 (int)
        recognitions = zero_times(recognize("this is a test ten", graph, examples))

        self.assertEqual(len(recognitions), 1)
        recognition = recognitions[0]
        self.assertTrue(recognition.intent)

        entities = {e.entity: e for e in recognition.entities}
        self.assertIn("number", entities)
        number = entities["number"]
        self.assertEqual(number.value, 10)

    def test_entity_converter(self):
        """Check sentence with an entity converter."""
        intents = parse_ini(
            """
        [TestIntent]
        this is a test (four: point: two:4.2){number!float}
        """
        )

        graph = intents_to_graph(intents)
        examples = train(graph)

        # "four point two" -> 4.2
        recognitions = zero_times(
            recognize("this is a test four point two", graph, examples)
        )

        self.assertEqual(len(recognitions), 1)
        recognition = recognitions[0]
        self.assertTrue(recognition.intent)

        entities = {e.entity: e for e in recognition.entities}
        self.assertIn("number", entities)
        number = entities["number"]
        self.assertEqual(number.value, 4.2)

    def test_entity_converters_both(self):
        """Check sentence with an entity converter and a converter inside the entity."""
        intents = parse_ini(
            """
        [TestIntent]
        this is a test (four:4 point: two:2){number!floatify}
        """
        )

        graph = intents_to_graph(intents)
        examples = train(graph)

        # "four two" -> 4.2
        recognitions = zero_times(
            recognize(
                "this is a test four point two",
                graph,
                examples,
                extra_converters={"floatify": lambda a, b: [float(f"{a}.{b}")]},
            )
        )

        self.assertEqual(len(recognitions), 1)
        recognition = recognitions[0]
        self.assertTrue(recognition.intent)

        entities = {e.entity: e for e in recognition.entities}
        self.assertIn("number", entities)
        number = entities["number"]
        self.assertEqual(number.value, 4.2)

    def test_sequence_converters(self):
        """Check sentence with sequence converters."""
        intents = parse_ini(
            """
        [TestIntent]
        this (is a test)!upper
        """
        )

        graph = intents_to_graph(intents)
        examples = train(graph)

        # Should upper-case "is a test"
        recognitions = zero_times(recognize("this is a test", graph, examples))
        self.assertEqual(
            recognitions,
            [
                Recognition(
                    intent=Intent(name="TestIntent", confidence=1),
                    text="this IS A TEST",
                    raw_text="this is a test",
                    tokens=["this", "IS", "A", "TEST"],
                    raw_tokens=["this", "is", "a", "test"],
                )
            ],
        )


# # -----------------------------------------------------------------------------


def zero_times(recognitions):
    """Set times to zero so they can be easily compared in assertions"""
    for recognition in recognitions:
        recognition.recognize_seconds = 0

    return recognitions
