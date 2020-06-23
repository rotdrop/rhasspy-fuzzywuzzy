# Rhasspy Fuzzywuzzy

[![Continous Integration](https://github.com/rhasspy/rhasspy-fuzzywuzzy/workflows/Tests/badge.svg)](https://github.com/rhasspy/rhasspy-fuzzywuzzy/actions)
[![PyPI package version](https://img.shields.io/pypi/v/rhasspy-fuzzywuzzy.svg)](https://pypi.org/project/rhasspy-fuzzywuzzy)
[![Python versions](https://img.shields.io/pypi/pyversions/rhasspy-fuzzywuzzy.svg)](https://www.python.org)
[![GitHub license](https://img.shields.io/github/license/rhasspy/rhasspy-fuzzywuzzy.svg)](https://github.com/rhasspy/rhasspy-fuzzywuzzy/blob/master/LICENSE)

Intent recognition for Rhasspy using [rapidfuzz](https://github.com/rhasspy/rapidfuzz).

## Requirements

* Python 3.7

## Installation

```bash
$ git clone https://github.com/rhasspy/rhasspy-fuzzywuzzy
$ cd rhasspy-fuzzywuzzy
$ ./configure
$ make
$ make install
```

## Deployment

```bash
$ make dist
```

## Running

```bash
$ bin/rhasspy-fuzzywuzzy <ARGS>
```

## Command-Line Options

```
usage: rhasspy-fuzzywuzzy [-h] [--debug] {recognize,train} ...

positional arguments:
  {recognize,train}
    recognize        Do intent recognition
    train            Generate intent examples from sentences and slots

optional arguments:
  -h, --help         show this help message and exit
  --debug            Print DEBUG messages to the console
```

### Recognize

```
usage: rhasspy-fuzzywuzzy recognize [-h] --examples EXAMPLES --intent-graph
                                    INTENT_GRAPH [--replace-numbers]
                                    [--language LANGUAGE]
                                    [--word-casing {upper,lower,ignore}]
                                    [query [query ...]]

positional arguments:
  query                 Query input sentences

optional arguments:
  -h, --help            show this help message and exit
  --examples EXAMPLES   Path to examples JSON file
  --intent-graph INTENT_GRAPH
                        Path to intent graph JSON file
  --replace-numbers     Automatically replace numbers in query text
  --language LANGUAGE   Language used for number replacement
  --word-casing {upper,lower,ignore}
                        Case transformation applied to query text
```

### Train

```
usage: rhasspy-fuzzywuzzy train [-h] [--examples EXAMPLES]
                                [--intent-graph INTENT_GRAPH]
                                [--sentences SENTENCES]

optional arguments:
  -h, --help            show this help message and exit
  --examples EXAMPLES   Path to write examples JSON file
  --intent-graph INTENT_GRAPH
                        Path to write intent graph JSON file
  --sentences SENTENCES
                        Paths to sentences ini files
```
