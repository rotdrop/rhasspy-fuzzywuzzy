# Rhasspy Fuzzywuzzy

Intent recognition for Rhasspy using [fuzzywuzzy](https://github.com/seatgeek/fuzzywuzzy).

## Running With Docker

```bash
docker run -it rhasspy/rhasspy-fuzzywuzzy:<VERSION> <ARGS>
```

## Building From Source

Clone the repository and create the virtual environment:

```bash
git clone https://github.com/rhasspy/rhasspy-fuzzywuzzy.git
cd rhasspy-fuzzywuzzy
make venv
```

Run the `bin/rhasspy-fuzzywuzzy` script to access the command-line interface:

```bash
bin/rhasspy-fuzzywuzzy --help
```

## Building the Debian Package

Follow the instructions to build from source, then run:

```bash
source .venv/bin/activate
make debian
```

If successful, you'll find a `.deb` file in the `dist` directory that can be installed with `apt`.

## Building the Docker Image

Follow the instructions to build from source, then run:

```bash
source .venv/bin/activate
make docker
```

This will create a Docker image tagged `rhasspy/rhasspy-fuzzywuzzy:<VERSION>` where `VERSION` comes from the file of the same name in the source root directory.

NOTE: If you add things to the Docker image, make sure to whitelist them in `.dockerignore`.

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
                                [--sentences SENTENCES] [--slots SLOTS]
                                [--slot-programs SLOT_PROGRAMS]
                                [--replace-numbers] [--language LANGUAGE]
                                [--word-casing {upper,lower,ignore}]

optional arguments:
  -h, --help            show this help message and exit
  --examples EXAMPLES   Path to write examples JSON file
  --intent-graph INTENT_GRAPH
                        Path to write intent graph JSON file
  --sentences SENTENCES
                        Paths to sentences ini files
  --slots SLOTS         Directories with static slot text files
  --slot-programs SLOT_PROGRAMS
                        Directories with slot programs
  --replace-numbers     Automatically replace numbers and number ranges in
                        sentences/slots
  --language LANGUAGE   Language used for number replacement
  --word-casing {upper,lower,ignore}
                        Case transformation applied to words
```
