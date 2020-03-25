SHELL := bash
PYTHON_NAME = rhasspyfuzzywuzzy
PACKAGE_NAME = rhasspy-fuzzywuzzy
SOURCE = $(PYTHON_NAME)
PYTHON_FILES = $(SOURCE)/*.py tests/*.py *.py
SHELL_FILES = bin/* debian/bin/*

.PHONY: reformat check dist venv test pyinstaller debian deploy

version := $(shell cat VERSION)
architecture := $(shell bash architecture.sh)

all: venv

# -----------------------------------------------------------------------------
# Python
# -----------------------------------------------------------------------------

reformat:
	scripts/format-code.sh $(PYTHON_FILES)

check:
	scripts/check-code.sh $(PYTHON_FILES)

test:
	scripts/run-tests.sh

coverage:
	coverage report -m

venv:
	scripts/create-venv.sh

dist: sdist debian

sdist:
	python3 setup.py sdist

# -----------------------------------------------------------------------------
# Docker
# -----------------------------------------------------------------------------

docker: pyinstaller
	docker build . -t "rhasspy/$(PACKAGE_NAME):$(version)" -t "rhasspy/$(PACKAGE_NAME):latest"

deploy:
	echo "$$DOCKER_PASSWORD" | docker login -u "$$DOCKER_USERNAME" --password-stdin
	docker push "rhasspy/$(PACKAGE_NAME):$(version)"

# -----------------------------------------------------------------------------
# Debian
# -----------------------------------------------------------------------------

pyinstaller:
	scripts/build-pyinstaller.sh "${architecture}" "${version}"

debian:
	scripts/build-debian.sh "${architecture}" "${version}"
