###############################################################################
# COMMANDS
###############################################################################
.PHONY: clean
## Clean python cache file.
clean:
	find . -name '*.pyo' -delete
	find . -name '*.pyc' -delete
	find . -name __pycache__ -delete
	find . -name '*~' -delete
	find . -name .coverage -delete
	find . -name 'coverage.*' -delete
	find . -name 'requirements*.txt' -delete
	find . -type d -name .pytest_cache -exec rm -r {} +
	find . -type d -name .ruff_cache -exec rm -r {} +
	find . -type d -name .mypy_cache -exec rm -r {} +

.PHONY: install-pdm
## install pdm before environment setup
install-pdm:
	python -m pip install -U \
	    pip setuptools wheel pdm

.PHONY: update-lock
## update pdm.lock
update-lock:
	pdm update --no-sync

.PHONY: deploy-dev-x86
## deploy x86 dev environment
deploy-dev-x86:
	pdm sync -G dev -G repl -G x86 --clean

.PHONY: deploy-dev-osx
## deploy OSX dev environment
deploy-dev-osx:
	pdm sync -G dev -G repl -G osx --clean

.PHONY: format
## isort and yapf formatting
format:
	isort src tests
	yapf -i -r src tests

.PHONY: lint
## pylint check
lint:
	ruff check src/rcnn --show-source --show-fixes \
	    --exit-zero

.PHONY: test
test:
	PYTHONPATH=./src \
	    pytest -s -v --cov=app --cov-config=pyproject.toml \
	    > coverage.txt
