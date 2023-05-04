.DEFAULT_GOAL := help


## This help message
.PHONY: help
help:
	@printf "Usage\n";

	@awk '{ \
			if ($$0 ~ /^.PHONY: [a-zA-Z\-\_0-9]+$$/) { \
				helpCommand = substr($$0, index($$0, ":") + 2); \
				if (helpMessage) { \
					printf "\033[36m%-20s\033[0m %s\n", \
						helpCommand, helpMessage; \
					helpMessage = ""; \
				} \
			} else if ($$0 ~ /^[a-zA-Z\-\_0-9.]+:/) { \
				helpCommand = substr($$0, 0, index($$0, ":")); \
				if (helpMessage) { \
					printf "\033[36m%-20s\033[0m %s\n", \
						helpCommand, helpMessage; \
					helpMessage = ""; \
				} \
			} else if ($$0 ~ /^##/) { \
				if (helpMessage) { \
					helpMessage = helpMessage"\n                     "substr($$0, 3); \
				} else { \
					helpMessage = substr($$0, 3); \
				} \
			} else { \
				if (helpMessage) { \
					print "\n                     "helpMessage"\n" \
				} \
				helpMessage = ""; \
			} \
		}' \
		$(MAKEFILE_LIST)

## -- QA Task Runners --

## Run linter
.PHONY: lint
lint:
	poetry run isort imagined_handwriting tests
	poetry run black imagined_handwriting tests
	poetry run flake8 imagined_handwriting tests
	poetry run mypy imagined_handwriting tests

## Run lint check (used for ci/cd)
.PHONY: lint-check
lint-check:
	poetry run isort --check imagined_handwriting tests
	poetry run black --check imagined_handwriting tests
	poetry run flake8 imagined_handwriting tests
	poetry run mypy imagined_handwriting tests

## -- Testing --

## Run tests
.PHONY: test
test:
	poetry run pytest tests/
