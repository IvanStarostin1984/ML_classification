train-logreg:
	python -m src.models.logreg

train-cart:
	python -m src.models.cart

train:
	python -m src.train

eval:
	python -m src.evaluate

test:
	PYTHONPATH=$(PWD) python -m pytest -q

docs:
	sphinx-build -b html docs docs/_build

lint-docs:
	npx markdownlint-cli '**/*.md' --ignore node_modules
