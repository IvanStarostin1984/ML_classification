train-logreg:
	python -m src.models.logreg

train-cart:
	python -m src.models.cart

train:
	python -m src.train

eval:
	python -m src.evaluate
