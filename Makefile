train-logreg:
	python -m src.models.logreg

train-cart:
	python -m src.models.cart

train: train-logreg train-cart
