create_training_env:
	python3 -m venv venv
	source venv/bin/activate && \
	pip install -r requirements.txt && \
	mkdir -p data && \
	python src/tiny_shakespeare.py

.PHONY: create_training_env 