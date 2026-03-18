.PHONY: install dev ingest test lint docker-up docker-down

install:
	pip install -r requirements.txt
	python -m nltk.downloader punkt punkt_tab

dev:
	uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

ingest:
	curl -X POST http://localhost:8000/api/v1/ingest \
		-H "Content-Type: application/json" \
		-d '{"arxiv_id": "2307.09288", "strategy": "sentence"}'

test:
	pytest tests/ -v -s

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down
