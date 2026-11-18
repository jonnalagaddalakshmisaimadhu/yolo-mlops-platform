.PHONY: setup up down logs preprocess split augment train eval export

setup:        ## python env for ml/ (install CUDA torch first: https://pytorch.org/get-started)
	python -m venv .venv && . .venv/bin/activate && pip install -r ml/requirements.txt
up:           ## start postgres, minio, mlflow, api, frontend
	docker compose up -d --build
down:
	docker compose down
logs:
	docker compose logs -f --tail=100
preprocess:
	python ml/src/preprocess.py --src data/raw --dst data/processed --max-size 1280
split:
	python ml/src/split_dataset.py --out data/datasets/v1
augment:
	python ml/src/augment.py --dataset data/datasets/v1 --copies 2
train:
	python ml/src/train.py --config ml/configs/train.yaml
eval:
	python ml/src/evaluate.py --weights models/weights/best.pt --data data/datasets/v1/data.yaml
export:
	python ml/src/export.py --weights models/weights/best.pt --format onnx
