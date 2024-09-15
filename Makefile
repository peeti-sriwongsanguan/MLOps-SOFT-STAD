.PHONY: setup run train test docker-build docker-run clean

setup:
	conda env create -f environment.yml
	conda activate walmart-time-series

run:
	conda run -n walmart-time-series python src/train.py

train: run

test:
	conda run -n walmart-time-series pytest tests/

docker-build:
	docker build -t walmart-time-series .

docker-run:
	docker run -it walmart-time-series

clean:
	conda env remove -n walmart-time-series
	rm -rf __pycache__
	rm -rf .pytest_cache