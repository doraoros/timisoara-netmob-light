.PHONY: setup dataset validate train experiment robustness sequential explainability \
        interpretability routemap routemap-static routemap-interactive save-model predict evaluate test thesis dissertation \
        all run-all notebook

PY ?= python

setup:
	$(PY) -m venv .venv
	.venv/Scripts/pip install -e . 2>/dev/null || .venv/bin/pip install -e .

dataset:
	$(PY) -m src.utils.build_radio_gps_dataset

validate:
	$(PY) -m src.utils.validate_dataset

train:
	$(PY) -m src.models.train_handoff_models

experiment:
	$(PY) scripts/experiment_feature_engineering.py

robustness:
	$(PY) scripts/experiment_robustness.py

sequential:
	$(PY) scripts/experiment_sequential.py

explainability:
	$(PY) scripts/experiment_explainability.py

interpretability:
	$(PY) scripts/interpretability_classic.py

routemap:
	$(PY) scripts/generate_route_map.py
	$(PY) scripts/generate_interactive_route_map.py

routemap-static:
	$(PY) scripts/generate_route_map.py

routemap-interactive:
	$(PY) scripts/generate_interactive_route_map.py

save-model:
	$(PY) scripts/train_and_save_model.py

predict:
	$(PY) scripts/predict.py $(ARGS)

evaluate:
	$(PY) scripts/evaluate_on_new_session.py $(ARGS)

test:
	$(PY) -m pytest tests/ -q

thesis:
	$(PY) -m src.reports.generate_thesis_figures

dissertation:
	$(PY) -m src.reports.generate_dissertation_bundle

all: dataset validate train experiment robustness interpretability routemap save-model thesis dissertation

run-all:
	bash run_all.sh

notebook:
	$(PY) -m jupyter notebook notebooks
