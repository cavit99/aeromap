.PHONY: setup doctor lint format test figures demo aeromap-replay aeromap-3d-bridge core-replay

setup:
	uv sync --all-groups

doctor:
	uv run aeromap doctor

lint:
	uv run ruff format --check .
	uv run ruff check .
	uv run mypy src tests

format:
	uv run ruff format .
	uv run ruff check --fix .

test:
	uv run pytest

figures:
	uv run python scripts/generate_aeromap_portfolio_figures.py

demo:
	open docs/demo/aeromap_mission_control.html

aeromap-replay:
	uv run aeromap benchmark aeromap-decision-replay-v03 \
		--config configs/benchmark/aeromap_mission_control_v03.yaml \
		--dataset-npz docs/evidence/aeromap/airfrans_geometry_scalar_dataset.npz \
		--out docs/evidence/aeromap/airfrans_decision_replay_v03.json \
		--svg-dir docs/evidence/aeromap

aeromap-3d-bridge:
	uv run aeromap benchmark aeromap-3d-triage \
		--out docs/evidence/aeromap3d/metadata_triage.json
	uv run aeromap benchmark aeromap-3d-drivaerml-scalars \
		--cache-dir artifacts/benchmark/aeromap3d/drivaerml \
		--out docs/evidence/aeromap3d/drivaerml_scalar_bridge_dataset.json
	uv run aeromap benchmark aeromap-decision-replay-v03 \
		--config configs/benchmark/aeromap_3d_bridge.yaml \
		--dataset-npz docs/evidence/aeromap3d/drivaerml_scalar_bridge_dataset.npz \
		--out docs/evidence/aeromap3d/drivaerml_scalar_bridge_replay.json \
		--svg-dir docs/evidence/aeromap3d

core-replay:
	uv run scripts/run_venturi_core_2d_response_map_replay.py
