SHELL := /bin/bash
.DEFAULT_GOAL := help

# Resolve every target from this Makefile's application directory.
MAKEFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
APP_DIR := $(patsubst %/,%,$(dir $(MAKEFILE_PATH)))
SRC_DIR := $(APP_DIR)/src
TRAIN_DIR := $(APP_DIR)/train
DATA_DIR := $(APP_DIR)/data
EVAL_DIR := $(APP_DIR)/eval
RUNTIME_TEST_DIR := $(SRC_DIR)/tests
DATA_TEST_DIR := $(DATA_DIR)/tests
TRAIN_TEST_DIR := $(TRAIN_DIR)/tests
EVAL_TEST_DIR := $(EVAL_DIR)/tests
TEST_DIRS := $(DATA_TEST_DIR) $(TRAIN_TEST_DIR) $(EVAL_TEST_DIR) $(RUNTIME_TEST_DIR)
PYTEST_CONFIG := $(APP_DIR)/pytest.ini
APP_ENV := $(SRC_DIR)/.env
ENV_FILE_FLAG := $(if $(wildcard $(APP_ENV)),--env-file "$(APP_ENV)",)
RUN_ENV := PYTHONPATH="$(SRC_DIR):$(TRAIN_DIR):$(APP_DIR)" PLAYWRIGHT_BROWSERS_PATH="$(SRC_DIR)/.runtime/ms-playwright"

APP := cn_travel
TRAIN_PYTHON ?= python3
VERL_PYTHON ?= python3
TESTS ?= $(TEST_DIRS)
PYTEST_ARGS ?=
EVAL_ARGS ?=
PROXY_PORT ?= 8765
MODEL_NAMES ?= qwen embedding

QWEN_SFT := $(TRAIN_DIR)/qwen3_5_0_8b_lora/SFT/run.sh
LFM_SFT := $(TRAIN_DIR)/lfm2_5_350m_lora/SFT/run.sh
QWEN_TRL_STEP := $(TRAIN_DIR)/qwen3_5_0_8b_lora/GRPO_TRL/run.sh
QWEN_TRL_TURN := $(TRAIN_DIR)/qwen3_5_0_8b_lora/GRPO_TRL/run_traj.sh
LFM_TRL_STEP_CONFIG := $(TRAIN_DIR)/lfm2_5_350m_lora/GRPO_TRL/train_step_config.json
LFM_TRL_TURN_CONFIG := $(TRAIN_DIR)/lfm2_5_350m_lora/GRPO_TRL/train_traj_config.json
QWEN_VERL := $(TRAIN_DIR)/qwen3_5_0_8b_lora/GRPO_VERL/run_verl.sh
LFM_VERL := $(TRAIN_DIR)/lfm2_5_350m_lora/GRPO_VERL/run_verl.sh
LFM_VERL_RUN2_CONFIG := $(TRAIN_DIR)/lfm2_5_350m_lora/GRPO_VERL/verl_traj_config.json
LFM_VERL_RUN3_CONFIG := $(TRAIN_DIR)/lfm2_5_350m_lora/GRPO_VERL/verl_traj_config_run3.json
EVALUATOR := $(EVAL_DIR)/evaluate.py
MODEL_FETCHER := $(TRAIN_DIR)/fetch_models.py

.PHONY: help install install-browser fetch-models promote-runtime validate build-kb serve-policy serve chat eval \
	claude-proxy codex-proxy \
	test test-required test-offline test-strict test-boundary test-standalone \
	train-qwen-sft train-lfm-sft \
	train-qwen-grpo-step train-qwen-grpo-turn \
	train-lfm-grpo-step train-lfm-grpo-turn \
	train-qwen-grpo-verl train-lfm-grpo-verl-run2 train-lfm-grpo-verl-run3

help: ## Show CN Travel commands
	@awk 'BEGIN {FS = ":.*## "; printf "CN Travel\n\n"} /^[a-z0-9-]+:.*## / {printf "  \033[36m%-30s\033[0m %s\n", $$1, $$2}' "$(MAKEFILE_PATH)"

install: ## Install the application, local RAG, and offline-test dependencies
	cd "$(SRC_DIR)" && uv sync --extra test

install-browser: ## Install Chromium into this application's runtime directory
	cd "$(SRC_DIR)" && $(RUN_ENV) uv run --project "$(SRC_DIR)" playwright install chromium

fetch-models: ## Fetch base and embedding models into this application
	cd "$(APP_DIR)" && $(RUN_ENV) uv run --project "$(SRC_DIR)" --extra models python "$(MODEL_FETCHER)" $(MODEL_NAMES)

promote-runtime: ## Copy and verify the selected policy and retrieval artifacts
	cd "$(APP_DIR)" && $(RUN_ENV) uv run --project "$(SRC_DIR)" python "$(TRAIN_DIR)/promote_runtime.py" --write

validate: ## Validate the CN Travel application contract
	cd "$(SRC_DIR)" && $(RUN_ENV) uv run --project "$(SRC_DIR)" $(ENV_FILE_FLAG) python -m cn_travel.cli validate

build-kb: ## Build the CN Travel vector store from the guide corpus
	cd "$(APP_DIR)" && $(RUN_ENV) uv run --project "$(SRC_DIR)" $(ENV_FILE_FLAG) python "$(DATA_DIR)/scripts/build_knowledge.py"

serve-policy: ## Serve the promoted policy through an OpenAI-compatible endpoint
	cd "$(SRC_DIR)" && $(RUN_ENV) uv run --project "$(SRC_DIR)" --extra inference $(ENV_FILE_FLAG) cn-travel-policy

serve: ## Start the CN Travel HTTP API
	cd "$(SRC_DIR)" && $(RUN_ENV) uv run --project "$(SRC_DIR)" $(ENV_FILE_FLAG) python -m cn_travel serve

chat: ## Start the CN Travel interactive client
	cd "$(SRC_DIR)" && $(RUN_ENV) uv run --project "$(SRC_DIR)" $(ENV_FILE_FLAG) python -m cn_travel.cli chat

eval: ## Run the sealed evaluator; pass EVAL_ARGS='--system ... --model ... --base-url ...'
	@test -n "$(strip $(EVAL_ARGS))" || { echo "EVAL_ARGS is required; run 'make help' for the invocation contract" >&2; exit 2; }
	cd "$(APP_DIR)" && $(RUN_ENV) uv run --project "$(SRC_DIR)" $(ENV_FILE_FLAG) python "$(EVALUATOR)" $(EVAL_ARGS)

claude-proxy: ## Serve the Claude CLI through the evaluator's local protocol
	cd "$(APP_DIR)" && $(RUN_ENV) uv run --project "$(SRC_DIR)" $(ENV_FILE_FLAG) python eval/claude_proxy.py --port "$(PROXY_PORT)"

codex-proxy: ## Serve the Codex CLI through the evaluator's local protocol
	cd "$(APP_DIR)" && $(RUN_ENV) uv run --project "$(SRC_DIR)" $(ENV_FILE_FLAG) python eval/codex_proxy.py --port "$(PROXY_PORT)"

test: ## Run all tests whose declared capabilities are available
	cd "$(APP_DIR)" && $(RUN_ENV) uv run --project "$(SRC_DIR)" --extra test $(ENV_FILE_FLAG) pytest -c "$(PYTEST_CONFIG)" $(TESTS) $(PYTEST_ARGS)

test-required: ## Require every live, local-model, retrieval, and provider capability
	cd "$(APP_DIR)" && CN_TRAVEL_REQUIRED_CAPABILITIES="live,model,policy,reviews,rag,amap,hotel_backend_real" $(RUN_ENV) uv run --project "$(SRC_DIR)" --extra test $(ENV_FILE_FLAG) pytest -c "$(PYTEST_CONFIG)" $(TESTS) $(PYTEST_ARGS)

test-offline: ## Run deterministic tests without API, network, RAG, or model capabilities
	cd "$(APP_DIR)" && $(RUN_ENV) uv run --project "$(SRC_DIR)" --extra test $(ENV_FILE_FLAG) pytest -c "$(PYTEST_CONFIG)" $(TESTS) -m "not live and not model and not policy and not reviews and not rag and not amap" $(PYTEST_ARGS)

test-strict: ## Run all available tests with expected defects promoted to failures
	cd "$(APP_DIR)" && STRICT_DEFECTS=1 $(RUN_ENV) uv run --project "$(SRC_DIR)" --extra test $(ENV_FILE_FLAG) pytest -c "$(PYTEST_CONFIG)" $(TESTS) $(PYTEST_ARGS)

test-boundary: ## Verify that active application code owns every dependency
	cd "$(APP_DIR)" && $(RUN_ENV) uv run --project "$(SRC_DIR)" --extra test pytest -c "$(PYTEST_CONFIG)" "$(RUNTIME_TEST_DIR)/test_repository_boundary.py" -m "not live and not model and not policy and not reviews and not rag and not amap"

test-standalone: ## Validate the complete application in a fresh temporary application directory
	bash "$(RUNTIME_TEST_DIR)/standalone_check.sh" "$(SRC_DIR)"

train-qwen-sft: ## Train Qwen3.5-0.8B SFT with its sealed JSON config
	cd "$(APP_DIR)" && PYTHON="$(TRAIN_PYTHON)" bash "$(QWEN_SFT)"

train-lfm-sft: ## Train LFM2.5-350M SFT with its sealed JSON config
	cd "$(APP_DIR)" && PYTHON="$(TRAIN_PYTHON)" bash "$(LFM_SFT)"

train-qwen-grpo-step: ## Train Qwen per-step GRPO with TRL
	cd "$(APP_DIR)" && PYTHON="$(TRAIN_PYTHON)" bash "$(QWEN_TRL_STEP)"

train-qwen-grpo-turn: ## Train Qwen per-user-turn GRPO with TRL
	cd "$(APP_DIR)" && PYTHON="$(TRAIN_PYTHON)" bash "$(QWEN_TRL_TURN)"

train-lfm-grpo-step: ## Train LFM per-step GRPO with TRL
	cd "$(APP_DIR)" && PYTHON="$(TRAIN_PYTHON)" CONFIG="$(LFM_TRL_STEP_CONFIG)" bash "$(QWEN_TRL_STEP)"

train-lfm-grpo-turn: ## Train LFM per-user-turn GRPO with TRL
	cd "$(APP_DIR)" && PYTHON="$(TRAIN_PYTHON)" CONFIG="$(LFM_TRL_TURN_CONFIG)" bash "$(QWEN_TRL_TURN)"

train-qwen-grpo-verl: ## Train Qwen GRPO with VERL
	cd "$(APP_DIR)" && PYTHON="$(VERL_PYTHON)" bash "$(QWEN_VERL)"

train-lfm-grpo-verl-run2: ## Train LFM whole-trajectory GRPO with the Run 2 configuration
	cd "$(APP_DIR)" && PYTHON="$(VERL_PYTHON)" CONFIG="$(LFM_VERL_RUN2_CONFIG)" bash "$(LFM_VERL)"

train-lfm-grpo-verl-run3: ## Train LFM whole-trajectory GRPO with the Run 3 configuration
	cd "$(APP_DIR)" && PYTHON="$(VERL_PYTHON)" CONFIG="$(LFM_VERL_RUN3_CONFIG)" bash "$(LFM_VERL)"
