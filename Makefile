# Makefile

# Default target
all: help

# Help target
help:
	@echo "Usage: make [target]"
	@echo "Targets:"
	@echo "  train      Train the model"
	@echo "  data       Download data"
	@echo "  preprocess Preprocess data"
	@echo "  submission Generate submission"
	@echo "  optimize   Optimize with sweep ID"
	@echo ""
	@echo "Variables:"
	@echo "  sweepid       Sweep ID for optimization"

# Train target
train:
	python src/models/make_train.py -m

# Data target
data:
	python src/data/make_data.py

# Submission target
submission:
	python src/models/make_submission.py

# Preprocess target
preprocess:
	python src/data/make_preprocessing.py

# Optimize target
optimize:
	@wandb agent winged-bull/crop-forecasting/$(sweepid)
