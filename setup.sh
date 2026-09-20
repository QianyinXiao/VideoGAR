#!/usr/bin/env bash

# source setup.sh
export VIDEOGAR_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${PYTHONPATH}:${VIDEOGAR_ROOT}"
else
  export PYTHONPATH="${VIDEOGAR_ROOT}"
fi

# Avoid Hugging Face tokenizers warnings in DataLoader workers.
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

echo "${PYTHONPATH}"
