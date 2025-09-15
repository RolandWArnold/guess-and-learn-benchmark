# scripts/reproduce_smoke.sh
#!/usr/bin/env bash
set -euo pipefail
python src/run_all.py --datasets mnist --models cnn \
  --strategies entropy --tracks 'G&L-SO' \
  --seeds 0 --subset 100 --output_dir ./runs --devices cpu
