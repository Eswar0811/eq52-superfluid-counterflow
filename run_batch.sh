#!/bin/bash
# usage: run_batch.sh <parallel> <config:seed> ...   (runs inverse.py jobs, N at a time)
cd "$(dirname "$0")/src"; N=$1; shift
printf '%s\n' "$@" | xargs -P "$N" -I{} bash -c 'c=${1%%:*}; s=${1##*:}; PINN_THREADS=2 ../.venv/bin/python inverse.py $c $s > ../logs/inv_${c}_s${s}.log 2>&1' _ {}
