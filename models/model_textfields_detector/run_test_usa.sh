#!/usr/bin/env bash
# Запуск тестов для всех штатов USA (модели runs/usa_<state>/weights/best.pt).

set -e
cd "$(dirname "$0")/../.."
ROOT="$PWD"
SCRIPT="$ROOT/models/model_textfields_detector/test_usa.py"

python "$SCRIPT" all
