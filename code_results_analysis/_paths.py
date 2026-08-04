#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

DATA = os.environ.get("DATA_DIR", os.path.join(ROOT, "data_results_analysis"))
RAW = os.environ.get("SRC", os.path.join(DATA, "source_responses.csv.gz"))
FIGS = os.environ.get("FIG_DIR", os.path.join(ROOT, "figures"))

def d(name):
    return os.path.join(DATA, name)
