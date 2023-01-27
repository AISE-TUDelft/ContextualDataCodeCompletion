#!/bin/bash
set -e

###############
# evaluate.sh #
############################
# Evaluate all predictions #
############################

./node_modules/.bin/tsc -b

node ./dist/postprocessing/postprocess.ts unixcoder
python py/evaluate.py unixcoder
python py/evaluate.py unixcoder single
python py/statistical-analysis.py unixcoder
python py/statistical-analysis.py unixcoder single

node ./dist/postprocessing/postprocess.ts codegpt
python py/evaluate.py codegpt
python py/evaluate.py codegpt single
python py/statistical-analysis.py codegpt
python py/statistical-analysis.py codegpt single

node ./dist/postprocessing/postprocess.ts incoder
python py/evaluate.py incoder
python py/evaluate.py incoder single
python py/statistical-analysis.py incoder
python py/statistical-analysis.py incoder single
