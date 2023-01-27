#!/bin/bash
set -e

###################
# create-datasets #
#################################################
# The datasets should already have been fetched #
# This script the datasets with markers, with   #
# different type explicitness rates: TS704-NT,  #
# TS704-OT, and TS704-AT                        #
#################################################

# build all the typescript code
./node_modules/.bin/tsc -b

# 1) add markers
echo "[SCRIPT] [1/5] Adding Markers"
node ./dist/markers/master.js

# 2) add type annotations
echo "[SCRIPT] [2/5] Installing Dependencies & Adding Types"
node ./dist/add-types/copy-folder.js

# install third-party dependencies
set +e
CUR_PATH="$(pwd)"
for PACKAGE_JSON_PATH in $(find "./data/dataset-marked-explicit" -type f -name "package.json" -not -path "*/node_modules/*"); do
for PACKAGE_JSON_PATH in $(cat locs.txt | tail -n +10465); do
 PACKAGE_JSON_FOLDER_PATH="$(dirname "$PACKAGE_JSON_PATH")"
 echo "Installing dependencies for $PACKAGE_JSON_FOLDER_PATH"
 cd "$PACKAGE_JSON_FOLDER_PATH"
 npm install --ignore-scripts
 cd "$CUR_PATH"
done
set -e

node --max-old-space-size=8192 ./dist/add-types/master.js

./data/get_valid_explicit_projects.sh


# 3) get all file paths (only valid projects are considered)
echo "[SCRIPT] [3/5] Getting all file paths"
node ./dist/get-file-paths/master.js

# 4) create untyped dataset
echo "[SCRIPT] [4/5] Creating untyped dataset"
node ./dist/remove-types/copy-folder.js
node --max-old-space-size=8192 ./dist/remove-types/master.js

# 5) analyze dataset
node ./dist/analyze-dataset/master.js

# 5) split data
echo "[SCRIPT] [4/5] Splitting Data"
node ./dist/split-data/split-data.js

# 6) create sets for UniXcoder and InCoder
echo "[SCRIPT] [5/5] Creating Sets for UniXcoder and InCoder"
node ./dist/create-model-files/create-model-files.js unixcoder
node ./dist/create-model-files/create-model-files.js incoder
