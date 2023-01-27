#!/bin/bash

# Script that reads `inferredProjectPaths.txt`, and creates a new file called `explicitProjectPaths.txt`
# This script is used to filter out projects that are not valid (i.e. have missing dependencies)
# This is done using the exit code of `npm ls`

CURRENT_DIR="$(pwd)"
DATASET_MARKED_EXPLICIT_NAME="dataset-marked-explicit"
VALID_PROJECT_PATHS_FILE_PATH="$CURRENT_DIR/validProjectPaths.txt"

if [[ ! -f "$CURRENT_DIR/inferredProjectPaths.txt" ]]
then
    echo "inferredProjectPaths.txt does not exist. Run create-datasets.sh first to ensure that dataset-marked-explicit and inferredProjectPaths.txt are created"
    exit 1
fi

if [[ -f "$VALID_PROJECT_PATHS_FILE_PATH" ]]
then
    echo "$VALID_PROJECT_PATHS_FILE_PATH already exists. Delete it to override."
    exit 1
fi

PROJECT_PATH_COUNT=$(cat inferredProjectPaths.txt | wc -l)
i=1

# split the file on newlines
IFS="\n"

cat inferredProjectPaths.txt | while read REL_TSCONFIG_PATH
do
  REL_PROJECT_PATH=$(dirname "$REL_TSCONFIG_PATH")
  ABS_PROJECT_PATH="$CURRENT_DIR/$DATASET_MARKED_EXPLICIT_NAME/$REL_PROJECT_PATH"

  # log which project, and the process
  echo "[$i/$PROJECT_PATH_COUNT] $ABS_PROJECT_PATH"
  i=$((i+1))

  cd "$ABS_PROJECT_PATH"

  # npm ls lists all dependencies, and returns 0 if succesful
  # this code checks whether it indeed returns 0

  npm ls &> /dev/null
  if [[ $? -eq 0 ]]; then
      echo "$REL_PROJECT_PATH" >> "$VALID_PROJECT_PATHS_FILE_PATH"
  fi
done
