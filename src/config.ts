import os from 'os';
import { readFileSync } from 'fs';
import path from 'path';
import { clamp } from './utils';

const config: Record<string, any> = JSON.parse(
  readFileSync(path.resolve(__dirname, '../config.json'), 'utf8'),
);

export const DATA_FOLDER = path.resolve(config.DATA_FOLDER);

const DATASET_FOLDER_NAME = 'dataset';
const DATASET_MARKED_FOLDER_NAME = 'dataset-marked';
const DATASET_MARKED_EXPLICIT_FOLDER_NAME = 'dataset-marked-explicit';
const DATASET_MARKED_UNTYPED_FOLDER_NAME = 'dataset-marked-untyped';

// dataset is in this folder
export const DATASET_FOLDER: string = path.resolve(DATA_FOLDER, DATASET_FOLDER_NAME);

// then the dataset is copied here, and <marker:i> comments are added
export const DATASET_MARKED_FOLDER: string = path.resolve(DATA_FOLDER, DATASET_MARKED_FOLDER_NAME);

// then the marked dataset is copied here, dependencies are installed, and types are inferred
export const DATASET_MARKED_EXPLICIT_FOLDER: string = path.resolve(
  DATA_FOLDER,
  DATASET_MARKED_EXPLICIT_FOLDER_NAME,
);

export const DATASET_MARKED_UNTYPED_FOLDER: string = path.resolve(
  DATA_FOLDER,
  DATASET_MARKED_UNTYPED_FOLDER_NAME,
);

export const SETS_FOLDER: string = path.resolve(DATA_FOLDER, 'sets');

// the train/test/dev files with all the data (instead of just file names)
export const SETS_FILES_FOLDER: string = path.resolve(DATA_FOLDER, 'sets-files');

// folder has structure predictions-folder/model/dataset/predictions.txt and predictions-folder/model/dataset/test.json
export const PREDICTIONS_FOLDER: string = path.resolve(DATA_FOLDER, 'predictions');

export const ALLOWED_CPUS: number = Math.floor(
  clamp(1, os.cpus().length, os.cpus().length * config.ALLOWED_CPUS),
);
export const LINE_MASK_CHANCE: number = config.LINE_MASK_CHANCE;
export const RANDOM_SEED: string = config.RANDOM_SEED;

export const DATASET_SPLIT: {
  TRAIN: number;
  TEST: number;
  DEV: number;
} = config.DATASET_SPLIT;

export const NUM_MARKERS: number = config.NUM_MARKERS;

if (DATASET_SPLIT.TRAIN + DATASET_SPLIT.TEST + DATASET_SPLIT.DEV !== 1.0) {
  throw new Error('Dataset split must total to 1');
}
