import { collect, collectSet, randChoiceWeighted, setIntersection } from '../utils';
import { getFileLines, pathExists } from '../file-utils';
import { DATA_FOLDER, DATASET_SPLIT, RANDOM_SEED, SETS_FOLDER } from '../config';
import { createWriteStream } from 'fs';
import path from 'path';
import random from 'random-seed';
import { mkdir } from 'fs/promises';
import { filePathsFilePath } from '../get-file-paths/master';

export const TRAIN_FILE_PATH = path.resolve(SETS_FOLDER, 'train.txt');
export const TEST_FILE_PATH = path.resolve(SETS_FOLDER, 'test.txt');
export const DEV_FILE_PATH = path.resolve(SETS_FOLDER, 'dev.txt');

async function split() {
  if (!(await pathExists(filePathsFilePath))) {
    console.log(
      `File ${filePathsFilePath} does not exist. Make sure it exists by running ./src/get-file-paths/master.ts`,
    );
    process.exit(1);
  }

  if (await pathExists(SETS_FOLDER)) {
    console.log(`Folder ${SETS_FOLDER} already exists. Delete to override`);
    process.exit(1);
  }

  await mkdir(SETS_FOLDER);

  const files = await collect(getFileLines(filePathsFilePath));

  console.log(`Found ${files.length} files`);

  console.log(
    `Splitting them into train/test/dev = ${DATASET_SPLIT.TRAIN}/${DATASET_SPLIT.TEST}/${DATASET_SPLIT.DEV}`,
  );

  const train = createWriteStream(TRAIN_FILE_PATH, 'utf8');
  const test = createWriteStream(TEST_FILE_PATH, 'utf8');
  const dev = createWriteStream(DEV_FILE_PATH, 'utf8');

  const prng = random.create(RANDOM_SEED);
  const streams = { train, test, dev } as const;
  const counts = { train: 0, test: 0, dev: 0 };
  for (const file of files) {
    const set = randChoiceWeighted<keyof typeof streams>(
      ['train', 'test', 'dev'],
      [DATASET_SPLIT.TRAIN, DATASET_SPLIT.TEST, DATASET_SPLIT.DEV],
      prng.random.bind(prng),
    );
    streams[set].write(`${file}\n`);
    counts[set]++;
  }

  train.end();
  test.end();
  dev.end();

  console.log('Done');
  console.log(counts);
}

split();
