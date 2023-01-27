import { batchBySize, collect, mapAsync } from '../utils';
import path from 'path';
import {
  ALLOWED_CPUS,
  DATASET_MARKED_EXPLICIT_FOLDER,
  DATASET_MARKED_FOLDER,
  DATASET_MARKED_UNTYPED_FOLDER,
} from '../config';
import { getFileLines, pathExists } from '../file-utils';
import { multithread } from '../threading';
import { filePathsFilePath } from '../get-file-paths/master';

const folders = [
  DATASET_MARKED_UNTYPED_FOLDER,
  DATASET_MARKED_EXPLICIT_FOLDER,
  DATASET_MARKED_FOLDER,
];

export async function master() {
  for (const folder of folders) {
    if (!(await pathExists(folder))) {
      console.error(`Folder ${folder} does not exist. Make sure to create datasets first`);
      process.exit(1);
    }
  }

  if (!(await pathExists(filePathsFilePath))) {
    throw new Error('File paths file does not exist. Run ./src/get-file-paths/master.ts first.');
  }

  if (!(await pathExists(DATASET_MARKED_UNTYPED_FOLDER))) {
    console.log(
      `${DATASET_MARKED_UNTYPED_FOLDER} does not exist. Copy the dataset-marked folder to this location first with ./copy-folder.ts`,
    );
    process.exit(1);
  }

  const files = await collect(
    mapAsync(getFileLines(filePathsFilePath), (filePath) =>
      path.resolve(DATASET_MARKED_UNTYPED_FOLDER, filePath),
    ),
  );

  console.log(`Removing types from ${files.length} files`);

  await multithread(
    batchBySize(files, 250),
    path.resolve(__dirname, './worker.js'),
    () => 0,
    0,
    ALLOWED_CPUS,
  );

  console.log('Done!');
}

master();
