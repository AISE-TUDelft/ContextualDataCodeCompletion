import { ALLOWED_CPUS, DATASET_FOLDER, DATASET_MARKED_FOLDER } from '../config';
import { copyFolder, findFilesRecursively, pathExists } from '../file-utils';
import { batchBySize, collect } from '../utils';
import { multithread } from '../threading';
import path from 'path';

async function master() {
  console.log(`Copying folder ${DATASET_FOLDER} to ${DATASET_MARKED_FOLDER}`);

  if (await pathExists(DATASET_MARKED_FOLDER)) {
    console.log(`Folder ${DATASET_MARKED_FOLDER} already exists. Delete it to override`);
    process.exit(1);
  }

  if (!(await pathExists(DATASET_FOLDER))) {
    console.log(`Folder ${DATASET_FOLDER} does not exist. Create a dataset first`);
    process.exit(1);
  }

  await copyFolder(DATASET_FOLDER, DATASET_MARKED_FOLDER);

  console.log('Gathering TS file paths...');
  const filePaths = await collect(
    findFilesRecursively(DATASET_MARKED_FOLDER, ['.ts'], ['.d.ts'], ['node_modules']),
  );

  console.log(`Adding marker comments to ${filePaths.length} TS files in ${DATASET_MARKED_FOLDER}`);

  await multithread(
    batchBySize(filePaths, 25),
    path.resolve(__dirname, './worker.js'),
    () => 0,
    0,
    ALLOWED_CPUS,
  );

  console.log('Done!');
}

master();
