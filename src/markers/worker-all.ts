import path from 'path';
import { DATASET_FOLDER, DATASET_MARKED_FOLDER } from '../config';
import { readFile, writeFile } from 'fs/promises';
import { addMarkerComments } from './statement-marking';
import { copyFolder, findFilesRecursively, pathExists } from '../file-utils';
import { collect } from '../utils';

const log = (...args: unknown[]) => console.log(`[${new Date().toISOString()}]`, ...args);

async function worker() {
  log(`Copying folder ${DATASET_FOLDER} to ${DATASET_MARKED_FOLDER}`);

  if (await pathExists(DATASET_MARKED_FOLDER)) {
    log(`Folder ${DATASET_MARKED_FOLDER} already exists. Delete it to override`);
    process.exit(1);
  }

  if (!(await pathExists(DATASET_FOLDER))) {
    log(`Folder ${DATASET_FOLDER} does not exist. Create a dataset first`);
    process.exit(1);
  }

  await copyFolder(DATASET_FOLDER, DATASET_MARKED_FOLDER);

  log('Gathering TS file paths...');
  const relTsFilePaths = await collect(
    findFilesRecursively(DATASET_MARKED_FOLDER, ['.ts'], ['.d.ts'], ['node_modules']),
  );

  log(`Adding marker comments to ${relTsFilePaths.length} TS files in ${DATASET_MARKED_FOLDER}`);

  for (const tsFilePath of relTsFilePaths) {
    const absTsFilePath = path.resolve(DATASET_MARKED_FOLDER, tsFilePath);
    const code = await readFile(absTsFilePath, 'utf8');

    try {
      const markedCode = addMarkerComments(code, tsFilePath);
      await writeFile(absTsFilePath, markedCode);
    } catch (e) {
      // error: the old file still exists so it's fine
    }
  }
}

worker();
