import { getFileLines, pathExists } from '../file-utils';
import { DATASET_MARKED_UNTYPED_FOLDER, DATASET_MARKED_FOLDER } from '../config';
import { filePathsFilePath } from '../get-file-paths/master';
import { mkdir, copyFile } from 'fs/promises';
import path from 'path';
import { batchPromises, collect } from '../utils';

async function main() {
  console.log('Creating folder for untyped marked dataset');

  if (await pathExists(DATASET_MARKED_UNTYPED_FOLDER)) {
    console.log(`Folder ${DATASET_MARKED_UNTYPED_FOLDER} already exists. Delete it to override`);
    process.exit(1);
  }

  if (!(await pathExists(DATASET_MARKED_FOLDER))) {
    console.log(`Folder ${DATASET_MARKED_FOLDER} does not exist. Create a marked dataset first`);
    process.exit(1);
  }

  await mkdir(DATASET_MARKED_UNTYPED_FOLDER);

  async function task(relFilePath: string): Promise<void> {
    const fromPath = path.resolve(DATASET_MARKED_FOLDER, relFilePath);
    const toPath = path.resolve(DATASET_MARKED_UNTYPED_FOLDER, relFilePath);
    const toDir = path.dirname(toPath);
    await mkdir(toDir, { recursive: true });
    await copyFile(fromPath, toPath);
  }

  const relFilePaths = await collect(getFileLines(filePathsFilePath));
  await batchPromises(task, relFilePaths, 100);
}

if (require.main === module) {
  main();
}
