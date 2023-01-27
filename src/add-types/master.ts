import { batchBySize, collect, collectSet, shuffle } from '../utils';
import path from 'path';
import { ALLOWED_CPUS, DATA_FOLDER, DATASET_MARKED_EXPLICIT_FOLDER } from '../config';
import { findFilesRecursively, findTsConfigs, getFileLines, pathExists } from '../file-utils';
import { multithread } from '../threading';
import { createWriteStream } from 'fs';

export async function master() {
  if (!(await pathExists(DATASET_MARKED_EXPLICIT_FOLDER))) {
    console.log(
      `${DATASET_MARKED_EXPLICIT_FOLDER} does not exist. Copy the dataset-marked folder to this location first with ./copy-folder.ts`,
    );
    process.exit(1);
  }

  const tsConfigs = await collect(findTsConfigs(DATASET_MARKED_EXPLICIT_FOLDER));
  console.log(
    `Found ${tsConfigs.length} TypeScript projects in ${DATASET_MARKED_EXPLICIT_FOLDER}. Inferring types...`,
  );

  const inferredProjectPathsFilePath = path.resolve(DATA_FOLDER, 'inferredProjectPaths.txt');

  let handledTsConfigs = new Set<string>();

  if (await pathExists(inferredProjectPathsFilePath)) {
    handledTsConfigs = await collectSet(getFileLines(inferredProjectPathsFilePath));
  }

  console.log(`Found ${handledTsConfigs.size} already inferred projects`);

  const tsConfigsToInfer = tsConfigs.filter((tsConfig) => !handledTsConfigs.has(tsConfig));
  shuffle(tsConfigsToInfer);

  const handledTsConfigsWriteStream = createWriteStream(inferredProjectPathsFilePath, {
    flags: 'a',
    encoding: 'utf8',
  });

  await multithread(
    batchBySize(tsConfigsToInfer, 10),
    path.resolve(__dirname, './worker.js'),
    (handledTsConfigPath: string) => {
      handledTsConfigsWriteStream.write(`${handledTsConfigPath}\n`);
      return 0;
    },
    0,
    ALLOWED_CPUS,
  );

  handledTsConfigsWriteStream.end();

  console.log('Done!');
}

master();
