/*
 * This script gets all project file paths and puts it in ./data/file-paths.txt
 * It only considers valid projects, and only considers files that are in all projects
 */

import path from 'path';
import {
  ALLOWED_CPUS,
  DATA_FOLDER,
  DATASET_MARKED_EXPLICIT_FOLDER,
  DATASET_MARKED_FOLDER,
} from '../config';
import { getFileLines, pathExists } from '../file-utils';
import { createWriteStream } from 'fs';
import { batchBySize, collectSet } from '../utils';
import { multithread } from '../threading';

export const validProjectPathsPath = path.resolve(DATA_FOLDER, 'validProjectPaths.txt');

const folders = [DATASET_MARKED_EXPLICIT_FOLDER, DATASET_MARKED_FOLDER];

export const filePathsFilePath = path.resolve(DATA_FOLDER, 'file-paths.txt');

async function main() {
  if (!(await pathExists(validProjectPathsPath))) {
    console.error(
      `${validProjectPathsPath} does not exist. Run data/get_valid_explicit_projects.sh first.`,
    );
    process.exit(1);
  }

  for (const folder of folders) {
    if (!(await pathExists(folder))) {
      console.log(
        `${folder} does not exist. Make sure to create the marked and marked-explicit datasets first`,
      );
      process.exit(1);
    }
  }

  if (await pathExists(filePathsFilePath)) {
    console.log(`${filePathsFilePath} already exists. Delete it first`);
    process.exit(1);
  }

  const stream = createWriteStream(filePathsFilePath, 'utf-8');

  const validProjectPaths = await collectSet(getFileLines(validProjectPathsPath));

  for (const projectPath of validProjectPaths) {
    for (const folder of folders) {
      const absPath = path.resolve(folder, projectPath);
      if (!(await pathExists(absPath))) {
        console.log(`Project ${projectPath} does not exist in dataset ${path.basename(folder)}`);
        process.exit(1);
      }
    }
  }

  const tsConfigs: string[] = [];
  for (const relProjectPath of validProjectPaths) {
    tsConfigs.push(path.join(relProjectPath, 'tsconfig.json'));
  }

  console.log(`Getting file paths for ${tsConfigs.length} projects`);

  const batches = batchBySize(tsConfigs, 10);

  const known = new Set<string>();
  const fileCount = await multithread<string[], string[], number>(
    batches,
    path.resolve(__dirname, './worker.js'),
    (relFilePaths, fileCount) => {
      for (const relFilePath of relFilePaths) {
        if (!known.has(relFilePath)) {
          stream.write(`${relFilePath}\n`);
          known.add(relFilePath);
          fileCount++;
        }
      }
      return fileCount;
    },
    0,
    ALLOWED_CPUS,
  );

  stream.end();

  console.log(`Done. Found ${fileCount} files`);
}

if (require.main === module) {
  main();
}
