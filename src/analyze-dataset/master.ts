/*
 * This script analyzes all marked datasets. It counts #loc, #files, and creates a list of all file paths for each dataset
 * Only files in TS projects are considered
 */

import {
  DATASET_MARKED_UNTYPED_FOLDER,
  DATASET_MARKED_EXPLICIT_FOLDER,
  DATASET_MARKED_FOLDER,
  ALLOWED_CPUS,
} from '../config';
import path from 'path';
import { getFileLines, pathExists } from '../file-utils';
import { batchBySize, collect, round } from '../utils';
import { multithread } from '../threading';
import { filePathsFilePath } from '../get-file-paths/master';

const folders = [
  DATASET_MARKED_UNTYPED_FOLDER,
  DATASET_MARKED_EXPLICIT_FOLDER,
  DATASET_MARKED_FOLDER,
];

export type Statistics = {
  loc: number;
  files: number;
  typeSlots: number;
  typeAnnotations: number;
  // type explicitness = typeAnnotations / typeSlots
};

const mergeStatistics = (stats1: Statistics, stats2: Statistics): Statistics => {
  return {
    loc: stats1.loc + stats2.loc,
    files: stats1.files + stats2.files,
    typeSlots: stats1.typeSlots + stats2.typeSlots,
    typeAnnotations: stats1.typeAnnotations + stats2.typeAnnotations,
  };
};

async function main() {
  if (!(await pathExists(filePathsFilePath))) {
    console.error(`${filePathsFilePath} does not exist. Run ./src/get-file-paths/master.ts first.`);
    process.exit(1);
  }

  for (const folder of folders) {
    if (!(await pathExists(folder))) {
      console.log(`${folder} does not exist. Make sure to create the datasets first`);
      process.exit(1);
    }
  }

  const files = await collect(getFileLines(filePathsFilePath));

  const repos = new Set<string>();
  for (const file of files) {
    const [repo] = file.split('/');
    repos.add(repo);
  }
  const repoCount = repos.size;
  repos.clear();

  const batches = batchBySize(files, 100);

  const stats = await multithread<
    string[],
    { normal: Statistics; explicit: Statistics; untyped: Statistics },
    { normal: Statistics; explicit: Statistics; untyped: Statistics }
  >(
    batches,
    path.resolve(__dirname, './worker.js'),
    (result, aggregate) => ({
      normal: mergeStatistics(result.normal, aggregate.normal),
      explicit: mergeStatistics(result.explicit, aggregate.explicit),
      untyped: mergeStatistics(result.untyped, aggregate.untyped),
    }),
    {
      normal: { loc: 0, files: 0, typeSlots: 0, typeAnnotations: 0 },
      explicit: { loc: 0, files: 0, typeSlots: 0, typeAnnotations: 0 },
      untyped: { loc: 0, files: 0, typeSlots: 0, typeAnnotations: 0 },
    },
    ALLOWED_CPUS,
  );

  console.log(`Repo count: ${repoCount}`);
  for (const key of ['normal', 'explicit', 'untyped'] as const) {
    const { loc, files, typeSlots, typeAnnotations } = stats[key];
    console.log(`Dataset ${key}:`);
    console.log(`LOC: ${loc}`);
    console.log(`Files: ${files}`);
    console.log(`Type explicitness: ${round((100 * typeAnnotations) / typeSlots, 2)}`);
  }
  console.log('--------------');
  console.log('If there is a difference in the number of files, something is wrong');
  console.log(
    'A difference between the LOC is not strange, as types sometimes take up multiple lines',
  );
  console.log('Type explicitness should match the dataset');
}

if (require.main === module) {
  main();
}
