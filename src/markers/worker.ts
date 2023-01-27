import { workerData } from 'worker_threads';
import { reportProgress, reportTotal } from '../threading';
import path from 'path';
import { DATASET_MARKED_FOLDER } from '../config';
import { readFile, writeFile } from 'fs/promises';
import { addMarkerComments } from './statement-marking';

async function worker() {
  const relTsFilePaths = workerData as string[];
  reportTotal(relTsFilePaths.length);
  for (const tsFilePath of relTsFilePaths) {
    const absTsFilePath = path.resolve(DATASET_MARKED_FOLDER, tsFilePath);
    const code = await readFile(absTsFilePath, 'utf8');

    try {
      const markedCode = addMarkerComments(code, tsFilePath);
      await writeFile(absTsFilePath, markedCode);
      reportProgress('increment');
    } catch (e) {
      // error: the old file still exists so it's fine
    }
  }
}

worker();
