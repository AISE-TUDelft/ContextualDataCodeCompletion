import { workerData } from 'worker_threads';
import { reportProgress, reportTotal } from '../threading';
import { Project } from 'ts-morph';
import { removeTypes } from './remove-types';
import { readFile, writeFile } from 'fs/promises';

async function worker(): Promise<void> {
  const files = workerData as string[];
  reportTotal(files.length);

  for (const absFilePath of files) {
    const project = new Project({
      useInMemoryFileSystem: true,
    });
    const sourceFile = project.createSourceFile('temp.ts', await readFile(absFilePath, 'utf-8'));
    removeTypes(sourceFile);
    await writeFile(absFilePath, sourceFile.getFullText());
    reportProgress('increment');
  }
}

worker();
