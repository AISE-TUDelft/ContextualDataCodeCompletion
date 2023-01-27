import { workerData } from 'worker_threads';
import { reportProgress, reportResult, reportTotal } from '../threading';
import { Project, ScriptKind } from 'ts-morph';
import path from 'path';
import { DATASET_MARKED_EXPLICIT_FOLDER, DATASET_MARKED_FOLDER } from '../config';
import { pathExists } from '../file-utils';

async function worker() {
  const data = workerData as string[];
  reportTotal(data.length);

  for (const relTsConfigPath of data) {
    try {
      const relFilePaths = await handleProject(relTsConfigPath);
      reportResult(relFilePaths);
    } catch (e: any) {
      console.error(`Error while handling ${relTsConfigPath}: ${e.message}`);
    }
    reportProgress('increment');
  }
}

async function handleProject(relTsConfigPath: string): Promise<string[]> {
  const explicitTsConfigPath = path.resolve(DATASET_MARKED_EXPLICIT_FOLDER, relTsConfigPath);
  const project = new Project({
    tsConfigFilePath: explicitTsConfigPath,
  });
  const relFilePaths = project
    .getSourceFiles()
    .filter((sf) => sf.getScriptKind() === ScriptKind.TS && !sf.isDeclarationFile())
    .map((file) => path.relative(DATASET_MARKED_EXPLICIT_FOLDER, file.getFilePath()));

  for (let i = relFilePaths.length - 1; i >= 0; i--) {
    const relFilePath = relFilePaths[i];
    const filePath = path.resolve(DATASET_MARKED_FOLDER, relFilePath);
    if (!(await pathExists(filePath))) {
      // This file does not exist in this dataset, so we remove it from the list
      relFilePaths.splice(i, 1);
      break;
    }
  }

  return relFilePaths;
}

worker();
