import { workerData } from 'worker_threads';
import { reportProgress, reportResult, reportTotal } from '../threading';
import { Project } from 'ts-morph';
import { addTypes } from './add-types';
import path from 'path';
import { DATASET_MARKED_EXPLICIT_FOLDER } from '../config';

async function worker(): Promise<void> {
  const tsConfigs = workerData as string[];
  reportTotal(tsConfigs.length);

  for (const tsConfigFilePath of tsConfigs) {
    const project = new Project({
      tsConfigFilePath: path.resolve(DATASET_MARKED_EXPLICIT_FOLDER, tsConfigFilePath),
    });
    addTypes(project);
    await project.save();
    reportProgress('increment');
    reportResult(tsConfigFilePath);
  }
}

worker();
