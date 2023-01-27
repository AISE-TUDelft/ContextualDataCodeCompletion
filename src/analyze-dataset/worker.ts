import { workerData } from 'worker_threads';
import { Project, SourceFile, Node } from 'ts-morph';
import { Statistics } from './master';
import { readFile } from 'fs/promises';
import { reportProgress, reportResult, reportTotal } from '../threading';
import {
  DATASET_MARKED_EXPLICIT_FOLDER,
  DATASET_MARKED_FOLDER,
  DATASET_MARKED_UNTYPED_FOLDER,
} from '../config';
import path from 'path';

const datasets = {
  normal: DATASET_MARKED_FOLDER,
  explicit: DATASET_MARKED_EXPLICIT_FOLDER,
  untyped: DATASET_MARKED_UNTYPED_FOLDER,
} as const;

const baseStats = (): Statistics => ({
  loc: 0,
  files: 0,
  typeSlots: 0,
  typeAnnotations: 0,
});

async function worker() {
  const files = workerData as string[];
  reportTotal(Object.keys(datasets).length * files.length);

  for (const [dataset, folder] of Object.entries(datasets)) {
    for (const relFilePath of files) {
      const absFilePath = path.resolve(folder, relFilePath);
      const stats = await getStats(absFilePath);
      reportResult({
        normal: baseStats(),
        explicit: baseStats(),
        untyped: baseStats(),
        [dataset]: stats,
      });
      reportProgress('increment');
    }
  }
}

async function getStats(absFilePath: string): Promise<Statistics> {
  const stats = baseStats();
  const content = await readFile(absFilePath, 'utf-8');
  const project = new Project({
    useInMemoryFileSystem: true,
  });
  const sourceFile = project.createSourceFile('temp.ts', content);
  stats.loc += sourceFile.getFullText().split(/\n+/g).length;
  stats.files++;
  const { typeSlots, typeAnnotations } = getTypeStats(sourceFile);
  stats.typeSlots += typeSlots;
  stats.typeAnnotations += typeAnnotations;
  return stats;
}

function getTypeStats(sourceFile: SourceFile): { typeSlots: number; typeAnnotations: number } {
  let typeSlots = 0;
  let typeAnnotations = 0;

  sourceFile.forEachDescendant((node) => {
    if (Node.isVariableDeclaration(node)) {
      if (node.getTypeNode() !== undefined) {
        typeAnnotations++;
      }
      typeSlots++;
    } else if (Node.isFunctionDeclaration(node) || Node.isArrowFunction(node)) {
      typeSlots++; // return type
      typeSlots += node.getParameters().length;
      if (node.getReturnTypeNode() !== undefined) {
        typeAnnotations++;
      }
      for (const param of node.getParameters()) {
        if (param.getTypeNode() !== undefined) {
          typeAnnotations++;
        }
      }
    }
  });

  return { typeSlots, typeAnnotations };
}

worker();
