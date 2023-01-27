import { readFile } from 'fs/promises';
import path from 'path';
import { countTypes } from './count-types';
import { collect, enumerate, round } from '../utils';
import { findFilesRecursively } from '../file-utils';

const folder = process.argv[2];

if (typeof folder !== 'string') {
  console.log('You must provide a folder to search as argv1');
  process.exit(1);
}

async function computeTypeExplicitness() {
  console.log('Computing type explicitness');

  const files = await collect(findFilesRecursively(folder, ['.ts'], ['.d.ts'], ['node_modules']));

  let potential = 0;
  let annotations = 0;
  for (const [i, file] of enumerate(files)) {
    const filePath = path.resolve(folder, file);
    const code = await readFile(filePath, 'utf8');
    const result = countTypes(code);
    potential += result.potential;
    annotations += result.annotations;
    if (i !== 0 && i % 10000 === 0) {
      console.log(
        `[${round(((i + 1) / files.length) * 100, 2)}%] Type Explicitness: ${
          annotations / potential
        } = ${round((annotations / potential) * 100, 2)}%`,
      );
    }
  }

  console.log(
    `Done computing type explicitness: ${annotations} / ${potential} = ${
      annotations / potential
    } = ${round((annotations / potential) * 100, 2)}%`,
  );
}

computeTypeExplicitness();
