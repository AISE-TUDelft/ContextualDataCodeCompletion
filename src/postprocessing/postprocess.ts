import { createReadStream, createWriteStream } from 'fs';
import path from 'path';
import { createInterface } from 'readline';
import { zipAsync } from '../utils';
import { pathExists } from '../file-utils';
import tokenize from 'js-tokens';
import { countTypes } from '../type-explicitness/count-types';
import { removeComments } from '../markers/statement-marking';
import { opendir } from 'fs/promises';
import { PREDICTIONS_FOLDER } from '../config';

const model = process.argv[2];

if (typeof model !== 'string') {
  console.log('Invalid model!');
  process.exit(1);
}

const modelFolder = path.resolve(PREDICTIONS_FOLDER, model);

async function main() {
  console.log(`Starting postprocess. Folder = ${modelFolder}`);

  if (!(await pathExists(modelFolder))) {
    console.log(`Folder missing: '${modelFolder}'. Did you train and test models yet?`);
    process.exit(1);
  }

  for await (const file of await opendir(modelFolder)) {
    if (file.isDirectory()) {
      await go(file.name);
    }
  }
}

async function go(dataset: string) {
  console.log('Post processing dataset', dataset);

  const predictionsPath = path.resolve(modelFolder, dataset, 'predictions.txt');
  if (!(await pathExists(predictionsPath))) {
    console.log(`No predictions.txt for ${dataset}`);
    process.exit(1);
  }

  const testPath = path.resolve(modelFolder, dataset, 'test.json');
  if (!(await pathExists(testPath))) {
    console.log(`No test.json at ${testPath}`);
    return;
  }

  const outPath = path.resolve(modelFolder, dataset, 'postprocessed.txt');
  const postprocessed = createWriteStream(outPath, 'utf8');
  let predictions: AsyncIterable<string>;
  let inputs: AsyncIterable<string>;
  try {
    predictions = createInterface({
      input: createReadStream(predictionsPath, 'utf8'),
      crlfDelay: Infinity,
    });
    inputs = createInterface({
      input: createReadStream(testPath, 'utf8'),
      crlfDelay: Infinity,
    });
  } catch (e: unknown) {
    console.log('Cant read input/predictions');
    return;
  }

  for await (const [pred, json] of zipAsync(predictions, inputs)) {
    const obj = JSON.parse(json) as { input: string; gt: string };
    const gt = postprocess(obj.gt);
    const input = obj.input;
    const prediction = postprocess(pred);
    postprocessed.write(
      `${JSON.stringify({
        gt,
        prediction,
        input,
        inputTokens: [...tokenize(input)].map((token) => token.value),
        gtTokens: [...tokenize(gt)].map((token) => token.value),
        predictionTokens: [...tokenize(prediction)].map((token) => token.value),
      })}\n`,
    );
  }

  console.log('Finished post processing', dataset);
}

function postprocess(code: string): string {
  if (code.includes('<s>')) code = code.slice(code.indexOf('<s>') + 1);
  if (code.includes('</s>')) code = code.slice(0, code.indexOf('</s>'));

  // remove all comments. replace literals with constants. normalize spacing
  code = [...tokenize(code)]
    .filter((token) => {
      if (token.type === 'SingleLineComment') return false;
      if (token.type === 'MultiLineComment') return false;
      return true;
    })
    .map((token) => {
      if (token.type === 'NumericLiteral') return '0';
      if (token.type === 'StringLiteral' || token.type === 'NoSubstitutionTemplate') return '""';
      if (token.type === 'WhiteSpace') return ' ';
      return token.value;
    })
    .join('');

  code = code
    .replace(/['"`]<STR_LIT>['"`]/g, '""')
    .replace(/<NUM_LIT>/g, '0')
    .replace(/\s+/g, ' ')
    .replace(/[\r\n]+/, '\n');

  code = removeComments(code).trim();

  return code;
}

if (require.main === module) {
  main();
}
