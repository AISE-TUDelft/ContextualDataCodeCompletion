import { getFileLines, pathExists } from '../file-utils';
import {
  SETS_FOLDER,
  SETS_FILES_FOLDER,
  DATASET_MARKED_FOLDER,
  DATASET_MARKED_EXPLICIT_FOLDER,
  DATASET_MARKED_UNTYPED_FOLDER,
  NUM_MARKERS,
  RANDOM_SEED,
} from '../config';
import { mkdir, readFile } from 'fs/promises';
import path from 'path';
import { createWriteStream, WriteStream } from 'fs';
import { preprocess } from '../preprocessing/preprocess';
import { getMarkedVariants, MARKER_COMMENT_REGEX } from '../markers/statement-marking';
import { collect, randChoices, setIntersection } from '../utils';
import random from 'random-seed';

export enum PreservedComments {
  NONE = 0,
  SINGLE_LINE = 1 << 0,
  MULTI_LINE = 1 << 1,
  DOCBLOCK = 1 << 2,
  ALL = PreservedComments.SINGLE_LINE | PreservedComments.MULTI_LINE | PreservedComments.DOCBLOCK,
}

const PRESERVED_COMMENTS = [
  // PreservedComments.ALL,
  // PreservedComments.NONE,
  PreservedComments.SINGLE_LINE,
  PreservedComments.MULTI_LINE,
  PreservedComments.DOCBLOCK,
];

type Model = 'unixcoder' | 'incoder';

type Options = {
  shouldReplaceLiterals: boolean;
};

async function createFiles() {
  const args = process.argv.slice(2);
  if (args.length < 1) {
    console.error('Please arguments: [model]');
    process.exit(1);
  }

  const options: Options = {
    shouldReplaceLiterals: true,
  };

  const [model] = process.argv.slice(2);

  if (model === 'unixcoder') {
    options.shouldReplaceLiterals = true;
  } else if (model === 'incoder') {
    options.shouldReplaceLiterals = false;
  } else {
    console.log('Invalid model');
    process.exit(1);
  }

  const outDirPath = path.resolve(SETS_FILES_FOLDER, model);

  if (!(await pathExists(SETS_FILES_FOLDER))) {
    await mkdir(SETS_FILES_FOLDER);
  }

  if (await pathExists(outDirPath)) {
    console.log(`Folder ${outDirPath} already exists. Delete it to override`);
    process.exit(1);
  }

  if (!(await pathExists(SETS_FOLDER))) {
    console.log(`Folder ${SETS_FOLDER} does not exist. Create it first with split-data`);
    process.exit(1);
  }

  await mkdir(outDirPath);

  console.log('Creating files');
  for (const comments of PRESERVED_COMMENTS) {
    console.log(`Creating files with ${PreservedComments[comments]} comments`);
    await Promise.all([
      // createTrainFiles(outDirPath, comments, options),
      // createDevFiles(outDirPath, comments, options),
      createTestFiles(outDirPath, comments, options),
    ]);
  }

  console.log('Done!');
}

if (require.main === module) {
  createFiles();
}

type TypeExplicitness = 'normal' | 'explicit' | 'untyped';

const TYPE_EXPLICITNESS: TypeExplicitness[] = ['normal', 'explicit', 'untyped'];

function getInFolder(typeExplicitness: TypeExplicitness): string {
  if (typeExplicitness === 'normal') {
    return DATASET_MARKED_FOLDER;
  } else if (typeExplicitness === 'explicit') {
    return DATASET_MARKED_EXPLICIT_FOLDER;
  } else if (typeExplicitness === 'untyped') {
    return DATASET_MARKED_UNTYPED_FOLDER;
  }

  throw new Error('Invalid typeExplicitness');
}

function getOutFolderName(typeExplicitness: TypeExplicitness, comments: PreservedComments): string {
  return `${typeExplicitness}-${PreservedComments[comments].toLowerCase()}`;
}

async function createTrainFiles(outDirPath: string, comments: PreservedComments, options: Options) {
  const files = await collect(getFileLines(path.resolve(SETS_FOLDER, 'train.txt')));

  let i = 0;
  let I = TYPE_EXPLICITNESS.length * files.length;

  for (const typeExplicitness of TYPE_EXPLICITNESS) {
    const inFolder = getInFolder(typeExplicitness);
    const outFolder = path.resolve(outDirPath, getOutFolderName(typeExplicitness, comments));

    await mkdir(outFolder, { recursive: true });

    const writeStream = createWriteStream(path.resolve(outFolder, 'train.txt'));

    for (const file of files) {
      try {
        const content = await readFile(path.resolve(inFolder, file), 'utf8');
        const processedContent = preprocess(content, comments, options.shouldReplaceLiterals);
        writeStream.write(`${processedContent}\n`);
        i++;

        if (i % 500 === 0) {
          console.log(`[TRAIN] Processed ${i}/${I} files`);
        }
      } catch (e) {
        //
      }
    }

    writeStream.end();
  }
}

function createTestFiles(outPath: string, comments: PreservedComments, options: Options) {
  return createMaskedFiles('test', outPath, comments, options);
}

function createDevFiles(outPath: string, comments: PreservedComments, options: Options) {
  return createMaskedFiles('dev', outPath, comments, options);
}

async function createMaskedFiles(
  name: string,
  outDirPath: string,
  comments: PreservedComments,
  options: Options,
) {
  const files = await collect(getFileLines(path.resolve(SETS_FOLDER, `${name}.txt`)));

  let i = 0;
  let I = TYPE_EXPLICITNESS.length * files.length;

  const outFolders: Record<TypeExplicitness, string> = {
    normal: path.resolve(outDirPath, getOutFolderName('normal', comments)),
    explicit: path.resolve(outDirPath, getOutFolderName('explicit', comments)),
    untyped: path.resolve(outDirPath, getOutFolderName('untyped', comments)),
  };

  const outFileName = `${name}.json`;

  for (const folderPath of Object.values(outFolders)) {
    await mkdir(folderPath, { recursive: true });
  }

  const writeStreams: Record<TypeExplicitness, WriteStream> = {
    normal: createWriteStream(path.resolve(outFolders.normal, outFileName)),
    explicit: createWriteStream(path.resolve(outFolders.explicit, outFileName)),
    untyped: createWriteStream(path.resolve(outFolders.untyped, outFileName)),
  };

  // for each file:
  // 1. get each code variant
  // 2. get common masks
  // 3. only use common masks to create tasks
  // 4. ???
  // 5. profit
  for (const file of files) {
    const variantCode: Partial<Record<TypeExplicitness, string>> = {};
    const variantMarkers: Partial<Record<TypeExplicitness, Set<string>>> = {};

    for (const typeExplicitness of TYPE_EXPLICITNESS) {
      const inFolder = getInFolder(typeExplicitness);

      // get the code for each type explicitness
      const content = await readFile(path.resolve(inFolder, file), 'utf8');
      variantCode[typeExplicitness] = content;

      // get all valid marker strings
      const markers =
        content.match(MARKER_COMMENT_REGEX)?.filter((marker) => {
          try {
            // only keep markers that have a valid left context and gt
            const markedVariants = getMarkedVariants(content, marker);

            if (markedVariants.length !== 1) {
              return false;
            }

            const [{ gt, input }] = markedVariants;

            return input.length > 0 && gt.length > 0;
          } catch (e: any) {
            return false;
          }
        }) ?? [];
      variantMarkers[typeExplicitness] = new Set(markers);
    }

    // determine which markers are in all the variants
    let commonMarkers = setIntersection(...Object.values(variantMarkers));

    if (commonMarkers.size > NUM_MARKERS) {
      // Limit the common markers to the max amount
      const commonMarkersArray = Array.from(commonMarkers);
      const prng = random.create(`${file}-${RANDOM_SEED}`);
      commonMarkers = new Set(randChoices(commonMarkersArray, NUM_MARKERS, prng.random.bind(prng)));
    }

    try {
      // for each variant, create the tasks for the markers that are in all the files
      const objs: Record<TypeExplicitness, object[]> = {
        normal: [],
        explicit: [],
        untyped: [],
      };

      for (const typeExplicitness of TYPE_EXPLICITNESS) {
        const content = variantCode[typeExplicitness]!;
        const markedVariants = getMarkedVariants(content, ...commonMarkers);

        for (const { input, marker, gt } of markedVariants) {
          const obj = {
            input: preprocess(input, comments, options.shouldReplaceLiterals),
            gt: preprocess(gt, comments, options.shouldReplaceLiterals),
            marker,
          };
          objs[typeExplicitness].push(obj);
        }

        i++;
        if (i % 500 === 0) {
          console.log(`[${name.toUpperCase()}] Processed ${i}/${I} files`);
        }
      }

      // write the tasks to the appropriate files
      for (const typeExplicitness of TYPE_EXPLICITNESS) {
        const writeStream = writeStreams[typeExplicitness];
        const objsForType = objs[typeExplicitness];
        for (const obj of objsForType) {
          writeStream.write(`${JSON.stringify(obj)}\n`);
        }
      }
    } catch (e: any) {
      //
    }
  }

  for (const writeStream of Object.values(writeStreams)) {
    writeStream.end();
  }
}
