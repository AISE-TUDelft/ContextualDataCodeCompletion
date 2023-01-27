import { copyFolder, pathExists } from '../file-utils';
import { DATASET_MARKED_EXPLICIT_FOLDER, DATASET_MARKED_FOLDER } from '../config';

async function main() {
  console.log('Creating folder for explicit marked dataset');

  if (await pathExists(DATASET_MARKED_EXPLICIT_FOLDER)) {
    console.log(`Folder ${DATASET_MARKED_EXPLICIT_FOLDER} already exists. Delete it to override`);
    process.exit(1);
  }

  if (!(await pathExists(DATASET_MARKED_FOLDER))) {
    console.log(`Folder ${DATASET_MARKED_FOLDER} does not exist. Create a marked dataset first`);
    process.exit(1);
  }

  console.log(`Copying folder ${DATASET_MARKED_FOLDER} to ${DATASET_MARKED_EXPLICIT_FOLDER}`);
  await copyFolder(DATASET_MARKED_FOLDER, DATASET_MARKED_EXPLICIT_FOLDER);
}

if (require.main === module) {
  main();
}
