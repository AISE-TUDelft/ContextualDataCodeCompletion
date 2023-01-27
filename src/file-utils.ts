import { access, opendir, readFile } from 'fs/promises';
import path from 'path';
import { exec } from 'child_process';
import { createReadStream } from 'fs';
import { createInterface } from 'readline';

/**
 * Explores folders recursively and calls a handler on specific files
 * @param folderPath Folder to explore recursively
 * @param handler Handler to apply to specific files
 * @param filter Filter used to select which files to handle
 */
export async function exploreFolder(
  folderPath: string,
  handler: (fileContent: string, filePath: string) => unknown,
  filter: (filePath: string) => boolean,
): Promise<void> {
  const dir = await opendir(folderPath);
  const promises = [];

  for await (const file of dir) {
    const filePath = path.resolve(folderPath, file.name);
    if (file.isFile()) {
      if (filter(filePath))
        await readFile(filePath, 'utf8').then((fileContent) => handler(fileContent, filePath));
    } else if (file.isDirectory()) {
      promises.push(exploreFolder(filePath, handler, filter));
    }
  }

  await Promise.all(promises);
}

export async function getFolderPaths(folderPath: string): Promise<string[]> {
  const dir = await opendir(folderPath);
  const folders = [];

  for await (const file of dir) {
    if (file.isDirectory()) {
      folders.push(path.resolve(folderPath, file.name));
    }
  }

  return folders;
}

export async function pathExists(folderPath: string): Promise<boolean> {
  try {
    await access(folderPath);
    return true;
  } catch (e) {
    return false;
  }
}

export async function* findFoldersAtDepth(rootPath: string, depth: number): AsyncIterable<string> {
  if (depth === 0) {
    yield rootPath;
  } else {
    for await (const file of await opendir(rootPath)) {
      if (file.isDirectory()) {
        yield* findFoldersAtDepth(path.resolve(rootPath, file.name), depth - 1);
      }
    }
  }
}

export async function copyFolder(folder: string, destination: string) {
  if (await pathExists(destination)) {
    throw new Error(`Folder ${destination} already exists`);
  }

  return new Promise<void>((resolve) =>
    exec(`cp -R ${folder} ${destination}`, { cwd: process.cwd() }, () => resolve()),
  );
}

/**
 * Finds files recursively. Only returns files if there is not a file with the same name in a deeper level of that directory
 */
export async function* findDeepestFiles(
  folderPath: string,
  fileName: string,
  ignoredFolders: string[] = [],
  foundFileRef: { value: boolean } = { value: false },
): AsyncIterable<string> {
  const dir = await opendir(folderPath);

  const filePaths: string[] = [];
  const foundDeeperFileRef = { value: false };

  for await (const file of dir) {
    const filePath = path.resolve(dir.path, file.name);
    if (file.isFile() && file.name === fileName && !foundDeeperFileRef.value) {
      filePaths.push(filePath);
    } else if (file.isDirectory() && !ignoredFolders.includes(file.name)) {
      yield* findDeepestFiles(
        path.resolve(dir.path, file.name),
        fileName,
        ignoredFolders,
        foundDeeperFileRef,
      );
    }
  }

  if (foundDeeperFileRef.value) {
    foundFileRef.value = true;
  } else {
    yield* filePaths;
  }
}

/**
 * Iterates through relative file paths in some folder (non recursive)
 */
export async function* findFiles(
  folderPath: string,
  allowedExtension: string[] = [],
  bannedExtensions: string[] = [],
): AsyncIterable<string> {
  const dir = await opendir(folderPath);

  for await (const file of dir) {
    if (
      file.isFile() &&
      (allowedExtension.length === 0 ||
        allowedExtension.some((extension) => file.name.endsWith(extension))) &&
      (bannedExtensions.length === 0 ||
        !bannedExtensions.some((extension) => file.name.endsWith(extension)))
    ) {
      yield file.name;
    }
  }
}

/**
 * Iterates through relative file paths in some folder (recursively)
 */
export async function* findFilesRecursively(
  folderPath: string,
  allowedExtension: string[] = [],
  bannedExtensions: string[] = [],
  bannedFolders: string[] = [],
): AsyncIterable<string> {
  const dir = await opendir(folderPath);

  for await (const file of dir) {
    if (file.isDirectory()) {
      if (bannedFolders.length === 0 || !bannedFolders.includes(file.name)) {
        for await (const filePath of findFilesRecursively(
          path.resolve(folderPath, file.name),
          allowedExtension,
          bannedExtensions,
          bannedFolders,
        )) {
          yield path.join(file.name, filePath);
        }
      }
    } else if (file.isFile()) {
      if (
        (allowedExtension.length === 0 ||
          allowedExtension.some((extension) => file.name.endsWith(extension))) &&
        (bannedExtensions.length === 0 ||
          !bannedExtensions.some((extension) => file.name.endsWith(extension)))
      ) {
        yield file.name;
      }
    }
  }
}

export async function* getFileLines(filePath: string): AsyncIterable<string> {
  const readStream = createReadStream(filePath, { encoding: 'utf8' });
  yield* createInterface({ input: readStream, crlfDelay: Infinity });
  readStream.close();
}

/**
 * Yields RELATIVE file paths
 */
export function findTsConfigs(folderPath: string): AsyncIterable<string> {
  return findFilesRecursively(folderPath, ['tsconfig.json'], [], ['node_modules']);
}
