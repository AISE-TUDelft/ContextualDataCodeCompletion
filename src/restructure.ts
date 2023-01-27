import {mkdir, opendir, readFile, rm, writeFile} from "fs/promises";
import path from "path";
import { createHash } from "crypto";

// script that can be used to restructure Functions_Inferred into the new format

async function rec(folder: string) {
    const dir = await opendir(folder);
    for await (const file of dir) {
        if (file.isDirectory()) {
            await rec(path.resolve(folder, file.name));
        } else {
            const [name, fn] = file.name.split('.');
            await mkdir(path.resolve(folder, name), { recursive: true });
            const data = await readFile(path.resolve(folder, file.name), 'utf8')
            await rm(path.resolve(folder, file.name));
            await writeFile(path.resolve(folder, name, `${fn}.ts`), data);
        }
    }
}

rec('./data/Functions_Inferred')