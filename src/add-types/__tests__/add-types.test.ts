import { ModuleKind, Project } from 'ts-morph';
import { addTypes } from '../add-types';

const fixSpacing = (str: string) => str.trim().split(/\s+/).join(' ');

it('simple project', () => {
  const project = new Project({
    compilerOptions: {
      module: ModuleKind.CommonJS,
      rootDir: './',
    },
    useInMemoryFileSystem: true,
  });

  project.createSourceFile(
    './a.ts',
    `
export function a() {
 return 1;
}
`.trim(),
  );

  project.createSourceFile(
    './b.ts',
    `
import { a } from './a';

export function b() {
    return a() + 1;
}
  `.trim(),
  );

  addTypes(project);

  expect(
    project
      .getSourceFile('a.ts')
      ?.print()
      ?.trim()
      ?.split(/[\s\n\r]+/)
      ?.join(' '),
  ).toBe('export function a(): number { return 1; }');

  expect(
    project
      .getSourceFile('b.ts')
      ?.print()
      ?.trim()
      ?.split(/[\s\n\r]+/)
      ?.join(' '),
  ).toBe("import { a } from './a'; export function b(): number { return a() + 1; }");
});

it('lambda parameters', () => {
  const project = new Project();
  const sf = project.createSourceFile(
    'temp.ts',
    `
const x: (a: number) => number = (a) => a + 1;
`.trim(),
  );

  addTypes(project);

  expect(sf.getVariableDeclaration('x')?.getType()?.getText()).toBe('(a: number) => number');
});

it('importing types', () => {
  const project = new Project({
    compilerOptions: {
      module: ModuleKind.CommonJS,
      rootDir: './',
    },
    useInMemoryFileSystem: true,
  });

  project.createSourceFile(
    './a.ts',
    `
export type A = {
  a: number;
};
`.trim(),
  );

  project.createSourceFile(
    './b.ts',
    `
import { A } from './a';

export function b() {
    return { a: 1 } as A;
}
  `.trim(),
  );

  addTypes(project);

  expect(
    project
      .getSourceFile('b.ts')
      ?.print()
      ?.trim()
      ?.split(/[\s\n\r]+/)
      ?.join(' '),
  ).toBe("import { A } from './a'; export function b(): A { return { a: 1 } as A; }");
});

it('destructuring', () => {
  const project = new Project();
  const sf = project.createSourceFile(
    'temp.ts',
    `
const x = 1;
const { a, b } = { a: 1, b: '2' }
`.trim(),
  );

  expect(() => addTypes(project)).not.toThrow();

  expect(fixSpacing(sf.print())).toBe(
    fixSpacing("const x: number = 1; const { a, b }: { a: number, b: number } = { a: 1, b: '2' };"),
  );
});
