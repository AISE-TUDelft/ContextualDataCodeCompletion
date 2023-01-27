import { Node, Project, ScriptKind } from 'ts-morph';

export function addTypes(project: Project) {
  for (const sourceFile of project.getSourceFiles()) {
    if (sourceFile.getScriptKind() !== ScriptKind.TS) {
      continue;
    }

    sourceFile.forEachDescendant((node) => {
      try {
        if (Node.isVariableDeclaration(node) && node.getTypeNode() === undefined) {
          const type = node
            .getType()
            .getBaseTypeOfLiteralType()
            .getText(node)
            .replace(/import\(.+\)\./g, '');

          node.setType(type);
        } else if (
          (Node.isFunctionDeclaration(node) && node.isImplementation()) ||
          Node.isArrowFunction(node)
        ) {
          node.getParameters().forEach((param) => {
            if (param.getTypeNode() === undefined) {
              param.setType(param.getType().getBaseTypeOfLiteralType().getText(node));
            }
          });

          if (node.getReturnTypeNode() === undefined) {
            node.setReturnType(node.getReturnType().getBaseTypeOfLiteralType().getText(node));
          }
        }
      } catch (e: any) {}
    });
  }
}
