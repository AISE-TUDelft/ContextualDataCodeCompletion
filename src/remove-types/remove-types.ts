import { ScriptKind, ts, SourceFile } from 'ts-morph';

export function removeTypes(sourceFile: SourceFile) {
  if (sourceFile.getScriptKind() !== ScriptKind.TS) {
    return;
  }

  // sourceFile.forEachDescendant((node) => {
  //   try {
  //     if (Node.isVariableDeclaration(node) && node.getTypeNode() !== undefined) {
  //       node.removeType();
  //     } else if (Node.isFunctionDeclaration(node)) {
  //       node.removeReturnType();
  //       node.getTypeParameters().forEach((typeParameter) => {
  //         typeParameter.remove();
  //       });
  //       node.getParameters().forEach((param) => {
  //         param.removeType();
  //       });
  //     } else if (Node.isArrowFunction(node)) {
  //       node.removeReturnType();
  //       node.getTypeParameters().forEach((typeParameter) => {
  //         typeParameter.remove();
  //       });
  //       node.getParameters().forEach((param) => {
  //         param.removeType();
  //       });
  //     }
  //   } catch (e: any) {}
  // });

  sourceFile.transform((traversal) => {
    // remove the type annotation if there is any
    const node = traversal.visitChildren();
    if (ts.isVariableDeclaration(node)) {
      const { type, initializer } = node;
      if (type) {
        return ts.factory.updateVariableDeclaration(
          node,
          node.name,
          node.exclamationToken,
          undefined,
          node.initializer,
        );
      }
    } else if (ts.isFunctionDeclaration(node)) {
      // remove the parameter types and return types
      const { parameters, type } = node;
      if (parameters.length > 0 || type) {
        return ts.factory.updateFunctionDeclaration(
          node,
          node.decorators,
          node.modifiers,
          node.asteriskToken,
          node.name,
          undefined,
          parameters.map((param) =>
            ts.factory.updateParameterDeclaration(
              param,
              param.decorators,
              param.modifiers,
              param.dotDotDotToken,
              param.name,
              param.questionToken,
              undefined,
              param.initializer,
            ),
          ),
          undefined,
          node.body,
        );
      }
    } else if (ts.isArrowFunction(node)) {
      // remove the parameter types and return types
      const { parameters, type } = node;
      if (parameters.length > 0) {
        return ts.factory.updateArrowFunction(
          node,
          node.modifiers,
          undefined,
          parameters.map((param) => {
            return ts.factory.updateParameterDeclaration(
              param,
              param.decorators,
              param.modifiers,
              param.dotDotDotToken,
              param.name,
              param.questionToken,
              undefined,
              param.initializer,
            );
          }),
          undefined,
          node.equalsGreaterThanToken,
          node.body,
        );
      } else if (type) {
        return ts.factory.updateArrowFunction(
          node,
          node.modifiers,
          undefined,
          parameters.map((param) => {
            return ts.factory.updateParameterDeclaration(
              param,
              param.decorators,
              param.modifiers,
              param.dotDotDotToken,
              param.name,
              param.questionToken,
              undefined,
              param.initializer,
            );
          }),
          undefined,
          node.equalsGreaterThanToken,
          node.body,
        );
      }
    }

    return node;
  });
}
