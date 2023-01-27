import tokenize, { Token } from 'js-tokens';

/**
 * Try to read a token sequence with specific types. Returns an empty array if it fails.
 */
export function readTokenSeqOfType(tokens: Token[], types: Token['type'][]) {
  const result: Token[] = [];

  for (const token of tokens) {
    if (types.length === result.length) break;
    const currentType = types[result.length];
    if (token.type === currentType) {
      result.push(token);
    } else {
      return [];
    }
  }

  if (types.length !== result.length) return [];

  return result;
}

/**
 * Finds all tokens between two delimiter tokens. Skips any leading tokens
 */
export function findTokensBetween(
  tokens: Token[],
  tokenLeft: Token,
  tokenRight: Token,
): [] | [tokens: Token[], left: number, right: number] {
  for (let i = 0; i < tokens.length; i++) {
    let token = tokens[i];
    if (token.type === tokenLeft.type && token.value === tokenLeft.value) {
      const result: Token[] = [];

      let j = i + 1;
      token = tokens[j];
      for (
        ;
        j < tokens.length &&
        (tokens[j].type !== tokenRight.type || tokens[j].value !== tokenRight.value);
        j++
      ) {
        result.push(tokens[j]);
      }

      return [result, i, j];
    }
  }

  return [];
}

/**
 * Splits on some token
 */
function splitOnToken(tokens: Token[], splitToken: Token) {
  const result: Token[][] = [[]];

  for (const token of tokens) {
    if (token.type === splitToken.type && token.value === splitToken.value) {
      result.push([]);
    } else {
      result[result.length - 1].push(token);
    }
  }

  return result;
}

/**
 * Checks whether some sequence of tokens has a type annotation.
 * First token should be a variable/param name
 */
function hasTypeAnnotation(tokens: Token[]): boolean {
  const tks = readTokenSeqOfType(tokens, ['IdentifierName', 'Punctuator', 'IdentifierName']);
  const hasType = tks.length === 3 && tks[1].value === ':';

  return hasType;
}

//TODO: Arrow function
/**
 * Count the amount of type annotations in some code, and the amount of places where they could have been added
 */
export function countTypes(code: string): { annotations: number; potential: number } {
  const tokens = [...tokenize(code)].filter(
    (token) => !['WhiteSpace', 'LineTerminatorSequence'].includes(token.type),
  );
  const result = { annotations: 0, potential: 0 };

  for (let i = 0; i < tokens.length; i++) {
    const token = tokens[i];
    if (token.type === 'IdentifierName' && ['const', 'let', 'var'].includes(token.value)) {
      result.annotations += Number(hasTypeAnnotation(tokens.slice(i + 1)));
      result.potential++;
    } else if (token.type === 'IdentifierName' && token.value === 'function') {
      const [betweenBrackets, left, right] = findTokensBetween(
        tokens.slice(i + 1),
        { type: 'Punctuator', value: '(' },
        { type: 'Punctuator', value: ')' },
      );

      if (betweenBrackets === undefined || right === undefined || left === undefined) {
        continue;
      }

      const params = splitOnToken(betweenBrackets, { type: 'Punctuator', value: ',' });

      // in case we have no params
      if (params.length === 1 && params[0].length === 0) {
        params.length = 0;
      }

      const paramTypes = params.reduce(
        (total, param) => total + Number(hasTypeAnnotation(param)),
        0,
      );

      const hasReturnType =
        findTokensBetween(
          tokens.slice(i + 1 + right),
          { type: 'Punctuator', value: ':' },
          { type: 'Punctuator', value: '{' },
        ).length > 0;

      result.annotations += paramTypes + Number(hasReturnType);
      result.potential += params.length + 1;
    }
  }

  return result;
}
