import {
  getFunctionBodyBounds,
  removeComments,
  groupPotentialTypes,
  ModifiedToken,
  isMaskableToken,
} from '../statement-marking';
import tokenize, { Token } from 'js-tokens';

const findMultilineComment = (tokens: Token[], comment: string) => {
  const idx = tokens.findIndex(
    (token) => token.type === 'MultiLineComment' && token.value === `/*${comment}*/`,
  );

  if (idx === -1) {
    throw new Error(`Could not find comment block: ${comment}`);
  }

  return idx;
};

describe('getFunctionBodyBounds', () => {
  it.each([
    `function() /*start*/{
/*end*/}`,
    `function() /*start*/{
    /* {{{ */
/*end*/}`,
    `function() /*start*/{
    /** {{{ */
/*end*/}`,
    `function() /*start*/{
    const x = \`{{{\`
/*end*/}`,
    `function() /*start*/{
    const a = '123';
    const x = \`{\${a}{{\`
/*end*/}`,
    `function() /*start*/{
    const a = '123';
    const x = \`\${a}{{\`
/*end*/}`,
    `function() /*start*/{
    const a = '123';
    const x = \`\${a}{\${a}{\`
/*end*/}`,
    `function() /*start*/{
    const a = '123';
    const x = \`{{\${a}\`
/*end*/}`,
    `/* {{{}}}} */ function() /*start*/{
    const a = '123';
    const x = \`{{\${a}\`
/*end*/}`,
    `/* }}}}{{{ */ function() /*start*/{
    const a = '123';
    const x = \`{{\${a}\`
/*end*/}`,
  ])('should return indices of the starting and ending { } of a function', (code) => {
    const tokens = [...tokenize(code)];
    const startIndex = 1 + findMultilineComment(tokens, 'start');
    const endIndex = 1 + findMultilineComment(tokens, 'end');
    const result = getFunctionBodyBounds(tokens);
    expect(result).toEqual({ startIndex, endIndex });
  });
});

describe('removeComments', () => {
  it('single line comments', () => {
    expect(removeComments('lets go // 123')).toBe('lets go ');
  });

  it('multi line comments', () => {
    expect(removeComments('lets go /* 123 */ yea lets go')).toBe('lets go  yea lets go');
  });
});

describe('groupTokens', () => {
  it('1', () => {
    const tokens = [...tokenize('const x: number = 1')];
    groupPotentialTypes(tokens);
    expect(tokens.map((token) => token.type)).toEqual<ModifiedToken['type'][]>([
      'IdentifierName',
      'WhiteSpace',
      'IdentifierName',
      'PotentialType',
      'Punctuator',
      'WhiteSpace',
      'NumericLiteral',
    ]);
  });

  it('2', () => {
    const tokens = [...tokenize('const x: Record<string, number> = 1')];
    groupPotentialTypes(tokens);
    expect(tokens.map((token) => token.type)).toEqual<ModifiedToken['type'][]>([
      'IdentifierName',
      'WhiteSpace',
      'IdentifierName',
      'PotentialType',
      'Punctuator',
      'WhiteSpace',
      'NumericLiteral',
    ]);
  });

  it('3', () => {
    const a = 'const x = [1,2,3,4];';
    const b = 'const x: number[] = [1,2,3,4];';
    const at = [...tokenize(a)];
    const bt = [...tokenize(b)];
    groupPotentialTypes(at);
    groupPotentialTypes(bt);
    expect(at.filter((token) => isMaskableToken(token))).toEqual(
      bt.filter((token) => isMaskableToken(token)),
    );
  });

  it('4', () => {
    const a = 'const file: string = fs.readFileSync(path).toString();';
    const t = [...tokenize(a)];
    groupPotentialTypes(t);
    expect(t.map((token) => token.value).join('')).toEqual(a);
  });

  it('5', () => {
    const a = 'console.log(...args);';
    const t = [...tokenize(a)];
    groupPotentialTypes(t);
    expect(t.map((token) => token.value).join('')).toEqual(a);
  });
});
