import tokenize, { Token } from 'js-tokens';
import { and, filter, not } from '../utils';
import { PreservedComments } from '../create-model-files/create-model-files';
import { MARKER_COMMENT_REGEX } from '../markers/statement-marking';

export function preprocess(
  code: string,
  preservedComments: PreservedComments,
  replaceLiterals: boolean,
) {
  let tokens = tokenize(code);
  tokens = replaceTemplateStrings(tokens);

  const filters: ((tok: Token) => boolean)[] = [not(isMarkerComment)];

  if ((preservedComments & PreservedComments.MULTI_LINE) === 0) {
    filters.push(not(isMultiLineNonDocblockComment));

    if ((preservedComments & PreservedComments.DOCBLOCK) === 0) {
      filters.push(not(isDocBlock));
    }
  }

  if ((preservedComments & PreservedComments.SINGLE_LINE) === 0) {
    filters.push(not(isSingleLineComment));
  }

  tokens = filter(tokens, ...filters);
  tokens = removeDoubleWhitespaces(tokens);
  tokens = removeDoubleNewLines(tokens);

  return [...stringifyTokens(tokens, replaceLiterals)].join('');
}

function* replaceTemplateStrings(tokens: Iterable<Token>): Iterable<Token> {
  let level = 0;
  for (const token of tokens) {
    if (token.type === 'TemplateHead') {
      level++;
    } else if (token.type === 'TemplateTail') {
      level--;
    }

    if (level === 0) {
      if (token.type === 'TemplateTail') {
        yield { type: 'NoSubstitutionTemplate', value: '``', closed: true };
      } else {
        yield token;
      }
    }
  }
}

export function isComment(token: Token) {
  return token.type === 'MultiLineComment' || token.type === 'SingleLineComment';
}

export function isSingleLineComment(token: Token): boolean {
  return token.type === 'SingleLineComment';
}

export function isMultiLineComment(token: Token): boolean {
  return token.type === 'MultiLineComment';
}

export function isDocBlock(token: Token): boolean {
  return (
    isMultiLineComment(token) &&
    token.value.trimStart().startsWith('/**') &&
    token.value
      .split('\n')
      .slice(1)
      .every((line) => line.trimStart().startsWith('*'))
  );
}

export function isMultiLineNonDocblockComment(token: Token): boolean {
  return and(isMultiLineComment, not(isDocBlock))(token);
}

export function isMarkerComment(token: Token): boolean {
  return isMultiLineComment(token) && token.value.match(MARKER_COMMENT_REGEX) !== null;
}

export function isLineTerminator(token: Token): boolean {
  return token.type === 'LineTerminatorSequence';
}

function* stringifyTokens(tokens: Iterable<Token>, replaceLiterals: boolean) {
  for (const token of tokens) {
    if (
      (token.type === 'StringLiteral' || token.type === 'NoSubstitutionTemplate') &&
      replaceLiterals
    ) {
      const startQuote = token.value[0];
      const endQuote = token.value[token.value.length - 1];
      yield `${startQuote}<STR_LIT>${endQuote}`;
    } else if (token.type === 'MultiLineComment' && replaceLiterals) {
      yield token.value.replace(/\n/g, '<EOL>');
    } else if (token.type === 'NumericLiteral' && replaceLiterals) {
      yield '<NUM_LIT>';
    } else if (token.type === 'LineTerminatorSequence' && replaceLiterals) {
      yield '<EOL>';
    } else if (
      token.type === 'TemplateHead' ||
      token.type === 'TemplateMiddle' ||
      token.type === 'TemplateTail'
    ) {
      throw new Error(
        'Can not stringify TemplateHead/TemplateMiddle/TemplateTail. These should first be removed by replaceTemplateStrings',
      );
    } else {
      yield token.value;
    }
  }
}

/**
 * Removing comments can introduce trailing whitespaces.
 * This function removes all duplicate+ whitespaces given that it is not directly following a newline.
 */
function* removeDoubleWhitespaces(tokens: Iterable<Token>): Iterable<Token> {
  let isNewLine = false;
  let prevIsWhiteSpace = false;
  for (const token of tokens) {
    isNewLine =
      token.type === 'LineTerminatorSequence' || (isNewLine && token.type === 'WhiteSpace');

    if (token.type === 'WhiteSpace' && prevIsWhiteSpace && !isNewLine) {
      continue;
    }

    yield token;
    prevIsWhiteSpace = token.type === 'WhiteSpace';
  }
}

/**
 * Removes duplicate newlines
 */
function* removeDoubleNewLines(tokens: Iterable<Token>): Iterable<Token> {
  let prevIsNewLine = false;
  for (const token of tokens) {
    if (token.type === 'LineTerminatorSequence' && prevIsNewLine) {
      continue;
    }

    yield token;
    prevIsNewLine = token.type === 'LineTerminatorSequence';
  }
}
