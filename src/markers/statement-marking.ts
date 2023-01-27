import tokenize, { Token } from 'js-tokens';
import {
  clamp,
  enumerate,
  findIndexRight,
  multisplit,
  randChoice,
  randChoices,
  range,
} from '../utils';
import { LINE_MASK_CHANCE, RANDOM_SEED } from '../config';
import random from 'random-seed';

const createMarkerComment = (i: number) => `/*<marker:${i}>*/`;
export const MARKER_COMMENT_REGEX = /\/\*<marker:\d+>\*\//g;

export type ModifiedToken =
  | Token
  | { type: 'PotentialType'; value: string }
  | { type: 'TemplateString'; value: string };

/**
 * Adds a <marker:i> comment to some code to indicate that masking should happen there.
 * This should be applied to TypeScript code, such that it can be compiled to JS and then masked to get equivalent masked tokens.
 * @param code TypeScript code
 * @param hashString if you want to hash based on e.g. a filename
 */
export function addMarkerComments(code: string, hashString: string = '') {
  const tokens = [...tokenize(code)];

  groupTemplateStrings(tokens);
  groupPotentialTypes(tokens);

  // const { startIndex, endIndex } = getFunctionBodyBounds(tokens);
  const [startIndex, endIndex] = [0, tokens.length - 1];
  const lineRanges = getLineRanges(tokens, startIndex + 1, endIndex - 1);

  if (lineRanges.length === 0) {
    return code;
  }

  // Make sure to mask each identical function the same way
  const prng = random.create(`${hashString}-${RANDOM_SEED}`);
  const suitableLineRanges = lineRanges.filter((lineRange) =>
    lineRangeHasMaskableTokens(tokens, ...lineRange),
  );
  const choiceCount = clamp(
    1,
    suitableLineRanges.length,
    Math.floor(suitableLineRanges.length * LINE_MASK_CHANCE),
  );
  for (const [i, [from, to]] of enumerate(
    randChoices(suitableLineRanges, choiceCount, prng.random.bind(prng)).reverse(),
  )) {
    let j = choiceCount - i - 1; // reverse the counter since we are going in reverse
    maskRange(tokens, j, from, to, prng.random.bind(prng));
  }

  return tokens.map((v) => v.value).join('');
}

export function groupTemplateStrings(tokens: ModifiedToken[]) {
  for (let i = 0; i < tokens.length; i++) {
    const token = tokens[i];
    // if we find a templateHead, replace it with a TemplateString token that captures the entire template string
    if (token.type === 'TemplateHead') {
      let level = 1;
      const newToken: ModifiedToken = { type: 'TemplateString', value: token.value };
      for (let j = i + 1; j < tokens.length; j++) {
        const token = tokens[j];
        newToken.value += token.value;
        if (token.type === 'TemplateHead') {
          level++;
        } else if (token.type === 'TemplateTail') {
          level--;
        }

        if (level === 0) {
          // level is 0, so we have found the end of the template string
          // it starts at i, and ends at j (inclusive!)
          // so we have j - 1 + 1 tokens to delete, and we shall insert newToken
          tokens.splice(i, j - i + 1, newToken);
          break;
        }
      }
    }
  }
}

/**
 * Groups tokens where there are potentially types such that we don't mask types
 * Not a 100% accurate method, but better than 0%
 */
export function groupPotentialTypes(tokens: ModifiedToken[]) {
  type LevelPunctuator = '{' | '<' | '[' | '(';
  type OppositeLevelPunctuator = '}' | '>' | ']' | ')';
  const isLevelPunctuator = (value: string): value is LevelPunctuator =>
    [...'{<[('].includes(value);
  const isOppositeLevelPunctuator = (value: string): value is OppositeLevelPunctuator =>
    [...'}>])'].includes(value);
  const oppositeLevelPunctuator: Record<OppositeLevelPunctuator, LevelPunctuator> = {
    '}': '{',
    ']': '[',
    '>': '<',
    ')': '(',
  };
  let levels: Record<LevelPunctuator, number> = {
    '{': 0,
    '<': 0,
    '[': 0,
    '(': 0,
  };

  for (let i = 0; i < tokens.length; i++) {
    const current = tokens[i];
    // type annotations come after :
    if (current.type === 'Punctuator' && current.value === ':') {
      const typeToken: ModifiedToken = { type: 'PotentialType', value: ':' };
      // find the end of this potential type by going until we find a character =/,/)/; at level 0, ie not in {}/[]/()
      for (let j = i + 1; j < tokens.length; j++) {
        if (tokens[j].type === 'Punctuator') {
          const value = tokens[j].value;
          if (isLevelPunctuator(value)) {
            levels[value]++;
          } else if (isOppositeLevelPunctuator(value)) {
            levels[oppositeLevelPunctuator[value]]--;
          } else if (
            [...'=,)}];'].includes(value) &&
            Object.values(levels).every((level) => level === 0)
          ) {
            // Probably end of type
            tokens.splice(i, j - i, typeToken);
            break;
          }
        }

        typeToken.value += tokens[j].value;
      }
    }
  }
}

function lineRangeHasMaskableTokens(tokens: ModifiedToken[], from: number, to: number): boolean {
  return range(from, to + 1).some((i) => isMaskableToken(tokens[i]));
}

/**
 * Inserts a mask comment before a token in some range (e.g. a line)
 */
function maskRange(
  tokens: ModifiedToken[],
  i: number,
  from: number,
  to: number,
  random = Math.random,
): void {
  const maskableTokenIndices = range(from, to + 1).filter((i) => isMaskableToken(tokens[i]));
  if (maskableTokenIndices.length === 0) {
    throw new Error('No maskable tokens.');
  }
  const tokenToMask = randChoice(maskableTokenIndices, random);
  tokens.splice(tokenToMask, 0, {
    type: 'MultiLineComment',
    value: createMarkerComment(i),
    closed: true,
  });
}

/**
 * Gets the start and end indices of non-empty lines
 */
function getLineRanges(
  tokens: ModifiedToken[],
  from: number,
  to: number,
): [from: number, to: number][] {
  const lines: [from: number, to: number][] = [];

  const newLineIndices = range(from, to + 1).filter(
    (i) => tokens[i].type === 'LineTerminatorSequence',
  );

  let prevIdx = from;
  for (const idx of newLineIndices) {
    if (prevIdx >= idx) {
      prevIdx = Math.max(prevIdx, idx + 1);
      continue;
    }

    lines.push([prevIdx, idx - 1]);
    prevIdx = idx + 1;
  }

  return lines;
}

/**
 * Returns the indices of the start and end tokens of the function body ({ and }).
 */
export function getFunctionBodyBounds(tokens: ModifiedToken[]) {
  try {
    const endIndex = findIndexRight(
      tokens,
      (token) => token.type === 'Punctuator' && token.value === '}',
    );

    let indent = 0;
    const startIndex = findIndexRight(tokens, (token) => {
      if (token.type !== 'Punctuator') return false;
      if (token.value === '}') {
        indent++;
      } else if (token.value === '{') {
        indent--;
        return indent === 0;
      }
      return false;
    });

    return { startIndex, endIndex };
  } catch (e) {
    console.log(tokens.map((t) => t.value).join(''));
    throw e;
  }
}

/**
 * Determines whether we should be able to place a <marker> before this token.
 */
export function isMaskableToken(token: ModifiedToken): boolean {
  switch (token.type) {
    case 'StringLiteral':
    case 'RegularExpressionLiteral':
    case 'IdentifierName':
    case 'NumericLiteral':
    case 'NoSubstitutionTemplate':
    case 'Punctuator':
    case 'PrivateIdentifier':
      return true;

    // Template is grouped, so these are in practice impossible
    case 'TemplateHead':
    case 'TemplateMiddle':
    case 'TemplateTail':
    // This one is possible
    case 'TemplateString':
    // Comments are removed later (sometimes) and are not under test
    case 'MultiLineComment':
    case 'SingleLineComment':
    case 'WhiteSpace':
    case 'LineTerminatorSequence':
    case 'Invalid':
    case 'PotentialType':
    default:
      return false;
  }
}

/**
 * Retrieves masked variants of the given code.
 * Only considers the given markers
 */
export function getMarkedVariants(
  code: string,
  ...markers: string[]
): { input: string; gt: string; marker: string }[] {
  const parts = multisplit(code, ...markers); // these parts may have other markers, but they are removed in pre-processing

  if (parts.length === 1) return [];

  if (parts.length !== markers.length + 1) {
    // this error may occur when there are duplicate markers in the code
    // the typescript compiler does this when removing types sometimes
    throw new Error(
      `Marker count does not match part count ${markers.length} markers, ${parts.length} parts. Incorrect markers?`,
    );
  }

  const truths = parts
    .slice(1)
    .map((part) => part.match(/^.+/)?.[0])
    // nothing should be filtered here!
    .filter((truth, i): truth is string => truth !== undefined)
    .map((truth) => removeComments(truth).trim());

  if (truths.length !== parts.length - 1) {
    throw new Error('Truth count does not match part count. Maybe an empty truth?');
  }

  return truths.map((gt, i) => ({ input: parts.slice(0, i + 1).join(''), gt, marker: markers[i] }));
}

/**
 * Removes comments from some piece of code
 */
export function removeComments(code: string): string {
  return [...tokenize(code)]
    .filter((token) => token.type !== 'SingleLineComment' && token.type !== 'MultiLineComment')
    .map((token) => token.value)
    .join('');
}
