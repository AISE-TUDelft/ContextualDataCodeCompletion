import { concat, zipOne } from '../utils';

describe('iterables', () => {
  it('concat, zipOne', () => {
    const iterable = concat(zipOne('a', [1, 2, 3]), zipOne('b', [4, 5, 6]));
    expect([...iterable]).toEqual([
      ['a', 1],
      ['a', 2],
      ['a', 3],
      ['b', 4],
      ['b', 5],
      ['b', 6],
    ]);
  });
});
