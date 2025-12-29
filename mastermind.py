import itertools
import numpy as np
import sys

M = 4  # number of slots
N = 6  # number of symbols


def main():
  set_globals()
  queries = itertools.product(range(1, N + 1), repeat=M)
  queries = np.array(list(queries))
  valid_candidates = np.ones(queries.shape[0]).astype(bool)

  exact_hits, misplaced_hits = build_hits_matrices(queries)

  while np.sum(valid_candidates) >= 2:
    print(f'Number of candidates remaining: {np.sum(valid_candidates)}')

    query_idx = compute_query(exact_hits, misplaced_hits, valid_candidates)
    query = queries[query_idx]
    str_query = ''.join([str(q) for q in query])
    print(f'Query: {str_query}')

    response = ask_for_hits()
    valid_candidates = valid_candidates & filter_candidates(
      exact_hits,
      misplaced_hits,
      query_idx,
      response,
    )

  if np.sum(valid_candidates) == 0:
    print('Something is wrong; there appear to be no valid solutions.')
    return

  solution = [str(x) for x in queries[valid_candidates][0]]
  solution = ''.join(solution)
  print(f'Solution: {solution}')


def set_globals():
  global M, N
  if len(sys.argv) >= 2:
    try:
      M, N = sys.argv[1:3]
      M, N = int(M), int(N)
    except ValueError:
      print('Expected two integers.')
      raise


def build_hits_matrices(queries):
  exact_hits = np.sum(queries[:, None, :] == queries[None, :, :], axis=2)
  values = np.arange(1, N + 1)
  M = (queries[:, :, None] == values)
  M1 = np.sum(M, axis=1)
  total_hits = np.minimum(M1[None, :, :], M1[:, None, :]).sum(axis=2)
  return exact_hits, total_hits - exact_hits


def compute_query(exact_hits, misplaced_hits, valid_candidates):
  results = np.array([
    exact_hits[valid_candidates],
    misplaced_hits[valid_candidates],
  ]).T
  U = np.array(list(itertools.product(range(M + 1), repeat=2)))
  eq = (results[..., None, :] == U[None, None, :, :]).all(axis=-1)
  return np.argmin(np.max(np.sum(eq, axis=1), axis=1))


def ask_for_hits():
  while True:
    try:
      exact_hits, misplaced_hits = input('Exact/misplaced: ').strip().split()
      exact_hits = int(exact_hits)
      misplaced_hits = int(misplaced_hits)
      return exact_hits, misplaced_hits
    except ValueError:
      print('I expected two integers separated by a space.')


def filter_candidates(
  exact_hits,
  misplaced_hits,
  query_idx,
  response,
):
  results = np.array([
    exact_hits[query_idx],
    misplaced_hits[query_idx]
  ]).T
  response = np.array(response)
  return (results == response).all(axis=-1)


if __name__ == '__main__':
  main()
