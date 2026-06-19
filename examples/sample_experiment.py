"""Example experiment: empirically test a CS hypothesis.

Hypothesis: 'Python's built-in sorted() is stable (equal keys keep input order).'
Convention: exit 0 if the hypothesis holds, non-zero (AssertionError) if it is false.
"""

data = [(1, "a"), (2, "b"), (1, "c"), (2, "d"), (1, "e")]
result = sorted(data, key=lambda t: t[0])
# If stable, the second elements of the key==1 group stay in input order: a, c, e
ones = [second for first, second in result if first == 1]
assert ones == ["a", "c", "e"], f"sort was NOT stable: {ones}"
print(f"stable sort confirmed: {ones}")
