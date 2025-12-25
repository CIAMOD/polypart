from time import perf_counter

from polypart.arrangements import get_moduli_arrangement
from polypart.delres import number_of_regions
from polypart.polytopes import get_product_of_simplices
from polypart.ppart import build_partition_tree

n, r = 3, 3
P = get_product_of_simplices(n, r - 1)
P.extreme()
A = get_moduli_arrangement(n, r, 0)
start = perf_counter()
# num_regions = number_of_regions(A, P)
_, num_regions = build_partition_tree(P, A)
end = perf_counter()
print(f"Found {num_regions} regions in {end - start:.4f} seconds")
