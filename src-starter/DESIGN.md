## Start Time

Challenge opened: 2026-04-21 00:30:57 -04:00 (America/Toronto)

## Routing Topology

Current code maps one PE program over a `P x P` rectangle. The host shards `D`
by contiguous row ranges in row-major PE order, copies the full query `q` to
every PE, and copies one shard of rows plus a `valid_rows` count to each PE.
Each PE therefore owns rows
`[pe_linear * rows_per_pe, (pe_linear + 1) * rows_per_pe)`, clipped by `N`.

The current implemented path does not yet route candidate pairs on the fabric.
Instead, each PE computes a local top-`K` and exposes two output arrays:
`local_distances[K]` and `local_indices[K]`. The host reads back all `P * P`
local candidate lists and performs the final merge.

Planned on-wafer merge: keep the same sharding and candidate representation, but
send sorted local candidate lists toward a designated root PE using a row/column
reduction. A simple version is:

1. Merge west-to-east within each row until the rightmost PE in each row holds
   the row top-`K`.
2. Merge north-to-south within the final column until the southeast PE holds
   the global top-`K`.

At every stage, the payload is a sorted list of candidate pairs represented as
`(distance, global_row_index)`. This makes the merge rule independent of wavelet
arrival order.

## Local Top-K Algorithm

Each PE stores:

- `D_shard[rows_per_pe * d_dim]`
- `q[d_dim]`
- `valid_rows[1]`
- `local_indices[K]`
- `local_distances[K]`

For each valid local row, the PE computes squared L2 distance:

`dist(row) = sum_j (D_shard[row, j] - q[j])^2`

Squared distance is used because it preserves ordering and avoids a square root.

After computing one row distance, the PE inserts the candidate into a sorted
local top-`K` buffer. The current code keeps two parallel arrays rather than a
struct array, but logically each slot is a pair `(distance, global_row_index)`.
Insertion is done by linear search followed by shifting worse elements one slot
to the right.

Per row work is:

- `O(d_dim)` for distance computation
- `O(K)` for insertion into the top-`K` buffer

So the local PE work is `O(valid_rows * (d_dim + K))`.

## Fabric Bandwidth Accounting

In the code that exists now, fabric traffic is minimal because final merging is
still host-side. The host sends:

- one shard of `rows_per_pe * d_dim` floats to each PE
- one copy of `q[d_dim]` to each PE
- one `valid_rows` integer to each PE

Then it reads back:

- `K` local indices from each PE
- `K` local distances from each PE

The planned on-wafer merge changes the traffic pattern. If every PE first
computes a sorted local top-`K`, then each merge edge only needs to move `K`
candidates, not the full shard. If candidates are represented as
`(distance, global_row_index)`, each merge message contains `K` float/index
pairs. For a west-to-east row reduction followed by a north-to-south column
reduction, the worst-case total traffic scales with the number of merge edges:

- row phase: `Ph * (Pw - 1)` merges
- column phase: `Pw_final * (Ph - 1)` merges, where `Pw_final = 1`

So the dominant communication cost is proportional to `K * (P * (P - 1) + (P - 1))`
candidate pairs rather than to the full database size `N * d`.

## Tie-Breaking Argument

The current PE-local code already uses deterministic comparison. A candidate
`A = (dist_a, idx_a)` is better than `B = (dist_b, idx_b)` iff:

1. `dist_a < dist_b`, or
2. `dist_a == dist_b` and `idx_a < idx_b`

The important design choice is that `idx` is the original global row index in
`D`, not a local row number. Because row shards are assigned in contiguous
row-major order, every PE can compute:

`global_row_index = pe_linear * rows_per_pe + local_row`

This means every candidate has a globally meaningful total order under the key
`(distance, global_row_index)`.

That same comparator should be used in every on-wafer merge step. If two sorted
candidate lists are merged by repeatedly choosing the smaller of the two front
elements under the key `(distance, global_row_index)`, then the result depends
only on the set of candidates, not on message arrival order. Arrival timing may
change when a pair becomes available, but it does not change which pair wins a
comparison. This is the core determinism argument for the final design.

## If I Had 2x More Time

1. Move the final merge fully on-wafer. The current code already has correct
   PE-local top-`K`, row sharding, and host/runtime wiring. The next step is to
   replace host-side `merge_candidates()` with a real fabric reduction to a root
   PE using sorted candidate-list merges.
2. Reduce memory and traffic by packing candidate metadata more tightly and by
   streaming merges rather than storing extra temporary buffers per PE. That
   would improve SRAM headroom and likely help with cycle count on the baseline.
