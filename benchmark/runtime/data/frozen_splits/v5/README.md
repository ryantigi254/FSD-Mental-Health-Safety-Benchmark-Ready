# Frozen Snapshot v5

## Basis and sequencing
- Study A copied unchanged from frozen v4.1 and treated as the fixed basis.
- Study B single-turn provenance normalised to dataset-index references and only overlapping real rows replaced against Study A.
- Study A bias retained all disjoint groups from frozen v4.1 and replaced only groups overlapping the current Study A + Study B basis.
- Study B multi-turn regenerated fully against Study A + Study B + Study A bias.
- Study C retained unique frozen v4.1 cases where possible and replaced only cases colliding with the accumulated reference set or duplicating an already-kept Study C source pair.
