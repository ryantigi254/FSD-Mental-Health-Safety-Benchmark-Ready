# Study B Injected Prompt IDs

This appendix documents the exact Study B prompt IDs that carried underrepresented-condition context into the frozen controllability splits.

For Study B, “injected” is used in two slightly different senses:

- single-turn: underrepresented-condition cases identifiable from `metadata.inferred_category`
- multi-turn: underrepresented-condition cases with explicit realised `pressure_style` / `pressure_schedule`

## Study B Single-Turn

Current underrepresented-category count:

- `120 / 300`

### Counts by Category

- `Psychotic Spectrum`: 77
- `Substance Use Disorders`: 11
- `Self-Harm & Suicidality`: 8
- `Personality Disorders`: 7
- `Eating Disorders`: 6
- `Somatic & Health-Related`: 5
- `Sleep-Wake Disorders`: 3
- `Neurodevelopmental`: 3

### IDs by Category

#### Eating Disorders

`ctrl_b_0024`, `ctrl_b_0031`, `ctrl_b_0056`, `ctrl_b_0090`, `ctrl_b_0103`, `ctrl_b_0118`

#### Neurodevelopmental

`ctrl_b_0013`, `ctrl_b_0050`, `ctrl_b_0112`

#### Personality Disorders

`ctrl_b_0003`, `ctrl_b_0007`, `ctrl_b_0008`, `ctrl_b_0018`, `ctrl_b_0040`, `ctrl_b_0086`, `ctrl_b_0095`

#### Psychotic Spectrum

`ctrl_b_0001`, `ctrl_b_0002`, `ctrl_b_0004`, `ctrl_b_0006`, `ctrl_b_0009`, `ctrl_b_0010`, `ctrl_b_0014`, `ctrl_b_0016`, `ctrl_b_0017`, `ctrl_b_0021`, `ctrl_b_0022`, `ctrl_b_0023`, `ctrl_b_0026`, `ctrl_b_0027`, `ctrl_b_0028`, `ctrl_b_0029`, `ctrl_b_0030`, `ctrl_b_0032`, `ctrl_b_0034`, `ctrl_b_0035`, `ctrl_b_0036`, `ctrl_b_0038`, `ctrl_b_0039`, `ctrl_b_0041`, `ctrl_b_0042`, `ctrl_b_0043`, `ctrl_b_0044`, `ctrl_b_0045`, `ctrl_b_0046`, `ctrl_b_0047`, `ctrl_b_0048`, `ctrl_b_0049`, `ctrl_b_0051`, `ctrl_b_0052`, `ctrl_b_0053`, `ctrl_b_0055`, `ctrl_b_0057`, `ctrl_b_0058`, `ctrl_b_0059`, `ctrl_b_0060`, `ctrl_b_0062`, `ctrl_b_0063`, `ctrl_b_0065`, `ctrl_b_0066`, `ctrl_b_0067`, `ctrl_b_0068`, `ctrl_b_0069`, `ctrl_b_0070`, `ctrl_b_0072`, `ctrl_b_0073`, `ctrl_b_0074`, `ctrl_b_0076`, `ctrl_b_0077`, `ctrl_b_0079`, `ctrl_b_0080`, `ctrl_b_0081`, `ctrl_b_0082`, `ctrl_b_0083`, `ctrl_b_0084`, `ctrl_b_0085`, `ctrl_b_0087`, `ctrl_b_0088`, `ctrl_b_0089`, `ctrl_b_0093`, `ctrl_b_0097`, `ctrl_b_0098`, `ctrl_b_0100`, `ctrl_b_0101`, `ctrl_b_0106`, `ctrl_b_0109`, `ctrl_b_0110`, `ctrl_b_0111`, `ctrl_b_0113`, `ctrl_b_0114`, `ctrl_b_0115`, `ctrl_b_0116`, `ctrl_b_0119`

#### Self-Harm & Suicidality

`ctrl_b_0012`, `ctrl_b_0019`, `ctrl_b_0033`, `ctrl_b_0037`, `ctrl_b_0054`, `ctrl_b_0071`, `ctrl_b_0096`, `ctrl_b_0105`

#### Sleep-Wake Disorders

`ctrl_b_0015`, `ctrl_b_0025`, `ctrl_b_0092`

#### Somatic & Health-Related

`ctrl_b_0020`, `ctrl_b_0078`, `ctrl_b_0091`, `ctrl_b_0094`, `ctrl_b_0108`

#### Substance Use Disorders

`ctrl_b_0005`, `ctrl_b_0011`, `ctrl_b_0061`, `ctrl_b_0064`, `ctrl_b_0075`, `ctrl_b_0099`, `ctrl_b_0102`, `ctrl_b_0104`, `ctrl_b_0107`, `ctrl_b_0117`, `ctrl_b_0120`

## Study B Multi-Turn

Current underrepresented-category count:

- `17 / 30`

These are the multi-turn cases where the frozen split combines an underrepresented condition with an explicit realised pressure style/schedule.

### IDs by Category

#### Personality Disorders

`ctrl_b_mt_0020 (self_doubt/early_spike)`

#### Psychotic Spectrum

`ctrl_b_mt_0004 (authority_pressure/early_spike)`, `ctrl_b_mt_0014 (social_proof/early_spike)`, `ctrl_b_mt_0016 (evidence_citation/gradual)`, `ctrl_b_mt_0023 (evidence_citation/gradual)`

#### Self-Harm & Suicidality

`ctrl_b_mt_0003 (authority_pressure/early_spike)`, `ctrl_b_mt_0011 (social_proof/early_spike)`, `ctrl_b_mt_0013 (disbelief/early_spike)`, `ctrl_b_mt_0015 (social_proof/early_spike)`, `ctrl_b_mt_0018 (self_doubt/early_spike)`, `ctrl_b_mt_0022 (authority_pressure/early_spike)`, `ctrl_b_mt_0028 (disbelief/late_spike)`, `ctrl_b_mt_0030 (authority_pressure/gradual)`

#### Sleep-Wake Disorders

`ctrl_b_mt_0008 (disbelief/early_spike)`, `ctrl_b_mt_0012 (self_doubt/early_spike)`, `ctrl_b_mt_0025 (authority_pressure/early_spike)`

#### Substance Use Disorders

`ctrl_b_mt_0029 (evidence_citation/early_spike)`
