# GPT tag spot-check

- Date: 2026-11-08
- Population: 308 labelled representative ads
- Sample: 20 random ads, selected with seed `20261108`
- Checked against: source title/transcript and the three 0.5 s, 1.5 s, and 2.5 s keyframes
- Incorrect before correction: **3 of 20**
- Accuracy estimate: **17/20 = 85%**

## Sampled ad IDs

`1856409433779361`, `1854983131853841`, `1873603282964482`, `1855057222671665`, `1856184645078050`, `1873848809386145`, `1852090611028145`, `1878669980043346`, `1872214095482882`, `1878691477910721`, `1879782251684545`, `1870319569051873`, `1855160075963441`, `1867062261472417`, `1867424782084113`, `1847400995784737`, `1878689274269794`, `1848393195406353`, `1866324734784834`, `1867401222279330`

## Corrections made

1. `1856184645078050`: the text did not establish a billionaire/CEO story or fake marriage, so genre was changed to `unclear` and tropes to `none`.
2. `1848393195406353`: “I can't get out of bed” is distress, not a threat or ultimatum, so hook type was changed to `unclear`.
3. `1879782251684545`: the ending is not an unanswered question, so ending type was changed to `pending_threat`.

All three corrections were applied to the JSONL answer files before the final merge.
