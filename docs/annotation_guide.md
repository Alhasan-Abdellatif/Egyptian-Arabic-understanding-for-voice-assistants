# Egyptian test set: annotation guide

You are writing the **evaluation data**. Every headline number in the project comes from it, so
natural Egyptian and consistent labels matter more than speed.

## Before you start
- **Write the test set before you read any synthetic (Claude-generated) data.** Otherwise your
  phrasing drifts toward the generator's style and the scores get inflated.
- Open the CSVs in Google Sheets, Numbers, or Excel (UTF-8). Only edit `egy_annot`, `notes`,
  and (challenge set only) `intent` / `category`.

## Format: MASSIVE bracket notation
```
صحيني [date : بكرة] الساعة [time : ستة الصبح]
شغلي أي حاجة ل[artist_name : عمرو دياب]           ← attached preposition stays outside (see below)
الجو عامل ايه في [place_name : اسكندرية] [date : النهارده]
```
`make testset` checks the brackets, intents, and slot types, and prints errors with line numbers.

## egy_test.csv (200 rows)
For each row, read `source_annot` (a Saudi/MSA MASSIVE test item) and write what **you** would
naturally say to your phone in Egyptian to get the same thing.
- Keep the **same intent**. Try to keep the **same slot types** (the validator warns if they
  differ, which is fine when the Egyptian phrasing naturally drops or adds one).
- Rephrase freely; don't translate word by word. If the source is odd or a fragment, write the
  natural Egyptian request with that intent.
- Localize where an Egyptian would (الرياض → القاهرة, ريال → جنيه). Keep names of people,
  artists, and apps.
- Leave `egy_annot` empty to skip a row you can't do naturally (say why in `notes`).

## egy_challenge.csv (50 rows)
Write your own commands for any intent in `configs/schema.json`, and fill `intent` and
`category`. Aim for roughly 10 per category:

| category | example |
|---|---|
| `code_switch` | اعملي [event_name : meeting] مع [person : أحمد] [date : بكرة] |
| `arabizi` | 3ayez asma3 [artist_name : Amr Diab] |
| `digits` | صحيني الساعة [time : ٦:٣٠] |
| `time_expression` | فكرني أصلي [time : بعد العصر] |
| `local_entity` | الجو عامل ايه في [place_name : الساحل] |

## Slot conventions (follow MASSIVE)
- Bracket **only the words that fill the slot**: `الساعة [time : ستة]`, not `[time : الساعة ستة]`.
- **Clitics:** keep attached prepositions outside the slot by opening the bracket right after
  them, with no space: `شغلي حاجة ل[artist_name : عمرو دياب]` becomes the text
  "شغلي حاجة لعمرو دياب" with the slot value "عمرو دياب".
- Dates and times: bracket the expression as spoken (`[date : بكرة]`, `[time : ستة ونص]`).
