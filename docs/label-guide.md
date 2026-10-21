# Label guide — WallVoice24BD

Version: 1.0 (2026-10-21). This guide defines the controlled values used by the
KNN baseline and the metadata file `metadata/images.csv`.

## 1. Prediction target

The baseline predicts **`area`** (the collection area / neighbourhood where the
wall was photographed), derived from the folder an image lives in. It does not
predict language, transcription, topic, or stance; those fields remain blank
until a human annotates them.

`area` is a *collection* label, not a claim about where the graffiti was made.
It is inferred from the folder and must not be “corrected” from visual style.

### Area codes

| Code | Folder | Images | What the label means |
|---|---|---:|---|
| `DUS` | `DU-Shahbagh/` | 428 | Dhaka University / Shahbagh area |
| `DMD` | `Dhanmondi/` | 312 | Dhanmondi area |
| `AZN` | `Azimpur-Nilkhet/` | 172 | Azimpur–Nilkhet area |
| `BRV` | `BaileyRoad-Viqarunnisa/` | 136 | Bailey Road / Viqarunnisa area |
| `BUP` | `Buet-Palashi/` | 90 | BUET / Palashi area |
| `MUT` | `Mirpur-Uttara-Tejgaon/` | 87 | Mirpur / Uttara / Tejgaon area |

Examples (one per label):

- `DUS` — `DU-Shahbagh/DUS (42).png`: cartoon-style political wall painting.
- `DMD` — `Dhanmondi/DMD (127).png`: lettering on a boundary wall.
- `AZN` — `Azimpur-Nilkhet/AZN (18).png`: wall writing along a roadside wall.
- `BRV` — `BaileyRoad-Viqarunnisa/BRV (33).png`: painted slogan.
- `BUP` — `Buet-Palashi/BUP (81).png`: “KILLING Da FLOWERS WONT DELAY SPRING”.
- `MUT` — `Mirpur-Uttara-Tejgaon/MUT (8).png`: wall graffiti on a roadside wall.

Rules:

1. Use exactly one code from the table. No free text, no combinations.
2. Keep `area` consistent with `relative_path`; the folder is the source of truth.
3. Never move an image between folders to change its label. Record uncertainty
   in `notes`, not by inventing a new class.
4. The classes are **imbalanced by collection design** (428 vs 87). Report
   macro-F1 alongside accuracy and do not interpret a high accuracy as balanced
   performance.
5. The locations are coarse. Do not add precise coordinates without consent.

## 2. Other metadata fields

These are part of the inventory but are **not** used as prediction targets in
this baseline.

| Field | Controlled values / format |
|---|---|
| `image_id` | `DUS-0001`, `DMD-0001`, …; stable, never reused |
| `relative_path` | POSIX path relative to the repo root, e.g. `DU-Shahbagh/DUS (1).png` |
| `area` | folder name (see table) |
| `sub_area` | short place name, only if known and non-sensitive |
| `capture_date` | `YYYY-MM-DD`; blank if unknown |
| `source` | `field-collection`, `shared-with-permission`, `public-web`, or blank |
| `photographer` | collector code, not a real name unless consent is recorded |
| `license` | e.g. `CC-BY-4.0`, `permission-research-only`, blank if unknown |
| `latitude`, `longitude` | leave blank unless publication is safe and consented |
| `language` | ISO 639-1/639-3 code (`bn`, `en`, `mixed`), blank if unreviewed |
| `content_type` | `text`, `drawing`, `mixed`, `sticker`, `other` |
| `transcription` | exact visible text; blank when unreadable |
| `translation` | English translation; mark uncertainty with `[?]` |
| `topics` | semicolon-separated from the topic list below |
| `sentiment_or_stance` | `supportive`, `critical`, `neutral`, `unclear` |
| `review_status` | see below |
| `notes` | short, non-sensitive notes |

Topic list (optional, for later annotations): `politics`, `religion`,
`education`, `labour`, `gender`, `health`, `environment`, `sports`, `memorial`,
`advertising`, `personal`, `other`.

### `review_status`

| Value | Meaning |
|---|---|
| `unreviewed` | no human has checked the row (current state of all rows) |
| `ai-labelled` | machine output only; must not be treated as ground truth |
| `human-verified` | a person checked the image and the fields against this guide |
| `disputed` | reviewers disagree; resolve before modelling on the field |
| `excluded` | reviewed and deliberately removed from modelling (original kept) |

The KNN baseline uses folder-derived `area`, so it is **not** a human-verified
annotation. Any future use of `content_type`, `topics`, or `transcription`
needs at least one human verification pass.

## 3. Splitting and duplicate policy

- Images are grouped when they are exact duplicates (MD5), perceptual-hash
  near-duplicates (pHash Hamming ≤ 6), or CNN near-duplicates (ResNet-18
  ImageNet embedding cosine ≥ 0.93).
- A whole group goes to one split. No group may span train/val/test.
- Split ratio is approximately 70/15/15 per area, seed 42.
- The test split is frozen and is evaluated once with `scripts/evaluate.py`.
- Do not delete originals to “clean” the dataset. Record exclusions in
  `review_status=excluded` and keep the file.

## 4. Privacy and release

Before publishing images or annotations: check for faces, vehicle plates,
phone numbers, and precise locations; blur only a derivative copy and document
it. Do not publish coordinates or photographer identities without consent.
The baseline itself publishes no images and no new personal data.
