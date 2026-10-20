# WallVoice24BD

WallVoice24BD is a collection of wall-graffiti images from several areas of Dhaka, Bangladesh. The current collection contains **1,225 PNG images in six area folders**. Keep the original images unchanged; use the metadata file to describe and find them.

## Current contents

| Folder | Suggested area code | Images |
|---|---:|---:|
| `DU-Shahbagh/` | `DUS` | 428 |
| `Dhanmondi/` | `DMD` | 312 |
| `Azimpur-Nilkhet/` | `AZN` | 172 |
| `BaileyRoad-Viqarunnisa/` | `BRV` | 136 |
| `Buet-Palashi/` | `BUP` | 90 |
| `Mirpur-Uttara-Tejgaon/` | `MUT` | 87 |

The existing filenames (for example, `DUS (1).png`) are left as-is. Their numbering is only an identifier, not a capture date or an annotation.

## Recommended project structure

```text
WallVoice24BD/
├── README.md
├── metadata/
│   └── images.csv             # One record per image; annotation fields start blank
├── DU-Shahbagh/                # Original images, unchanged
├── Dhanmondi/
├── Azimpur-Nilkhet/
├── BaileyRoad-Viqarunnisa/
├── Buet-Palashi/
└── Mirpur-Uttara-Tejgaon/
```

As the project grows, keep derived or annotated material separate from the original photos:

```text
├── annotations/                # Reviewed labels/transcriptions, if needed
├── processed/                  # Crops or resized copies; never overwrite originals
├── splits/                     # Train/validation/test image lists
└── docs/                       # Collection protocol, label guide, and release notes
```

Only add those folders when you have a use for them. Do not duplicate images into train/validation/test folders; store their paths in split files instead.

## Image metadata

`metadata/images.csv` is the image inventory, with one row per image and paths relative to this project folder, such as `DU-Shahbagh/DUS (1).png`. The existing image IDs and paths are prefilled; fill in verified metadata and annotations as you collect or review them. Keep `image_id` unique and stable; do not use a person's name or other identifying information in an ID.

The starter CSV includes these fields:

| Field | What to record |
|---|---|
| `image_id`, `relative_path` | Stable ID and exact path to the image |
| `area`, `sub_area` | Known collection area and optional broader/narrower location |
| `capture_date` | Date taken (`YYYY-MM-DD`); leave blank if unknown |
| `source`, `photographer`, `license` | Image source, collector/photographer code, and permission or license status |
| `latitude`, `longitude` | Only if collected and safe to publish; otherwise leave blank or use a less precise location |
| `language`, `content_type` | Observed language and a documented category such as text, drawing, or mixed |
| `transcription`, `translation` | Exact visible text and translation; use blank when unreadable or not reviewed |
| `topics`, `sentiment_or_stance` | Optional labels, following a written label guide; do not present interpretation as fact |
| `review_status`, `notes` | Annotation/review state and brief non-sensitive notes |

Use blank cells for unknown or not-yet-reviewed values; do not guess. Agree on controlled values before multiple people annotate (for example, language codes and review-status values), and record those definitions in a label guide. Preserve the original image and record any processing separately.

## Collection and research practices

- Keep a collection log: when and where images were collected (at an appropriate privacy level), who collected them, how they were obtained, and the applicable permission/license.
- Confirm that you have the right to use and share each image. Public visibility does not automatically mean unrestricted reuse.
- Avoid collecting or publishing faces, vehicle plates, private contact details, or precise coordinates that could identify a person or expose a sensitive location. Blur/redact only a derivative copy and document that change.
- Retain the original filenames and back up the original images. Check for unreadable files, accidental duplicates, and images that do not show graffiti.
- For model evaluation, split by wall/site or collection route when possible, rather than randomly splitting individual photos. Near-duplicate views of the same wall in different splits can inflate results. Check that each area and important label is represented in the evaluation data.
- Document the intended task (for example, image classification, text transcription, or area-level analysis), label definitions, known collection biases, and any limitations before releasing or training on the dataset.

## Before sharing or using the dataset

Complete the metadata, document permissions and intended use, review privacy risks, and include a label guide and dataset version/date. State that the locations, collection methods, and image counts may be uneven; do not claim the collection represents all graffiti or public opinion in Dhaka.
