<div align="center">

# orena-focus

[![Tests](https://img.shields.io/github/actions/workflow/status/IMSY-DKFZ/orena-focus/tests.yml?branch=main&label=tests)](https://github.com/IMSY-DKFZ/orena-focus/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/orena-focus?color=blue)](https://pypi.org/project/orena-focus/)
[![Python](https://img.shields.io/pypi/pyversions/orena-focus)](https://pypi.org/project/orena-focus/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MICCAI 2026](https://img.shields.io/badge/Challenge-MICCAI%202026-blue)](https://or-arena.org/)

**HeiCo-FOCUS dataset**

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/Data-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-orena--dkfz%2Fheico--focus--vqa-blue)](https://huggingface.co/datasets/orena-dkfz/heico-focus-vqa)

**LapChole-FOCUS dataset**

[![License: Data Usage Agreement](https://img.shields.io/badge/License-CC%20Data--Usage--Agreement%204.0-lightgrey.svg)](https://huggingface.co/datasets/orena-dkfz/lapchole-focus-vqa/blob/main/LICENSE)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-orena--dkfz%2Flapchole--focus--vqa-blue)](https://huggingface.co/datasets/orena-dkfz/lapchole-focus-vqa)


</div>

<br>

Python utilities for the **FOCUS datasets and challenge** — *Foreign Object Contextual Understanding for Surgery*.

The library provides dataset loaders, preprocessing pipelines, answer-format handling, and an evaluation framework for working with the FOCUS surgical VQA datasets. It can be used independently for research on foreign-object understanding in minimally invasive surgery, and also serves as the official toolkit for the [ORena SAVE FOCUS challenge](https://or-arena.org/) at MICCAI 2026.

> **Challenge open for registration.** Submit your results and compete on the leaderboard at [orena-focus-challenge.org](https://orena-focus-challenge.org/).

![Clinical use case](https://github.com/IMSY-DKFZ/orena-focus/blob/main/src/focus/assets/motivation.png?raw=true)

Retained foreign objects are a life-threatening and preventable surgical complication. FOCUS benchmarks vision-language models on clinically relevant VQA tasks around detecting, counting, and reasoning about foreign objects in endoscopic video.

## Tracks

FOCUS offers three participation tracks, each requiring a different type of visual context:

| Track | `Track` enum | Visual input | Description |
|-------|-------------|--------------|-------------|
| **FRAME** | `Track.FRAME` | Single frame | Answer questions from one extracted video frame. The simplest entry point — no temporal modelling required. |
| **SEGMENT** | `Track.SEGMENT` | <= 5min clip | Answer questions from a multi-second video segment surrounding the relevant event. Requires understanding of motion and temporal context. |
| **PROCEDURE** | `Track.PROCEDURE` | Up to full video | Answer questions that may require reasoning over an entire surgical procedure, including events that happened well before or after the queried moment. |

Participants may enter any subset of tracks. Each track is evaluated independently with the same hierarchical capability taxonomy.

## Installation

```bash
pip install orena-focus
```

## Quick start

```python
from focus import FocusDataset, DatasetSplit, Track

ds = FocusDataset("heico", DatasetSplit.TEST, Track.SEGMENT)

request, reference = ds[0]
print(request.question)        # "How many sponges are visible?"
print(reference.answer)        # "2"
print(reference.format.type)   # "number"
```

## Data preparation

Download, preprocess, and split the dataset in one script — see **[`examples/data_preparation.py`](examples/data_preparation.py)** for the full walkthrough.

```python
from focus import download
from focus.preprocessing import VideoTimestampOverlayPreprocessor, FrameExtractorPreprocessor

download("heico")

VideoTimestampOverlayPreprocessor().process(dataset="heico")
FrameExtractorPreprocessor(stride=1).process(dataset="heico")
```

QA annotations are fetched automatically from HuggingFace when you construct a `FocusDataset`.

## Inference & evaluation

See **[`examples/inference.py`](examples/inference.py)** for an end-to-end example with Qwen3-VL.

```python
from focus import Evaluator, Response

responses = [Response(qID=req.qID, content=my_model(req)) for req, _ in ds]

results_df, summary_df = Evaluator().run(
    requests=ds.requests,
    references=ds.references,
    responses=responses,
)
print(summary_df)
```

## Capability taxonomy

Five capability groups, each composed of leaf capabilities assigned to questions.

![SAVE FOCUS capability taxonomy with example questions](https://github.com/IMSY-DKFZ/orena-focus/blob/main/src/focus/assets/taxonomy.png?raw=true)

## Answer formats

| Format | Accepts | Returns |
|--------|---------|---------|
| `Binary` | `"yes"` / `"no"` | `bool` |
| `Number` | Non-negative integer strings | `int` |
| `Percentage` | Numeric percentage strings | `float` |
| `FOClass` | Registered FO class names | `str` |
| `OpenEnded` | Free text (≤ 300 chars) | `str` |
| `Matching` | Regex-validated text | `str` |
| `MultipleChoice` | One of predefined options | `str` |
| `Time` | `hh:mm:ss` timestamps | `timedelta` |

## Datasets

FOCUS builds upon two datasets: **HeiCo-FOCUS** (30 videos) and LapChole-FOCUS (170 videos).

### HeiCo-FOCUS

The QA annotations are publicly available on HuggingFace: **[orena-dkfz/heico-focus-vqa](https://huggingface.co/datasets/orena-dkfz/heico-focus-vqa)**.

 **HeiCo-FOCUS** is built on the **HeiCo** dataset. If you use this data, please cite the original publication:

> Maier-Hein, L., et al. (2021). *Heidelberg colorectal data set for surgical data science in the sensor operating room*. [https://doi.org/10.1038/s41597-021-00882-2](https://doi.org/10.1038/s41597-021-00882-2)

The HeiCo data is released under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) — non-commercial use only, with attribution and share-alike conditions.

### LapChole-FOCUS

The **LapChole-FOCUS** videos comprises unpublished laparoscopic cholecystectomy recordings. It is currently only available within the scope of the FOCUS-Challenge and will be publicly released after the Challenge has ended.

To access it, send a request on HuggingFace: **[orena-dkfz/lapchole-focus-vqa](https://huggingface.co/datasets/orena-dkfz/lapchole-focus-vqa)**. To access and download the data you need to register yourself with HuggingFace. 

Please find the data usage agreement at the dataset card or in the `LICENSE` file on HuggingFace.

## License

The code is licensed under the permissive [MIT license](https://github.com/IMSY-DKFZ/orena-focus/blob/main/LICENSE). The underlying data is licensed independently, see [Dataset](#dataset) for the data licenses details.
