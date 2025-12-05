# Layout

## Install


```bash
virtualenv -p python3.10 env
source env/bin/activate
pip install -r requirements
```

## Run

⚠️ Remember to adjust the path and other variables at the top of each script.

To compute the quantity of several zones over centuries

```bash
python zones_count.py
```
output: `graph_zones_count.pdf`

To compute the relative percentage between two concurrent forms

```bash
python tokens_count.py
```
output: `graph_tokens_count.png`

To compute the average number of abbreviations (for one language, repeat for each language)

```bash
python count_abbr.py
```
output: `graph_count_abbr.png`

To compute the lexical richness

```bash
python lexical_richeness.py
```
output: `graph_count_abbr.png`

To generate synthetic representations of the pages, one per century (experimental)

```bash
python create_types.py
```
output: `pages_type_overlay/`

## Cite

```bibtex
@unpublished{clerice:hal-05299220,
  TITLE = {{CoMMA, a Large-scale Corpus of Multilingual Medieval Archives}},
  AUTHOR = {Cl{\'e}rice, Thibault and Gabay, Simon and Vlachou-Efstathiou, Malamatenia and Pinche, Ariane and Sagot, Beno{\^i}t},
  URL = {https://inria.hal.science/hal-05299220},
  NOTE = {working paper or preprint},
  YEAR = {2025},
  MONTH = Oct,
  KEYWORDS = {Automatic Text Recognition ; Medieval manuscripts ; Latin ; French ; Digital humanities ; Corpus},
  PDF = {https://inria.hal.science/hal-05299220v1/file/Latin_and_Old_French_Manuscripts-8.pdf},
  HAL_ID = {hal-05299220},
  HAL_VERSION = {v1},
}
```
