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

To generate synthetic representations of the pages, one per century

```bash
python create_types.py
```
output: `pages_type_overlay/`

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

## Cite

Simon Gabay, UniGE