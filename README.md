# HierGeo project page

Static project page for **HierGeo: POI-Grounded Satellite Captions and a Diagnostic Benchmark for Geographically Constrained Earth Observation Retrieval**. Code: <https://github.com/shubhamrpatle/HierGeo>.

The page is plain HTML, CSS and JavaScript, so no build step is needed.

## Preview

```bash
python -m http.server 8000
```

Then open <http://localhost:8000>. The place-parser demo loads `assets/data/places.json` with `fetch`, so open the page through a server rather than as a file.

## Files

| File | Purpose |
| --- | --- |
| `index.html` | Page content |
| `styles.css` | Shared MBZUAI-Oryx academic page theme |
| `hiergeo.css` | Additions for this page |
| `script.js` | Navigation, reveal animations, lightbox, BibTeX copy, parser demo |
| `place-parser.js` | JavaScript port of `hiergeo/routing.py`. It gives the same route as the Python parser on all 23,470 benchmark queries. |
| `assets/data/places.json` | Gazetteer and per-city patch counts exported from the corpus metadata |
| `assets/tables/*.png`, `assets/figures/teaser.png` | Rendered from the paper's LaTeX sources |

## Deploy with GitHub Pages

Push the contents of this folder to the `gh-pages` branch of `shubhamrpatle/HierGeo`. The page is then served at <https://shubhamrpatle.github.io/HierGeo/>.
