# Data sources

| Source | Type | Access | Licence | Notes |
|---|---|---|---|---|
| Roboflow Universe — vitiligo sets (`vitiligo-rzgah/viti-main`, `futureofikigai/vitiligo_1`, `sedki/vitiligo-amw4r`) | clinical photos, seg masks | public | per-project (check) | small, variable label quality |
| DermaCon-IN | Indian skin disorders, multi-concept annotations | on request | research | most relevant for Fitzpatrick IV–VI |
| DermNet | clinical photos | public | CC BY-NC-ND (check) | vitiligo category |
| ISIC / HAM10000 | dermoscopy | public | CC BY-NC | almost no vitiligo; negatives only |
| VR Foundation CloudBank | longitudinal patient records | academic request | research | ~2,800 datasets; for response prediction |
| VR Foundation Biobank | serum / hair / DNA | academic request | research | biomarker work |

Add a row for every new source. Images are never committed; store a manifest CSV in `data/public/<source>/manifest.csv` with `filename,url,licence,fitzpatrick,body_site,image_type`.
