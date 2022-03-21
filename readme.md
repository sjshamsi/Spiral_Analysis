## What are all these files?

`initial_sample_filter.ipynb`

(We don't need to have user do this, just refer it.)
- This is where we initially filter our sample. We first identify all `gz3d` galaxies which have a corresponding MaNGA observations. This comes out to about 9,000 galaxies.
- We then identify all spiral galaxies from this sample. A spiral galaxy is defined as a galaxy where at least one pixel is identified as a spiral arm by at least one person. This (admittedly) lose definition gives us about 2,300 spiral galaxies.
- We save all these lists in `numpy` and `pandas` pickle files.

`manga_population_map.ipynb`

(Should have the user reference this in the tutorial after threshold selection)

- We display our sample relative to the entire MaNGA sample