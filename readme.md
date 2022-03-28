## What are all these files?

`initial_sample_filter.ipynb`

(We don't need to have user do this, just refer it.)
- This is where we initially filter our sample. We first identify all `gz3d` galaxies which have a corresponding MaNGA observations. This comes out to about 9,000 galaxies.
- We then identify all spiral galaxies from this sample. A spiral galaxy is defined as a galaxy where at least one pixel is identified as a spiral arm by at least one person. This (admittedly) lose definition gives us about 2,300 spiral galaxies.
- We save all these lists in `numpy` and `pandas` pickle files.

`manga_population_map.ipynb`

(Should have the user reference this in the tutorial after threshold selection)

- We display our sample relative to the entire MaNGA sample

## Some questions that I have
- What if our sample has a galaxy which 'does not fit' into the MaNGA Plate IFU? Like it is too 'big' and/or 'near' and thus it seems that a very low $r/r_e$ radius exists within our hexagon. In this case, maybe it is a good idea to remove these galaxies. Is this worth it? Is it going to be a lot of galaxies at all? IDK.
- I'm still concerned that some galaxies in our sample might have just a couple of spaxels identified as spiral and then being used for analysis. Maybe I'll do a % spiral spaxels count and yeet out the ones that are super low, if they at all exist
- I really want to go through all these galaxies image by image and view the spiral regions differing with threshold. Just to see if different galaxy subsets have a particular threshold they work best with, and it would be cool to mark galaxies as grand design, or ones with three spiral arms etc. I bet these classification have already been done though.