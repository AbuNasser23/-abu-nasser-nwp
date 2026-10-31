# Abu Nasser NWP publisher

Public, zero-cost publisher for the observatory's 0.25° packs.

- ECMWF Open Data and NOAA GFS on AWS. No API key.
- GitHub Actions `ubuntu-latest` and GitHub Pages only.
- Each model is a separate job. One failure does not change the other pointer.
- `current.json` is replaced only after the cycle directory re-reads as complete.
- Horizon is 240 hours. ECMWF 00/12 UTC cycles: 3-hourly to +144h, then 6-hourly to +240h. GFS cycles: 3-hourly to +120h, then 6-hourly to +240h. No synthetic forecast times are inserted. Grid is 173×137.
- The last two successful cycles are kept. Older directories are removed only after the pointer moves.
