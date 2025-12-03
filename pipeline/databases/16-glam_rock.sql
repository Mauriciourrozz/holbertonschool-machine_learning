-- This script lists all Glam rock bands ranked by their longevity
SELECT band_name,
       (IF(split IS NULL, 2020, split) - formed) AS lifespan
FROM metal_bands
WHERE style = 'Glam rock'
ORDER BY lifespan DESC;
