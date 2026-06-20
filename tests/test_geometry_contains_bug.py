
import pytest
import numpy as np
import pandas as pd
import mikeio

def test_geometry_contains_mixed_element_sizes_bug():
    # 1. Load the reproducing dfsu subset and coordinate track from issue #697
    dfsu_path = "tests/testdata/Model_subset.dfsu"
    csv_path = "tests/testdata/Altmetry_data_debug.csv"
    
    ds = mikeio.read(dfsu_path)
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    
    # 2. Extract easting/northing coordinates of the points 
    # that are inside the grid but are being misclassified
    points = df[['easting', 'northing']].to_numpy()
    
    # Get the 2D Flexible Mesh geometry
    geom = ds.geometry
    
    # 3. Perform the containment check
    # Under the current implementation (using nearest 10 centroids),
    # this will incorrectly evaluate to False or raise an error for points near mixed element sizes.
    contains_result = geom.contains(points)
    
    # 4. Assert that all these points are correctly recognized as inside the mesh (True)
    assert np.all(contains_result == True), (
        "GeometryFM2D.contains failed to identify points inside the domain. "
        "This is likely due to the KDTree centroid check being limited to only 10 nearest elements."
    )
