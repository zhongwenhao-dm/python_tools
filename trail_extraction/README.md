## Trail Extraction

### 1. Extract Centerline
Extract centerline from processed pointcloud file(remove outliers).

For the Cocopark scene, extract the orange points as trail points, convert them into polar coordinates, fit a closed curve in polar angle order, and then sample to get the centerline (point set form)

```
python extract_center_line_from_pointcloud.py
```

### 2. Click to select points
Select proper points and fit a closed curse.
Can be used multiple times to get a centerline coarse to fine.

```
python click_and_simulate_centerline.py
```



### 3. Generate pose file from centerline
Use ordered positions to calculate rotations of all the points, and generate a trail map file which can be used in localization algorithm.

```
python generate_poses.py
```