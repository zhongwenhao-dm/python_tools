## 将轨道模型投影到图片上
在有fbx模型的前提下，先将其转为obj格式，然后以某个视角投影到图片上。

通过调整里面的scale和translationhnh投影参数来让投影轨道和图片吻合

```
python project_obj_to_image.py
```



## 重建过山车轨道

从2d图纸中重建出3d的过山车轨道：
- 手动标注2d轨迹点然后保存```python trajectory_2d_generation.py```
- 标注高度点并平滑拟合轨迹```python trajectory_3d_generation.py```
- 可视化编辑器编辑3d轨迹点```python edit_3d_trajectory.py```


查看3d生成结果：```python visualize_3d_traj.py --file trajectory_3d.csv```



## 模拟位姿

根据过山车各个路段的速度以及过山车的3d轨道图片，模拟出在轨迹上的位姿。

注意：这个位姿是按照帧率进行输出的位姿，是根据轨道点和大致速度估计出轨道点速度之后，再计算出加速度，并且按照时间间隔，从开始点重新生成的轨迹
```
python simulate_pose.py
```

可视化位姿结果进行验证
```
# 将四元数再次转为angle再可视化
python validate_pose.py

# 直接用四元数可视化，以及一些异常角速度分析
python play_traj_test.py
```

