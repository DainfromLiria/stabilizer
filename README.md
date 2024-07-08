# Real-time video stabilization

>[NOTE] Now works ONLY `stabilizer` package. `stabilizer_v2` is in development!

## About
This project provides two Python packages for real-time digital image stabilisation for the OAK-D Pro camera.

## Stabilizer
These packages use feature detection techniques for key point tracking. Based on key points, 
create 3x3 homography rotation matrix. Then using running average filter, smooth frame path and warp input frame.

## Stabiizer_v2
This package will work on quaternion rotation vector using data from MEMS-sensor in camera. 
This package is currently under development.

## How run
1. Install all library's and packages

>[NOTE] User's PC must have already installed python 3.11 or higher.

```shell 
pip install -r requirements.txt
```
2. Connect camera to PC.
3. Run application.
>[NOTE] If you don't want to run stabilisation in preview mode (show original and 
> stabilised video, but in cropped resolution), change the `preview_mode` parameter in `main.py` to `False`.
```shell
python main.py
```
4. Close window on `q` or `esc`
