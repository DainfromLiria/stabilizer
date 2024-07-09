"""
    Main project settings.
"""
BUFFER_SIZE = 5
FEATURE_PARAMS = dict(maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
TEXT_PARAMS = dict(org=(10, 30), fontFace=0, fontScale=1, color=(0, 255, 0), thickness=2)
CROP_MARGIN = 0.1  # float number from 0 to 1
PREVIEW_RESOLUTION = (700, 700)
MIN_DISTANCE = -100
MAX_DISTANCE = 100
BP_THRESHOLD = 2
BP_THRESHOLD_VIEW = 0.5
