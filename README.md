# img_enhancement
This is an image enhancement project on medical videos.

### Package structure

```
med_video_enhancement
├── __init__.py
├── util
│   └── evaluate_img.py
├── img_util
│   ├── __init__.py
│   ├── constrast_stretch.py
│   ├── dynamic_range_compress.py
│   ├── gamma.py
│   ├── retinex.py
│   ├── histo_eq.py
│   ├── histo_match.py
│   └── freq.py
├── vid_util
│   ├── __init__.py
│   ├── vid_splitter.py
│   ├── pc_trainer.py
│   └── pc_model.py
└── output
    ├── vid
    └── img
```
--------------------------------
#### [hackMD notes](https://hackmd.io/SkVk1-R0TeGcZOkP3kn0xw)


### Problem Description

Problem: Endoscopic surgeries require clean views of the organs and tissue, however, videos recorded via the endoscopes often contain a lot of extraneous noise in the form of blood and surgical smoke occuring from the use of electrosurgical unit (ESU).

Objective: Apply image enhancement technique on endoscopic videos to obtain processed video with clean view of the tissues and organs, potentially removing the noises of blood and surgical smoke.


### Design
Input: Endoscopic surgical videos with noises

Desired Output: Processed endoscopic surgical videos with clean views (potentially with the blood and surgical smoke removed)
