# Dataset File Tree and Extension Information

## MMPD
- Base path: `MMPD/subject{idx}/`
- Contents: `p{idx}_0.mat`

## UBFC-rPPG
- Base path: `UBFC-rPPG/subject{idx}/`
- Contents:
  - `vid.avi`
  - `ground_truth.txt`

## V4V
- Base path (Train/Val): `V4V/train_val/data/F{idx}_T{idx2}/`
- Contents (Train/Val):
  - `video.mkv`
  - `label.txt`
- Base path (Test Video): `V4V/test/Videos/Test/test/`
- Contents (Test Video): `{idx}.mkv`
- Base path (Test Label): `V4V/test/`
- Contents (Test Label): `validation_set_gt_release.txt`

## PURE
- Base path: `PURE/{idx}/`
- Contents:
  - `{idx}.json`
  - `{idx2}.png`

## VIPL_HR
- Base path: `VIPL_HR/data/p{idx}/v{idx2}/source{idx3}/`
- Contents:
  - `video.avi`
  - `gt_HR.csv`
  - `gt_SpO2.csv`
  - `time.txt`
  - `wave.csv`

## cohface
- Base path: `cohface/{idx}/{idx2}`
- Contents:
  - `data.avi`
  - `data.mkv`
  - `data.hdf5`

## LGGI
- Base path: `LGGI/{idx}/{idx}_{idx2}/`
- Contents:
  - `cms50_stream_handler.xml`
  - `cv_camera_sensor_stream_handler.avi`
  - `cv_camera_sensor_timer_stream_handler.xml`

## MAHNOB_HCI
- Base path: `MAHNOB_HCI/{idx}/`
- Contents:
  - `P{idx2}-Rec1-{date}_C1 trigger _C_Section_{idx3}.avi`
  - `Part_{idx2}_S_Trial{idx3//2}_emotion.bdf`

## vv100
- Base path: `vv100/`
- Contents:
  - `{idx}_{idx2}.mp4`
  - `{idx}.json`

# TODO
- Datasets to be added: 
  - UBFC-phys
  - VIPL_HR_V2
