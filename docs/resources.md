# External resources

Dataset, backbone, and pose-estimation resources are listed below.

| Resource | Source | Expected local input |
|---|---|---|
| UP-Fall | https://sites.google.com/up.edu.mx/har-up/ ; https://doi.org/10.3390/s19091988 | Complete 17-subject RGB streams, timestamps and state labels |
| UR Fall | https://fenix.ur.edu.pl/~mkepski/ds/uf.html ; https://doi.org/10.1016/j.cmpb.2014.09.005 | Camera 0 image sequences, sync timestamps, labels |
| UniFormerV2 | https://github.com/OpenGVLab/UniFormerV2 | upstream architecture reference |
| MMAction2 B/16 K710 integration | https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/uniformerv2/uniformerv2-base-p16-res224_clip-kinetics710-pre_8xb32-u8_kinetics400-rgb.py | installed mmaction2 1.2.0 and K710 checkpoint |
| K710 checkpoint | https://download.openmmlab.com/mmaction/v1.0/recognition/uniformerv2/kinetics710/uniformerv2-base-p16-res224_clip-pre_u8_kinetics710-rgb_20221219-77d34f81.pth | `pretrained/uniformerv2_k710_b16_8x224.pth` |
| RTMPose-m COCO | https://github.com/open-mmlab/mmpose/blob/main/configs/body_2d_keypoint/rtmpose/README.md | matching 17-joint model config and COCO pretrained checkpoint |
| MMPose API | https://github.com/open-mmlab/mmpose/blob/main/docs/en/user_guides/inference.md | `init_model`, `inference_topdown` |

MMAction2 and MMPose distribution/code terms apply to those packages. Review pretrained-weight terms separately. UR Fall's source states CC BY-NC-SA 4.0 and noncommercial academic use. Dataset bytes are not bundled. Verify UP-Fall's current access and use terms at its source. Checkpoint digests are not supplied by the PFT-ET manuscript; record a downloaded file's SHA256 locally rather than assuming one.
