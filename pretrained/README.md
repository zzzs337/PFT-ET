# Pretrained resources

Place the MMAction2 Kinetics-710 UniFormerV2-B/16 **8-frame** checkpoint at:

`pretrained/uniformerv2_k710_b16_8x224.pth`

Use the source in `docs/resources.md`. This is K710 pretraining, not a K400-finetuned classification checkpoint. The adapter removes `module.` and `backbone.` prefixes and strictly loads every backbone key. Classification-head keys from a full recognizer checkpoint are discarded. Shape/key mismatch raises an error. Record the SHA256 of the retrieved bytes in the run record. No digest is invented here.

RTMPose-m must use COCO's 17-keypoint output. Pass its local `.py` config and matching checkpoint to `scripts/extract_pose.py`; different keypoint taxonomies are rejected. AIC+COCO or body7 checkpoints are not silently presented as the paper's unspecified COCO-only checkpoint.

Trained PFT-ET checkpoints are produced by `scripts/train.py` and are not included in this package.

Run `python scripts/verify_backbone.py --help` after downloading the checkpoint. The verification output records the local file digest and strict-load result for that exact file.
