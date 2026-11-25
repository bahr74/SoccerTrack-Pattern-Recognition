# SoccerTrack Phase 2 Scientific Report

Integrated validation and baseline comparison

SBES426 Medical Pattern Recognition | Team Members1–4 |25November2026

## Abstract

We evaluated four detectors on598identical validation frames and controlled/real trackers on1500contiguous frames. Existing SVM/CNN training and verified YOLO cache were reused. Selected scratch method: scratch_cnn. Results are validation-only; custom HOG reference parity remains unverified.

## Project objective

We detect and track players in panoramic soccer video. Phase 2 compares rule-based vision, a trained HOG/SVM, a trained small CNN and pretrained tiled YOLOv 8 n, using one shared validation protocol. We estimate image-space player boxes and temporary identities, not physiological fatigue.

## Reference and approved scope

We use the paper’s curated bounding-box subset for image-space MOT. GSR/BAS benchmarks differ in task and protocol; no direct leaderboard claim is made. Primary reference: https://arxiv.org/html/2508.01802v2.

## Dataset organization

A clip and ten-column MOT file are joined by match ID. Native images are 4096 x 1080 at 25 FPS. Internal frames are zero-based and boxes use x,y,width,height. Rows describe the 22 on-pitch players. Ball actions, jersey recognition and pitch-space reconstruction are outside the executed scope.

## Data preparation and cleaning

MOT parsing preserves integer identities, rejects malformed/nonfinite values, removes duplicates/degenerate geometry, clips valid intersections and reports issue counts. Sequential decoding supplies requested frames and verifies coverage. Source annotations and pretrained weights remain read-only.

## Exploratory data analysis

The EDA tables summarize only the five approved training matches and validation 117093. Small distant players motivate native-resolution processing, perspective-aware proposals and tiled inference. Linear empirical quantiles use rank(n-1)q. EDA frame counts below distinguish header/dense-annotation agreement from the newly decoded evaluation union.

![eda/boxes_per_match.png](../figures/eda/boxes_per_match.png)

![eda/box_size_hist.png](../figures/eda/box_size_hist.png)

## Match split and leakage prevention

Training matches are 118575,118576,118577,118578,128058; validation is 117093. Protected 128057/132831 and disabled 117092 were not opened. Split checks precede data resolution. Selection on validation is disclosed; no independent test performance is claimed.

## Patch sampling and imbalance

Historical training contains 3432 patches:858 positive and 2574 negative, sampled at 39 match/frame pairs; validation contains 704 patches:176/528 over 8 frames. Crops are RGB64 x 32 with context. Proposal negatives repeat: unique pixels are 2782/580. We preserve the fitted data rather than silently deduplicating history.

![patch_examples.png](../figures/patch_examples.png)

![imbalance.png](../figures/imbalance.png)

## Background and pitch segmentation

The scratch pipeline compares each frame to a label-free median background, masks the pitch in HSV/Lab processing and cleans foreground differences. Background samples span the clip, including future frames: this is an offline, noncausal method. Cached background reuse avoids unnecessary decoding.

![scratch_stages.png](../figures/scratch_stages.png)

## Morphology and candidate proposals

Thresholded differences, median filtering, opening/closing, vertical dilation and connected components form regions. A size model fitted to training data accounts for player height versus image row; filtering and wide-component splitting reduce background/merged blobs. Selected candidate settings are compared with the starting settings on identical eval 10 frames.

![proposal_comparison.png](../figures/proposal_comparison.png)

## Rule based baseline

Shape, fill and expected perspective size produce a score without a learned classifier. This baseline tests how far domain-specific image processing can go. It is a detection baseline and must not be mislabeled as the majority patch classifier.

## HOG feature representation

Unsigned orientation votes and spatial cells form 756 custom HOG features per 64 x 32 crop. Block normalization reduces contrast dependence. The feature implementation and trained SVM are unchanged. Installed OpenCV lacks HOGDescriptor, so reference numerical parity remains unverified despite internal reproducibility.

![hog_features.png](../figures/hog_features.png)

## SVM training and calibration

Balanced LinearSVC uses the squared-hinge objective. Training-only stratified folds calibrate scores with a sigmoid; validation patch AP selects C from 0.01/0.1/1/10, with historical C0.01 retained. A calibrated score ranks proposals; good sampled-patch performance alone does not establish detector recall.

## CNN architecture

Three convolution blocks (16/32/64 channels), pooling, global aggregation, dropout and one binary output contain 23873 trainable parameters. RGB normalization is estimated from training patches only. Learned visual features replace fixed HOG features while using the same proposal source.

![cnn_architecture.png](../figures/cnn_architecture.png)

## CNN optimization and historical training

The existing model was trained for 8 epochs with Adam (learning rate 0.001, weight decay 0.0001) and weighted BCEWithLogits to address 1:3 imbalance. Training augmentation includes horizontal flips and brightness/contrast changes. Validation AP selected epoch 7. The displayed curves are historical training evidence; no model was retrained during integration.

![cnn_training.png](../figures/cnn_training.png)

## Pretrained YOLO comparison

YOLOv 8 n is a pretrained COCO-person reference, not a locally trained Phase 2 network. Its multiscale backbone/head predicts boxes and scores. The source model uses classification and box/distribution losses during pretraining; this project performs inference only and does not invent local YOLO training curves.

## Native tiling and global NMS

1280 x 1080 tiles with 128 px overlap and inference size 1280 retain tiny-player detail. Tile-local boxes are translated/clipped to native coordinates and global NMS at IoU0.5 removes duplicates. Pitch filtering remains off in the accepted selected configuration. Raw cache identity/coverage was verified before shared serialization.

![yolo_stages/tiles.jpg](../figures/yolo_stages/tiles.jpg)

![yolo_stages/tile_local.jpg](../figures/yolo_stages/tile_local.jpg)

![yolo_stages/post_nms.jpg](../figures/yolo_stages/post_nms.jpg)

## IoU Hungarian tracking

IoU measures intersection area divided by union. A gated Hungarian assignment links current boxes to tracks; confirmation and ageing handle transient misses. Every contiguous frame receives an update, including no detections. Position-only association can fragment or swap identities under occlusion.

## ByteTrack association

ByteTrack uses high-confidence detections and lower-score candidates to retain tracks through uncertain observations. Detections down to the 0.05 source floor remain serialized. Reported 0.50 detector threshold is separate from association thresholds. Track IDs are temporary labels; unique-ID counts do not equal identity switches.

## Shared evaluation protocol

E1 classifies sampled patches; E2 classifies every scored proposal and reports many-to-one GT coverage; E3 matches native-frame detections one-to-one in confidence order; E4 evaluates continuous tracks with TrackEval. Tables are read back after 2-decimal boxes/4-decimal scores, so all final metrics use the same representation.

## E1 patch classification

We compare majority negative, HOG/SVM and CNN on the same 704 validation patches. Accuracy, precision, recall, specificity, F1, ROC-AUC and classification AP answer different questions. Majority prediction has 75%accuracy but zero positive recall, illustrating why accuracy alone is misleading.

| method | accuracy | sensitivity | specificity | f1 | roc_auc | pr_auc |
| --- | --- | --- | --- | --- | --- | --- |
| Majority negative (auxiliary) | 0.7500 | 0.0000 | 1.0000 | 0.0000 | 0.5000 | 0.2500 |
| scratch_hog_svm | 0.9418 | 0.8750 | 0.9640 | 0.8825 | 0.9779 | 0.9155 |
| scratch_cnn | 0.8764 | 0.9886 | 0.8390 | 0.8000 | 0.9923 | 0.9788 |

![eval/roc_classifiers_val.png](../figures/eval/roc_classifiers_val.png)

## E2 proposal evaluation

All proposals, including those removed by downstream NMS, are retained for 598 eval 10 frames. Coverage at IoU0.5/0.3 measures whether any candidate can localize each GT box. Classification on natural proposal imbalance is separate. Proposal recall bounds downstream recall under that candidate/localization protocol.

| method | proposals | gt | sensitivity | f1 |
| --- | --- | --- | --- | --- |
| scratch_hog_svm | 17360 | 13156 | 0.8726 | 0.8634 |
| scratch_cnn | 17360 | 13156 | 0.9921 | 0.8527 |

[{"method": "scratch_hog_svm", "0.5": 0.7083, "0.3": 0.8867}, {"method": "scratch_cnn", "0.5": 0.7083, "0.3": 0.8867}]

## E3 common frame detection

Rules, HOG/SVM, CNN and YOLO use the same 598 frame IDs and GT. Precision/recall/F1 use score 0.50 and IoU0.5, with IoU0.3 diagnostics. AP50 and AP50:95 use all detections retained at floor 0.05 and confidence ordering. Our AP is the approved all-point implementation, not an unqualified claim of identical COCO tooling.

| method | tp | fp | fn | precision | recall | f1 | ap50 | ap50_95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scratch_mwb0 | 5881 | 1904 | 7275 | 0.7554 | 0.4470 | 0.5617 | 0.5104 | 0.2344 |
| scratch_hog_svm | 7977 | 1360 | 5179 | 0.8543 | 0.6063 | 0.7093 | 0.6248 | 0.3020 |
| scratch_cnn | 9070 | 3061 | 4086 | 0.7477 | 0.6894 | 0.7174 | 0.6327 | 0.2888 |
| yolov8n_coco | 11127 | 2983 | 2029 | 0.7886 | 0.8458 | 0.8162 | 0.7739 | 0.5552 |

![detection_comparison.png](../figures/detection_comparison.png)

![eval/pr_detection_val.png](../figures/eval/pr_detection_val.png)

## E4 controlled and real tracking

Oracle and seeded noisy GT isolate association behavior for both trackers. Noise drops 10%of boxes and perturbs coordinates by Gaussian 2 px with seed 42; repaired boxes stay valid. Real selected scratch+IoU and YOLO+ByteTrack use 1500 contiguous frames. TrackEval supplies HOTA/DetA/AssA/LocA/IDF1/MOTA/IDSW. Controlled ByteTrack rows use starting settings; the real YOLO row selects conservative settings (high/new 0.4,low 0.1,buffer 30,match 0.8) from the three preplanned candidates by HOTA, then IDF1/MOTA. Explicit tracker overrides are stored separately from the base shared configuration; no detector reinference or new grid was introduced.

| method | hota | deta | assa | idf1 | mota | idsw |
| --- | --- | --- | --- | --- | --- | --- |
| gt_oracle+iou_hungarian | 0.9680 | 0.9950 | 0.9417 | 0.9648 | 0.9984 | 4 |
| gt_oracle+bytetrack | 0.9699 | 0.9841 | 0.9559 | 0.9772 | 0.9998 | 7 |
| gt_noisy+iou_hungarian | 0.2525 | 0.6325 | 0.1020 | 0.2254 | 0.7732 | 1722 |
| gt_noisy+bytetrack | 0.4744 | 0.6195 | 0.3653 | 0.5986 | 0.8038 | 669 |
| scratch_cnn+iou_hungarian | 0.2551 | 0.4375 | 0.1501 | 0.2680 | 0.4058 | 465 |
| yolov8n_coco+bytetrack | 0.4321 | 0.4797 | 0.3903 | 0.4327 | 0.3255 | 172 |

![tracking_comparison.png](../figures/tracking_comparison.png)

## Model comparison and selection

The selected scratch detector is scratch_cnn, using highest E3 F1 at the fixed 0.50 reporting threshold on validation. This is validation selection, not an independent generalization result. We retain the existing models and selected proposal configuration instead of introducing a new optimization search.

## Error analysis

Synchronized native-frame views expose distant players, background components, merged players and localization errors. Error-example CSVs use the shared one-to-one matcher. Red boxes mark missed GT; crops magnify actual validation pixels. TrackEval summary errors are global sequence measures; a single still image does not establish an identity switch.

![synchronized_frame_0.png](../figures/synchronized_frame_0.png)

![representative_detection_error.png](../figures/representative_detection_error.png)

![missed_player_crop.png](../figures/missed_player_crop.png)

![tracking_timeline.png](../figures/tracking_timeline.png)

![tracking_transition.png](../figures/tracking_transition.png)

## Runtime and resources

Scratch methods share decoding and proposals. Per-method processing seconds include measured shared preparation plus scoring/NMS; published frameset FPS allocates union processing proportionally and excludes decode/background/setup/write. Union wall time is measured separately. Historical YOLO warm processing 8.1282 FPS excludes setup/decode/write. These are different scopes; no realtime or fair end-to-end speed ranking is claimed.

## Reference comparison

Paper Table 3 reports GS-HOTA12.25 over 20 halves and 10.96 on test halves. Our 60 second image-space HOTA differs in coordinates, attributes and sequence length. Direct numerical superiority is not claimed.

## Scientific interpretation

YOLO achieves E3 F1 0.8162, versus CNN 0.7174 and HOG/SVM 0.7093 on identical frames. The scratch selection is scratch_cnn; compare precision and recall as well as this narrow validation F1 difference. Proposal coverage changes from 0.5412 to 0.7083 at IoU0.5. Selected YOLO/ByteTrack gives HOTA 0.4321, IDF1 0.4327 and MOTA 0.3255; selected scratch/IoU gives 0.2551, 0.268 and 0.4058, respectively. Different metrics measure different errors; no universal winner or independent generalization claim follows.

## Limitations

One validation match, limited sampled training coverage, repeated negatives, validation-based selection, offline future-frame background, unresolved OpenCV HOG reference verification, small/occluded players, temporary tracking identities and differing runtime scopes limit conclusions. No physiological fatigue labels or held-out results support clinical or generalization claims.

## Phase 3 and presentation

Future work may improve proposals/appearance association, test explainability, perform authorized held-out assessment and investigate calibrated pitch-space workload proxies. None is implemented as Phase 3 work here. The final PowerPoint is intentionally pending, so complete academic submission readiness is not claimed.

## Reproduction and artifact identity

Normal notebook execution only loads accepted saved artifacts. Reproduction uses the production config/registry, existing hash-checked models, verified YOLO migration and checkpointed missing scratch inference. EXECUTION_JOURNAL and stage.json record progress. All historical experiments remain immutable and the user manages Git.

## Conclusion

The integrated project provides reproducible validation comparisons across classical vision, trained handcrafted/learned patch classifiers, pretrained detection and two trackers. Conclusions below are derived from accepted E1–E4 artifacts. Manual review should inspect the application, synchronized video, notebook and report before presentation.

## References

Scott et al. SoccerTrack v2, arXiv2508.01802v2, https://arxiv.org/html/2508.01802v2 ; dataset/code https://github.com/AtomScott/SoccerTrack-v2 ; Luiten et al. HOTA, IJCV2021 ; Zhang et al. ByteTrack, ECCV2022 ; Dalal and Triggs HOG, CVPR2005 ; Ultralytics installed YOLOv8 implementation. Paper abstract/HTML and official repository documentation were read online; no dataset/weight/package download.
