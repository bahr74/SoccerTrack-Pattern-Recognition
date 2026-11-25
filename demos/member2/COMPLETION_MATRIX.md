# Member 2 requirement coverage

| Requirement | Status | Evidence / boundary |
| --- | --- | --- |
| M1 tiling, coordinates, global NMS | COMPLETE | Synthetic full pixel coverage, alternate geometry, clipping, border duplicates, invalid/empty boxes |
| M2 native-resolution real person inference | IMPLEMENTED — LOCAL VALIDATION ONLY | Verified local weights; frame 0 has 69 detections; GPU acceptance passed; independent overlay inspected; shared Member 4 review pending |
| M3 full eval10 and win60s detection | IMPLEMENTED — LOCAL VALIDATION ONLY | Exact 598/1500 coverage, low score cache, runtime and configuration; official validators/writers/E3 blocked |
| M4 ByteTrack adapter and continuous YOLO sequence | IMPLEMENTED — LOCAL VALIDATION ONLY | Installed API, synthetic motion/misses/reset/crossing/index tests; all 1500 frames; official oracle/noisy/E4 blocked |
| M5 bounded detection/threshold/tracker experiments | PARTIAL | Four paired candidates and three tracker settings; selected detector config; tracker defaults retained pending official evaluation; pitch-on comparison blocked |
| M6 held-out testing | NOT RUN | Prohibited; no held-out clip/annotation access |
| Course data preparation and split | COMPLETE within Member 2 scope | Sequential 0-based validation; native geometry; existing match-level split protected; no new training |
| Course imbalance handling | NOT APPLICABLE to new training | Pretrained inference only; person/player mismatch and threshold trade-offs disclosed |
| Course baseline model training | NOT APPLICABLE to Member 2 Phase 2 | Zero-shot pretrained reference; from-scratch training belongs to Member 1 and was not changed |
| Course architecture/training mechanism explanation | COMPLETE | Loaded model 3,157,200 parameters, Conv/C2f/SPPF, strides 8/16/32; installed BCE/CIoU/DFL/TaskAlignedAssigner inspected; no invented losses |
| Course evaluation/reference comparison | PARTIAL | Local PR/F1 and measured runtime; historical reference differences disclosed; official AP/tracking and certified A/B blocked |
| Course specificity/ROC/confusion matrix | NOT APPLICABLE to detector box background TN | TP/FP/FN reported; patch-classification E1/E2 belong elsewhere; no fabricated TN/specificity |
| Documented notebook and reproducibility | COMPLETE locally | 16 scientific sections, 17 executed code cells, cached artifact verification, HTML and execution evidence |
| Real visual evidence | COMPLETE locally | Original/GT/tiles/local/global/pre/post NMS, paired experiments, score curves, TP/FP/FN, trajectories/lengths, decoded 150-frame video |
| Final shared team notebook/report/slides | BLOCKED | Member 4 integration; isolated notebook/report section supplied |
| Official AP/mAP/HOTA/IDF1/MOTA/ID switches | BLOCKED | Member 3 shared production placeholders on current branch; remote branch deliberately not reviewed/merged |
| Fine-tuning, workload and fatigue | PHASE 3 ONLY / outside scope | No execution or physiological claims |
