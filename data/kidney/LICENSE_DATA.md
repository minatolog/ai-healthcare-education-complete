# KiTS19 Data Attribution and License

The CT images and reference segmentation masks in this folder are derived from **KiTS19**, created by Nicholas Heller and collaborators at the University of Minnesota. The challenge acknowledges support from Climb 4 Kidney Cancer (C4KC).

The dataset is provided under **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)**. Retain attribution, identify adaptations, use the data for noncommercial purposes and preserve the applicable ShareAlike terms when distributing adapted data. The license and its legal text are available at:

- <https://creativecommons.org/licenses/by-nc-sa/4.0/>
- <https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en>

Source and references:

- Official dataset description: <https://kits19.grand-challenge.org/data/>
- Official source repository: <https://github.com/neheller/kits19>
- Heller et al. (2019), *The KiTS19 Challenge Data: 300 Kidney Tumor Cases with Clinical Context, CT Semantic Segmentations, and Surgical Outcomes*: <https://arxiv.org/abs/1904.00445>
- Heller et al. (2021), *The state of the art in kidney and kidney tumor segmentation in contrast-enhanced CT imaging: Results of the KiTS19 challenge*, Medical Image Analysis 67, 101821: <https://doi.org/10.1016/j.media.2020.101821>

Adaptations in this educational project: the existing local exporter converted axial CT and masks to 256x256 PNGs, clipped/scaled CT intensities, and selected foreground slices plus sampled background. This handoff selects the already locked 40-patient subset and copies its PNGs without changing their bytes or labels. Original case/slice IDs, metadata, cohort lock and audit are retained. The additional role tables and checksum catalogue are project handoff metadata.

The upstream repository's MIT software license explicitly excludes its data directory. Do not describe these CT images or reference masks as MIT-licensed synthetic data. This notice concerns the dataset; it does not relicense the team's model code.
