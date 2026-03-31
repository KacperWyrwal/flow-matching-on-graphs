"""Distribution-level conditional flow matching: datasets, sampling, calibration."""

from otfm.distribution.dataset import (
    MetaFlowMatchingDataset,
    ConditionalMetaFlowMatchingDataset,
    InpaintingDataset,
    TopologyGeneralizationDataset,
    CubeBoundaryDataset,
    PosteriorSamplingDataset,
    CubePosteriorDataset,
)
from otfm.distribution.sample import sample_posterior_film
