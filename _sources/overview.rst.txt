Overview
========

Click on a nut name for more details.


**Network wrapping**

- :class:`KerasNetwork <nutsml.network.KerasNetwork>` :
  wrapper for Keras networks.

- :class:`LasagneNetwork <nutsml.network.LasagneNetwork>` :
  wrapper for Lasagne networks.


**Data reading**

- :class:`DplyToList <nutsml.reader.DplyToList>` :
  convert DplyDataframe to list.

- :class:`ReadImage <nutsml.reader.ReadImage>` :
  load image as numpy array from file system.

- :class:`ReadLabelDirs <nutsml.reader.ReadLabelDirs>` :
  read file paths from label directories.

- :class:`ReadPandas <nutsml.reader.ReadPandas>` :
  read data as Pandas table from file system.


**Data writing**

- :class:`WriteImage <nutsml.writer.XWriteImageXX>` :
  write images to file system.


**Data viewing**

- :class:`PrintColType <nutsml.viewer.PrintColType>` :
  print type and other information for sample columns.

- :class:`ViewImage <nutsml.viewer.ViewImage>` :
  display image in window.

- :class:`ViewImageAnnotation <nutsml.viewer.ViewImageAnnotation>` :
  display image and annotation in window.


**Sample processing**

- :class:`CheckNaN <nutsml.common.CheckNaN>` :
  raise exception if data contains NaNs.

- :class:`PartitionByCol <nutsml.common.PartitionByCol>` :
  partition samples depending on column value.

- :class:`SplitRandom <nutsml.common.SplitRandom>` :
  randomly split iterable into partitions, e.g. training, validation, test.

- :class:`Stratify <nutsml.stratify.Stratify>` :
  stratifies samples by down-sampling or up-sampling.


**Transforming & Augmenting**

- :class:`AugmentImage <nutsml.transformer.AugmentImage>` :
  augment images using random transformations, e.g. rotation.

- :class:`ImageAnnotationToMask <nutsml.transformer.ImageAnnotationToMask>` :
  return bit mask for geometric image annotation.

- :class:`ImageChannelMean <nutsml.transformer.ImageChannelMean>` :
  compute per-channel means over images and subtract from images.

- :class:`ImageMean <nutsml.transformer.ImageMean>` :
  compute mean over images and subtract from images.

- :class:`ImagePatchesByAnnotation <nutsml.transformer.ImagePatchesByAnnotation>` :
  randomly sample patches from image based on geometric annotation.

- :class:`ImagePatchesByMask <nutsml.transformer.XXImagePatchesByMask>` :
  randomly sample patches from image based on annotation mask.

- :class:`RandomImagePatches <nutsml.transformer.RandomImagePatches>` :
  extract patches at random locations from images.

- :class:`RegularImagePatches <nutsml.transformer.RegularImagePatches>` :
  extract patches in a regular grid from images.

- :class:`TransformImage <nutsml.transformer.TransformImage>` :
  transform images, e.g. crop, translate, rotate.


**Boosting**

- :class:`Boost <nutsml.booster.Boost>` :
  boost samples with high softmax probability for incorrect class.


**Batching**

- :class:`BuildBatch <nutsml.batcher.BuildBatch>` :
  build batches for GPU-based training.


**Plotting**

- :class:`PlotLines <nutsml.plotter.PlotLines>` :
  plot lines for selected data columns, e.g. accuracy, loss.


**Logging**

- :class:`LogCols <nutsml.logger.LogCols>` :
  log sample columns to file.