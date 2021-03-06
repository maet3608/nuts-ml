Overview
========

Click on a nut name for more details.

**nuts-ml** is based on `nuts-flow <https://github.com/maet3608/nuts-flow>`_,
which provides additional nuts, see 
`nuts-flow overview <https://maet3608.github.io/nuts-flow/overview.html>`_.

All ``nutsflow`` functions can be imported from ``nutsml`` as well,
e.g. ``from nutsflow import Collect`` or ``from nutsml import Collect``
both work.


**Network wrapping**

- :class:`KerasNetwork <nutsml.network.KerasNetwork>` :
  wrapper for Keras networks.
  
- :class:`PytorchNetwork <nutsml.network.PytorchNetwork>` :
  wrapper for Pytorch networks.  

- :class:`LasagneNetwork <nutsml.network.LasagneNetwork>` :
  wrapper for Lasagne networks.


**Data reading**

- :class:`ReadLabelDirs <nutsml.reader.ReadLabelDirs>` :
  read file paths from label directories.

- :class:`ReadPandas <nutsml.reader.ReadPandas>` :
  read data via Pandas from file system.

- :class:`ReadNumpy <nutsml.reader.ReadNumpy>` :
  load numpy array from file system.
  
- :class:`ReadImage <nutsml.reader.ReadImage>` :
  load image as numpy array from file system.
  

**Data writing**

- :class:`WriteImage <nutsml.writer.WriteImage>` :
  write images to file system.


**Data viewing**

- :class:`ViewImage <nutsml.viewer.ViewImage>` :
  display image in window.

- :class:`ViewImageAnnotation <nutsml.viewer.ViewImageAnnotation>` :
  display image and annotation in window.


**Data printing** (from nuts-flow)

- :class:`Print <nutsflow.function.Print>` :
  print data to console. 
  
- :class:`PrintType <nutsflow.function.PrintType>` :
  print data type

- :class:`PrintColType <nutsflow.function.PrintColType>` :
  print column data, eg. tuples    
 
- :class:`PrintProgress <nutsflow.processor.PrintProgress>` :
  print progress on iterable.


**Sample processing**

- :class:`ConvertLabel <nutsml.common.ConvertLabel>` :
  convert between string labels and integer class ids.

- :class:`CheckNaN <nutsml.common.CheckNaN>` :
  raise exception if data contains NaNs.

- :class:`PartitionByCol <nutsml.common.PartitionByCol>` :
  partition samples depending on column value.

- :class:`SplitRandom <nutsml.common.SplitRandom>` :
  randomly split iterable into partitions, e.g. training, validation, test.
  
- :class:`SplitLeaveOneOut <nutsml.common.SplitLeaveOneOut>` :
  split iterable into leave-one-out train and test sets.  

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
  
- :class:`Mixup <nutsml.batcher.Mixup>` :
  mixup augmentation, see 
  `mixup: Beyond Empirical Risk Minimization <https://arxiv.org/abs/1710.09412>`_ 


**Boosting**

- :class:`Boost <nutsml.booster.Boost>` :
  boost samples with high confidence for incorrect class.


**Batching**

- :class:`BuildBatch <nutsml.batcher.BuildBatch>` :
  build batches for GPU-based training.


**Plotting**

- :class:`PlotLines <nutsml.plotter.PlotLines>` :
  plot lines for selected data columns, e.g. accuracy, loss.


**Logging**

- :class:`LogToFile <nutsml.logger.LogToFile>` :
  log sample columns to file.