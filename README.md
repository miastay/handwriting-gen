# handwriting-gen

Computing Handwriting: Conversion of Typeface to Handwritten Text Using an EMNIST-Trained GAN and YOLO Semantic Object Detection
Ryan Taylor

Split into multiple steps:
Training of a CNN to classify images from the EMNIST database
Must identify class with high accuracy
Training of a CGAN to produce new images in accordance with EMNIST form
Generation of classed bounding boxes through YOLOv3 based in input text image
Superimposition of CGAN-generated characters on given bounding boxes by class

More information can be found in the "Handwriting Synthesis" PDF.
