XTrans Develop 

This project is meant to offer a high-quality alternative to FUJI XTrans raw (.RAF) processing available in common raw conversion software suites using a pre-trained deep CNN machine-learning model (Gharbi et al). The output is a high-quality tiff image with exif metadata cloned from the original raw image. For instance, this means the as-photographed orientation will be stored. The output tiff, while highly compatible, may not render in your default OS viewer.

For optimal results, you will want to post-process the output to apply sharpening and adjust the color curve, etc. While the exif metadata is cloned, it is regrettable that you may not find FUJI lens correction support in your image postprocessing app. Overall, I've been able to get fantastic results with this tool and post-processing with RawTherapee so commercial software is completely optional.

Beware that this tool is best used with GPU acceleration. 

Running a 16MP Xtrans RAF through CPU decoding takes me ~3mins, while the GPU acceleration is only 2-3 seconds. In this case compressing and storing the output tiff takes substantially longer (~6-8 secs). You are on your own with finding and installing appropriate CUDA drivers for your GPU. The included requirements.txt will install the CPU-version of PyTorch if your Python environment needs a PyTorch installation. The chances are that if you've gotten this far, you can likely source the GPU-enabled PyTorch on your own as well, but here's a link to get started: <https://pytorch.org/get-started/locally/> .


Example #1 (result is saved as test.tiff):

     python3 ./xtrans-develop.py --input DSCF4199.RAF --output test.tiff

Example #1a (using gpu; result is saved as test.tiff):

     python3 ./xtrans-develop.py --input DSCF4199.RAF --output test.tiff --gpu

Example #2 (finds all .RAF in /Data and each output is made with .tiff extension):

     python3 ./xtrans-develop.py --input /Data

Example #3 (finds all .RAF in /Data and each output is made with .tiff extension in the /Processed folder):

     python3 ./xtrans-develop.py --input /Data --output /Processed


----
Sources:

The pretrained model and some of the source code is sourced from the work of Gharbi et al. and Ehret and Facciolo. Additional raw image processing ideas and code are sourced from H. Bauke's blog.

Pytorch implementation (using pretrained weights) of:

     Deep Joint Demosaicking and Denoising, SiGGRAPH Asia 2016.
     Michaël Gharbi, Gaurav Chaurasia, Sylvain Paris, Frédo Durand

This code (based on <https://github.com/mgharbi/demosaicnet_torch> and <https://github.com/mgharbi/demosaicnet_caffe>) was created to [reproduce the results of demosaicnet for this publication](https://doi.org/10.5201/ipol.2019.274)

    T. Ehret, and G. Facciolo, "A Study of Two CNN Demosaicking Algorithms", 
    Image Processing On Line, 9 (2019), pp. 220–230. https://doi.org/10.5201/ipol.2019.274

Raw image processing pipeline Python code:

    H. Bauke, https://www.numbercrunch.de/blog/2020/12/from-numbers-to-images-raw-image-processing-with-python/
    