# Speech & Audio tools on Python 3

Included Extension Library: Numpy, Librosa, Audiolazy

Due to the limited efficiency of Audiolazy library, the multiprocessing version isrecommended for files over 20s.

The multithread version is designed for comparisons, the global interpreter lock makes multithread meansless in practise.


1. 50%-Overlap FFT -- FFTStereo.py -- It can process stereo wav files at any with any sample rate.
2. LPC (One Thread) -- SimpleLPC.py -- For Mono
3. LPC (MultiThread) -- ThreadBasedLPC.py -- For Mono
4. LPC (MultiProcessing) -- ProcessBasedLPC.py -- For Mono
5. CmpMCEP2MFCC -- the comparison of MFCCs and MCEPs
