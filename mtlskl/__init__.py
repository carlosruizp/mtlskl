# from .sample.text import joke
from .data.DataLoader import *

from .model.svm.ctl.CTLSVM import CTLSVC, CTLSVR
from .model.svm.itl.ITLSVM import ITLSVC, ITLSVR
from .model.svm.convexmtl.ConvexMTLSVM import ConvexMTLSVC, ConvexMTLSVR
from .model.svm.adapglmtl.AdapGLMTLSVM import AdapGLMTLSVC, AdapGLMTLSVR