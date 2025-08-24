from yacs.config import CfgNode as CN
import platform

_C = CN()
_C.PLATFORM = CN()
_C.DATASET = CN()

_C.DATASET.DATATYPE = 'Brain_OASIS'

_C.PLATFORM.basefile = ''
if "Win" in platform.system():
    _C.PLATFORM.isWin = True
    _C.PLATFORM.server = 'Computer'


else:
    _C.PLATFORM.isWin = False
    _C.PLATFORM.server = 'Server'


_C.DATASET.NUM_EPOCHS = 300 #LPBA40
_C.DATASET.Iter = 30001 #OASIS

_C.DATALOADER = CN()
_C.DATALOADER.BATCH_SIZE = 1
_C.DATALOADER.NUM_WORKERS = 6
_C.DATALOADER.NORM = False

_C.MODEL = CN()
_C.MODEL.DROPOUT = 0.2
_C.MODEL.GPU_NUM = 0
_C.MODEL.PARALLEL_GPUS = None


_C.MODEL.FDoubleConv= True
_C.MODEL.BDoubleConv= True
_C.MODEL.BestModelCheckRatio= 0.98

_C.MODEL.Start_Channel=6
_C.MODEL.Ch_magnitude=2

if _C.DATASET.DATATYPE == 'Brain_OASIS':
    _C.DATASET.DATA_PATH = 'YOUR PATH'
    _C.DATASET.DATA_PATH_IMGS = _C.DATASET.DATA_PATH + 'IMAGE PATH'
    _C.DATASET.DATA_PATH_LABELS = _C.DATASET.DATA_PATH + 'IMAGE LABEL PATH'
    _C.DATASET.PAIR = False


elif _C.DATASET.DATATYPE == 'LPBA40_small':
    _C.DATASET.DATA_PATH = 'YOUR PATH'
    _C.DATASET.DATA_PATH_IMGS = _C.DATASET.DATA_PATH + 'IMG PATH'
    _C.DATASET.DATA_PATH_LABELS = _C.DATASET.DATA_PATH + 'IMG LABEL PATH'
    _C.DATASET.PAIR = False

_C.SOLVER = CN()
_C.SOLVER.LEARNING_RATE = 1e-5



_C.SOLVER.CHECKPOINT = 5000
_C.SOLVER.LOAD_MODEL = False
_C.SOLVER.RANGE_FLOW = 0.1

_C.SOLVER.hyp_loss_jacobin = 0
_C.SOLVER.hyp_loss_smooth = 3.5
_C.SOLVER.hyp_loss_CL = 10
_C.SOLVER.hyp_loss_kl = 1e-3
_C.SOLVER.ANTIFOLD=0


_C.SOLVER.FREEZE_STEP =2000


