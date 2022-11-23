from .code import CodeUtil
from .mol import MolUtil
from .tud import TUUtil, TUUtil_v2

DATASET_UTILS = {
    'ogbg-code': CodeUtil,
    'ogbg-code2': CodeUtil,
    'ogbg-molhiv': MolUtil,
    'ogbg-molpcba': MolUtil,
    'NCI1': TUUtil_v2,
    'NCI109': TUUtil_v2,
    'IMDB-BINARY':TUUtil_v2,
    'IMDB-MULTI':TUUtil_v2
}
