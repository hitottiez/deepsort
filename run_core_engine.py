# コアエンジンメインプログラム
# ==============================================================================

from app.writer.FileWriter import FileWriter
from app.main_loop import MainLoop
from app.action_master import ActionMaster

import sys
import signal
import argparse
import traceback
import logging
logger = logging.getLogger(__name__)



def main(args):
    writer = FileWriter(**vars(args))
    master = ActionMaster(args.dataset_type)

    # シグナルハンドリング
    def handle_signal(signum, frame):
        logger.info('Process killed (signum={:d})'.format(signum))
        sys.exit(0)
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        main_loop = MainLoop(writer, master, args.input)
        main_loop(args.tsn_modality)
    except Exception:
        traceback.print_exc()