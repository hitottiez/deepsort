# -*- coding: utf-8 -*-

# 設定管理クラス
# ==============================================================================

import configparser
import logging

from .conf_util import createpath, getroot, get_project_root, loadini
from .Track import Track
import os
import sys


def getConfigure():
    """
    設定ファイルをロードする
    ※MULTI_ACTRECOG_CONFPATHが指定されている場合はそちらのパスを使用する
    ※MULTI_ACTRECOG_CONFPATHが指定されていない場合は/path/to/project_root/deepsort/config/multi_actrecog.iniを使用する
    """
    ini_filepath = os.getenv('MULTI_ACTRECOG_CONFPATH', createpath(
        os.path.join(os.path.dirname(__file__), './multi_actrecog.ini'), root=get_project_root()))
    try:
        return Config(ini_filepath)
    except IOError as e:
        print('設定ファイルがロードできませんでした[file={}]'.format(
            ini_filepath), file=sys.stderr)
        raise e


class Config:
    """
    iniファイルを管理するクラス
    """

    def __init__(self, ini_filepath):
        self.ini = loadini(ini_filepath)
        self.track = Track(self.ini)

    @property
    def getroot(self):
        """
        アプリケーションのルートディレクトリを返す
        """
        return getroot()

    def log_setup(self):
        """
        ログ出力セットアップ
        """
        level = self.ini.get('logging', 'level')
        format = self.ini.get('logging', 'format')
        logging.basicConfig(level=level, format=format)
