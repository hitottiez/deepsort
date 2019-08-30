# -*- coding: utf-8 -*-

# 設定関連のユーティリティ関数
# ==============================================================================

import configparser
import os
import traceback

ROOTDIR = os.path.join(os.path.dirname(__file__), '../')
ROOTDIR = os.path.abspath(ROOTDIR)
PROJECT_ROOT = os.path.abspath(os.path.join(ROOTDIR, '..'))

def getroot():
    """
    アプリのルートディレクトリを返す
    :return: このアプリのルートディレクトリのパス
    """
    return ROOTDIR

def get_project_root():
    """
    プロジェクトのルートディレクトリを返す
    """
    return PROJECT_ROOT

def createpath(filepath, root=None):
    """
    ファイルパスを作成する
    ※相対パスで指定した場合はアプリルートからの相対パス
    ※絶対パスの場合は絶対パスとして出力
    :param filepath: ファイルパス
    :return: 作成したファイルパス
    """
    if root is None:
        return os.path.abspath(os.path.join(getroot(), filepath))
    else:
        return os.path.abspath(os.path.join(root, filepath))

def loadini(ini_filepath):
    """
    iniファイルをロードする
    """
    ini = configparser.ConfigParser()
    try:
        with open(ini_filepath) as f:
            ini.read_file(f)
            return ini
    except Exception as e:
        traceback.print_exc()
        raise IOError('設定ファイルのロードに失敗しました')


def boolcast(value):
    """
    bool型にキャストする
    """
    if isinstance(value, str) and value.lower() == 'false':
        return False
    else:
        return bool(value)
