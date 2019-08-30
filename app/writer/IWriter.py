# 結果を書き出すクラスのIFクラス
# ※評価ツール対応に伴い修正
# ----------------------------------------------------------------

class IWriter:

    def writeContextData(self, hist_id, frame_no, timestamp, context_data):
        """
        コンテキストデータを書き出し
        """
        pass

    def writeEA03Data(self, ea03_data):
        """
        EA03データを書き出し
        """
        pass

    def writeBenchmark(self, hist_id, frame_no, timestamp, benchmark_data):
        """
        ベンチマークデータを書き出し
        """
        pass

    def writeFrame(self, frame, contextData):
        """
        コンテキストデータ出力
        """
        pass

    def release(self):
        """
        開放するリソースがあればここに処理を書く
        """
        pass