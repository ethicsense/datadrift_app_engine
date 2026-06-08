import os
from pathlib import Path

class Config:
    """Flask 웹앱 설정 클래스"""

    def __init__(self):
        # 이 패키지(backend) 디렉터리를 앱 루트로 사용
        self.app_root = Path(__file__).parent.absolute()

        # 주요 디렉토리 경로들
        self.db_dir = self.app_root / "db"
        self.models_dir = self.app_root / "models"
        self.datasets_uploads = self.app_root / "datasets" / "uploads"
        self.datasets_exported = self.app_root / "datasets" / "exported_datasets"
        self.logs_dir = self.app_root / "logs"
        self.static_cam_results = self.app_root / "static" / "cam_results"
        self.static_perturbation_results = self.app_root / "static" / "perturbation_results"

        # 데이터베이스 설정
        self.milvus_db_name = "DAE_data.db"
        self.milvus_db_path = self.db_dir / self.milvus_db_name

        # Flask 앱 설정 (기본값)
        self.flask_port = 5555
        self.fiftyone_port = 8159
        self.tensorboard_port = 6006

        # FiftyOne Manager 설정
        self._fom_runner = None
        self._fiftyone_thread = None

        # 기본값들
        self.default_project = "runs"
        self.default_run = "exp"
        self.default_epochs = 100
        self.default_batch_size = 16
        self.default_img_size = 640
        self.default_learning_rate = 0.001

    def ensure_directories(self):
        """필수 디렉토리들이 존재하는지 확인하고 없으면 생성"""
        directories = [
            self.db_dir,
            self.models_dir,
            self.datasets_uploads,
            self.datasets_exported,
            self.logs_dir,
            self.static_cam_results,
            self.static_perturbation_results,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"✅ Directory ensured: {directory}")

    def get_milvus_connection_string(self):
        """Milvus 연결 문자열 반환"""
        return str(self.milvus_db_path)

    def get_project_paths(self):
        """프로젝트 경로들을 딕셔너리로 반환"""
        return {
            "app_root": str(self.app_root),
            "models_dir": str(self.models_dir),
            "datasets_uploads": str(self.datasets_uploads),
            "datasets_exported": str(self.datasets_exported),
            "logs_dir": str(self.logs_dir),
            "static_cam_results": str(self.static_cam_results),
            "static_perturbation_results": str(self.static_perturbation_results),
        }

    def get_fiftyone_manager(self):
        """FiftyOne Manager 인스턴스를 반환 (싱글톤 패턴)"""
        if self._fom_runner is None:
            from utils import FiftyoneManager

            self._fom_runner = FiftyoneManager(port=self.fiftyone_port)
            self._fiftyone_thread = self._fom_runner.start()
        return self._fom_runner, self._fiftyone_thread

    def set_fiftyone_port(self, port):
        """FiftyOne 포트를 동적으로 설정 (이미 초기화된 경우 재시작)"""
        if self.fiftyone_port != port:
            self.fiftyone_port = port
            # 이미 초기화된 경우 재시작
            if self._fom_runner is not None:
                print(f"🔄 Restarting FiftyOne Manager with new port: {port}")
                self._fom_runner = None
                self._fiftyone_thread = None


# 전역 설정 인스턴스
config = Config()
