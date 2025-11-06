"""
配置管理模块
用于保存和加载FTIR处理参数配置
"""
import json
from datetime import datetime
from typing import Dict, Any, Optional


class ConfigManager:
    """配置管理器 - 用于保存和加载处理参数"""

    def __init__(self):
        self.config_version = "1.0"

    def save_config(self, config_data: Dict[str, Any], file_path: str) -> bool:
        """
        保存配置到JSON文件

        Args:
            config_data: 配置数据字典
            file_path: 保存路径

        Returns:
            bool: 是否成功保存
        """
        try:
            # 添加元数据
            full_config = {
                "version": self.config_version,
                "created_at": datetime.now().isoformat(),
                "config": config_data
            }

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(full_config, f, indent=4, ensure_ascii=False)

            return True
        except Exception as e:
            raise Exception(f"保存配置失败: {str(e)}")

    def load_config(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        从JSON文件加载配置

        Args:
            file_path: 配置文件路径

        Returns:
            配置数据字典，如果失败则返回None
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                full_config = json.load(f)

            # 检查版本兼容性
            if "version" in full_config:
                return full_config["config"]
            else:
                # 兼容旧版本配置文件
                return full_config

        except Exception as e:
            raise Exception(f"加载配置失败: {str(e)}")

    @staticmethod
    def create_config_from_gui(gui_instance) -> Dict[str, Any]:
        """
        从GUI实例创建配置字典

        Args:
            gui_instance: SpectralProcessorGUI实例

        Returns:
            配置字典
        """
        config = {
            "y_label": gui_instance.y_label_var.get(),
            "smoothing": {
                "method": gui_instance.smooth_method.get(),
                "params": {}
            },
            "baseline": {
                "method": gui_instance.baseline_method.get(),
                "params": {},
                "data_source": gui_instance.data_source_var.get()
            },
            "peak_analysis": {
                "threshold": gui_instance.peak_threshold_var.get(),
                "distance": gui_instance.peak_distance_var.get(),
                "data_source": gui_instance.peak_data_var.get()
            }
        }

        # 获取平滑参数
        method = gui_instance.smooth_method.get()
        if method == "savgol":
            config["smoothing"]["params"] = {
                "window_length": gui_instance.window_length_var.get(),
                "polyorder": gui_instance.polyorder_var.get()
            }
        elif method == "moving_average":
            config["smoothing"]["params"] = {
                "window_length": gui_instance.window_length_var.get()
            }
        elif method == "gaussian":
            config["smoothing"]["params"] = {
                "sigma": gui_instance.sigma_var.get()
            }
        elif method == "median":
            config["smoothing"]["params"] = {
                "window_length": gui_instance.window_length_var.get()
            }
        elif method == "lowess":
            config["smoothing"]["params"] = {
                "frac": gui_instance.frac_var.get(),
                "it": gui_instance.it_var.get()
            }

        # 获取基线校正参数
        baseline_method = gui_instance.baseline_method.get()
        if baseline_method == "rubberband":
            config["baseline"]["params"] = {
                "num_points": gui_instance.num_points_var.get()
            }
        elif baseline_method == "modpoly":
            config["baseline"]["params"] = {
                "poly_order": gui_instance.poly_order_var.get()
            }
        elif baseline_method == "imodpoly":
            config["baseline"]["params"] = {
                "poly_order": gui_instance.poly_order_var.get(),
                "num_iter": gui_instance.num_iter_var.get()
            }
        elif baseline_method == "asls":
            config["baseline"]["params"] = {
                "lam": gui_instance.lam_var.get(),
                "p": gui_instance.p_var.get()
            }
        elif baseline_method == "mixture_model":
            config["baseline"]["params"] = {
                "num_knots": gui_instance.num_knots_var.get()
            }

        return config

    @staticmethod
    def apply_config_to_gui(config: Dict[str, Any], gui_instance) -> None:
        """
        将配置应用到GUI实例

        Args:
            config: 配置字典
            gui_instance: SpectralProcessorGUI实例
        """
        try:
            # 应用Y轴标签
            if "y_label" in config:
                gui_instance.y_label_var.set(config["y_label"])

            # 应用平滑设置
            if "smoothing" in config:
                gui_instance.smooth_method.set(config["smoothing"]["method"])
                gui_instance.update_param_frame()

                # 应用平滑参数
                params = config["smoothing"]["params"]
                method = config["smoothing"]["method"]

                if method == "savgol":
                    gui_instance.window_length_var.set(params.get("window_length", "11"))
                    gui_instance.polyorder_var.set(params.get("polyorder", "3"))
                elif method == "moving_average":
                    gui_instance.window_length_var.set(params.get("window_length", "5"))
                elif method == "gaussian":
                    gui_instance.sigma_var.set(params.get("sigma", "1.0"))
                elif method == "median":
                    gui_instance.window_length_var.set(params.get("window_length", "5"))
                elif method == "lowess":
                    gui_instance.frac_var.set(params.get("frac", "0.2"))
                    gui_instance.it_var.set(params.get("it", "3"))

            # 应用基线校正设置
            if "baseline" in config:
                gui_instance.baseline_method.set(config["baseline"]["method"])
                gui_instance.update_baseline_params()

                # 应用基线参数
                params = config["baseline"]["params"]
                method = config["baseline"]["method"]

                if method == "rubberband":
                    gui_instance.num_points_var.set(params.get("num_points", "100"))
                elif method == "modpoly":
                    gui_instance.poly_order_var.set(params.get("poly_order", "2"))
                elif method == "imodpoly":
                    gui_instance.poly_order_var.set(params.get("poly_order", "3"))
                    gui_instance.num_iter_var.set(params.get("num_iter", "100"))
                elif method == "asls":
                    gui_instance.lam_var.set(params.get("lam", "1e7"))
                    gui_instance.p_var.set(params.get("p", "0.01"))
                elif method == "mixture_model":
                    gui_instance.num_knots_var.set(params.get("num_knots", "10"))

                # 应用数据源
                if "data_source" in config["baseline"]:
                    gui_instance.data_source_var.set(config["baseline"]["data_source"])

            # 应用峰分析设置
            if "peak_analysis" in config:
                gui_instance.peak_threshold_var.set(config["peak_analysis"].get("threshold", "0.1"))
                gui_instance.peak_distance_var.set(config["peak_analysis"].get("distance", "10"))
                if "data_source" in config["peak_analysis"]:
                    gui_instance.peak_data_var.set(config["peak_analysis"]["data_source"])

        except Exception as e:
            raise Exception(f"应用配置失败: {str(e)}")


class ProcessingPipeline:
    """处理流程管道 - 用于自动化批量处理"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化处理流程

        Args:
            config: 配置字典
        """
        self.config = config
        self.steps = []
        self._build_pipeline()

    def _build_pipeline(self):
        """根据配置构建处理流程"""
        # 确定处理步骤顺序
        if self.config.get("smoothing", {}).get("method"):
            self.steps.append("smoothing")

        if self.config.get("baseline", {}).get("method"):
            self.steps.append("baseline")

        if self.config.get("peak_analysis", {}).get("threshold"):
            self.steps.append("peak_analysis")

    def get_steps(self):
        """返回处理步骤列表"""
        return self.steps

    def get_config(self):
        """返回配置"""
        return self.config
