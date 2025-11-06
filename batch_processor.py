"""
批量处理模块
用于批量处理多个FTIR光谱文件
"""
import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Callable, Optional
from datetime import datetime
import json


class BatchProcessor:
    """批量文件处理器"""

    def __init__(self, config: Dict[str, Any], processing_functions: Dict[str, Callable]):
        """
        初始化批量处理器

        Args:
            config: 处理配置字典
            processing_functions: 处理函数字典 {'smoothing': func, 'baseline': func, ...}
        """
        self.config = config
        self.processing_functions = processing_functions
        self.results = []
        self.errors = []

    def process_files(self, file_paths: List[str],
                     output_dir: str,
                     progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        批量处理文件

        Args:
            file_paths: 文件路径列表
            output_dir: 输出目录
            progress_callback: 进度回调函数 callback(current, total, filename)

        Returns:
            处理结果摘要
        """
        self.results = []
        self.errors = []
        total_files = len(file_paths)

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        for i, file_path in enumerate(file_paths):
            try:
                # 更新进度
                if progress_callback:
                    progress_callback(i + 1, total_files, os.path.basename(file_path))

                # 处理单个文件
                result = self._process_single_file(file_path, output_dir)
                self.results.append(result)

            except Exception as e:
                self.errors.append({
                    "file": file_path,
                    "error": str(e)
                })

        # 生成处理报告
        return self._generate_summary()

    def _process_single_file(self, file_path: str, output_dir: str) -> Dict[str, Any]:
        """
        处理单个文件

        Args:
            file_path: 输入文件路径
            output_dir: 输出目录

        Returns:
            处理结果字典
        """
        # 加载数据
        data = pd.read_csv(file_path)
        if len(data.columns) < 2:
            raise ValueError(f"数据文件格式不正确！需要至少两列数据。")

        x_data = data.iloc[:, 0].values
        y_data = data.iloc[:, 1].values

        # 初始化处理结果
        result = {
            "input_file": file_path,
            "filename": os.path.basename(file_path),
            "data_points": len(x_data),
            "processing_steps": [],
            "output_files": []
        }

        # 应用处理步骤
        current_y = y_data.copy()
        smoothed_data = None
        baseline_corrected_data = None

        # 1. 平滑处理
        if "smoothing" in self.config and self.config["smoothing"].get("method"):
            if "smoothing" in self.processing_functions:
                smoothed_data = self.processing_functions["smoothing"](x_data, current_y, self.config["smoothing"])
                current_y = smoothed_data
                result["processing_steps"].append("smoothing")

        # 2. 基线校正
        if "baseline" in self.config and self.config["baseline"].get("method"):
            if "baseline" in self.processing_functions:
                # 确定使用哪个数据源
                if self.config["baseline"].get("data_source") == "smoothed" and smoothed_data is not None:
                    baseline_corrected_data = self.processing_functions["baseline"](x_data, smoothed_data, self.config["baseline"])
                else:
                    baseline_corrected_data = self.processing_functions["baseline"](x_data, y_data, self.config["baseline"])
                current_y = baseline_corrected_data
                result["processing_steps"].append("baseline")

        # 3. 峰检测
        peaks_data = None
        if "peak_analysis" in self.config:
            if "peak_analysis" in self.processing_functions:
                # 确定使用哪个数据源
                peak_source = self.config["peak_analysis"].get("data_source", "original")
                if peak_source == "corrected" and baseline_corrected_data is not None:
                    peak_y = baseline_corrected_data
                elif peak_source == "smoothed" and smoothed_data is not None:
                    peak_y = smoothed_data
                else:
                    peak_y = y_data

                peaks_data = self.processing_functions["peak_analysis"](x_data, peak_y, self.config["peak_analysis"])
                result["processing_steps"].append("peak_analysis")
                result["peaks_found"] = len(peaks_data) if peaks_data is not None else 0

        # 保存处理结果
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_files = self._save_results(
            output_dir, base_name, x_data, y_data,
            smoothed_data, baseline_corrected_data, peaks_data
        )
        result["output_files"] = output_files

        return result

    def _save_results(self, output_dir: str, base_name: str,
                     x_data, y_data, smoothed_data, corrected_data, peaks_data) -> List[str]:
        """
        保存处理结果

        Args:
            output_dir: 输出目录
            base_name: 基础文件名
            x_data: X轴数据
            y_data: 原始Y轴数据
            smoothed_data: 平滑后数据
            corrected_data: 校正后数据
            peaks_data: 峰数据

        Returns:
            生成的文件列表
        """
        output_files = []

        # 保存主要数据
        main_data = {"wavenumber": x_data, "raw": y_data}

        if smoothed_data is not None:
            main_data["smoothed"] = smoothed_data

        if corrected_data is not None:
            main_data["corrected"] = corrected_data

        main_file = os.path.join(output_dir, f"{base_name}_processed.csv")
        pd.DataFrame(main_data).to_csv(main_file, index=False)
        output_files.append(main_file)

        # 保存峰数据
        if peaks_data is not None and len(peaks_data) > 0:
            peaks_file = os.path.join(output_dir, f"{base_name}_peaks.csv")
            pd.DataFrame(peaks_data).to_csv(peaks_file, index=False)
            output_files.append(peaks_file)

        return output_files

    def _generate_summary(self) -> Dict[str, Any]:
        """
        生成处理摘要

        Returns:
            摘要字典
        """
        summary = {
            "total_files": len(self.results) + len(self.errors),
            "successful": len(self.results),
            "failed": len(self.errors),
            "timestamp": datetime.now().isoformat(),
            "results": self.results,
            "errors": self.errors
        }

        return summary

    def save_summary(self, output_dir: str, filename: str = "batch_summary.json"):
        """
        保存处理摘要到文件

        Args:
            output_dir: 输出目录
            filename: 摘要文件名
        """
        summary = self._generate_summary()
        summary_file = os.path.join(output_dir, filename)

        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)

        return summary_file


class ExportManager:
    """增强的导出管理器"""

    @staticmethod
    def export_to_excel(file_path: str, x_data, y_data,
                       smoothed_data=None, corrected_data=None,
                       peaks_data=None, config=None):
        """
        导出到Excel格式（多个工作表）

        Args:
            file_path: 输出文件路径
            x_data: X轴数据
            y_data: 原始Y轴数据
            smoothed_data: 平滑后数据（可选）
            corrected_data: 校正后数据（可选）
            peaks_data: 峰数据（可选）
            config: 处理配置（可选）
        """
        try:
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                # 工作表1: 原始数据
                df_raw = pd.DataFrame({
                    '波数(cm^-1)': x_data,
                    '吸光度': y_data
                })
                df_raw.to_excel(writer, sheet_name='原始数据', index=False)

                # 工作表2: 平滑数据
                if smoothed_data is not None:
                    df_smooth = pd.DataFrame({
                        '波数(cm^-1)': x_data,
                        '原始吸光度': y_data,
                        '平滑后吸光度': smoothed_data
                    })
                    df_smooth.to_excel(writer, sheet_name='平滑数据', index=False)

                # 工作表3: 基线校正数据
                if corrected_data is not None:
                    df_corrected = pd.DataFrame({
                        '波数(cm^-1)': x_data,
                        '校正后吸光度': corrected_data
                    })
                    df_corrected.to_excel(writer, sheet_name='基线校正数据', index=False)

                # 工作表4: 峰分析结果
                if peaks_data is not None:
                    df_peaks = pd.DataFrame(peaks_data)
                    df_peaks.to_excel(writer, sheet_name='峰分析', index=False)

                # 工作表5: 处理参数
                if config is not None:
                    config_text = json.dumps(config, indent=2, ensure_ascii=False)
                    df_config = pd.DataFrame({
                        '配置项': ['处理参数'],
                        '值': [config_text]
                    })
                    df_config.to_excel(writer, sheet_name='处理参数', index=False)

            return True
        except Exception as e:
            raise Exception(f"导出Excel失败: {str(e)}")

    @staticmethod
    def export_with_metadata(file_path: str, x_data, y_data,
                           smoothed_data=None, corrected_data=None,
                           config=None):
        """
        导出CSV文件并附带元数据

        Args:
            file_path: 输出文件路径
            x_data: X轴数据
            y_data: 原始Y轴数据
            smoothed_data: 平滑后数据（可选）
            corrected_data: 校正后数据（可选）
            config: 处理配置（可选）
        """
        try:
            # 创建数据DataFrame
            data_dict = {'wavenumber': x_data, 'raw': y_data}

            if smoothed_data is not None:
                data_dict['smoothed'] = smoothed_data

            if corrected_data is not None:
                data_dict['corrected'] = corrected_data

            df = pd.DataFrame(data_dict)

            # 写入文件
            with open(file_path, 'w', encoding='utf-8') as f:
                # 写入元数据作为注释
                f.write(f"# FTIR Spectral Data\n")
                f.write(f"# Export Date: {datetime.now().isoformat()}\n")
                f.write(f"# Data Points: {len(x_data)}\n")

                if config:
                    f.write(f"# Processing Config: {json.dumps(config, ensure_ascii=False)}\n")

                f.write("#\n")

                # 写入数据
                df.to_csv(f, index=False)

            return True
        except Exception as e:
            raise Exception(f"导出CSV失败: {str(e)}")
