"""
FTIR光谱数据处理工具

该程序提供了完整的FTIR光谱数据处理功能，包括：
- 数据加载和可视化
- 多种平滑算法（Savitzky-Golay、LOWESS、移动平均、高斯滤波、中值滤波）
- 多种基线校正方法（Rubberband、修正多项式、Whittaker-ASLS等）
- 特征峰自动识别和分析

邮箱: sl-xiao@zjnu.cn
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import SpanSelector
from matplotlib.font_manager import FontProperties
from scipy.signal import find_peaks
import logging
from typing import Optional, Tuple, Dict, Any, List

# 导入专业处理类
from src.data_manager import DataManager
from src.smoothing_processor import SmoothingProcessor
from src.baseline_corrector import BaselineCorrector
from src.peak_analyzer import PeakAnalyzer

# 配置日志
import os
import sys

log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 【修复】设置控制台输出编码为 UTF-8，解决 Windows 中文乱码问题
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        # Python < 3.7 不支持 reconfigure
        pass

# 创建控制台处理器，设置编码为 UTF-8
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'ftir_processor.log'), encoding='utf-8'),
        console_handler
    ]
)
logger = logging.getLogger(__name__)


class SpectralProcessorGUI:
    """
    FTIR光谱数据处理图形用户界面主类

    该类提供了完整的GUI界面，用于FTIR光谱数据的加载、处理、分析和导出。

    Attributes:
        root: Tkinter根窗口
        x_data: X轴数据（波数，单位cm^-1）
        y_data: Y轴数据（吸光度/透射率）
        smoothed_data: 平滑处理后的数据
        corrected_data: 基线校正后的数据
    """
    # 作者信息常量
    AUTHOR_NAME = "zjnuxsl"
    AUTHOR_EMAIL = "sl-xiao@zjnu.cn"

    def __init__(self, root):
        """
        初始化FTIR光谱处理GUI

        Args:
            root: Tkinter根窗口对象
        """
        self.root = root
        logger.info("初始化FTIR光谱处理器")

        # 设置matplotlib中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        # 配置Treeview样式
        self.setup_treeview_styles()

        # 初始化专业处理类
        self.data_manager = DataManager()  # 数据管理器
        self.smoothing_processor = SmoothingProcessor()  # 平滑处理器
        self.baseline_corrector = BaselineCorrector()  # 基线校正器
        self.peak_analyzer = PeakAnalyzer()  # 峰分析器

        # 初始化变量
        self.data_source_var = tk.StringVar(value="original")  # 数据源选择
        self.y_label_var = tk.StringVar(value="吸光度")  # Y轴标签
        self.smoothed_data_history = []  # 平滑数据历史（用于撤销）
        self.current_file_path = None  # 当前加载的文件路径
        self.current_file_name = None  # 当前加载的文件名（不含扩展名）
        self.smooth_ranges = []  # 存储选中的平滑区间 [(start1, end1), (start2, end2), ...]
        self.range_spans = []  # 存储图形上的区间高亮对象
        self.range_annotations = []  # 存储区间标签对象
        self.interactive_mode = False  # 交互式选择模式开关
        self.span_selector = None  # SpanSelector对象
        self.selected_range_index = None  # 当前选中的区间索引
        self.preview_timer = None  # 实时预览定时器（用于防抖）
        self.auto_preview_var = tk.BooleanVar(value=False)  # 实时预览开关（提前初始化）
        self.preview_in_progress = False  # 标志：是否正在执行预览

        # 区间边界拖动相关
        self.dragging_boundary = None  # 正在拖动的边界 (range_index, 'start'/'end')
        self.boundary_drag_threshold = 20  # 边界检测阈值（像素）

        # 峰分析交互式选择相关
        self.peak_interactive_mode = False  # 峰分析交互式选择模式开关
        self.peak_span_selector = None  # 峰分析SpanSelector对象
        self.peak_analysis_results = []  # 存储峰分析结果列表
        self.analyzed_ranges = []  # 存储已分析的区间 [(lower, upper, peak_number, file_name), ...]
        self.peak_range_artists = []  # 存储区间可视化对象（用于在图形上绘制）
        self.peak_selected_range = None  # 存储当前交互式选择的区域 (xmin, xmax)
        self.peak_context_menu = None  # 峰分析右键菜单

        # 多数据集对比分析相关
        self.loaded_datasets = []  # 存储多个数据集 [{'name': str, 'x_data': array, 'y_data': array, 'checked': bool}, ...]
        self.max_datasets = 10  # 最大加载数据集数量
        self.fixed_integration_range = tk.BooleanVar(value=False)  # 固定积分区间开关
        self.dataset_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']  # 数据集颜色列表
        self.has_performed_peak_finding = False  # 标志：是否已执行过寻峰操作
        self.dataset_switched = False  # 标志：数据集是否已切换（用于控制Y轴范围重置）

        # 初始化图形属性
        self.smooth_ax1 = None  # 平滑页面图1
        self.smooth_ax2 = None  # 平滑页面图2
        self.baseline_ax1 = None  # 基线页面图1
        self.baseline_ax2 = None  # 基线页面图2
        self.smooth_canvas = None  # 平滑页面画布
        self.baseline_canvas = None  # 基线页面画布

        # 创建数据文件夹
        self.input_dir = os.path.join('data', 'input')
        self.output_dir = os.path.join('data', 'output')
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        # 创建主框架
        self.create_main_frame()

        # 【修复】绑定窗口关闭事件，确保程序正确退出
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        logger.info("GUI初始化完成")

    def on_closing(self):
        """
        窗口关闭事件处理函数

        清理资源并正确退出程序，避免后台进程残留
        """
        try:
            logger.info("正在关闭程序...")

            # 取消所有定时器
            if hasattr(self, 'preview_timer') and self.preview_timer is not None:
                self.root.after_cancel(self.preview_timer)
                self.preview_timer = None

            # 关闭所有 matplotlib 图形
            plt.close('all')

            # 销毁主窗口
            self.root.destroy()

            logger.info("程序已正常关闭")

        except Exception as e:
            logger.error(f"关闭程序时出错: {str(e)}")
            # 强制退出
            self.root.destroy()

    def setup_treeview_styles(self):
        """配置Treeview样式（网格线和标题行背景色）"""
        style = ttk.Style()

        # 配置Treeview样式
        # 设置标题行背景色为浅蓝色
        style.configure("Treeview.Heading",
                       background="#E8F4F8",
                       foreground="black",
                       relief="flat",
                       font=('SimHei', 9, 'bold'))

        # 鼠标悬停在标题上时的样式
        style.map("Treeview.Heading",
                 background=[('active', '#D0E8F0')])

        # 配置Treeview行样式
        style.configure("Treeview",
                       background="white",
                       foreground="black",
                       rowheight=25,
                       fieldbackground="white",
                       font=('SimHei', 9))

        # 配置选中行的样式
        style.map("Treeview",
                 background=[('selected', '#0078D7')],
                 foreground=[('selected', 'white')])

        logger.info("Treeview样式配置完成")

    def create_author_label(self, parent_frame):
        """
        创建作者信息标签（消除重复代码）

        Args:
            parent_frame: 父容器框架

        Returns:
            ttk.Frame: 包含作者信息的框架
        """
        author_frame = ttk.Frame(parent_frame)
        author_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        author_label = ttk.Label(
            author_frame,
            text=f"作者: {self.AUTHOR_NAME}\n邮箱: {self.AUTHOR_EMAIL}",
            justify=tk.LEFT,
            font=('SimHei', 9)
        )
        author_label.pack(side=tk.LEFT)

        return author_frame

    def check_data_loaded(self, data_type='original'):
        """
        检查所需数据是否已加载

        Args:
            data_type (str): 数据类型，可选值：'original', 'smoothed', 'corrected'

        Returns:
            bool: 如果数据已加载返回True，否则显示错误消息并返回False
        """
        is_loaded, error_msg = self.data_manager.check_data_loaded(data_type)
        if not is_loaded:
            messagebox.showerror("错误", error_msg)
        return is_loaded

    @property
    def x_data(self):
        """获取X轴数据"""
        return self.data_manager.x_data

    @property
    def y_data(self):
        """获取Y轴原始数据"""
        return self.data_manager.y_data

    @property
    def smoothed_data(self):
        """获取平滑后的数据"""
        return self.data_manager.smoothed_data

    @property
    def corrected_data(self):
        """获取基线校正后的数据"""
        return self.data_manager.corrected_data



    def update_plots(self):
        """更新所有图形的 y 轴标题"""
        y_label = self.y_label_var.get()

        # 更新平滑处理页面的图形
        if hasattr(self, 'smooth_ax1') and self.smooth_ax1 is not None:
            self.smooth_ax1.set_ylabel(y_label)
        if hasattr(self, 'smooth_ax2') and self.smooth_ax2 is not None:
            self.smooth_ax2.set_ylabel(y_label)
        if hasattr(self, 'smooth_canvas'):
            self.smooth_canvas.draw()

        # 更新基线校正页面的图形
        if hasattr(self, 'baseline_ax1') and self.baseline_ax1 is not None:
            self.baseline_ax1.set_ylabel(y_label)
        if hasattr(self, 'baseline_ax2') and self.baseline_ax2 is not None:
            self.baseline_ax2.set_ylabel(y_label)
        if hasattr(self, 'baseline_canvas'):
            self.baseline_canvas.draw()
    
    def create_main_frame(self):
        """创建主界面框架"""
        # 创建标签页
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建页面
        self.smooth_frame = ttk.Frame(self.notebook)
        self.baseline_frame = ttk.Frame(self.notebook)
        self.peak_analysis_frame = ttk.Frame(self.notebook)
        self.log_management_frame = ttk.Frame(self.notebook)

        self.notebook.add(self.smooth_frame, text="平滑处理")
        self.notebook.add(self.baseline_frame, text="基线校正")
        self.notebook.add(self.peak_analysis_frame, text="特征峰分析")
        self.notebook.add(self.log_management_frame, text="日志管理")

        # 创建各页面内容
        self.create_smooth_page()
        self.create_baseline_page()
        self.create_peak_analysis_page()
        self.create_log_management_page()
        
    def create_smooth_page(self):
        """创建平滑处理页面"""
        # 创建左侧控制面板（增加宽度以完整显示所有控件）
        control_frame = ttk.LabelFrame(self.smooth_frame, text="控制面板", width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        control_frame.pack_propagate(False)  # 固定宽度

        # 加载数据按钮（放在最顶部，更显眼）
        load_btn = ttk.Button(control_frame, text="📁 加载数据",
                  command=self.load_data)
        load_btn.pack(fill=tk.X, padx=5, pady=5)

        # 局部区间平滑选择
        range_frame = ttk.LabelFrame(control_frame, text="局部区间平滑")
        range_frame.pack(fill=tk.X, padx=5, pady=5)

        # 说明文字
        ttk.Label(range_frame, text="选择需要平滑的波数区间",
                 font=('', 8), foreground='gray').pack(padx=5, pady=2)

        # 区间列表
        self.ranges_listbox = tk.Listbox(range_frame, height=4)
        self.ranges_listbox.pack(fill=tk.X, padx=5, pady=5)

        # 区间输入（改进布局）
        range_input_frame1 = ttk.Frame(range_frame)
        range_input_frame1.pack(fill=tk.X, padx=5, pady=2)

        ttk.Label(range_input_frame1, text="起始值:", width=8).pack(side=tk.LEFT)
        self.range_start_var = tk.StringVar()
        ttk.Entry(range_input_frame1, textvariable=self.range_start_var, width=12).pack(side=tk.LEFT, padx=2)

        range_input_frame2 = ttk.Frame(range_frame)
        range_input_frame2.pack(fill=tk.X, padx=5, pady=2)

        ttk.Label(range_input_frame2, text="终止值:", width=8).pack(side=tk.LEFT)
        self.range_end_var = tk.StringVar()
        ttk.Entry(range_input_frame2, textvariable=self.range_end_var, width=12).pack(side=tk.LEFT, padx=2)

        # 范围操作按钮
        range_btn_frame = ttk.Frame(range_frame)
        range_btn_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(range_btn_frame, text="添加", command=self.add_range, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Button(range_btn_frame, text="删除", command=self.delete_range, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Button(range_btn_frame, text="清空", command=self.clear_ranges, width=8).pack(side=tk.LEFT, padx=2)

        # 交互式选择模式
        interactive_frame = ttk.Frame(range_frame)
        interactive_frame.pack(fill=tk.X, padx=5, pady=5)

        self.interactive_mode_var = tk.BooleanVar(value=False)
        self.interactive_btn = ttk.Checkbutton(
            interactive_frame,
            text="🖱️ 交互式选择",
            variable=self.interactive_mode_var,
            command=self.toggle_interactive_mode
        )
        self.interactive_btn.pack(side=tk.LEFT)

        # 提示标签
        self.interactive_hint_label = ttk.Label(
            interactive_frame,
            text="",
            font=('', 8),
            foreground='blue'
        )
        self.interactive_hint_label.pack(side=tk.LEFT, padx=5)
        
        # 平滑方法选择（使用下拉框，更节省空间）
        method_frame = ttk.LabelFrame(control_frame, text="平滑方法")
        method_frame.pack(fill=tk.X, padx=5, pady=5)

        self.smooth_method = tk.StringVar(value="savgol")
        # 方法列表：(显示名称, 内部值)
        self.smooth_methods = [
            ("Savitzky-Golay（全局平滑推荐）", "savgol"),
            ("LOWESS（局部平滑推荐）", "lowess"),
            ("移动平均", "moving_average"),
            ("高斯滤波", "gaussian"),
            ("中值滤波", "median")
        ]

        # 创建显示名称到内部值的映射
        self.smooth_method_display_to_value = {display: value for display, value in self.smooth_methods}
        self.smooth_method_value_to_display = {value: display for display, value in self.smooth_methods}

        # 使用下拉框代替单选按钮，显示方法名称
        self.smooth_method_display = tk.StringVar(value="Savitzky-Golay（全局推荐）")
        method_combo = ttk.Combobox(method_frame, textvariable=self.smooth_method_display,
                                    values=[m[0] for m in self.smooth_methods], state='readonly', width=25)
        method_combo.pack(fill=tk.X, padx=5, pady=5)

        # 绑定方法切换事件
        def on_method_change(e):
            # 将显示名称转换为内部值
            display_name = self.smooth_method_display.get()
            self.smooth_method.set(self.smooth_method_display_to_value[display_name])
            self.update_param_frame()

        method_combo.bind('<<ComboboxSelected>>', on_method_change)
        
        # 参数设置框架
        self.param_frame = ttk.LabelFrame(control_frame, text="参数设置")
        self.param_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 初始化参数设置
        self.update_param_frame()
        
        # 添加实时预览选项
        preview_frame = ttk.Frame(control_frame)
        preview_frame.pack(fill=tk.X, padx=5, pady=5)

        # auto_preview_var 已在 __init__ 中初始化
        ttk.Checkbutton(preview_frame, text="实时预览",
                       variable=self.auto_preview_var,
                       command=self.toggle_auto_preview).pack(side=tk.LEFT)

        ttk.Label(preview_frame, text="⚡", foreground="orange").pack(side=tk.LEFT, padx=2)

        # 执行和导出按钮（使用更醒目的样式）
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=10)

        self.smooth_btn = ttk.Button(button_frame, text="✓ 应用平滑",
                                     command=self.smooth_data)
        self.smooth_btn.pack(fill=tk.X, pady=2)

        ttk.Button(button_frame, text="💾 导出数据",
                  command=self.export_smooth_data).pack(fill=tk.X, pady=2)

        # 撤销按钮已移除（后端功能保留，可在需要时重新启用）
        # ttk.Button(button_frame, text="↶ 撤销",
        #           command=self.undo_smooth).pack(fill=tk.X, pady=2)
        
        # 创建右侧图形区
        plot_frame = ttk.Frame(self.smooth_frame)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.smooth_fig = plt.Figure(figsize=(10, 6))
        # 使用 gridspec 调整子图间距，避免标签重叠
        self.smooth_ax1 = self.smooth_fig.add_subplot(211)
        self.smooth_ax2 = self.smooth_fig.add_subplot(212)
        # 调整子图间距，增加垂直间距避免重叠
        self.smooth_fig.subplots_adjust(hspace=0.35)

        self.smooth_canvas = FigureCanvasTkAgg(self.smooth_fig, master=plot_frame)

        # 初始化空图，设置默认横坐标范围（FTIR标准：4000-400 cm⁻¹）
        self.smooth_ax1.set_xlabel('波数 (cm$^{-1}$)')
        self.smooth_ax1.set_ylabel('吸光度')
        self.smooth_ax1.set_xlim(4000, 400)  # 左大右小
        self.smooth_ax1.grid(True)

        self.smooth_ax2.set_xlabel('波数 (cm$^{-1}$)')
        self.smooth_ax2.set_ylabel('吸光度')
        self.smooth_ax2.set_xlim(4000, 400)  # 左大右小
        self.smooth_ax2.grid(True)

        self.smooth_canvas.draw()
        self.smooth_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 添加工具栏
        toolbar = NavigationToolbar2Tk(self.smooth_canvas, plot_frame)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        # 初始化SpanSelector（默认不激活）
        self.span_selector = SpanSelector(
            self.smooth_ax1,
            self.on_span_select,
            'horizontal',
            useblit=True,
            props=dict(alpha=0.3, facecolor='cyan'),
            interactive=False,
            drag_from_anywhere=True
        )
        self.span_selector.set_active(False)

        # 绑定鼠标事件（点击、移动、释放）
        self.smooth_canvas.mpl_connect('button_press_event', self.on_canvas_click)
        self.smooth_canvas.mpl_connect('motion_notify_event', self.on_canvas_motion)
        self.smooth_canvas.mpl_connect('button_release_event', self.on_canvas_release)

        # 绑定键盘事件（Delete键删除选中区间）
        self.smooth_canvas.get_tk_widget().bind('<Delete>', self.on_delete_key)
        self.smooth_canvas.get_tk_widget().bind('<BackSpace>', self.on_delete_key)

        # 绑定区间列表的选择事件
        self.ranges_listbox.bind('<<ListboxSelect>>', self.on_range_listbox_select)

        # 在控制面板最下方添加作者信息
        self.create_author_label(control_frame)
        
    def create_baseline_page(self):
        """创建基线校正页面"""
        # 创建左侧控制面板
        control_frame = ttk.LabelFrame(self.baseline_frame, text="控制面板")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # 加载数据按钮
        load_btn = ttk.Button(control_frame, text="加载数据",
                             command=self.load_data)
        load_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # 数据源选择部分
        data_frame = ttk.LabelFrame(control_frame, text="数据选择")
        data_frame.pack(fill=tk.X, padx=5, pady=5)

        # 数据源选择按钮
        ttk.Radiobutton(data_frame, text="原始数据", value="original",
                        variable=self.data_source_var, command=self.update_baseline_plot).pack(anchor=tk.W)
        ttk.Radiobutton(data_frame, text="平滑后数", value="smoothed",
                        variable=self.data_source_var, command=self.update_baseline_plot).pack(anchor=tk.W)
        # 移除重复的"加载新数据"按钮（已在控制面板顶部有"加载数据"按钮）
        # ttk.Button(data_frame, text="加载新数据", command=self.load_data).pack(pady=5)
        
        # 基线校正方法选择
        method_frame = ttk.LabelFrame(control_frame, text="校正方法")
        method_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.baseline_method = tk.StringVar(value="rubberband")
        methods = [
            ("Rubberband(推荐)", "rubberband"),  # FTIR最常用
            ("修正多项式", "modpoly"),  # 简单基线
            ("自适应迭代多项式", "imodpoly"),  # 适合非线性基线
            ("Whittaker-ASLS", "asls"),  # 处理基线漂移
            ("平滑样条", "mixture_model"),  # 复杂基线
        ]
        for text, value in methods:
            ttk.Radiobutton(method_frame, text=text, value=value, 
                          variable=self.baseline_method,
                          command=self.update_baseline_params).pack(anchor=tk.W)
        
        # 创建参数设框架
        self.baseline_param_frame = ttk.LabelFrame(control_frame, text="参数设置")
        self.baseline_param_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 初始化数设置
        self.update_baseline_params()
        
        # 执行和导出按钮
        ttk.Button(control_frame, text="执行校正", command=self.correct_baseline).pack(pady=5)
        ttk.Button(control_frame, text="导出数据", command=self.export_baseline_data).pack(pady=5)
        
        # 创建右侧图形区域
        plot_frame = ttk.Frame(self.baseline_frame)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.baseline_fig = plt.Figure(figsize=(10, 6))
        self.baseline_ax1 = self.baseline_fig.add_subplot(211)
        self.baseline_ax2 = self.baseline_fig.add_subplot(212)
        # 调整子图间距，避免标签重叠
        self.baseline_fig.subplots_adjust(hspace=0.35)

        self.baseline_canvas = FigureCanvasTkAgg(self.baseline_fig, master=plot_frame)

        # 初始化空图，设置默认横坐标范围（FTIR标准：4000-400 cm⁻¹）
        self.baseline_ax1.set_xlabel('波数 (cm$^{-1}$)')
        self.baseline_ax1.set_ylabel('吸光度')
        self.baseline_ax1.set_xlim(4000, 400)  # 左大右小
        self.baseline_ax1.grid(True)

        self.baseline_ax2.set_xlabel('波数 (cm$^{-1}$)')
        self.baseline_ax2.set_ylabel('吸光度')
        self.baseline_ax2.set_xlim(4000, 400)  # 左大右小
        self.baseline_ax2.grid(True)

        self.baseline_canvas.draw()
        self.baseline_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 添加工具栏
        toolbar = NavigationToolbar2Tk(self.baseline_canvas, plot_frame)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 在控制面板最下方添加作者信息
        self.create_author_label(control_frame)

    def switch_data(self, data_type):
        """
        切换基线校正使用的数据源

        Args:
            data_type (str): 数据类型，'original'表示原始数据，'smoothed'表示平滑后数据
        """
        if not self.check_data_loaded(data_type):
            return

        self.data_source_var.set(data_type)
        self.update_baseline_plot()
        data_type_text = "原始数据" if data_type == "original" else "平滑后数据"
        logger.info(f"切换数据源到: {data_type_text}")
        messagebox.showinfo("成功", f"已切换到{data_type_text}")

    def load_data(self):
        """
        从CSV文件加载FTIR光谱数据

        该方法会打开文件选择对话框，允许用户选择CSV格式的光谱数据文件。
        文件应至少包含两列：第一列为波数（cm^-1），第二列为吸光度/透射率。

        加载成功后会自动绘制光谱图。
        """
        # 默认打开 data/input 文件夹
        initial_dir = self.input_dir if os.path.exists(self.input_dir) else os.getcwd()

        file_path = filedialog.askopenfilename(
            title="选择数据文件",
            initialdir=initial_dir,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if not file_path:
            return

        # 保存文件路径和文件名
        self.current_file_path = file_path
        self.current_file_name = os.path.splitext(os.path.basename(file_path))[0]

        # 使用DataManager加载数据
        success, message = self.data_manager.load_from_csv(file_path)

        if success:
            # 更新图形显示
            self.plot_data()
            logger.info(f"成功加载文件: {self.current_file_name}")

            # 将加载的数据添加到特征峰分析的数据集列表中
            # 先清空现有数据集（单文件加载模式）
            self.loaded_datasets.clear()

            # 重置原始视图范围（重要：确保能正确设置Y轴范围）
            self.peak_original_xlim = None
            self.peak_original_ylim = None

            self.loaded_datasets.append({
                'name': self.current_file_name,
                'x_data': self.x_data.copy(),
                'y_data': self.y_data.copy(),
                'checked': True
            })
            logger.info(f"已将数据添加到特征峰分析数据集列表: {self.current_file_name}")

            # 更新特征峰分析页面的数据集列表显示
            self.update_datasets_tree()

            # 更新特征峰分析页面的图形
            self.update_peak_plot()

            messagebox.showinfo("成功", message)

            # 自动寻峰（仅在特征峰分析页面激活时）
            try:
                current_tab = self.notebook.tab(self.notebook.select(), "text")
                if current_tab == "特征峰分析":
                    self.find_peaks()
                    logger.info("数据加载后自动寻峰完成")
                else:
                    logger.info(f"当前在'{current_tab}'页面，跳过自动寻峰")
            except Exception as e:
                # 【修复】为自动寻峰失败添加用户提示
                error_msg = f"自动寻峰失败: {str(e)}"
                logger.warning(error_msg)
                messagebox.showwarning("自动寻峰失败",
                    f"数据加载成功，但自动寻峰失败。\n\n"
                    f"错误信息: {str(e)}\n\n"
                    f"您可以手动调整寻峰参数后重新寻峰。")
        else:
            messagebox.showerror("错误", message)

    def load_multiple_datasets(self):
        """
        加载多个数据集用于对比分析

        该方法允许用户一次性选择多个CSV文件，将它们加载到数据集列表中。
        每个数据集包含文件名、x_data和y_data。
        """
        # 检查是否已达到最大数量
        if len(self.loaded_datasets) >= self.max_datasets:
            messagebox.showwarning("警告", f"已达到最大数据集数量限制（{self.max_datasets}个）")
            return

        # 默认打开 data/input 文件夹
        initial_dir = self.input_dir if os.path.exists(self.input_dir) else os.getcwd()

        file_paths = filedialog.askopenfilenames(
            title="选择数据文件（支持多选）",
            initialdir=initial_dir,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if not file_paths:
            return

        # 检查是否超过最大数量
        if len(self.loaded_datasets) + len(file_paths) > self.max_datasets:
            messagebox.showwarning("警告",
                f"选择的文件数量过多，最多只能加载{self.max_datasets - len(self.loaded_datasets)}个文件")
            file_paths = file_paths[:self.max_datasets - len(self.loaded_datasets)]

        success_count = 0
        failed_files = []

        for file_path in file_paths:
            try:
                # 读取CSV文件
                data = pd.read_csv(file_path)
                if data.shape[1] < 2:
                    failed_files.append(f"{os.path.basename(file_path)}: 列数不足")
                    continue

                x_data = data.iloc[:, 0].values
                y_data = data.iloc[:, 1].values
                file_name = os.path.splitext(os.path.basename(file_path))[0]

                # 添加到数据集列表（默认勾选）
                self.loaded_datasets.append({
                    'name': file_name,
                    'x_data': x_data,
                    'y_data': y_data,
                    'checked': True  # 默认勾选
                })
                success_count += 1
                logger.info(f"成功加载数据集: {file_name}")

            except Exception as e:
                failed_files.append(f"{os.path.basename(file_path)}: {str(e)}")
                logger.error(f"加载文件失败 {file_path}: {str(e)}")

        # 如果只加载了一个文件，同时更新单数据变量（兼容原有功能）
        if success_count == 1:
            dataset = self.loaded_datasets[-1]  # 获取最后添加的数据集
            self.data_manager.x_data = dataset['x_data']
            self.data_manager.y_data = dataset['y_data']
            # 更新当前文件名，用于分析结果表格显示
            self.current_file_name = dataset['name']
            logger.info(f"单文件加载：同时更新了data_manager的x_data和y_data，文件名: {self.current_file_name}")

        # 更新数据集列表显示
        if hasattr(self, 'datasets_tree'):
            self.update_datasets_tree()

        # 更新图形显示
        if hasattr(self, 'peak_ax'):
            self.update_peak_plot()

        # 检查并更新寻峰按钮状态
        if hasattr(self, 'find_peaks_btn'):
            self.check_find_peaks_button_state()

        # 显示结果
        if success_count > 0:
            msg = f"成功加载 {success_count} 个数据集"
            if failed_files:
                msg += f"\n\n失败 {len(failed_files)} 个:\n" + "\n".join(failed_files[:5])
                if len(failed_files) > 5:
                    msg += f"\n... 还有 {len(failed_files) - 5} 个"
            messagebox.showinfo("加载完成", msg)
        else:
            messagebox.showerror("错误", "没有成功加载任何数据集")

    def update_datasets_tree(self):
        """更新数据集列表显示（Treeview）"""
        if not hasattr(self, 'datasets_tree'):
            return

        logger.info(f"update_datasets_tree: 开始更新，数据集数量: {len(self.loaded_datasets)}")

        # 清空现有项
        for item in self.datasets_tree.get_children():
            self.datasets_tree.delete(item)

        # 添加所有数据集
        for idx, dataset in enumerate(self.loaded_datasets):
            checked = dataset.get('checked', True)
            checkbox_symbol = '☑' if checked else '☐'
            logger.info(f"  数据集 {idx}: '{dataset['name']}', checked={checked}, 符号={checkbox_symbol}")

            # 获取对应的颜色（使用彩色线条符号，和右侧曲线风格一致）
            color = self.dataset_colors[idx % len(self.dataset_colors)]
            color_symbol = '━'  # 使用横线符号作为图例

            # 交替行背景色（斑马纹效果）
            row_tag = 'evenrow' if idx % 2 == 0 else 'oddrow'

            # 插入数据集项
            item_id = self.datasets_tree.insert('', 'end', text=checkbox_symbol,
                                     values=(color_symbol, dataset['name']),
                                     tags=(str(idx), row_tag))

            # 为颜色列设置前景色（文字颜色）
            self.datasets_tree.tag_configure(f'color_{idx}', foreground=color)
            # 为该项添加颜色标签
            current_tags = list(self.datasets_tree.item(item_id, 'tags'))
            current_tags.append(f'color_{idx}')
            self.datasets_tree.item(item_id, tags=tuple(current_tags))

        # 配置斑马纹行背景色
        self.datasets_tree.tag_configure('evenrow', background='white')
        self.datasets_tree.tag_configure('oddrow', background='#F5F5F5')
        logger.info("update_datasets_tree: 更新完成")

    def remove_selected_dataset(self):
        """移除选中的数据集"""
        if not hasattr(self, 'datasets_tree'):
            return

        selection = self.datasets_tree.selection()
        if not selection:
            messagebox.showwarning("警告", "请先选择要移除的数据集")
            return

        # 获取选中项的索引
        indices = []
        for item in selection:
            tags = self.datasets_tree.item(item, 'tags')
            if tags:
                indices.append(int(tags[0]))

        # 从后往前删除，避免索引变化
        for index in sorted(indices, reverse=True):
            dataset_name = self.loaded_datasets[index]['name']
            del self.loaded_datasets[index]
            logger.info(f"移除数据集: {dataset_name}")

        self.update_datasets_tree()
        self.update_peak_plot()  # 更新图形
        messagebox.showinfo("成功", f"已移除 {len(selection)} 个数据集")

    def clear_all_datasets(self):
        """清空所有数据集"""
        if not self.loaded_datasets:
            messagebox.showinfo("提示", "当前没有已加载的数据集")
            return

        if messagebox.askyesno("确认", f"确定要清空所有 {len(self.loaded_datasets)} 个数据集吗？"):
            count = len(self.loaded_datasets)
            self.loaded_datasets.clear()

            # 重置原始视图范围（重要：确保下次加载数据时能正确设置Y轴范围）
            self.peak_original_xlim = None
            self.peak_original_ylim = None
            logger.info("已重置原始视图范围")

            # 清空峰列表（因为数据集已清空）
            if hasattr(self, 'peaks_tree'):
                for item in self.peaks_tree.get_children():
                    self.peaks_tree.delete(item)
                logger.info("已清空峰列表")

            self.update_datasets_tree()
            self.update_peak_plot()  # 更新图形（会再次清空峰列表，但这是安全的）
            logger.info(f"清空了 {count} 个数据集")
            messagebox.showinfo("成功", f"已清空 {count} 个数据集")

    def on_dataset_click(self, event):
        """处理数据集列表的点击事件（切换复选框状态）"""
        if not hasattr(self, 'datasets_tree'):
            return

        # 获取点击的区域
        region = self.datasets_tree.identify_region(event.x, event.y)

        # 只处理点击在tree列（复选框列）的情况
        if region == 'tree':
            item = self.datasets_tree.identify_row(event.y)
            if item:
                # 获取数据集索引
                tags = self.datasets_tree.item(item, 'tags')
                if tags:
                    idx = int(tags[0])
                    # 切换复选框状态
                    old_state = self.loaded_datasets[idx].get('checked', True)
                    new_state = not old_state
                    self.loaded_datasets[idx]['checked'] = new_state
                    logger.info(f"数据集 '{self.loaded_datasets[idx]['name']}' 复选框状态: {old_state} -> {new_state}")

                    # 如果只有一个数据集被勾选，更新data_manager的数据
                    checked_datasets = [ds for ds in self.loaded_datasets if ds.get('checked', True)]
                    logger.info(f"当前勾选的数据集数量: {len(checked_datasets)}")

                    if len(checked_datasets) == 1:
                        self.data_manager.x_data = checked_datasets[0]['x_data']
                        self.data_manager.y_data = checked_datasets[0]['y_data']
                        # 更新当前文件名，用于分析结果表格显示
                        self.current_file_name = checked_datasets[0]['name']
                        logger.info(f"切换到单数据集模式：{checked_datasets[0]['name']}，更新current_file_name为: {self.current_file_name}")
                        # 设置数据集切换标志，用于重置Y轴范围
                        self.dataset_switched = True

                        # 【修复】清空峰分析区域的上下限输入框（避免在新数据集上显示旧的预览区域）
                        if hasattr(self, 'peak_lower_var') and hasattr(self, 'peak_upper_var'):
                            self.peak_lower_var.set("")
                            self.peak_upper_var.set("")
                            logger.info("已清空峰分析区域的上下限输入框")
                    elif len(checked_datasets) == 0:
                        logger.info("所有数据集都已取消勾选")
                        # 【修复】清空峰分析区域的上下限输入框
                        if hasattr(self, 'peak_lower_var') and hasattr(self, 'peak_upper_var'):
                            self.peak_lower_var.set("")
                            self.peak_upper_var.set("")
                            logger.info("已清空峰分析区域的上下限输入框")

                    # 更新显示
                    self.update_datasets_tree()
                    # 更新图形
                    self.update_peak_plot()
                    # 检查是否需要禁用寻峰按钮
                    self.check_find_peaks_button_state()

                    # 自动寻峰：如果已经执行过寻峰，且当前只有一个数据集被勾选，则自动更新峰列表
                    if self.has_performed_peak_finding and len(checked_datasets) == 1:
                        logger.info(f"检测到数据集切换，自动为数据集 '{checked_datasets[0]['name']}' 执行寻峰")
                        self._auto_find_peaks_for_dataset(checked_datasets[0])

    def get_checked_datasets_count(self):
        """获取当前勾选的数据集数量"""
        return sum(1 for dataset in self.loaded_datasets if dataset.get('checked', True))

    def check_find_peaks_button_state(self):
        """检查并更新寻峰按钮的状态"""
        if not hasattr(self, 'find_peaks_btn'):
            return

        checked_count = self.get_checked_datasets_count()

        if checked_count == 1:
            # 只有一个数据集被勾选，启用寻峰按钮
            self.find_peaks_btn.config(state=tk.NORMAL)
            if hasattr(self, 'peak_hint_label'):
                self.peak_hint_label.config(text="")
        elif checked_count == 0:
            # 没有数据集被勾选，禁用寻峰按钮
            self.find_peaks_btn.config(state=tk.DISABLED)
            if hasattr(self, 'peak_hint_label'):
                self.peak_hint_label.config(text="请勾选一个数据集")
        else:
            # 多个数据集被勾选，禁用寻峰按钮
            self.find_peaks_btn.config(state=tk.DISABLED)
            if hasattr(self, 'peak_hint_label'):
                self.peak_hint_label.config(text="请仅勾选一个数据集以进行峰分析")

    def update_param_frame(self):
        """根据选择的平滑方法更新参数设置框架（带滑块）"""
        # 清除所有参数设置
        for widget in self.param_frame.winfo_children():
            widget.destroy()

        method = self.smooth_method.get()

        if method == "savgol":
            self._create_param_with_slider(
                "窗口长度", "window_length_var", 5, 51, 11, 2
            )
            self._create_param_with_slider(
                "多项式阶数", "polyorder_var", 1, 9, 3, 1
            )

        elif method == "moving_average":
            self._create_param_with_slider(
                "窗口长度", "window_length_var", 3, 51, 5, 2
            )

        elif method == "gaussian":
            self._create_param_with_slider(
                "标准差 σ", "sigma_var", 0.5, 10.0, 1.0, 0.5
            )

        elif method == "median":
            self._create_param_with_slider(
                "窗口长度", "window_length_var", 3, 51, 5, 2
            )

        elif method == "lowess":
            self._create_param_with_slider(
                "平滑分数", "lowess_frac_var", 0.05, 0.5, 0.2, 0.05
            )
            self._create_param_with_slider(
                "迭代次数", "lowess_iterations_var", 1, 10, 3, 1
            )

    def _create_param_with_slider(self, label_text, var_name, min_val, max_val, default_val, step):
        """
        创建带滑块的参数控件

        Args:
            label_text: 参数标签文本
            var_name: 变量名
            min_val: 最小值
            max_val: 最大值
            default_val: 默认值
            step: 步长
        """
        # 创建框架
        frame = ttk.Frame(self.param_frame)
        frame.pack(fill=tk.X, padx=5, pady=3)

        # 标签和当前值
        header_frame = ttk.Frame(frame)
        header_frame.pack(fill=tk.X)

        ttk.Label(header_frame, text=label_text).pack(side=tk.LEFT)

        # 创建变量
        if isinstance(default_val, int):
            var = tk.IntVar(value=default_val)
        else:
            var = tk.DoubleVar(value=default_val)
        setattr(self, var_name, var)

        # 当前值标签
        value_label = ttk.Label(header_frame, text=f"{default_val}",
                               foreground="blue", font=('', 9, 'bold'))
        value_label.pack(side=tk.RIGHT)

        # 滑块
        if isinstance(default_val, int):
            slider = ttk.Scale(frame, from_=min_val, to=max_val,
                             variable=var, orient=tk.HORIZONTAL,
                             command=lambda v: self._on_param_change(v, value_label, True))
        else:
            slider = ttk.Scale(frame, from_=min_val, to=max_val,
                             variable=var, orient=tk.HORIZONTAL,
                             command=lambda v: self._on_param_change(v, value_label, False))
        slider.pack(fill=tk.X, pady=2)

    def _on_param_change(self, value, value_label, is_int):
        """
        参数滑块变化时的回调函数

        Args:
            value: 滑块当前值
            value_label: 显示值的标签
            is_int: 是否为整数类型
        """
        # 更新显示的值
        if is_int:
            value_label.config(text=f"{int(float(value))}")
        else:
            value_label.config(text=f"{float(value):.2f}")

        # 如果启用了实时预览，触发预览
        if self.auto_preview_var.get():
            self._schedule_preview()

    def _schedule_preview(self):
        """
        安排实时预览（带防抖机制）
        延迟500ms执行，避免频繁调用
        """
        # 取消之前的定时器
        if self.preview_timer is not None:
            self.root.after_cancel(self.preview_timer)

        # 设置新的定时器
        self.preview_timer = self.root.after(500, self._execute_preview)

    def _execute_preview(self):
        """执行实时预览（不保存到历史记录）"""
        # 【修复】检查是否已有预览正在执行
        if self.preview_in_progress:
            logger.debug("上一次预览尚未完成，跳过本次预览")
            return

        if not self.check_data_loaded():
            return

        try:
            # 设置预览进行中标志
            self.preview_in_progress = True

            method = self.smooth_method.get()
            ranges = self.get_selected_ranges()

            # 准备参数
            params = {}
            if method == "savgol":
                window_length = int(self.window_length_var.get())
                # 确保窗口长度是奇数
                if window_length % 2 == 0:
                    window_length += 1
                params['window_length'] = window_length
                params['polyorder'] = int(self.polyorder_var.get())
            elif method == "lowess":
                params['frac'] = float(self.lowess_frac_var.get())
                params['iterations'] = int(self.lowess_iterations_var.get())
            elif method in ["moving_average", "median"]:
                window_length = int(self.window_length_var.get())
                # 中值滤波器也需要奇数窗口
                if method == "median" and window_length % 2 == 0:
                    window_length += 1
                params['window_length'] = window_length
            elif method == "gaussian":
                params['sigma'] = float(self.sigma_var.get())

            # 使用SmoothingProcessor进行平滑
            success, smoothed_data, error_msg = self.smoothing_processor.smooth_data_in_ranges(
                self.x_data, self.y_data, ranges, method, **params
            )

            if success:
                # 临时更新平滑数据（不保存到历史）
                # 使用 data_manager 的方法来设置数据
                self.data_manager.set_smoothed_data(smoothed_data)

                # 重新绘制图形
                self.plot_smooth_result()

                logger.info("实时预览已更新")
            else:
                logger.warning(f"实时预览失败: {error_msg}")

        except Exception as e:
            logger.error(f"实时预览出错: {str(e)}")
        finally:
            # 【修复】确保标志被重置
            self.preview_in_progress = False

    def add_range(self):
        """添加数据处理范围（带边界检查）"""
        try:
            start_str = self.range_start_var.get().strip()
            end_str = self.range_end_var.get().strip()

            if not start_str or not end_str:
                messagebox.showerror("错误", "请输入起始值和终止值！")
                return

            start = float(start_str)
            end = float(end_str)

            if start >= end:
                messagebox.showerror("错误", "起始值必须小于终止值！")
                return

            # 检查数据是否已加载
            if self.x_data is None or len(self.x_data) == 0:
                messagebox.showerror("错误", "请先加载数据！")
                return

            # 获取数据的波数范围
            min_wavenumber = float(np.min(self.x_data))
            max_wavenumber = float(np.max(self.x_data))

            # 检查区间是否在数据范围内
            if start < min_wavenumber or end > max_wavenumber:
                messagebox.showerror(
                    "错误",
                    f"区间范围必须在 {min_wavenumber:.2f} - {max_wavenumber:.2f} cm⁻¹ 之间\n"
                    f"您输入的范围：{start:.2f} - {end:.2f} cm⁻¹"
                )
                return

            # 自动裁剪到有效范围（可选，这里选择报错而不是自动裁剪）
            # start = max(start, min_wavenumber)
            # end = min(end, max_wavenumber)

            range_str = f"{start:.2f} - {end:.2f}"
            self.ranges_listbox.insert(tk.END, range_str)

            logger.info(f"添加区间: {range_str}")

            # 清空输入框
            self.range_start_var.set("")
            self.range_end_var.set("")

            # 更新图形显示区间高亮
            if self.data_manager.y_data is not None:
                self._draw_smooth_ranges()
                self.smooth_canvas.draw()

                # 检查并合并重叠的区间
                merged = self._merge_overlapping_ranges()
                if merged:
                    self._draw_smooth_ranges()
                    self.smooth_canvas.draw()
                    logger.info("添加区间后自动合并了重叠区间")

                # 如果启用了实时预览，立即更新平滑效果
                if self.auto_preview_var.get():
                    self._execute_preview()
                    logger.info("区间变化，实时预览已更新")

        except ValueError:
            messagebox.showerror("错误", "请输入有效的数值！")

    def delete_range(self):
        """删除选中的范围"""
        selection = self.ranges_listbox.curselection()
        if selection:
            deleted_index = selection[0]
            self.ranges_listbox.delete(selection)

            # 更新选中索引
            if self.selected_range_index == deleted_index:
                self.selected_range_index = None
            elif self.selected_range_index is not None and self.selected_range_index > deleted_index:
                self.selected_range_index -= 1

            # 更新图形显示区间高亮
            if self.data_manager.y_data is not None:
                self._draw_smooth_ranges()
                self.smooth_canvas.draw()

                # 如果启用了实时预览，立即更新平滑效果
                if self.auto_preview_var.get():
                    self._execute_preview()
                    logger.info("区间变化，实时预览已更新")

    def clear_ranges(self):
        """清空所有范围"""
        self.ranges_listbox.delete(0, tk.END)
        self.selected_range_index = None
        # 更新图形显示区间高亮
        if self.data_manager.y_data is not None:
            self._draw_smooth_ranges()
            self.smooth_canvas.draw()

            # 如果启用了实时预览，立即更新平滑效果
            if self.auto_preview_var.get():
                self._execute_preview()
                logger.info("区间变化，实时预览已更新")

    def toggle_interactive_mode(self):
        """切换交互式选择模式"""
        self.interactive_mode = self.interactive_mode_var.get()

        if self.interactive_mode:
            # 启用交互模式
            self.span_selector.set_active(True)
            self.interactive_hint_label.config(text="拖动鼠标选择区间")
            logger.info("交互式选择模式已启用")
        else:
            # 禁用交互模式
            self.span_selector.set_active(False)
            self.interactive_hint_label.config(text="")
            logger.info("交互式选择模式已禁用")

    def on_span_select(self, xmin, xmax):
        """SpanSelector回调函数：当用户拖拽选择区间时调用（带边界检查）"""
        if not self.interactive_mode:
            return

        # 确保xmin < xmax（因为横坐标已倒置）
        if xmin > xmax:
            xmin, xmax = xmax, xmin

        # 检查数据是否已加载
        if self.x_data is None or len(self.x_data) == 0:
            return

        # 获取数据的波数范围
        min_wavenumber = float(np.min(self.x_data))
        max_wavenumber = float(np.max(self.x_data))

        # 自动裁剪到有效范围
        original_xmin, original_xmax = xmin, xmax
        xmin = max(xmin, min_wavenumber)
        xmax = min(xmax, max_wavenumber)

        # 如果裁剪后的区间太小，不添加
        if xmax - xmin < 1.0:
            logger.warning(f"选择的区间太小或超出范围，已忽略: {original_xmin:.2f} - {original_xmax:.2f}")
            return

        # 如果进行了裁剪，记录日志
        if abs(original_xmin - xmin) > 0.01 or abs(original_xmax - xmax) > 0.01:
            logger.info(f"区间已自动裁剪: {original_xmin:.2f}-{original_xmax:.2f} → {xmin:.2f}-{xmax:.2f}")

        # 添加选中的区间
        self.range_start_var.set(f"{xmin:.2f}")
        self.range_end_var.set(f"{xmax:.2f}")
        self.add_range()

        logger.info(f"通过拖拽添加区间: {xmin:.2f} - {xmax:.2f}")

    def on_canvas_click(self, event):
        """鼠标点击事件处理：用于选择、删除区间或开始拖动边界"""
        # 只在上图（原始数据图）中响应
        if event.inaxes != self.smooth_ax1:
            return

        # 如果不在交互模式，不处理点击
        if not self.interactive_mode:
            return

        # 获取点击位置的x坐标
        click_x = event.xdata
        if click_x is None:
            return

        # 检查是否点击在某个区间的边界附近
        ranges = self.get_selected_ranges()
        boundary_info = self._find_nearby_boundary(event.x, click_x, ranges)

        if boundary_info is not None and event.button == 1:
            # 开始拖动边界
            self.dragging_boundary = boundary_info

            # 禁用 SpanSelector，防止拖动边界时触发区间选择
            if hasattr(self, 'span_selector') and self.span_selector is not None:
                self.span_selector.set_active(False)

            logger.info(f"开始拖动区间 {boundary_info[0] + 1} 的 {boundary_info[1]} 边界")
            return

        # 检查是否点击在某个区间内
        clicked_range_index = None
        for idx, (start, end) in enumerate(ranges):
            if min(start, end) <= click_x <= max(start, end):
                clicked_range_index = idx
                break

        if clicked_range_index is not None:
            # 如果是右键点击，删除该区间
            if event.button == 3:  # 右键
                self.ranges_listbox.delete(clicked_range_index)
                self.selected_range_index = None
                self._draw_smooth_ranges()
                self.smooth_canvas.draw()
                logger.info(f"删除区间 {clicked_range_index + 1}")
            # 如果是左键点击，选中该区间
            elif event.button == 1:  # 左键
                if self.selected_range_index == clicked_range_index:
                    # 取消选中
                    self.selected_range_index = None
                else:
                    # 选中该区间
                    self.selected_range_index = clicked_range_index
                    self.ranges_listbox.selection_clear(0, tk.END)
                    self.ranges_listbox.selection_set(clicked_range_index)
                    self.ranges_listbox.see(clicked_range_index)

                self._draw_smooth_ranges()
                self.smooth_canvas.draw()
                logger.info(f"选中区间 {clicked_range_index + 1}")
        else:
            # 点击空白区域，取消选中
            if event.button == 1:
                self.selected_range_index = None
                self.ranges_listbox.selection_clear(0, tk.END)
                self._draw_smooth_ranges()
                self.smooth_canvas.draw()

    def _merge_overlapping_ranges(self):
        """
        合并重叠或相邻的区间

        Returns:
            是否进行了合并操作
        """
        ranges = self.get_selected_ranges()
        logger.info(f"开始检查区间合并，当前区间数: {len(ranges)}, 区间: {ranges}")

        if len(ranges) <= 1:
            logger.info("区间数 <= 1，无需合并")
            return False

        # 按起始位置排序
        sorted_ranges = sorted(ranges, key=lambda x: min(x[0], x[1]))
        logger.info(f"排序后的区间: {sorted_ranges}")

        merged = []
        current_start, current_end = sorted_ranges[0]

        # 确保 start < end
        if current_start > current_end:
            current_start, current_end = current_end, current_start

        has_merged = False

        for i in range(1, len(sorted_ranges)):
            next_start, next_end = sorted_ranges[i]

            # 确保 start < end
            if next_start > next_end:
                next_start, next_end = next_end, next_start

            # 检查是否重叠或相邻（允许5个单位的间隙）
            logger.info(f"检查区间 [{next_start:.2f}, {next_end:.2f}] 是否与当前区间 [{current_start:.2f}, {current_end:.2f}] 重叠")
            logger.info(f"判断条件: {next_start:.2f} <= {current_end:.2f} + 5.0 = {current_end + 5.0:.2f} ? {next_start <= current_end + 5.0}")

            if next_start <= current_end + 5.0:
                # 合并区间
                old_end = current_end
                current_end = max(current_end, next_end)
                has_merged = True
                logger.info(f"✓ 合并区间: [{current_start:.2f}, {old_end:.2f}] + [{next_start:.2f}, {next_end:.2f}] → [{current_start:.2f}, {current_end:.2f}]")
            else:
                # 保存当前区间，开始新区间
                logger.info(f"✗ 不合并，保存当前区间 [{current_start:.2f}, {current_end:.2f}]")
                merged.append((current_start, current_end))
                current_start, current_end = next_start, next_end

        # 添加最后一个区间
        merged.append((current_start, current_end))

        # 如果进行了合并，更新列表
        if has_merged:
            # 清空列表
            self.ranges_listbox.delete(0, tk.END)

            # 添加合并后的区间
            for start, end in merged:
                range_str = f"{start:.2f} - {end:.2f}"
                self.ranges_listbox.insert(tk.END, range_str)

            logger.info(f"区间合并完成: {len(ranges)} → {len(merged)}")
            return True

        return False

    def _find_nearby_boundary(self, pixel_x, data_x, ranges):
        """
        查找鼠标附近的区间边界

        Args:
            pixel_x: 鼠标的像素x坐标
            data_x: 鼠标的数据x坐标
            ranges: 区间列表

        Returns:
            (range_index, 'start'/'end') 如果找到边界，否则返回 None
        """
        if not ranges:
            return None

        # 将数据坐标转换为像素坐标的辅助函数
        def data_to_pixel(x_val):
            # 使用 transData 转换
            return self.smooth_ax1.transData.transform([(x_val, 0)])[0][0]

        # 检查每个区间的边界
        for idx, (start, end) in enumerate(ranges):
            start_pixel = data_to_pixel(start)
            end_pixel = data_to_pixel(end)

            # 检查是否靠近起始边界
            if abs(pixel_x - start_pixel) < self.boundary_drag_threshold:
                return (idx, 'start')

            # 检查是否靠近终止边界
            if abs(pixel_x - end_pixel) < self.boundary_drag_threshold:
                return (idx, 'end')

        return None

    def on_canvas_motion(self, event):
        """鼠标移动事件处理：拖动边界或更改光标"""
        if event.inaxes != self.smooth_ax1:
            # 恢复默认光标
            self.smooth_canvas.get_tk_widget().config(cursor="")
            return

        if not self.interactive_mode:
            return

        # 如果正在拖动边界
        if self.dragging_boundary is not None:
            if event.xdata is None:
                return

            range_idx, boundary_type = self.dragging_boundary
            ranges = self.get_selected_ranges()

            if range_idx >= len(ranges):
                return

            start, end = ranges[range_idx]
            new_x = event.xdata

            # 限制在数据范围内
            if self.x_data is not None:
                min_wavenumber = float(np.min(self.x_data))
                max_wavenumber = float(np.max(self.x_data))
                new_x = max(min_wavenumber, min(max_wavenumber, new_x))

            # 更新边界（允许双向拖动：扩大或缩小）
            if boundary_type == 'start':
                # 确保起始值小于终止值
                if new_x < end - 1.0:  # 至少保持1个单位的间隔
                    start = new_x
            else:  # 'end'
                # 确保终止值大于起始值
                if new_x > start + 1.0:
                    end = new_x

            # 更新列表中的区间
            range_str = f"{start:.2f} - {end:.2f}"
            self.ranges_listbox.delete(range_idx)
            self.ranges_listbox.insert(range_idx, range_str)
            self.ranges_listbox.selection_set(range_idx)

            # 注意：不在拖动过程中检查合并，只在拖动完成后检查
            # 这样可以避免拖动缩小区间时被误判为新增区间

            # 重新绘制
            self._draw_smooth_ranges()
            self.smooth_canvas.draw()

        else:
            # 检查是否靠近边界，更改光标
            if event.xdata is not None:
                ranges = self.get_selected_ranges()
                boundary_info = self._find_nearby_boundary(event.x, event.xdata, ranges)

                if boundary_info is not None:
                    # 靠近边界，显示左右箭头光标
                    self.smooth_canvas.get_tk_widget().config(cursor="sb_h_double_arrow")
                else:
                    # 恢复默认光标
                    self.smooth_canvas.get_tk_widget().config(cursor="")

    def on_canvas_release(self, event):
        """鼠标释放事件处理：结束边界拖动并检查合并"""
        if self.dragging_boundary is not None:
            range_idx, boundary_type = self.dragging_boundary
            logger.info(f"完成拖动区间 {range_idx + 1} 的 {boundary_type} 边界")
            self.dragging_boundary = None

            # 重新启用 SpanSelector
            if hasattr(self, 'span_selector') and self.span_selector is not None and self.interactive_mode:
                self.span_selector.set_active(True)

            # 最后再检查一次是否需要合并
            merged = self._merge_overlapping_ranges()
            if merged:
                self.selected_range_index = None
                self._draw_smooth_ranges()
                self.smooth_canvas.draw()

            # 如果启用了实时预览，立即更新平滑效果
            if self.auto_preview_var.get():
                self._execute_preview()
                logger.info("区间边界拖动完成，实时预览已更新")

            # 恢复光标
            self.smooth_canvas.get_tk_widget().config(cursor="")

    def on_delete_key(self, event):
        """Delete键或Backspace键处理：删除选中的区间"""
        if self.selected_range_index is not None:
            self.ranges_listbox.delete(self.selected_range_index)
            self.selected_range_index = None
            self._draw_smooth_ranges()
            self.smooth_canvas.draw()
            logger.info("通过键盘删除区间")

            # 如果启用了实时预览，立即更新平滑效果
            if self.auto_preview_var.get():
                self._execute_preview()
                logger.info("区间变化，实时预览已更新")
        elif self.ranges_listbox.curselection():
            # 如果列表中有选中项，删除它
            self.delete_range()

    def on_range_listbox_select(self, event):
        """区间列表选择事件：同步图形上的选中状态"""
        selection = self.ranges_listbox.curselection()
        if selection:
            self.selected_range_index = selection[0]
        else:
            self.selected_range_index = None

        # 更新图形显示
        if self.data_manager.y_data is not None:
            self._draw_smooth_ranges()
            self.smooth_canvas.draw()

    def get_selected_ranges(self):
        """获取所有选择的范围"""
        ranges = []
        for i in range(self.ranges_listbox.size()):
            range_str = self.ranges_listbox.get(i)
            start, end = map(float, range_str.split(" - "))
            ranges.append((start, end))
        return ranges

    def _draw_smooth_ranges(self):
        """在图形上绘制选中的平滑区间高亮（带标签和编号）"""
        # 清除之前的高亮
        for span in self.range_spans:
            try:
                span.remove()
            except Exception as e:
                logger.warning(f"移除区间高亮对象失败: {str(e)}")
        self.range_spans.clear()

        # 清除之前的标签
        for annotation in self.range_annotations:
            try:
                annotation.remove()
            except Exception as e:
                logger.warning(f"移除区间标签对象失败: {str(e)}")
        self.range_annotations.clear()

        # 获取选中的区间
        ranges = self.get_selected_ranges()
        if not ranges:
            return

        # 获取Y轴范围用于定位标签
        if self.y_data is not None:
            y_max = np.max(self.y_data)
            y_min = np.min(self.y_data)
            y_range = y_max - y_min
            label_y = y_max - y_range * 0.05  # 标签位置在顶部5%处
        else:
            label_y = 1.0

        # 在上图绘制高亮区间
        for idx, (start, end) in enumerate(ranges, 1):
            # 根据是否选中使用不同颜色
            if self.selected_range_index == idx - 1:
                color = 'lightblue'
                alpha = 0.4
                edgecolor = 'blue'
                linewidth = 2
            else:
                color = 'yellow'
                alpha = 0.2
                edgecolor = None
                linewidth = 0

            # 使用半透明颜色高亮显示选中区间
            span = self.smooth_ax1.axvspan(
                start, end,
                alpha=alpha,
                color=color,
                edgecolor=edgecolor,
                linewidth=linewidth,
                label='平滑区间' if idx == 1 else ''
            )
            self.range_spans.append(span)

            # 添加区间标签（显示区间编号和范围）
            mid_x = (start + end) / 2
            label_text = f"区间{idx}\n{start:.1f}-{end:.1f}"

            annotation = self.smooth_ax1.annotate(
                label_text,
                xy=(mid_x, label_y),
                xytext=(0, 10),
                textcoords='offset points',
                ha='center',
                va='bottom',
                fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7, edgecolor='gray'),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='gray', lw=1)
            )
            self.range_annotations.append(annotation)

        # 更新图例（只在第一次添加时）
        if ranges:
            self.smooth_ax1.legend(loc='upper right')

    def smooth_data(self):
        """
        执行数据平滑处理

        根据用户选择的平滑方法和参数对光谱数据进行平滑处理。
        支持的方法包括：
        - Savitzky-Golay滤波
        - LOWESS局部加权回归
        - 移动平均
        - 高斯滤波
        - 中值滤波

        可以选择特定波数范围进行处理，未选择范围时处理全部数据。
        """
        if not self.check_data_loaded():
            return

        try:
            method = self.smooth_method.get()
            ranges = self.get_selected_ranges()

            # 准备参数
            params = {}
            if method == "savgol":
                window_length = int(self.window_length_var.get())
                # 确保窗口长度是奇数
                if window_length % 2 == 0:
                    window_length += 1
                params['window_length'] = window_length
                params['polyorder'] = int(self.polyorder_var.get())
            elif method == "lowess":
                params['frac'] = float(self.lowess_frac_var.get())
                params['iterations'] = int(self.lowess_iterations_var.get())
            elif method in ["moving_average", "median"]:
                window_length = int(self.window_length_var.get())
                # 中值滤波器也需要奇数窗口
                if method == "median" and window_length % 2 == 0:
                    window_length += 1
                params['window_length'] = window_length
            elif method == "gaussian":
                params['sigma'] = float(self.sigma_var.get())

            # 使用SmoothingProcessor进行平滑
            success, smoothed_data, error_msg = self.smoothing_processor.smooth_data_in_ranges(
                self.x_data, self.y_data, ranges, method, **params
            )

            if success:
                # 保存当前数据到历史（用于撤销）
                # 如果已有平滑数据，保存到历史
                if self.smoothed_data is not None:
                    self.smoothed_data_history.append(self.smoothed_data.copy())
                    # 限制历史记录数量为10
                    if len(self.smoothed_data_history) > 10:
                        self.smoothed_data_history.pop(0)
                    logger.info(f"保存平滑历史，当前历史记录数: {len(self.smoothed_data_history)}")

                # 更新平滑数据
                self.data_manager.set_smoothed_data(smoothed_data)

                # 重新绘制图形
                self.plot_smooth_result()
                messagebox.showinfo("成功", "平滑处理完成！")
            else:
                messagebox.showerror("错误", error_msg)

        except ValueError as e:
            messagebox.showerror("参数错误", f"参数格式不正确：{str(e)}")
        except Exception as e:
            messagebox.showerror("错误", f"平滑处理出错：{str(e)}")

    def toggle_auto_preview(self):
        """切换实时预览模式"""
        if self.auto_preview_var.get():
            logger.info("启用实时预览模式")
            # 立即执行一次预览，显示当前参数下的效果
            self._execute_preview()
            messagebox.showinfo("提示", "实时预览已启用\n移动参数滑块时将自动更新图形")
        else:
            logger.info("禁用实时预览模式")
            # 取消待执行的预览
            if self.preview_timer is not None:
                self.root.after_cancel(self.preview_timer)
                self.preview_timer = None

    def undo_smooth(self):
        """撤销上一次平滑操作"""
        if not self.smoothed_data_history:
            messagebox.showinfo("提示", "没有可撤销的操作")
            return

        # 恢复上一次的数据
        previous_data = self.smoothed_data_history.pop()
        self.data_manager.set_smoothed_data(previous_data)

        # 重新绘制图形
        self.plot_smooth_result()

        logger.info(f"已撤销上一次平滑操作，剩余历史记录: {len(self.smoothed_data_history)}")
        messagebox.showinfo("成功", "已撤销上一次操作")

    def update_baseline_params(self):
        """根据选的基线校正法更新参数设置"""
        # 清除现有参数设置
        for widget in self.baseline_param_frame.winfo_children():
            widget.destroy()
            
        method = self.baseline_method.get()
        
        if method == "rubberband":
            # Rubberband方法参数
            ttk.Label(self.baseline_param_frame, text="点数:").pack()
            self.num_points_var = tk.StringVar(value="100")
            ttk.Entry(self.baseline_param_frame, textvariable=self.num_points_var).pack()

        elif method == "modpoly":
            # 修正多项式参数
            ttk.Label(self.baseline_param_frame, text="多项式阶数:").pack()
            self.poly_order_var = tk.StringVar(value="2")
            ttk.Entry(self.baseline_param_frame, textvariable=self.poly_order_var).pack()

            
        elif method == "imodpoly":
            # 自适应迭代多项式参数
            ttk.Label(self.baseline_param_frame, text="多项式阶数:").pack()
            self.poly_order_var = tk.StringVar(value="3")
            ttk.Entry(self.baseline_param_frame, textvariable=self.poly_order_var).pack()
            
            ttk.Label(self.baseline_param_frame, text="迭代次数:").pack()
            self.num_iter_var = tk.StringVar(value="100")
            ttk.Entry(self.baseline_param_frame, textvariable=self.num_iter_var).pack()
            
        elif method == "asls":
            # Whittaker-ASLS参数
            ttk.Label(self.baseline_param_frame, text="平滑参数():").pack()
            self.lam_var = tk.StringVar(value="1e7")
            ttk.Entry(self.baseline_param_frame, textvariable=self.lam_var).pack()
            
            ttk.Label(self.baseline_param_frame, text="非对称参数(p):").pack()
            self.p_var = tk.StringVar(value="0.01")
            ttk.Entry(self.baseline_param_frame, textvariable=self.p_var).pack()
            
        elif method == "mixture_model":
            # 平滑样条参数
            ttk.Label(self.baseline_param_frame, text="样条节点数:").pack()
            self.num_knots_var = tk.StringVar(value="10")
            ttk.Entry(self.baseline_param_frame, textvariable=self.num_knots_var).pack()
            
    def correct_baseline(self):
        """
        执行基线校正

        根据用户选择的基线校正方法对光谱数据进行基线校正。
        支持的方法包括：
        - Rubberband（橡皮筋法）
        - Modified Polynomial（修正多项式）
        - Iterative Modified Polynomial（自适应迭代多项式）
        - Whittaker-ASLS（Whittaker平滑与非对称最小二乘）
        - Mixture Model（混合模型/平滑样条）

        可以选择使用原始数据或平滑后的数据进行校正。
        """
        if not self.check_data_loaded():
            return

        try:
            method = self.baseline_method.get()
            data_source = self.data_source_var.get()

            # 选择数据源
            if data_source == "smoothed" and self.smoothed_data is not None:
                y_data = self.smoothed_data
            else:
                y_data = self.y_data

            # 准备参数
            params = {}
            if method == "modpoly":
                params['poly_order'] = int(self.poly_order_var.get())
            elif method == "imodpoly":
                params['poly_order'] = int(self.poly_order_var.get())
                params['max_iter'] = int(self.num_iter_var.get())
            elif method == "asls":
                params['lam'] = float(self.lam_var.get())
                params['p'] = float(self.p_var.get())
            elif method == "mixture_model":
                params['num_knots'] = int(self.num_knots_var.get())

            # 使用BaselineCorrector进行基线校正
            success, corrected_data, baseline, error_msg = self.baseline_corrector.correct_baseline(
                self.x_data, y_data, method, **params
            )

            if success:
                self.data_manager.set_corrected_data(corrected_data)
                self.plot_baseline_result(baseline, y_data, self.x_data)
                messagebox.showinfo("成功", "基线校正完成")
            else:
                messagebox.showerror("错误", error_msg)

        except ValueError as e:
            messagebox.showerror("参数错误", f"参数格式不正确：{str(e)}")
        except Exception as e:
            messagebox.showerror("错误", f"基线校正出错：{str(e)}")

    def plot_data(self):
        # 检查数据是否已加载
        if self.x_data is None or self.y_data is None:
            logger.warning("plot_data: 数据未加载，无法绘制图谱")
            return

        logger.info(f"plot_data: 开始绘制图谱，数据点数: {len(self.x_data)}")

        # 获取文件名用于图例
        file_name = self.current_file_name if self.current_file_name else '数据'

        # 更新平滑处理页面的图形
        self.smooth_ax1.clear()
        self.smooth_ax2.clear()
        self.smooth_ax1.plot(self.x_data, self.y_data, 'b-', label=file_name)
        self.smooth_ax1.set_title('原始数据')
        self.smooth_ax1.set_xlabel('波数 (cm$^{-1}$)')
        self.smooth_ax1.set_ylabel(self.y_label_var.get())
        self.smooth_ax1.legend()
        self.smooth_ax1.grid(True)
        # 倒置横坐标（FTIR标准：高波数在左，低波数在右）
        self.smooth_ax1.invert_xaxis()

        # 绘制选中的区间高亮
        self._draw_smooth_ranges()

        # 如果存在平滑数据，则显示
        if self.smoothed_data is not None:
            self.smooth_ax2.plot(self.x_data, self.smoothed_data, 'r-', label=f'{file_name}_平滑')
            self.smooth_ax2.set_title('平滑后数据')
            self.smooth_ax2.set_xlabel('波数 (cm$^{-1}$)')
            self.smooth_ax2.set_ylabel(self.y_label_var.get())
            self.smooth_ax2.legend()
            self.smooth_ax2.grid(True)
            # 倒置横坐标
            self.smooth_ax2.invert_xaxis()

        self.smooth_fig.tight_layout()
        self.smooth_canvas.draw()
        logger.info("plot_data: 平滑处理页面图谱绘制完成")

        # 更新基线校正页面的图形
        self.baseline_ax1.clear()
        self.baseline_ax2.clear()

        # 根据数据源选择显示的数据
        if self.data_source_var.get() == "smoothed" and self.smoothed_data is not None:
            plot_data = self.smoothed_data
            data_label = f'{file_name}_平滑'
        else:
            plot_data = self.y_data
            data_label = file_name

        self.baseline_ax1.plot(self.x_data, plot_data, 'b-', label=data_label)
        self.baseline_ax1.set_title(data_label)
        self.baseline_ax1.set_xlabel('波数 (cm$^{-1}$)')
        self.baseline_ax1.set_ylabel(self.y_label_var.get())
        self.baseline_ax1.legend()
        self.baseline_ax1.grid(True)
        # 倒置横坐标
        self.baseline_ax1.invert_xaxis()
        self.baseline_fig.tight_layout()
        self.baseline_canvas.draw()
        logger.info("plot_data: 基线校正页面图谱绘制完成")

    def plot_smooth_result(self):
        # 获取文件名用于图例
        file_name = self.current_file_name if self.current_file_name else '数据'

        self.smooth_ax2.clear()
        self.smooth_ax2.plot(self.x_data, self.smoothed_data, 'r-', label=f'{file_name}_平滑')
        self.smooth_ax2.set_title('平滑后数据')
        self.smooth_ax2.set_xlabel('波数 (cm$^{-1}$)')
        self.smooth_ax2.set_ylabel(self.y_label_var.get())
        self.smooth_ax2.legend()
        self.smooth_ax2.grid(True)
        # 倒置横坐标
        self.smooth_ax2.invert_xaxis()
        self.smooth_fig.tight_layout()
        self.smooth_canvas.draw()

    def plot_baseline_result(self, baseline, plot_data, x_data):
        self.baseline_ax1.clear()
        self.baseline_ax2.clear()

        # 获取文件名用于图例
        file_name = self.current_file_name if self.current_file_name else '数据'

        # 根据数据源选择显示正确的标签
        if self.data_source_var.get() == "smoothed":
            data_label = f'{file_name}_平滑'
        else:
            data_label = file_name

        # 绘制数据和基线
        self.baseline_ax1.plot(x_data, plot_data, 'b-', label=data_label)
        self.baseline_ax1.plot(x_data, baseline, 'r--', label='基线')
        self.baseline_ax1.set_title(f'{data_label}和基线')
        self.baseline_ax1.set_xlabel('波数 (cm$^{-1}$)')
        self.baseline_ax1.set_ylabel(self.y_label_var.get())
        self.baseline_ax1.legend()
        self.baseline_ax1.grid(True)
        # 倒置横坐标
        self.baseline_ax1.invert_xaxis()

        # 绘制校正后的数据
        self.baseline_ax2.plot(x_data, self.corrected_data, 'g-', label=f'{file_name}_基线校正')
        self.baseline_ax2.set_title('基线校正后的数据')
        self.baseline_ax2.set_xlabel('波数 (cm$^{-1}$)')
        self.baseline_ax2.set_ylabel(self.y_label_var.get())
        self.baseline_ax2.legend()
        self.baseline_ax2.grid(True)
        # 倒置横坐标
        self.baseline_ax2.invert_xaxis()

        self.baseline_fig.tight_layout()
        self.baseline_canvas.draw()

    def export_smooth_data(self):
        """导出平滑后的数据"""
        if not self.check_data_loaded('smoothed'):
            return

        # 生成默认文件名
        if self.current_file_name:
            default_filename = f"{self.current_file_name}_平滑处理.csv"
        else:
            default_filename = "平滑处理.csv"

        # 默认保存到 data/output 文件夹
        initial_dir = self.output_dir if os.path.exists(self.output_dir) else os.getcwd()
        default_path = os.path.join(initial_dir, default_filename)

        file_path = filedialog.asksaveasfilename(
            title="保存平滑处理后的数据",
            initialdir=initial_dir,
            initialfile=default_filename,
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if not file_path:
            return

        # 使用DataManager导出数据
        success, message = self.data_manager.export_to_csv(file_path, 'smoothed')

        if success:
            logger.info(f"平滑数据已导出: {os.path.basename(file_path)}")
            messagebox.showinfo("成功", message)
        else:
            messagebox.showerror("错误", message)

    def export_baseline_data(self):
        """导出基线校正后的数据"""
        if not self.check_data_loaded('corrected'):
            return

        # 生成默认文件名
        if self.current_file_name:
            default_filename = f"{self.current_file_name}_基线校正.csv"
        else:
            default_filename = "基线校正.csv"

        # 默认保存到 data/output 文件夹
        initial_dir = self.output_dir if os.path.exists(self.output_dir) else os.getcwd()
        default_path = os.path.join(initial_dir, default_filename)

        file_path = filedialog.asksaveasfilename(
            title="保存基线校正后的数据",
            initialdir=initial_dir,
            initialfile=default_filename,
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if not file_path:
            return

        # 使用DataManager导出数据
        success, message = self.data_manager.export_to_csv(file_path, 'corrected')

        if success:
            logger.info(f"基线校正数据已导出: {os.path.basename(file_path)}")
            messagebox.showinfo("成功", message)
        else:
            messagebox.showerror("错误", message)

    def update_baseline_plot(self):
        """更新基线校正页面的图形显示"""
        if not self.check_data_loaded():
            return

        self.baseline_ax1.clear()
        self.baseline_ax2.clear()

        # 获取文件名用于图例
        file_name = self.current_file_name if self.current_file_name else '数据'

        # 根据数据源选择显示的数据
        if self.data_source_var.get() == "smoothed" and self.smoothed_data is not None:
            plot_data = self.smoothed_data
            data_label = f'{file_name}_平滑'
        else:
            plot_data = self.y_data
            data_label = file_name

        self.baseline_ax1.plot(self.x_data, plot_data, 'b-', label=data_label)
        self.baseline_ax1.set_title(data_label)
        self.baseline_ax1.set_xlabel('波数 (cm$^{-1}$)')
        self.baseline_ax1.set_ylabel(self.y_label_var.get())
        self.baseline_ax1.legend()
        self.baseline_ax1.grid(True)
        # 倒置横坐标
        self.baseline_ax1.invert_xaxis()

        self.baseline_fig.tight_layout()
        self.baseline_canvas.draw()

    def update_file_display(self, filename):
        """更新文件名显示"""
        self.current_file_var.set(filename)
        if hasattr(self, 'file_label'):
            self.file_label.config(text=filename)
        self.root.update_idletasks()





    def create_peak_analysis_page(self):
        """创建特征峰分析页面"""
        # 创建主 PanedWindow（竖向分割：上方区域 + 下方分析结果）
        main_paned = ttk.PanedWindow(self.peak_analysis_frame, orient=tk.VERTICAL)
        main_paned.pack(fill=tk.BOTH, expand=True)

        # 创建上方区域（左侧控制面板 + 右侧图形）
        top_frame = ttk.Frame(main_paned)
        main_paned.add(top_frame, weight=3)  # 占60%空间

        # 创建左侧控制面板
        control_frame = ttk.Frame(top_frame)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # 创建右侧图形区域
        plot_frame = ttk.Frame(top_frame)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 数据加载框架
        load_frame = ttk.LabelFrame(control_frame, text="数据加载")
        load_frame.pack(fill=tk.X, padx=5, pady=5)

        # 统一的加载数据按钮（支持单选和多选）
        ttk.Button(load_frame, text="加载数据（可多选）",
                   command=self.load_multiple_datasets).pack(fill=tk.X, padx=5, pady=2)

        # 数据集管理框架
        datasets_frame = ttk.LabelFrame(control_frame, text="已加载数据集")
        datasets_frame.pack(fill=tk.X, padx=5, pady=5)

        # 数据集列表（使用Treeview支持复选框）
        datasets_list_frame = ttk.Frame(datasets_frame)
        datasets_list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 创建Treeview控件
        columns = ('color', 'name')
        self.datasets_tree = ttk.Treeview(datasets_list_frame, columns=columns,
                                         show='tree headings', height=4, selectmode=tk.EXTENDED)

        # 设置列标题
        self.datasets_tree.heading('#0', text='✓', anchor='w')  # 复选框列，标题居左
        self.datasets_tree.heading('color', text='图例')
        self.datasets_tree.heading('name', text='数据集名称')

        # 设置列宽
        self.datasets_tree.column('#0', width=35, anchor='w')  # 复选框列居左，稍微加宽以显示完整
        self.datasets_tree.column('color', width=40, anchor='center')
        self.datasets_tree.column('name', width=180, anchor='w')

        # 添加滚动条
        datasets_scrollbar = ttk.Scrollbar(datasets_list_frame, orient='vertical',
                                          command=self.datasets_tree.yview)
        self.datasets_tree.configure(yscrollcommand=datasets_scrollbar.set)

        self.datasets_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        datasets_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 绑定点击事件（用于切换复选框状态）
        self.datasets_tree.bind('<Button-1>', self.on_dataset_click)

        # 数据集管理按钮
        datasets_btn_frame = ttk.Frame(datasets_frame)
        datasets_btn_frame.pack(fill=tk.X, padx=5, pady=2)

        ttk.Button(datasets_btn_frame, text="移除选中",
                   command=self.remove_selected_dataset).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        ttk.Button(datasets_btn_frame, text="清空所有",
                   command=self.clear_all_datasets).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

        # 初始化数据选择变量（固定为原始数据，不显示UI选择框）
        self.peak_data_var = tk.StringVar(value="original")
        
        # 峰检测设置框架
        peak_settings_frame = ttk.LabelFrame(control_frame, text="峰检测设置")
        peak_settings_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 阈值设置
        threshold_frame = ttk.Frame(peak_settings_frame)
        threshold_frame.pack(fill=tk.X, pady=2)
        ttk.Label(threshold_frame, text="阈值:").pack(side=tk.LEFT)
        self.peak_threshold_var = tk.StringVar(value="0.02")
        ttk.Entry(threshold_frame, textvariable=self.peak_threshold_var, width=10).pack(side=tk.LEFT)
        
        # 最小距离设置
        distance_frame = ttk.Frame(peak_settings_frame)
        distance_frame.pack(fill=tk.X, pady=2)
        ttk.Label(distance_frame, text="最小距离:").pack(side=tk.LEFT)
        self.peak_distance_var = tk.StringVar(value="10")
        ttk.Entry(distance_frame, textvariable=self.peak_distance_var, width=10).pack(side=tk.LEFT)
        
        # 寻峰按钮
        self.find_peaks_btn = ttk.Button(peak_settings_frame, text="寻找峰",
                                        command=self.find_peaks)
        self.find_peaks_btn.pack(fill=tk.X, pady=2)

        # 提示标签（使用系统默认字体以保持一致性）
        self.peak_hint_label = ttk.Label(peak_settings_frame, text="", foreground="red")
        self.peak_hint_label.pack(fill=tk.X, pady=2)

        # ========== 峰列表区域 ==========
        peaks_frame = ttk.LabelFrame(control_frame, text="峰列表")
        peaks_frame.pack(fill=tk.X, padx=5, pady=5)

        # 创建带滚动条的Treeview列表
        list_frame = ttk.Frame(peaks_frame)
        list_frame.pack(fill=tk.BOTH, padx=5, pady=5)

        # 创建Treeview控件（高度从10增加到15，支持多选）
        columns = ('filename', 'wavenumber', 'height')
        self.peaks_tree = ttk.Treeview(list_frame, columns=columns, show='headings',
                                      height=15, selectmode=tk.EXTENDED)

        # 设置列标题
        self.peaks_tree.heading('filename', text='文件名')
        self.peaks_tree.heading('wavenumber', text='波数(cm⁻¹)')
        self.peaks_tree.heading('height', text='峰高')

        # 设置列宽和对齐方式
        self.peaks_tree.column('filename', width=150, anchor='w')
        self.peaks_tree.column('wavenumber', width=120, anchor='center')
        self.peaks_tree.column('height', width=100, anchor='center')

        # 添加滚动条
        scrollbar = ttk.Scrollbar(list_frame, orient='vertical', command=self.peaks_tree.yview)
        self.peaks_tree.configure(yscrollcommand=scrollbar.set)

        self.peaks_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 添加选择事件绑定
        self.peaks_tree.bind('<<TreeviewSelect>>', self.on_peak_select)

        # 峰列表操作按钮框架
        peaks_btn_frame = ttk.Frame(peaks_frame)
        peaks_btn_frame.pack(fill=tk.X, padx=5, pady=2)

        # 取消选择按钮（移除expand=True，让按钮自动调整宽度）
        ttk.Button(peaks_btn_frame, text="取消选择",
                   command=self.clear_peak_selection).pack(side=tk.LEFT, padx=(0, 2))

        # 导出按钮（移除expand=True，让按钮自动调整宽度）
        ttk.Button(peaks_btn_frame, text="导出峰列表",
                   command=self.export_peak_list).pack(side=tk.LEFT, padx=(2, 0))

        # ========== 峰分析设置区域 ==========
        analysis_frame = ttk.LabelFrame(control_frame, text="峰分析设置")
        analysis_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 交互式选择模式
        interactive_frame = ttk.Frame(analysis_frame)
        interactive_frame.pack(fill=tk.X, padx=5, pady=5)

        self.peak_interactive_mode_var = tk.BooleanVar(value=False)
        self.peak_interactive_btn = ttk.Checkbutton(
            interactive_frame,
            text="交互式选择",
            variable=self.peak_interactive_mode_var,
            command=self.toggle_peak_interactive_mode
        )
        self.peak_interactive_btn.pack(side=tk.LEFT)

        # 提示标签
        self.peak_interactive_hint_label = ttk.Label(
            interactive_frame,
            text="",
            font=('', 8),
            foreground='blue'
        )
        self.peak_interactive_hint_label.pack(side=tk.LEFT, padx=5)

        # 固定积分区间复选框
        fixed_range_frame = ttk.Frame(analysis_frame)
        fixed_range_frame.pack(fill=tk.X, pady=2)

        ttk.Checkbutton(fixed_range_frame, text="固定积分区间",
                       variable=self.fixed_integration_range).pack(side=tk.LEFT)

        # 波数范围选择
        lower_frame = ttk.Frame(analysis_frame)
        lower_frame.pack(fill=tk.X, pady=2)
        ttk.Label(lower_frame, text="下限:").pack(side=tk.LEFT)
        self.peak_lower_var = tk.StringVar()
        self.peak_lower_var.trace_add("write", self.on_range_change)  # 添加回调
        ttk.Entry(lower_frame, textvariable=self.peak_lower_var, width=10).pack(side=tk.LEFT, padx=2)

        upper_frame = ttk.Frame(analysis_frame)
        upper_frame.pack(fill=tk.X, pady=2)
        ttk.Label(upper_frame, text="上限:").pack(side=tk.LEFT)
        self.peak_upper_var = tk.StringVar()
        self.peak_upper_var.trace_add("write", self.on_range_change)  # 添加回调
        ttk.Entry(upper_frame, textvariable=self.peak_upper_var, width=10).pack(side=tk.LEFT, padx=2)

        # 分析按钮框架
        btn_frame = ttk.Frame(analysis_frame)
        btn_frame.pack(fill=tk.X, pady=2)

        # 添加到分析列表按钮（主要按钮，移除emoji和expand=True）
        ttk.Button(btn_frame, text="添加到分析列表",
                   command=self.add_peak_to_analysis).pack(side=tk.LEFT, padx=(0, 2))

        # 分析选中峰按钮（保留，用于重新分析，移除emoji和expand=True）
        ttk.Button(btn_frame, text="重新分析",
                   command=self.analyze_selected_peak).pack(side=tk.LEFT, padx=(2, 0))

        # ========== 下方分析结果区域 ==========
        result_frame = ttk.LabelFrame(main_paned, text="分析结果")
        main_paned.add(result_frame, weight=2)  # 占40%空间

        # 创建表格容器框架（包含表格和滚动条）
        result_tree_frame = ttk.Frame(result_frame)
        result_tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 创建表格（添加区间列）
        columns = ('文件名', '编号', '波数', '峰高', '校正峰高', '区间下限', '区间上限', '面积', '校正面积')
        self.result_tree = ttk.Treeview(result_tree_frame, columns=columns, show='headings', height=8)

        # 定义列标题和宽度（波数不显示单位）
        self.result_tree.heading('文件名', text='文件名')
        self.result_tree.heading('编号', text='编号')
        self.result_tree.heading('波数', text='波数')
        self.result_tree.heading('峰高', text='峰高')
        self.result_tree.heading('校正峰高', text='校正峰高')
        self.result_tree.heading('区间下限', text='区间下限')
        self.result_tree.heading('区间上限', text='区间上限')
        self.result_tree.heading('面积', text='面积')
        self.result_tree.heading('校正面积', text='校正面积')

        self.result_tree.column('文件名', width=120, anchor='w')
        self.result_tree.column('编号', width=50, anchor='center')
        self.result_tree.column('波数', width=70, anchor='center')
        self.result_tree.column('峰高', width=70, anchor='center')
        self.result_tree.column('校正峰高', width=80, anchor='center')
        self.result_tree.column('区间下限', width=80, anchor='center')
        self.result_tree.column('区间上限', width=80, anchor='center')
        self.result_tree.column('面积', width=70, anchor='center')
        self.result_tree.column('校正面积', width=80, anchor='center')

        # 添加滚动条
        tree_scroll = ttk.Scrollbar(result_tree_frame, orient='vertical', command=self.result_tree.yview)
        self.result_tree.configure(yscrollcommand=tree_scroll.set)

        self.result_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # 配置斑马纹背景色（模拟网格线效果）
        self.result_tree.tag_configure('evenrow', background='white')
        self.result_tree.tag_configure('oddrow', background='#F5F5F5')

        # 绑定双击事件，用于复制单元格值
        self.result_tree.bind('<Double-Button-1>', self.on_result_cell_double_click)

        # 绑定单击事件，用于自动填充参数
        self.result_tree.bind('<Button-1>', self.on_result_tree_click)

        # 绑定右键点击事件，用于显示删除菜单
        self.result_tree.bind('<Button-3>', self.on_result_tree_right_click)

        # 分析结果操作按钮框架（在表格下方）
        result_btn_frame = ttk.Frame(result_frame)
        result_btn_frame.pack(fill=tk.X, padx=5, pady=(2, 5))

        # 移除expand=True，让按钮自动调整宽度
        ttk.Button(result_btn_frame, text="清空表格",
                   command=self.clear_result_table).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Button(result_btn_frame, text="导出结果",
                   command=self.export_peak_analysis_results).pack(side=tk.LEFT, padx=(2, 0))

        # 创建缩放工具栏
        zoom_toolbar_frame = ttk.LabelFrame(plot_frame, text="图形缩放工具")
        zoom_toolbar_frame.pack(fill=tk.X, padx=5, pady=5)

        # 工具模式按钮（互斥）
        self.peak_tool_mode = tk.StringVar(value="")  # 默认不选择任何工具

        ttk.Radiobutton(zoom_toolbar_frame, text="🔲 矩形选框",
                       variable=self.peak_tool_mode, value="rect_zoom",
                       command=self.switch_peak_tool_mode).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(zoom_toolbar_frame, text="🖐️ 平移",
                       variable=self.peak_tool_mode, value="pan",
                       command=self.switch_peak_tool_mode).pack(side=tk.LEFT, padx=2)

        # 分隔线
        ttk.Separator(zoom_toolbar_frame, orient='vertical').pack(side=tk.LEFT, fill=tk.Y, padx=5)

        # 历史记录按钮
        self.peak_back_btn = ttk.Button(zoom_toolbar_frame, text="⬅️ 后退",
                                        command=self.zoom_history_back, state='disabled')
        self.peak_back_btn.pack(side=tk.LEFT, padx=2)

        self.peak_forward_btn = ttk.Button(zoom_toolbar_frame, text="➡️ 前进",
                                           command=self.zoom_history_forward, state='disabled')
        self.peak_forward_btn.pack(side=tk.LEFT, padx=2)

        ttk.Button(zoom_toolbar_frame, text="🏠 重置",
                   command=self.reset_zoom_peak).pack(side=tk.LEFT, padx=2)

        # 创建图形
        self.peak_fig, self.peak_ax = plt.subplots(figsize=(8, 6))
        self.peak_canvas = FigureCanvasTkAgg(self.peak_fig, master=plot_frame)
        self.peak_canvas.draw()
        self.peak_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 添加工具栏
        toolbar = NavigationToolbar2Tk(self.peak_canvas, plot_frame)

        # 保存原始数据范围（用于重置）
        self.peak_original_xlim = None
        self.peak_original_ylim = None

        # 初始化缩放历史记录
        self.peak_zoom_history = []  # 存储 (xlim, ylim) 元组
        self.peak_zoom_history_index = -1  # 当前历史记录索引

        # 初始化矩形选框工具
        from matplotlib.widgets import RectangleSelector
        self.peak_rect_selector = None

        # 绑定鼠标事件
        self.peak_canvas.mpl_connect('motion_notify_event', self.on_peak_mouse_move)
        self.peak_canvas.mpl_connect('scroll_event', self.on_peak_scroll)
        self.peak_canvas.mpl_connect('button_press_event', self.on_peak_button_press)
        self.peak_canvas.mpl_connect('button_release_event', self.on_peak_button_release)

        # 初始化峰信息提示框
        self.peak_tooltip = None

        # 初始化平移相关变量
        self.peak_pan_start = None  # 平移起始位置
        self.peak_is_panning = False  # 是否正在平移

        # 创建峰分析右键菜单
        self.create_peak_context_menu()

        # 【修复】移除 Tkinter 的右键事件绑定，避免与 Matplotlib 事件冲突
        # 右键事件已经在 on_peak_button_press() 中通过 Matplotlib 事件处理
        # self.peak_canvas.get_tk_widget().bind('<Button-3>', self.on_peak_canvas_right_click)

        # 初始化工具模式（默认不启用任何工具，用户可自行选择）
        self.switch_peak_tool_mode()
        toolbar.update()

        # 初始化显示原始数据
        self.update_peak_plot()

    def on_range_change(self, *args):  # args用于Tkinter变量trace回调
        """
        当上下限输入框的值改变时更新图形

        Args:
            *args: Tkinter变量trace回调的标准参数（未使用但必须保留）
        """
        self.update_peak_plot()

    def create_log_management_page(self):
        """创建日志管理页面"""
        # 主容器
        main_container = ttk.Frame(self.log_management_frame)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # ========== 操作按钮区域 ==========
        btn_frame = ttk.LabelFrame(main_container, text="操作")
        btn_frame.pack(fill=tk.X, pady=(0, 10))

        btn_inner_frame = ttk.Frame(btn_frame)
        btn_inner_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(btn_inner_frame, text="刷新日志",
                   command=self.refresh_log).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_inner_frame, text="清空日志",
                   command=self.clear_log).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_inner_frame, text="导出日志",
                   command=self.export_log).pack(side=tk.LEFT)

        # ========== 筛选区域 ==========
        filter_frame = ttk.LabelFrame(main_container, text="筛选")
        filter_frame.pack(fill=tk.X, pady=(0, 10))

        filter_inner_frame = ttk.Frame(filter_frame)
        filter_inner_frame.pack(fill=tk.X, padx=5, pady=5)

        # 日志级别筛选
        ttk.Label(filter_inner_frame, text="日志级别:").pack(side=tk.LEFT, padx=(0, 5))
        self.log_level_var = tk.StringVar(value="全部")
        log_level_combo = ttk.Combobox(filter_inner_frame, textvariable=self.log_level_var,
                                       values=["全部", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                                       state='readonly', width=12)
        log_level_combo.pack(side=tk.LEFT, padx=(0, 20))
        log_level_combo.bind('<<ComboboxSelected>>', lambda e: self.filter_log())

        # 搜索框
        ttk.Label(filter_inner_frame, text="搜索关键词:").pack(side=tk.LEFT, padx=(0, 5))
        self.log_search_var = tk.StringVar()
        self.log_search_var.trace_add("write", lambda *args: self.filter_log())
        search_entry = ttk.Entry(filter_inner_frame, textvariable=self.log_search_var, width=30)
        search_entry.pack(side=tk.LEFT)

        # ========== 日志显示区域 ==========
        log_display_frame = ttk.LabelFrame(main_container, text="日志内容")
        log_display_frame.pack(fill=tk.BOTH, expand=True)

        # 创建Text控件和滚动条的容器
        text_container = ttk.Frame(log_display_frame)
        text_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 使用Text控件显示日志
        # 使用系统默认字体，与其他界面元素保持一致
        self.log_text = tk.Text(text_container, wrap=tk.NONE)

        # 添加滚动条
        log_scroll_y = ttk.Scrollbar(text_container, orient='vertical', command=self.log_text.yview)
        log_scroll_x = ttk.Scrollbar(text_container, orient='horizontal', command=self.log_text.xview)
        self.log_text.configure(yscrollcommand=log_scroll_y.set, xscrollcommand=log_scroll_x.set)

        # 使用grid布局
        self.log_text.grid(row=0, column=0, sticky='nsew')
        log_scroll_y.grid(row=0, column=1, sticky='ns')
        log_scroll_x.grid(row=1, column=0, sticky='ew')

        text_container.grid_rowconfigure(0, weight=1)
        text_container.grid_columnconfigure(0, weight=1)

        # 配置日志文本颜色标签（使用系统默认字体）
        self.log_text.tag_configure('DEBUG', foreground='gray')
        self.log_text.tag_configure('INFO', foreground='black')
        self.log_text.tag_configure('WARNING', foreground='orange')
        self.log_text.tag_configure('ERROR', foreground='red')
        self.log_text.tag_configure('CRITICAL', foreground='darkred', font=('', 0, 'bold'))

        # 初始加载日志
        self.refresh_log()

    def update_peak_plot(self):
        """更新特征峰分析图形（支持多数据集显示）"""
        logger.info(f"update_peak_plot: 开始更新图形，总数据集数: {len(self.loaded_datasets)}")

        # 打印所有数据集的勾选状态
        for idx, ds in enumerate(self.loaded_datasets):
            logger.info(f"  数据集 {idx}: '{ds['name']}', checked={ds.get('checked', True)}")

        # 检查是否有勾选的数据集
        checked_datasets = [ds for ds in self.loaded_datasets if ds.get('checked', True)]
        logger.info(f"update_peak_plot: 勾选的数据集数量: {len(checked_datasets)}")

        # 在 clear() 之前保存当前的视图范围（只在有勾选的数据集时）
        # 关键：必须在 clear() 之前获取范围，否则会得到错误的默认范围
        if checked_datasets:
            current_xlim = self.peak_ax.get_xlim()
            current_ylim = self.peak_ax.get_ylim()

            # 检查是否是空图的默认 X 轴范围（通常是 (1.0, 0.0) 或 (0.0, 1.0)）
            # 注意：只检查 X 轴范围，不检查 Y 轴范围
            # 因为 Y 轴范围 (0.0, 1.0) 可能是有效的数据范围
            is_default_xlim = (current_xlim == (1.0, 0.0) or current_xlim == (0.0, 1.0))

            if is_default_xlim:
                logger.info(f"update_peak_plot: 检测到默认 X 轴范围 xlim={current_xlim}，不保存")
                current_xlim = None
                current_ylim = None
            else:
                logger.info(f"update_peak_plot: 保存当前视图范围 xlim={current_xlim}, ylim={current_ylim}")
        else:
            current_xlim = None
            current_ylim = None
            logger.info("update_peak_plot: 没有勾选的数据集，不保存视图范围")

        # 清空坐标轴
        self.peak_ax.clear()

        # 如果存在 SpanSelector，需要重新创建（因为 clear() 会移除它）
        need_recreate_span_selector = False
        if hasattr(self, 'peak_span_selector') and self.peak_span_selector is not None:
            if self.peak_interactive_mode:
                need_recreate_span_selector = True
            # 先移除旧的 SpanSelector
            self.peak_span_selector.set_active(False)
            self.peak_span_selector = None

        if not checked_datasets:
            # 没有勾选的数据集，显示空图
            logger.info("update_peak_plot: 显示空图")

            # 清空峰列表（因为没有数据集）
            if hasattr(self, 'peaks_tree'):
                for item in self.peaks_tree.get_children():
                    self.peaks_tree.delete(item)
                logger.info("update_peak_plot: 已清空峰列表（无勾选数据集）")

            self.peak_ax.set_xlabel('波数 (cm$^{-1}$)')
            self.peak_ax.set_ylabel('吸光度')
            self.peak_ax.set_xlim(4000, 400)  # 设置默认横坐标范围（左大右小）
            self.peak_ax.grid(True)
            self.peak_fig.tight_layout()
            self.peak_canvas.draw()
            return

        # 显示勾选的数据集
        logger.info(f"update_peak_plot: 开始绘制 {len(checked_datasets)} 个数据集")
        for dataset in checked_datasets:
            # 找到该数据集在 loaded_datasets 中的原始索引，以保持颜色一致
            original_idx = next(i for i, ds in enumerate(self.loaded_datasets) if ds['name'] == dataset['name'])
            color = self.dataset_colors[original_idx % len(self.dataset_colors)]
            logger.info(f"  绘制数据集: '{dataset['name']}', 原始索引: {original_idx}, 颜色: {color}, 数据点数: {len(dataset['x_data'])}")
            self.peak_ax.plot(dataset['x_data'], dataset['y_data'],
                            color=color, label=dataset['name'], linewidth=1.5)

        # 如果只有一个数据集被勾选，使用它作为当前数据
        if len(checked_datasets) == 1:
            self.data_manager.x_data = checked_datasets[0]['x_data']
            self.data_manager.y_data = checked_datasets[0]['y_data']
            y_data = checked_datasets[0]['y_data']
        else:
            # 多个数据集，不显示峰标记
            y_data = None
        
        # 只在单个数据集时显示峰标记
        if y_data is not None and hasattr(self, 'x_data') and self.x_data is not None:
            # 获取当前选中的峰
            selected_items = self.peaks_tree.selection()

            # 绘制所有峰值点
            all_items = self.peaks_tree.get_children()
            for idx, item in enumerate(all_items):
                values = self.peaks_tree.item(item, 'values')
                # values[0] 是文件名，values[1] 是波数，values[2] 是峰高
                peak_wavenumber = float(values[1])
                peak_idx = np.argmin(np.abs(self.x_data - peak_wavenumber))
                peak_height = y_data[peak_idx]

                if item in selected_items:
                    # 选中的峰用绿色圆点标记
                    self.peak_ax.plot(peak_wavenumber, peak_height, 'go',
                                    markersize=8, label='选中峰' if item == selected_items[0] else "")
                else:
                    # 未选中的峰用蓝色圆点标记
                    self.peak_ax.plot(peak_wavenumber, peak_height, 'bo',
                                    markersize=8, label='峰值' if idx == 0 and not selected_items else "")

            # 为所有选中的峰绘制垂直虚线（从X轴延伸到峰高位置）
            if selected_items:
                for idx, item in enumerate(selected_items):
                    values = self.peaks_tree.item(item, 'values')
                    # values[0] 是文件名，values[1] 是波数，values[2] 是峰高
                    peak_wavenumber = float(values[1])
                    peak_idx = np.argmin(np.abs(self.x_data - peak_wavenumber))
                    peak_height = y_data[peak_idx]

                    # 绘制垂直虚线（浅灰色，半透明）
                    self.peak_ax.axvline(x=peak_wavenumber, ymin=0, ymax=1,
                                       color='gray', linestyle=':', alpha=0.5, linewidth=1.5,
                                       label='选中峰标记' if idx == 0 else "")
        
            # 绘制上下限虚线、连接线和积分区域填充（实时预览）
            try:
                if self.peak_lower_var.get() and self.peak_upper_var.get():
                    lower = float(self.peak_lower_var.get())
                    upper = float(self.peak_upper_var.get())

                    # 确保lower < upper
                    if lower > upper:
                        lower, upper = upper, lower

                    # 获取积分范围内的数据
                    mask = (self.x_data >= lower) & (self.x_data <= upper)
                    x_range = self.x_data[mask]
                    y_range = y_data[mask]

                    if len(x_range) > 0:
                        # 找到上下限对应的y值
                        lower_idx = np.argmin(np.abs(self.x_data - lower))
                        upper_idx = np.argmin(np.abs(self.x_data - upper))
                        lower_y = y_data[lower_idx]
                        upper_y = y_data[upper_idx]

                        # 计算基线（连接两端点的直线）
                        baseline_slope = (upper_y - lower_y) / (upper - lower) if upper != lower else 0
                        baseline_intercept = lower_y - baseline_slope * lower
                        y_baseline = baseline_slope * x_range + baseline_intercept

                        # 填充积分区域（半透明黄色）
                        self.peak_ax.fill_between(x_range, y_baseline, y_range,
                                                 alpha=0.3, color='yellow', label='积分区域')

                        # 绘制竖向虚线（深灰色）
                        self.peak_ax.axvline(x=lower, color='dimgray', linestyle='--', alpha=0.8)
                        self.peak_ax.axvline(x=upper, color='dimgray', linestyle='--', alpha=0.8)

                        # 绘制基线（黑色虚线）
                        self.peak_ax.plot([lower, upper], [lower_y, upper_y],
                                        color='black', linestyle='--', alpha=0.8, label='基线')

                        # 计算并显示预估面积
                        # 【修复】兼容 NumPy 旧版本，使用 trapz 而不是 trapezoid
                        try:
                            corrected_area = np.trapezoid(y_range - y_baseline, x_range)
                        except AttributeError:
                            corrected_area = np.trapz(y_range - y_baseline, x_range)

                        # 在积分区域旁边显示面积值
                        mid_x = (lower + upper) / 2
                        mid_y = np.max(y_range) * 1.05  # 稍微高于峰顶
                        self.peak_ax.text(mid_x, mid_y, f'面积: {corrected_area:.2f}',
                                        ha='center', va='bottom', fontsize=9,
                                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))

            except ValueError:
                pass  # 忽略无效的输入值

            # 绘制已分析的区间
            self.draw_analyzed_ranges_on_plot()

        self.peak_ax.set_xlabel('波数 (cm$^{-1}$)')
        self.peak_ax.set_ylabel('吸光度')
        self.peak_ax.legend()
        self.peak_ax.grid(True)

        # 【修复】检查是否需要恢复或调整视图范围
        # 关键：不再使用 invert_xaxis()，而是直接设置 xlim 为倒序（左大右小）
        if self.peak_original_xlim is None:
            # 第一次绘制，手动设置 Y 轴范围以确保数据完整显示
            # 收集所有勾选数据集的 Y 值，计算全局的最小值和最大值
            all_y_values = []
            for dataset in checked_datasets:
                all_y_values.extend(dataset['y_data'])

            if len(all_y_values) > 0:
                y_min = np.min(all_y_values)
                y_max = np.max(all_y_values)
                # 添加 5% 的边距，确保数据不会紧贴坐标轴边缘
                y_range = y_max - y_min
                y_margin = y_range * 0.05 if y_range > 0 else 0.05
                y_lim_lower = y_min - y_margin
                y_lim_upper = y_max + y_margin
                self.peak_ax.set_ylim(y_lim_lower, y_lim_upper)
                logger.info(f"update_peak_plot: 第一次绘制，手动设置 Y 轴范围: [{y_lim_lower:.4f}, {y_lim_upper:.4f}]")

            # 【修复】第一次绘制时，直接设置 X 轴范围为 FTIR 标准倒序（左大右小）
            all_x_values = []
            for dataset in checked_datasets:
                all_x_values.extend(dataset['x_data'])
            if len(all_x_values) > 0:
                x_min = np.min(all_x_values)
                x_max = np.max(all_x_values)
                self.peak_ax.set_xlim(x_max, x_min)  # 左大右小（FTIR标准）
                logger.info(f"update_peak_plot: 第一次绘制，设置 X 轴范围: [{x_max:.2f}, {x_min:.2f}] (倒序)")

            # 在设置坐标轴范围后，调用 tight_layout()
            self.peak_fig.tight_layout()

            # 在 tight_layout 之后，重新设置 Y 轴范围（因为 tight_layout 可能会改变坐标轴范围）
            if len(all_y_values) > 0:
                self.peak_ax.set_ylim(y_lim_lower, y_lim_upper)
                # 【修复】同时确保 X 轴范围保持倒序
                if len(all_x_values) > 0:
                    self.peak_ax.set_xlim(x_max, x_min)
                logger.info(f"update_peak_plot: tight_layout 后重新设置范围: X=[{x_max:.2f}, {x_min:.2f}], Y=[{y_lim_lower:.4f}, {y_lim_upper:.4f}]")

                # 保存原始视图范围（在重新设置范围之后）
                self.peak_original_xlim = self.peak_ax.get_xlim()
                self.peak_original_ylim = self.peak_ax.get_ylim()
                logger.info(f"update_peak_plot: 第一次绘制，保存原始视图范围 xlim={self.peak_original_xlim}, ylim={self.peak_original_ylim}")

            # 初始化缩放历史记录（只在第一次绘制时）
            if len(self.peak_zoom_history) == 0:
                self.peak_zoom_history.append((self.peak_original_xlim, self.peak_original_ylim))
                self.peak_zoom_history_index = 0
                self.update_zoom_history_buttons()
        else:
            # 检查是否切换了数据集
            if hasattr(self, 'dataset_switched') and self.dataset_switched:
                # 数据集已切换，重置Y轴范围（使用自动范围）
                # 但保持X轴范围（如果用户之前缩放过）
                if current_xlim is not None and current_xlim[0] != current_xlim[1]:
                    # 【修复】确保 X 轴保持倒序（FTIR 标准：左大右小）
                    if current_xlim[0] < current_xlim[1]:
                        corrected_xlim = (current_xlim[1], current_xlim[0])
                        logger.info(f"update_peak_plot: 数据集切换，检测到正序X轴范围，已转换为倒序: {current_xlim} -> {corrected_xlim}")
                    else:
                        corrected_xlim = current_xlim
                    self.peak_ax.set_xlim(corrected_xlim)
                    logger.info(f"update_peak_plot: 数据集已切换，保持X轴范围 xlim={corrected_xlim}，Y轴使用自动范围")
                else:
                    # 【修复】没有保存的范围时，使用 FTIR 标准倒序
                    all_x_values = []
                    for dataset in checked_datasets:
                        all_x_values.extend(dataset['x_data'])
                    if len(all_x_values) > 0:
                        x_min = np.min(all_x_values)
                        x_max = np.max(all_x_values)
                        self.peak_ax.set_xlim(x_max, x_min)  # 左大右小
                    logger.info("update_peak_plot: 数据集已切换，X轴和Y轴都使用自动范围（X轴保持倒序）")

                # 重置数据集切换标志
                self.dataset_switched = False
            else:
                # 数据集未切换，恢复之前保存的视图范围（保持用户的缩放状态）
                # 只在有有效的保存范围时才恢复（current_xlim 和 current_ylim 不为 None）
                if current_xlim is not None and current_ylim is not None:
                    # 检查保存的范围是否有效（不是默认的自动范围）
                    if current_xlim[0] != current_xlim[1] and current_ylim[0] != current_ylim[1]:
                        # 【修复】确保 X 轴保持倒序（FTIR 标准：左大右小）
                        # 如果 current_xlim 是正序的，则强制转换为倒序
                        if current_xlim[0] < current_xlim[1]:
                            # 当前是正序（左小右大），需要转换为倒序
                            corrected_xlim = (current_xlim[1], current_xlim[0])
                            logger.info(f"update_peak_plot: 检测到正序X轴范围，已转换为倒序: {current_xlim} -> {corrected_xlim}")
                        else:
                            corrected_xlim = current_xlim
                        self.peak_ax.set_xlim(corrected_xlim)
                        self.peak_ax.set_ylim(current_ylim)
                        logger.info(f"update_peak_plot: 恢复视图范围 xlim={corrected_xlim}, ylim={current_ylim}")
                    else:
                        # 【修复】范围无效时，使用 FTIR 标准倒序
                        all_x_values = []
                        for dataset in checked_datasets:
                            all_x_values.extend(dataset['x_data'])
                        if len(all_x_values) > 0:
                            x_min = np.min(all_x_values)
                            x_max = np.max(all_x_values)
                            self.peak_ax.set_xlim(x_max, x_min)  # 左大右小
                        logger.info("update_peak_plot: 保存的范围无效，使用自动范围（X轴保持倒序）")
                else:
                    # 【修复】没有保存的范围时，使用 FTIR 标准倒序
                    all_x_values = []
                    for dataset in checked_datasets:
                        all_x_values.extend(dataset['x_data'])
                    if len(all_x_values) > 0:
                        x_min = np.min(all_x_values)
                        x_max = np.max(all_x_values)
                        self.peak_ax.set_xlim(x_max, x_min)  # 左大右小
                    logger.info("update_peak_plot: 没有保存的范围，使用自动范围（X轴保持倒序）")

            # 在设置坐标轴范围后，调用 tight_layout()
            self.peak_fig.tight_layout()

        # 如果需要重新创建 SpanSelector
        if need_recreate_span_selector:
            from matplotlib.widgets import SpanSelector
            self.peak_span_selector = SpanSelector(
                self.peak_ax,
                self.on_peak_span_select,
                'horizontal',
                useblit=True,
                props=dict(alpha=0.3, facecolor='yellow'),
                interactive=True,
                drag_from_anywhere=True
            )
            logger.info("update_peak_plot: 重新创建了 SpanSelector")

        self.peak_canvas.draw()
        logger.info("update_peak_plot: 图形绘制完成")

    def draw_analyzed_ranges_on_plot(self):
        """在峰分析图形上绘制已分析的区间（仅显示当前勾选数据集的区间）"""
        if not hasattr(self, 'analyzed_ranges') or not self.analyzed_ranges:
            return

        # 获取当前勾选的数据集名称列表
        checked_datasets = [ds for ds in self.loaded_datasets if ds.get('checked', True)]
        checked_file_names = [ds['name'] for ds in checked_datasets]

        if not checked_file_names:
            logger.info("draw_analyzed_ranges_on_plot: 没有勾选的数据集，不绘制区间")
            return

        logger.info(f"draw_analyzed_ranges_on_plot: 当前勾选的数据集: {checked_file_names}")

        # 为每个已分析的区间绘制标记（仅绘制当前勾选数据集的区间）
        drawn_count = 0
        for range_data in self.analyzed_ranges:
            # 兼容旧格式（三元组）和新格式（四元组）
            if len(range_data) == 3:
                lower, upper, peak_number = range_data
                file_name = None  # 旧数据没有文件名
            elif len(range_data) == 4:
                lower, upper, peak_number, file_name = range_data
            else:
                logger.warning(f"draw_analyzed_ranges_on_plot: 区间数据格式错误: {range_data}")
                continue

            # 【修复】检查区域标记是否属于当前勾选的数据集
            # 如果没有文件名信息（旧格式或错误数据），跳过该区间
            if file_name is None:
                logger.warning(f"draw_analyzed_ranges_on_plot: 区间没有文件名信息，跳过: 峰编号={peak_number}, 区间={lower:.2f}-{upper:.2f}")
                continue

            # 检查文件名是否在当前勾选的数据集中
            if file_name not in checked_file_names:
                logger.debug(f"draw_analyzed_ranges_on_plot: 跳过未勾选数据集的区间: 文件={file_name}, 峰编号={peak_number}")
                continue

            # 获取对应数据集的数据
            dataset = None
            for ds in checked_datasets:
                if ds['name'] == file_name:
                    dataset = ds
                    break

            if dataset is None:
                logger.warning(f"draw_analyzed_ranges_on_plot: 找不到数据集: {file_name}")
                continue

            x_data = dataset['x_data']
            y_data = dataset['y_data']
            # 获取区间内的数据
            mask = (x_data >= lower) & (x_data <= upper)
            x_range = x_data[mask]
            y_range = y_data[mask]

            if len(x_range) > 0:
                # 计算基线
                lower_idx = np.argmin(np.abs(x_data - lower))
                upper_idx = np.argmin(np.abs(x_data - upper))
                lower_y = y_data[lower_idx]
                upper_y = y_data[upper_idx]

                baseline_slope = (upper_y - lower_y) / (upper - lower) if upper != lower else 0
                baseline_intercept = lower_y - baseline_slope * lower
                y_baseline = baseline_slope * x_range + baseline_intercept

                # 填充积分区域（使用不同的颜色，更淡）
                self.peak_ax.fill_between(x_range, y_baseline, y_range,
                                         alpha=0.2, color='lightgreen', edgecolor='green', linewidth=1)

                # 绘制边界虚线
                self.peak_ax.axvline(x=lower, color='green', linestyle=':', alpha=0.6, linewidth=1)
                self.peak_ax.axvline(x=upper, color='green', linestyle=':', alpha=0.6, linewidth=1)

                # 添加峰编号标注
                mid_x = (lower + upper) / 2
                max_y = np.max(y_range)
                self.peak_ax.text(mid_x, max_y * 1.1, f'#{peak_number}',
                                ha='center', va='bottom', fontsize=10, fontweight='bold',
                                bbox=dict(boxstyle='circle,pad=0.3', facecolor='lightgreen',
                                        edgecolor='green', alpha=0.7))

                drawn_count += 1
                logger.debug(f"draw_analyzed_ranges_on_plot: 绘制区间: 文件={file_name}, 峰编号={peak_number}, 区间={lower:.2f}-{upper:.2f}")

        logger.info(f"draw_analyzed_ranges_on_plot: 共绘制了 {drawn_count} 个区间")

    def draw_analyzed_ranges(self):
        """更新峰分析图形，重新绘制已分析的区间"""
        self.update_peak_plot()

    def on_peak_mouse_move(self, event):
        """
        鼠标移动事件处理 - 显示峰信息提示框 + 平移图谱

        当鼠标移动到峰附近时，显示峰的波数和高度信息
        如果正在拖动，则平移图谱
        """
        if event.inaxes != self.peak_ax:
            # 鼠标不在图形区域内，移除提示框
            if self.peak_tooltip is not None:
                try:
                    self.peak_tooltip.set_visible(False)
                except Exception as e:
                    logger.debug(f"隐藏峰提示框失败: {str(e)}")
                self.peak_tooltip = None
                self.peak_canvas.draw_idle()
            return

        # 处理平移
        if self.peak_pan_start is not None and event.xdata is not None and event.ydata is not None:
            # 如果鼠标移动了一定距离，开始平移
            if not self.peak_is_panning:
                dx = abs(event.xdata - self.peak_pan_start[0])
                dy = abs(event.ydata - self.peak_pan_start[1])
                if dx > 0.01 or dy > 0.01:  # 移动阈值
                    self.peak_is_panning = True

            if self.peak_is_panning:
                # 计算偏移量
                dx = event.xdata - self.peak_pan_start[0]
                dy = event.ydata - self.peak_pan_start[1]

                # 获取当前范围
                xlim = self.peak_ax.get_xlim()
                ylim = self.peak_ax.get_ylim()

                # 计算新范围（注意x轴倒置）
                new_xlim = (xlim[0] - dx, xlim[1] - dx)
                new_ylim = (ylim[0] - dy, ylim[1] - dy)

                # 限制范围不超出数据范围
                if hasattr(self, 'x_data') and self.x_data is not None:
                    data_x_min = np.min(self.x_data)
                    data_x_max = np.max(self.x_data)

                    x_range = new_xlim[0] - new_xlim[1]

                    # 因为x轴倒置，xlim[0]是大值，xlim[1]是小值
                    if new_xlim[0] > data_x_max:
                        new_xlim = (data_x_max, data_x_max - x_range)
                    if new_xlim[1] < data_x_min:
                        new_xlim = (data_x_min + x_range, data_x_min)

                # 应用新范围
                self.peak_ax.set_xlim(new_xlim)
                self.peak_ax.set_ylim(new_ylim)

                # 更新起始位置
                self.peak_pan_start = (event.xdata, event.ydata)

                self.peak_canvas.draw_idle()
                return  # 平移时不显示提示框

        if len(self.peaks_tree.get_children()) == 0:
            return

        # 获取当前数据
        if self.peak_data_var.get() == "smoothed" and self.smoothed_data is not None:
            y_data = self.smoothed_data
        elif self.peak_data_var.get() == "corrected" and self.corrected_data is not None:
            y_data = self.corrected_data
        else:
            y_data = self.y_data

        # 【修复】检查数据是否有效
        if y_data is None or self.x_data is None:
            return

        # 获取鼠标位置（数据坐标）
        mouse_x = event.xdata
        mouse_y = event.ydata

        if mouse_x is None or mouse_y is None:
            return

        # 查找最近的峰
        min_distance = float('inf')
        nearest_peak = None
        nearest_peak_height = None

        try:
            for item in self.peaks_tree.get_children():
                values = self.peaks_tree.item(item, 'values')
                if len(values) < 3:
                    logger.warning(f"峰列表数据格式错误: {values}")
                    continue

                # values[0] 是文件名，values[1] 是波数，values[2] 是峰高
                peak_wavenumber = float(values[1])
                peak_idx = np.argmin(np.abs(self.x_data - peak_wavenumber))
                peak_height = y_data[peak_idx]

                # 计算距离（只考虑x方向，转换为像素坐标）
                # 使用transData将数据坐标转换为显示坐标
                peak_display = self.peak_ax.transData.transform([[peak_wavenumber, peak_height]])[0]
                mouse_display = self.peak_ax.transData.transform([[mouse_x, mouse_y]])[0]

                distance = abs(peak_display[0] - mouse_display[0])

                if distance < min_distance:
                    min_distance = distance
                    nearest_peak = peak_wavenumber
                    nearest_peak_height = peak_height

            # 如果距离小于阈值（20像素），显示提示框
            if min_distance <= 20 and nearest_peak is not None:
                # 移除旧的提示框
                if self.peak_tooltip is not None:
                    try:
                        self.peak_tooltip.set_visible(False)
                    except Exception as e:
                        logger.debug(f"隐藏旧峰提示框失败: {str(e)}")

                # 创建新的提示框
                # 只显示数值，格式：波数, 峰高
                tooltip_text = f"{nearest_peak:.2f}, {nearest_peak_height:.4f}"
                # 使用 Times New Roman 字体
                font_props = FontProperties(family='Times New Roman', size=9)
                self.peak_tooltip = self.peak_ax.annotate(
                    tooltip_text,
                    xy=(nearest_peak, nearest_peak_height),
                    xytext=(10, 10),
                    textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8, edgecolor='black'),
                    fontproperties=font_props,
                    ha='left',
                    va='bottom',
                    zorder=1000
                )
                self.peak_canvas.draw_idle()
                logger.debug(f"显示峰信息提示框: 波数={nearest_peak:.2f}, 峰高={nearest_peak_height:.4f}")
            else:
                # 距离太远，移除提示框
                if self.peak_tooltip is not None:
                    try:
                        self.peak_tooltip.set_visible(False)
                    except Exception as e:
                        logger.debug(f"隐藏峰提示框失败: {str(e)}")
                    self.peak_tooltip = None
                    self.peak_canvas.draw_idle()

        except Exception as e:
            logger.error(f"显示峰信息提示框时出错: {str(e)}")
            # 移除提示框
            if self.peak_tooltip is not None:
                try:
                    self.peak_tooltip.set_visible(False)
                except Exception as ex:
                    logger.debug(f"清理峰提示框失败: {str(ex)}")
                self.peak_tooltip = None
                self.peak_canvas.draw_idle()

    def on_peak_scroll(self, event):
        """
        鼠标滚轮事件处理 - 缩放图谱

        Ctrl + 滚轮：缩放图谱
        - 向上滚动：放大
        - 向下滚动：缩小
        - 缩放中心：选中的峰或鼠标位置
        """
        if event.inaxes != self.peak_ax:
            return

        # 检查是否按下Ctrl键
        if event.key != 'control':
            return

        # 获取当前坐标轴范围
        xlim = self.peak_ax.get_xlim()
        ylim = self.peak_ax.get_ylim()

        # 确定缩放中心
        # 如果有选中的峰，以选中峰为中心；否则以鼠标位置为中心
        selection = self.peaks_tree.selection()
        if selection:
            # 以选中峰为中心
            values = self.peaks_tree.item(selection[0], 'values')
            # values[0] 是文件名，values[1] 是波数，values[2] 是峰高
            peak_wavenumber = float(values[1])
            center_x = peak_wavenumber
        else:
            # 以鼠标位置为中心
            center_x = event.xdata if event.xdata is not None else (xlim[0] + xlim[1]) / 2

        center_y = event.ydata if event.ydata is not None else (ylim[0] + ylim[1]) / 2

        # 缩放因子
        zoom_factor = 1.2 if event.button == 'up' else 0.8

        # 计算新的范围
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]

        new_x_range = x_range * zoom_factor
        new_y_range = y_range * zoom_factor

        # 限制最大放大倍数（不超过数据范围的1/20）
        if hasattr(self, 'x_data') and self.x_data is not None:
            data_x_range = np.max(self.x_data) - np.min(self.x_data)
            min_x_range = data_x_range / 20
            if new_x_range < min_x_range:
                new_x_range = min_x_range

        # 限制最大缩小倍数（不超过数据范围）
        if hasattr(self, 'x_data') and self.x_data is not None:
            data_x_range = np.max(self.x_data) - np.min(self.x_data)
            if new_x_range > data_x_range * 1.1:
                new_x_range = data_x_range * 1.1

        # 计算新的坐标轴范围（保持中心点不变）
        # 注意：x轴是倒置的，所以xlim[0] > xlim[1]
        x_left_ratio = (center_x - xlim[1]) / x_range
        x_right_ratio = (xlim[0] - center_x) / x_range

        new_xlim_left = center_x + new_x_range * x_right_ratio
        new_xlim_right = center_x - new_x_range * x_left_ratio

        y_bottom_ratio = (center_y - ylim[0]) / y_range
        y_top_ratio = (ylim[1] - center_y) / y_range

        new_ylim_bottom = center_y - new_y_range * y_bottom_ratio
        new_ylim_top = center_y + new_y_range * y_top_ratio

        # 限制x范围不超出数据范围
        if hasattr(self, 'x_data') and self.x_data is not None:
            data_x_min = np.min(self.x_data)
            data_x_max = np.max(self.x_data)

            # 因为x轴倒置，xlim[0]是大值，xlim[1]是小值
            if new_xlim_left > data_x_max:
                new_xlim_left = data_x_max
            if new_xlim_right < data_x_min:
                new_xlim_right = data_x_min

        # 应用新的范围
        self.peak_ax.set_xlim(new_xlim_left, new_xlim_right)
        self.peak_ax.set_ylim(new_ylim_bottom, new_ylim_top)

        self.peak_canvas.draw_idle()
        logger.info(f"图谱缩放: x范围 {new_xlim_left:.2f} - {new_xlim_right:.2f}")

    def on_peak_button_press(self, event):
        """
        鼠标按下事件处理 - 开始平移或显示右键菜单
        """
        if event.inaxes != self.peak_ax:
            return

        # 处理右键点击（显示右键菜单）
        if event.button == 3:  # 右键
            self.on_peak_plot_right_click(event)
            return

        # 只处理左键
        if event.button != 1:
            return

        # 如果在交互式选择模式下，不处理平移（让SpanSelector处理）
        if self.peak_interactive_mode:
            return

        # 只在平移模式下处理
        if self.peak_tool_mode.get() != "pan":
            return

        # 记录起始位置
        self.peak_pan_start = (event.xdata, event.ydata)
        self.peak_is_panning = False  # 还未开始移动

    def on_peak_button_release(self, event):
        """
        鼠标释放事件处理 - 结束平移
        """
        # 如果刚完成平移，添加到历史记录
        if self.peak_is_panning:
            self.add_zoom_history(self.peak_ax.get_xlim(), self.peak_ax.get_ylim())

        self.peak_pan_start = None
        self.peak_is_panning = False

    def find_peaks(self):
        """
        自动寻峰功能

        使用scipy.signal.find_peaks算法自动识别光谱中的特征峰。
        用户可以设置阈值和最小距离参数来控制峰的识别。

        识别到的峰会显示在峰列表中，包括波数位置和峰高度。
        """
        try:
            # 检查勾选的数据集数量
            checked_count = self.get_checked_datasets_count()

            if checked_count == 0:
                messagebox.showwarning("警告", "请先勾选一个数据集！")
                return
            elif checked_count > 1:
                messagebox.showwarning("警告", "请仅勾选一个数据集以进行峰分析！\n当前勾选了 {} 个数据集。".format(checked_count))
                return

            # 获取唯一勾选的数据集
            checked_dataset = [ds for ds in self.loaded_datasets if ds.get('checked', True)][0]
            x_data = checked_dataset['x_data']
            y_data = checked_dataset['y_data']
            dataset_name = checked_dataset['name']  # 获取数据集名称

            # 设置当前文件名，用于分析结果表格显示
            self.current_file_name = dataset_name
            logger.info(f"设置当前文件名为: {self.current_file_name}")

            # 获取参数
            threshold = float(self.peak_threshold_var.get())
            distance = int(self.peak_distance_var.get())

            # 使用PeakAnalyzer进行寻峰
            success, peak_list, error_msg = self.peak_analyzer.find_peaks_auto(
                x_data, y_data, threshold, distance
            )

            if not success:
                messagebox.showerror("错误", error_msg)
                return

            # 清空现有峰列表
            for item in self.peaks_tree.get_children():
                self.peaks_tree.delete(item)

            if len(peak_list) == 0:
                messagebox.showinfo("提示", "未找到符合条件的峰！请调整阈值或最小距离参数。")
                return

            # 添加找到的峰（文件名、波数和峰高），并设置交替行背景色
            for idx, (wavenumber, height) in enumerate(peak_list):
                row_tag = 'evenrow' if idx % 2 == 0 else 'oddrow'
                self.peaks_tree.insert(
                    '', 'end',
                    values=(dataset_name, f"{wavenumber:.2f}", f"{height:.4f}"),
                    tags=(row_tag,)
                )

            # 配置峰列表的斑马纹背景色
            self.peaks_tree.tag_configure('evenrow', background='white')
            self.peaks_tree.tag_configure('oddrow', background='#F5F5F5')

            # 设置已执行寻峰标志
            self.has_performed_peak_finding = True
            logger.info("已设置寻峰标志，后续切换数据集将自动寻峰")

            # 更新图形
            self.update_peak_plot()
            messagebox.showinfo("成功", f"找到 {len(peak_list)} 个峰！")

        except ValueError as e:
            messagebox.showerror("参数错误", f"参数格式不正确：{str(e)}")
        except Exception as e:
            messagebox.showerror("错误", f"寻峰出错：{str(e)}")

    def _auto_find_peaks_for_dataset(self, dataset):
        """
        内部函数：自动为指定数据集执行寻峰操作（不显示消息框）

        Args:
            dataset: 数据集字典，包含 'name', 'x_data', 'y_data' 等字段
        """
        try:
            x_data = dataset['x_data']
            y_data = dataset['y_data']
            dataset_name = dataset['name']

            # 获取当前的寻峰参数
            threshold = float(self.peak_threshold_var.get())
            distance = int(self.peak_distance_var.get())

            logger.info(f"自动寻峰: 数据集='{dataset_name}', 阈值={threshold}, 最小距离={distance}")

            # 使用PeakAnalyzer进行寻峰
            success, peak_list, error_msg = self.peak_analyzer.find_peaks_auto(
                x_data, y_data, threshold, distance
            )

            if not success:
                logger.warning(f"自动寻峰失败: {error_msg}")
                # 清空峰列表
                for item in self.peaks_tree.get_children():
                    self.peaks_tree.delete(item)
                return

            # 清空现有峰列表
            for item in self.peaks_tree.get_children():
                self.peaks_tree.delete(item)

            if len(peak_list) == 0:
                logger.info(f"自动寻峰: 数据集 '{dataset_name}' 未找到符合条件的峰")
                return

            # 添加找到的峰（文件名、波数和峰高），并设置交替行背景色
            for idx, (wavenumber, height) in enumerate(peak_list):
                row_tag = 'evenrow' if idx % 2 == 0 else 'oddrow'
                self.peaks_tree.insert(
                    '', 'end',
                    values=(dataset_name, f"{wavenumber:.2f}", f"{height:.4f}"),
                    tags=(row_tag,)
                )

            # 配置峰列表的斑马纹背景色
            self.peaks_tree.tag_configure('evenrow', background='white')
            self.peaks_tree.tag_configure('oddrow', background='#F5F5F5')

            # 更新图形
            self.update_peak_plot()

            logger.info(f"自动寻峰成功: 数据集 '{dataset_name}' 找到 {len(peak_list)} 个峰")

        except ValueError as e:
            logger.error(f"自动寻峰参数错误: {str(e)}")
        except Exception as e:
            logger.error(f"自动寻峰出错: {str(e)}")

    def clear_peak_selection(self):
        """清除当前的峰选择（仅清空输入框，不清除峰列表选择）"""
        try:
            # 清空输入框
            self.peak_lower_var.set("")
            self.peak_upper_var.set("")

            # 重新绘制图形（移除选择标记）
            self.update_peak_plot()

            logger.info("已清除峰选择（清空输入框）")
        except Exception as e:
            logger.error(f"清除峰选择失败: {str(e)}")

    def export_peak_list(self):
        """导出峰列表"""
        if len(self.peaks_tree.get_children()) == 0:
            messagebox.showwarning("警告", "峰列表为空！请先进行寻峰。")
            return

        try:
            # 生成默认文件名
            if self.current_file_name:
                default_filename = f"{self.current_file_name}_峰列表.csv"
            else:
                default_filename = "峰列表.csv"

            # 默认保存到 data/output 文件夹
            initial_dir = self.output_dir if os.path.exists(self.output_dir) else os.getcwd()

            file_path = filedialog.asksaveasfilename(
                title="保存峰列表",
                initialdir=initial_dir,
                initialfile=default_filename,
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )

            if not file_path:
                return

            filenames = []
            peaks = []
            heights = []
            for item in self.peaks_tree.get_children():
                values = self.peaks_tree.item(item, 'values')
                filenames.append(values[0])
                peaks.append(float(values[1]))
                heights.append(float(values[2]))

            df = pd.DataFrame({
                "文件名": filenames,
                "峰位置(cm^-1)": peaks,
                "峰高度": heights
            })
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
            logger.info(f"峰列表已导出: {os.path.basename(file_path)}")
            messagebox.showinfo("成功", "峰列表导出成功！")

        except ValueError as e:
            messagebox.showerror("错误", f"峰数据格式错误：{str(e)}")
        except PermissionError:
            messagebox.showerror("错误", "没有权限写入该文件！请检查文件是否被其他程序占用。")
        except OSError as e:
            messagebox.showerror("错误", f"文件写入错误：{str(e)}")
        except Exception as e:
            messagebox.showerror("错误", f"导出出错：{str(e)}")

    def analyze_selected_peak(self):
        """
        分析选中的峰

        对用户选中的特征峰进行定量分析，计算：
        - 峰位置（波数）
        - 未校正峰高（原始高度）
        - 校正峰高（扣除基线后的高度）
        - 未校正峰面积
        - 校正峰面积（扣除基线后的面积）

        使用直线基线（连接分析范围两端点）进行校正。
        """
        selection = self.peaks_tree.selection()
        if not selection:
            messagebox.showwarning("警告", "请先选择要分析的峰！")
            return

        if not self.peak_lower_var.get() or not self.peak_upper_var.get():
            messagebox.showwarning("警告", "请先设置分析范围（上限和下限）！")
            return

        try:
            # 获取分析范围
            lower = float(self.peak_lower_var.get())
            upper = float(self.peak_upper_var.get())

            # 验证区间内的峰数量
            is_valid, peak_count, message = self.validate_peak_range(lower, upper)
            if not is_valid:
                messagebox.showwarning("区间验证失败", message)
                return

            # 获取当前选择的数据
            data_type = self.peak_data_var.get()
            y_data = self.data_manager.get_data(data_type)

            # 获取选中峰的波数
            values = self.peaks_tree.item(selection[0], 'values')
            # values[0] 是文件名，values[1] 是波数，values[2] 是峰高
            peak_wavenumber = float(values[1])

            # 使用PeakAnalyzer进行峰分析
            success, results, error_msg = self.peak_analyzer.analyze_peak(
                self.x_data, y_data, peak_wavenumber, lower, upper
            )

            if success:
                self.display_peak_results(results)
            else:
                messagebox.showerror("错误", error_msg)

        except ValueError as e:
            messagebox.showerror("参数错误", f"参数格式不正确：{str(e)}")
        except IndexError as e:
            messagebox.showerror("错误", f"数据索引错误：{str(e)}")
        except Exception as e:
            messagebox.showerror("错误", f"峰分析出错：{str(e)}")

    def add_peak_to_analysis(self):
        """
        添加峰到分析列表（逐个添加模式）

        验证区间内恰好有1个峰，然后分析该峰并添加到结果表格
        同时在图形上标记该区间
        """
        if not self.peak_lower_var.get() or not self.peak_upper_var.get():
            messagebox.showwarning("警告", "请先设置分析范围（上限和下限）！")
            return

        try:
            # 获取分析范围
            lower = float(self.peak_lower_var.get())
            upper = float(self.peak_upper_var.get())

            # 验证区间内的峰数量
            is_valid, peak_count, message = self.validate_peak_range(lower, upper)
            if not is_valid:
                messagebox.showwarning("区间验证失败", message)
                return

            # 确保lower < upper
            if lower > upper:
                lower, upper = upper, lower

            # 找到区间内的峰
            peak_wavenumber = None
            for item in self.peaks_tree.get_children():
                values = self.peaks_tree.item(item, 'values')
                # values[0] 是文件名，values[1] 是波数，values[2] 是峰高
                wn = float(values[1])
                if lower <= wn <= upper:
                    peak_wavenumber = wn
                    break

            if peak_wavenumber is None:
                messagebox.showerror("错误", "未找到区间内的峰")
                return

            # 获取当前选择的数据
            data_type = self.peak_data_var.get()
            y_data = self.data_manager.get_data(data_type)

            # 使用PeakAnalyzer进行峰分析
            success, results, error_msg = self.peak_analyzer.analyze_peak(
                self.x_data, y_data, peak_wavenumber, lower, upper
            )

            if success:
                # 获取峰编号
                peak_number = len(self.result_tree.get_children()) + 1

                # 获取当前勾选的数据集名称
                checked_datasets = [ds for ds in self.loaded_datasets if ds.get('checked', True)]
                current_file_name = checked_datasets[0]['name'] if len(checked_datasets) == 1 else self.current_file_name

                # 添加到结果表格
                self.add_result_to_table(peak_number, results, current_file_name)

                # 记录已分析的区间（包含文件名）
                self.analyzed_ranges.append((lower, upper, peak_number, current_file_name))
                logger.info(f"记录已分析区间: 文件={current_file_name}, 峰编号={peak_number}, 区间={lower:.2f}-{upper:.2f}")

                # 在图形上绘制该区间
                self.draw_analyzed_ranges()

                logger.info(f"峰 {peak_number} ({peak_wavenumber:.2f} cm⁻¹) 已添加到分析列表，区间: {lower:.2f} - {upper:.2f}")

                # 清空输入框，准备下一次输入
                self.peak_lower_var.set("")
                self.peak_upper_var.set("")
            else:
                messagebox.showerror("错误", error_msg)

        except ValueError as e:
            messagebox.showerror("参数错误", f"参数格式不正确：{str(e)}")
        except Exception as e:
            messagebox.showerror("错误", f"添加峰分析出错：{str(e)}")

    def display_peak_results(self, results):
        """显示峰分析结果（添加到表格）"""
        # 获取当前表格中的行数，作为峰编号
        peak_number = len(self.result_tree.get_children()) + 1

        # 获取当前勾选的数据集名称
        checked_datasets = [ds for ds in self.loaded_datasets if ds.get('checked', True)]
        current_file_name = checked_datasets[0]['name'] if len(checked_datasets) == 1 else self.current_file_name

        # 添加到结果表格
        self.add_result_to_table(peak_number, results, current_file_name)

        logger.info(f"峰 {peak_number} 的分析结果已添加到表格，文件名: {current_file_name}")

    def on_peak_select(self, event):  # event用于Tkinter事件绑定
        """
        当峰列表选择改变时更新图形

        Args:
            event: Tkinter事件对象（未使用但必须保留）
        """
        self.update_peak_plot()

    def toggle_peak_interactive_mode(self):
        """切换峰分析的交互式选择模式"""
        self.peak_interactive_mode = self.peak_interactive_mode_var.get()

        # 【修复】在切换模式前保存当前的视图范围，防止 X 轴方向被重置
        current_xlim = self.peak_ax.get_xlim()
        current_ylim = self.peak_ax.get_ylim()

        if self.peak_interactive_mode:
            # 启用交互式选择
            logger.info("峰分析交互式选择模式已启用")
            self.peak_interactive_hint_label.config(text="拖拽选择积分范围")

            # 禁用矩形选框工具（避免冲突）
            if self.peak_rect_selector is not None:
                self.peak_rect_selector.set_active(False)
                logger.info("矩形选框工具已禁用（交互式选择模式启用）")

            # 创建SpanSelector
            from matplotlib.widgets import SpanSelector
            self.peak_span_selector = SpanSelector(
                self.peak_ax,
                self.on_peak_span_select,
                'horizontal',
                useblit=True,
                props=dict(alpha=0.3, facecolor='yellow'),
                interactive=True,
                drag_from_anywhere=True
            )

            # 【修复】恢复视图范围，保持 FTIR 标准的倒序显示（左大右小）
            self.peak_ax.set_xlim(current_xlim)
            self.peak_ax.set_ylim(current_ylim)

            self.peak_canvas.draw()
        else:
            # 禁用交互式选择
            logger.info("峰分析交互式选择模式已禁用")
            self.peak_interactive_hint_label.config(text="")

            if self.peak_span_selector is not None:
                self.peak_span_selector.set_active(False)
                self.peak_span_selector = None

            # 如果当前工具模式是矩形选框，重新启用矩形选框工具
            if self.peak_tool_mode.get() == "rect_zoom" and self.peak_rect_selector is not None:
                self.peak_rect_selector.set_active(True)
                logger.info("矩形选框工具已重新启用")

            # 【修复】恢复视图范围，保持 FTIR 标准的倒序显示（左大右小）
            self.peak_ax.set_xlim(current_xlim)
            self.peak_ax.set_ylim(current_ylim)

            self.peak_canvas.draw()

    def on_peak_span_select(self, xmin, xmax):
        """峰分析SpanSelector回调函数"""
        # 确保xmin < xmax（因为x轴是倒置的）
        if xmin > xmax:
            xmin, xmax = xmax, xmin

        # 更新下限和上限输入框
        self.peak_lower_var.set(f"{xmin:.2f}")
        self.peak_upper_var.set(f"{xmax:.2f}")

        # 记录选中的区域，用于右键菜单
        self.peak_selected_range = (xmin, xmax)

        logger.info(f"通过拖拽选择峰分析范围: {xmin:.2f} - {xmax:.2f}")

    def create_peak_context_menu(self):
        """创建峰分析交互式选择的右键菜单"""
        self.peak_context_menu = tk.Menu(self.root, tearoff=0)
        self.peak_context_menu.add_command(label="取消选择", command=self.cancel_peak_selection)
        self.peak_context_menu.add_command(label="添加到分析列表", command=self.add_peak_from_context_menu)
        logger.info("峰分析右键菜单已创建")

    def on_peak_canvas_right_click(self, event):
        """处理峰分析画布的右键点击事件"""
        # 只在交互式选择模式下且有选中区域时显示菜单
        if not self.peak_interactive_mode:
            logger.debug("右键点击：交互式选择模式未启用")
            return

        if self.peak_selected_range is None:
            logger.debug("右键点击：没有选中的区域")
            return

        # 检查是否有有效的区间输入
        if not self.peak_lower_var.get() or not self.peak_upper_var.get():
            logger.debug("右键点击：区间输入框为空")
            return

        # 在鼠标位置弹出菜单
        try:
            self.peak_context_menu.tk_popup(event.x_root, event.y_root)
            logger.info(f"显示右键菜单，位置: ({event.x_root}, {event.y_root})")
        finally:
            # 确保菜单在点击外部时关闭
            self.peak_context_menu.grab_release()

    def cancel_peak_selection(self):
        """取消当前选中的峰分析区域"""
        # 清空输入框
        self.peak_lower_var.set("")
        self.peak_upper_var.set("")

        # 清空选中区域记录
        self.peak_selected_range = None

        # 更新图形（移除区域高亮）
        self.update_peak_plot()

        logger.info("已取消峰分析区域选择")

    def add_peak_from_context_menu(self):
        """从右键菜单添加峰到分析列表"""
        # 直接调用现有的添加功能
        self.add_peak_to_analysis()

        # 添加成功后清空选中区域记录
        self.peak_selected_range = None

        logger.info("通过右键菜单添加峰到分析列表")

    def validate_peak_range(self, lower, upper):
        """
        验证积分区间内的峰数量

        参数:
            lower: 区间下限（波数）
            upper: 区间上限（波数）

        返回:
            (is_valid, peak_count, message)
            - is_valid: 是否有效（恰好1个峰）
            - peak_count: 区间内的峰数量
            - message: 提示信息
        """
        if len(self.peaks_tree.get_children()) == 0:
            return False, 0, "请先寻找峰！"

        # 确保lower < upper
        if lower > upper:
            lower, upper = upper, lower

        # 统计区间内的峰数量
        peaks_in_range = []
        for item in self.peaks_tree.get_children():
            values = self.peaks_tree.item(item, 'values')
            # values[0] 是文件名，values[1] 是波数，values[2] 是峰高
            peak_wavenumber = float(values[1])
            if lower <= peak_wavenumber <= upper:
                peaks_in_range.append(peak_wavenumber)

        peak_count = len(peaks_in_range)

        if peak_count == 0:
            message = "所选区间内没有检测到峰，请重新选择"
            logger.warning(f"区间验证失败: {lower:.2f} - {upper:.2f}, {message}")
            return False, 0, message
        elif peak_count > 1:
            message = f"所选区间内包含多个峰（{peak_count}个），每个区间只能包含一个峰，请缩小范围"
            logger.warning(f"区间验证失败: {lower:.2f} - {upper:.2f}, {message}")
            return False, peak_count, message
        else:
            logger.info(f"区间验证通过: {lower:.2f} - {upper:.2f}, 包含1个峰: {peaks_in_range[0]:.2f}")
            return True, 1, "验证通过"

    def batch_analyze_peaks(self):
        """批量分析所有峰"""
        if len(self.peaks_tree.get_children()) == 0:
            messagebox.showwarning("警告", "请先寻找峰！")
            return

        try:
            # 获取当前选择的数据
            data_type = self.peak_data_var.get()
            y_data = self.data_manager.get_data(data_type)

            # 获取当前勾选的数据集名称
            checked_datasets = [ds for ds in self.loaded_datasets if ds.get('checked', True)]
            current_file_name = checked_datasets[0]['name'] if len(checked_datasets) == 1 else self.current_file_name

            # 清空结果表格
            self.clear_result_table()

            # 获取所有峰的波数
            peaks_wavenumbers = []
            for item in self.peaks_tree.get_children():
                values = self.peaks_tree.item(item, 'values')
                # values[0] 是文件名，values[1] 是波数，values[2] 是峰高
                peak_wavenumber = float(values[1])
                peaks_wavenumbers.append(peak_wavenumber)

            # 【修复】定义 peak_count 变量
            peak_count = len(peaks_wavenumbers)
            logger.info(f"开始批量分析 {peak_count} 个峰")

            # 为每个峰自动确定积分范围并分析
            for idx, peak_wavenumber in enumerate(peaks_wavenumbers):
                # 自动确定积分范围（使用相邻峰之间的中点）
                if idx == 0:
                    # 第一个峰：使用到下一个峰的中点
                    if peak_count > 1:
                        lower = peak_wavenumber - abs(peak_wavenumber - peaks_wavenumbers[idx + 1]) / 2
                    else:
                        lower = peak_wavenumber - 50  # 默认范围
                else:
                    # 使用到上一个峰的中点
                    lower = (peaks_wavenumbers[idx - 1] + peak_wavenumber) / 2

                if idx == peak_count - 1:
                    # 最后一个峰：使用到上一个峰的中点
                    if peak_count > 1:
                        upper = peak_wavenumber + abs(peak_wavenumber - peaks_wavenumbers[idx - 1]) / 2
                    else:
                        upper = peak_wavenumber + 50  # 默认范围
                else:
                    # 使用到下一个峰的中点
                    upper = (peak_wavenumber + peaks_wavenumbers[idx + 1]) / 2

                # 确保lower < upper（考虑x轴倒置）
                if lower > upper:
                    lower, upper = upper, lower

                # 分析峰
                success, results, error_msg = self.peak_analyzer.analyze_peak(
                    self.x_data, y_data, peak_wavenumber, lower, upper
                )

                if success:
                    # 添加到结果表格
                    self.add_result_to_table(idx + 1, results, current_file_name)
                else:
                    logger.warning(f"峰 {idx + 1} 分析失败: {error_msg}")

            messagebox.showinfo("成功", f"批量分析完成！共分析 {peak_count} 个峰。")
            logger.info(f"批量分析完成，共分析 {peak_count} 个峰")

        except Exception as e:
            messagebox.showerror("错误", f"批量分析出错：{str(e)}")
            logger.exception("批量分析出错")

    def batch_analyze_all_datasets(self):
        """
        批量分析所有已加载的数据集

        在固定的积分区间内对所有数据集进行峰面积分析，
        结果表格中显示每个数据集的分析结果。
        """
        # 检查是否有已加载的数据集
        if not self.loaded_datasets:
            messagebox.showwarning("警告", "请先加载多个数据集！")
            return

        # 检查是否勾选了固定积分区间
        if not self.fixed_integration_range.get():
            messagebox.showwarning("警告", "请先勾选'固定积分区间'选项！")
            return

        # 检查是否设置了积分区间
        if not self.peak_lower_var.get() or not self.peak_upper_var.get():
            messagebox.showwarning("警告", "请先设置积分区间（上限和下限）！")
            return

        try:
            # 获取积分区间
            lower = float(self.peak_lower_var.get())
            upper = float(self.peak_upper_var.get())

            # 确保lower < upper
            if lower > upper:
                lower, upper = upper, lower

            # 清空结果表格
            self.clear_result_table()

            success_count = 0
            failed_datasets = []

            # 对每个数据集进行分析
            for dataset in self.loaded_datasets:
                file_name = dataset['name']
                x_data = dataset['x_data']
                y_data = dataset['y_data']

                try:
                    # 找到区间内的峰位置（使用简单的最大值查找）
                    # 找到x_data在区间内的索引
                    mask = (x_data >= lower) & (x_data <= upper)
                    if not np.any(mask):
                        failed_datasets.append(f"{file_name}: 区间内无数据")
                        logger.warning(f"数据集 {file_name} 在区间 {lower:.2f}-{upper:.2f} 内无数据")
                        continue

                    # 找到区间内的最大值位置作为峰位置
                    y_in_range = y_data[mask]
                    x_in_range = x_data[mask]
                    peak_idx = np.argmax(y_in_range)
                    peak_wavenumber = x_in_range[peak_idx]

                    # 使用PeakAnalyzer进行峰分析
                    success, results, error_msg = self.peak_analyzer.analyze_peak(
                        x_data, y_data, peak_wavenumber, lower, upper
                    )

                    if success:
                        # 添加到结果表格（使用统一的峰编号1，因为都是同一个峰）
                        self.add_result_to_table(1, results, file_name)
                        success_count += 1
                        logger.info(f"数据集 {file_name} 分析成功")
                    else:
                        failed_datasets.append(f"{file_name}: {error_msg}")
                        logger.warning(f"数据集 {file_name} 分析失败: {error_msg}")

                except Exception as e:
                    failed_datasets.append(f"{file_name}: {str(e)}")
                    logger.error(f"分析数据集 {file_name} 时出错: {str(e)}")

            # 显示结果
            msg = f"批量分析完成！\n成功: {success_count}/{len(self.loaded_datasets)}"
            if failed_datasets:
                msg += f"\n\n失败的数据集:\n" + "\n".join(failed_datasets[:5])
                if len(failed_datasets) > 5:
                    msg += f"\n... 还有 {len(failed_datasets) - 5} 个"

            messagebox.showinfo("批量分析完成", msg)
            logger.info(f"批量分析所有数据集完成，成功 {success_count}/{len(self.loaded_datasets)}")

        except ValueError as e:
            messagebox.showerror("参数错误", f"参数格式不正确：{str(e)}")
        except Exception as e:
            messagebox.showerror("错误", f"批量分析出错：{str(e)}")
            logger.exception("批量分析所有数据集出错")

    def add_result_to_table(self, peak_number, results, file_name=None):
        """
        将分析结果添加到表格

        Args:
            peak_number: 峰编号
            results: 分析结果字典
            file_name: 文件名（可选，用于多数据集对比）
        """
        # 提取结果数据
        wavenumber = results.get('波数', 'N/A')
        uncorrected_height = results.get('未校正峰高', 0.0)
        corrected_height = results.get('校正峰高', 0.0)
        uncorrected_area = results.get('未校正峰面积', 0.0)
        corrected_area = results.get('校正峰面积', 0.0)
        lower_limit = results.get('区间下限', 'N/A')
        upper_limit = results.get('区间上限', 'N/A')

        # 如果没有提供文件名，使用当前文件名
        if file_name is None:
            file_name = self.current_file_name if self.current_file_name else "N/A"

        # 格式化波数：只显示数值，不显示单位
        if isinstance(wavenumber, (int, float)):
            wavenumber_str = f"{wavenumber:.2f}"
        else:
            wavenumber_str = str(wavenumber)

        # 格式化区间
        if isinstance(lower_limit, (int, float)):
            lower_limit_str = f"{lower_limit:.2f}"
        else:
            lower_limit_str = str(lower_limit)

        if isinstance(upper_limit, (int, float)):
            upper_limit_str = f"{upper_limit:.2f}"
        else:
            upper_limit_str = str(upper_limit)

        # 插入到表格（包含文件名和区间），并设置交替行背景色（模拟网格线效果）
        row_count = len(self.result_tree.get_children())
        row_tag = 'evenrow' if row_count % 2 == 0 else 'oddrow'

        self.result_tree.insert(
            '', 'end',
            values=(
                file_name,
                peak_number,
                wavenumber_str,
                f"{uncorrected_height:.4f}",
                f"{corrected_height:.4f}",
                lower_limit_str,
                upper_limit_str,
                f"{uncorrected_area:.4f}",
                f"{corrected_area:.4f}"
            ),
            tags=(row_tag,)
        )

    def on_result_cell_double_click(self, event):
        """
        处理分析结果表格的双击事件，复制单元格的值到剪贴板

        Args:
            event: Tkinter事件对象
        """
        try:
            # 获取点击的区域
            region = self.result_tree.identify_region(event.x, event.y)

            # 只处理点击在单元格上的情况
            if region == 'cell':
                # 获取点击的行和列
                item = self.result_tree.identify_row(event.y)
                column = self.result_tree.identify_column(event.x)

                if item and column:
                    # 获取列索引（column返回的是 '#1', '#2' 等）
                    column_index = int(column.replace('#', '')) - 1

                    # 获取该行的所有值
                    values = self.result_tree.item(item, 'values')

                    if values and 0 <= column_index < len(values):
                        # 获取单元格的值
                        cell_value = str(values[column_index])

                        # 复制到剪贴板
                        self.root.clipboard_clear()
                        self.root.clipboard_append(cell_value)
                        self.root.update()  # 确保剪贴板更新

                        # 获取列名
                        columns = ('文件名', '编号', '波数', '峰高', '校正峰高', '区间下限', '区间上限', '面积', '校正面积')
                        column_name = columns[column_index] if column_index < len(columns) else '未知'

                        logger.info(f"已复制单元格值到剪贴板: {column_name} = {cell_value}")

                        # 显示临时提示
                        self.show_copy_tooltip(event, cell_value)

        except Exception as e:
            logger.error(f"复制单元格值时出错: {str(e)}")

    def show_copy_tooltip(self, event, value):
        """
        显示复制成功的临时提示框

        Args:
            event: 鼠标事件对象
            value: 复制的值
        """
        try:
            # 创建一个顶层窗口作为提示框
            tooltip = tk.Toplevel(self.root)
            tooltip.wm_overrideredirect(True)  # 移除窗口边框
            tooltip.wm_attributes('-topmost', True)  # 置顶显示

            # 设置提示框内容
            # 限制显示的值的长度，避免提示框过长
            display_value = value if len(value) <= 30 else value[:27] + '...'
            label = tk.Label(
                tooltip,
                text=f"✓ 已复制: {display_value}",
                background='#4CAF50',  # 绿色背景
                foreground='white',
                font=('Arial', 9, 'bold'),
                padx=10,
                pady=5,
                relief=tk.SOLID,
                borderwidth=1
            )
            label.pack()

            # 计算提示框位置（在鼠标位置附近）
            x = event.x_root + 10
            y = event.y_root + 10
            tooltip.wm_geometry(f"+{x}+{y}")

            # 1.5秒后自动关闭提示框
            self.root.after(1500, tooltip.destroy)

        except Exception as e:
            logger.error(f"显示复制提示框时出错: {str(e)}")

    def clear_result_table(self):
        """清空结果表格和已分析区间"""
        for item in self.result_tree.get_children():
            self.result_tree.delete(item)

        # 清空已分析区间列表
        if hasattr(self, 'analyzed_ranges'):
            self.analyzed_ranges.clear()

        # 【修复】清空峰分析区域的上下限输入框（避免显示黄色预览区域）
        if hasattr(self, 'peak_lower_var') and hasattr(self, 'peak_upper_var'):
            self.peak_lower_var.set("")
            self.peak_upper_var.set("")
            logger.info("已清空峰分析区域的上下限输入框")

        # 重新绘制图形（移除区间标记）
        if hasattr(self, 'peak_ax') and self.peak_ax is not None:
            self.update_peak_plot()

        logger.info("结果表格和已分析区间已清空")

    def export_peak_analysis_results(self):
        """导出峰分析结果到CSV文件"""
        if len(self.result_tree.get_children()) == 0:
            messagebox.showwarning("警告", "没有可导出的分析结果！")
            return

        try:
            # 默认打开 data/output 文件夹
            initial_dir = self.output_dir if os.path.exists(self.output_dir) else os.getcwd()

            # 生成默认文件名
            # 检查是否是多数据集分析结果
            has_multiple_files = False
            file_names = set()
            for item in self.result_tree.get_children():
                values = self.result_tree.item(item)['values']
                if values:  # 确保有数据
                    file_names.add(values[0])  # 第一列是文件名

            if len(file_names) > 1:
                has_multiple_files = True
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                default_filename = f"多数据集峰分析结果_{timestamp}.csv"
            elif hasattr(self, 'current_file_name') and self.current_file_name:
                default_filename = f"{self.current_file_name}_峰分析结果.csv"
            else:
                default_filename = "峰分析结果.csv"

            file_path = filedialog.asksaveasfilename(
                initialdir=initial_dir,
                initialfile=default_filename,
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )

            if file_path:
                # 收集表格数据
                data = []
                for item in self.result_tree.get_children():
                    values = self.result_tree.item(item)['values']
                    data.append(values)

                # 创建DataFrame（包含文件名列和区间列）
                df = pd.DataFrame(data, columns=['文件名', '编号', '波数', '峰高', '校正峰高', '区间下限', '区间上限', '面积', '校正面积'])

                # 导出到CSV
                df.to_csv(file_path, index=False, encoding='utf-8-sig')

                msg = f"峰分析结果已导出到:\n{file_path}"
                if has_multiple_files:
                    msg += f"\n\n包含 {len(file_names)} 个数据集的分析结果"

                messagebox.showinfo("成功", msg)
                logger.info(f"峰分析结果导出到: {file_path}, 包含 {len(data)} 条记录")

        except Exception as e:
            messagebox.showerror("错误", f"导出出错：{str(e)}")
            logger.exception("导出峰分析结果出错")

    def switch_peak_tool_mode(self):
        """切换峰分析图形的工具模式（矩形选框 / 平移 / 无）"""
        try:
            mode = self.peak_tool_mode.get()

            if mode == "rect_zoom":
                # 启用矩形选框模式
                logger.info("切换到矩形选框模式")

                # 禁用矩形选框（如果存在）
                if self.peak_rect_selector is not None:
                    self.peak_rect_selector.set_active(False)

                # 创建新的矩形选框
                from matplotlib.widgets import RectangleSelector
                self.peak_rect_selector = RectangleSelector(
                    self.peak_ax,
                    self.on_rect_select,
                    useblit=True,
                    button=[1],  # 左键
                    minspanx=5,
                    minspany=5,
                    spancoords='pixels',
                    interactive=False,
                    props=dict(facecolor='blue', alpha=0.2, edgecolor='blue', linewidth=2)
                )

                # 检查交互式选择模式是否已启用
                if self.peak_interactive_mode:
                    # 如果交互式选择模式已启用，禁用矩形选框工具
                    self.peak_rect_selector.set_active(False)
                    logger.info("矩形选框工具未激活（交互式选择模式已启用）")

            elif mode == "pan":
                # 启用平移模式
                logger.info("切换到平移模式")

                # 禁用矩形选框
                if self.peak_rect_selector is not None:
                    self.peak_rect_selector.set_active(False)
                    self.peak_rect_selector = None

            else:
                # 默认模式（无工具启用）
                logger.info("切换到默认模式（无工具启用）")

                # 禁用所有工具
                if self.peak_rect_selector is not None:
                    self.peak_rect_selector.set_active(False)
                    self.peak_rect_selector = None

            self.peak_canvas.draw_idle()

        except Exception as e:
            logger.error(f"切换工具模式出错: {str(e)}")

    def on_rect_select(self, eclick, erelease):
        """矩形选框选择回调函数"""
        try:
            # 获取选框的坐标
            x1, x2 = sorted([eclick.xdata, erelease.xdata])
            y1, y2 = sorted([eclick.ydata, erelease.ydata])

            # 检查选框大小（避免误操作）
            if abs(x2 - x1) < 1 or abs(y2 - y1) < 0.001:
                logger.info("选框太小，忽略")
                return

            # 添加到历史记录
            self.add_zoom_history(self.peak_ax.get_xlim(), self.peak_ax.get_ylim())

            # 设置新的视图范围
            # 注意：X轴已倒置（FTIR标准：高波数在左，低波数在右）
            # 所以设置 xlim 时，较大的值在左侧，较小的值在右侧
            self.peak_ax.set_xlim(x2, x1)  # 倒置：左大右小
            self.peak_ax.set_ylim(y1, y2)

            self.peak_canvas.draw()
            logger.info(f"矩形选框缩放: X=[{x2:.2f}, {x1:.2f}] (倒置), Y=[{y1:.4f}, {y2:.4f}]")

        except Exception as e:
            logger.error(f"矩形选框缩放出错: {str(e)}")

    def add_zoom_history(self, xlim, ylim):
        """添加缩放状态到历史记录"""
        try:
            # 删除当前位置之后的所有历史记录
            self.peak_zoom_history = self.peak_zoom_history[:self.peak_zoom_history_index + 1]

            # 添加新的历史记录
            self.peak_zoom_history.append((tuple(xlim), tuple(ylim)))
            self.peak_zoom_history_index = len(self.peak_zoom_history) - 1

            # 限制历史记录数量（最多保留50个）
            if len(self.peak_zoom_history) > 50:
                self.peak_zoom_history.pop(0)
                self.peak_zoom_history_index -= 1

            # 更新按钮状态
            self.update_zoom_history_buttons()

            logger.debug(f"添加缩放历史: 索引={self.peak_zoom_history_index}, 总数={len(self.peak_zoom_history)}")

        except Exception as e:
            logger.error(f"添加缩放历史出错: {str(e)}")

    def update_zoom_history_buttons(self):
        """更新缩放历史按钮的状态"""
        try:
            # 后退按钮：如果当前索引 > 0，则可用
            if self.peak_zoom_history_index > 0:
                self.peak_back_btn.config(state='normal')
            else:
                self.peak_back_btn.config(state='disabled')

            # 前进按钮：如果当前索引 < 历史记录数量-1，则可用
            if self.peak_zoom_history_index < len(self.peak_zoom_history) - 1:
                self.peak_forward_btn.config(state='normal')
            else:
                self.peak_forward_btn.config(state='disabled')

        except Exception as e:
            logger.error(f"更新缩放历史按钮状态出错: {str(e)}")

    def zoom_history_back(self):
        """后退到上一个缩放状态"""
        try:
            if self.peak_zoom_history_index > 0:
                self.peak_zoom_history_index -= 1
                xlim, ylim = self.peak_zoom_history[self.peak_zoom_history_index]

                self.peak_ax.set_xlim(xlim)
                self.peak_ax.set_ylim(ylim)
                self.peak_canvas.draw()

                self.update_zoom_history_buttons()
                logger.info(f"后退到缩放历史: 索引={self.peak_zoom_history_index}")

        except Exception as e:
            logger.error(f"后退缩放历史出错: {str(e)}")

    def zoom_history_forward(self):
        """前进到下一个缩放状态"""
        try:
            if self.peak_zoom_history_index < len(self.peak_zoom_history) - 1:
                self.peak_zoom_history_index += 1
                xlim, ylim = self.peak_zoom_history[self.peak_zoom_history_index]

                self.peak_ax.set_xlim(xlim)
                self.peak_ax.set_ylim(ylim)
                self.peak_canvas.draw()

                self.update_zoom_history_buttons()
                logger.info(f"前进到缩放历史: 索引={self.peak_zoom_history_index}")

        except Exception as e:
            logger.error(f"前进缩放历史出错: {str(e)}")

    def reset_zoom_peak(self):
        """重置峰分析图形到原始视图"""
        try:
            if self.x_data is None:
                return

            # 如果有保存的原始范围，使用它；否则使用数据范围
            if self.peak_original_xlim is not None and self.peak_original_ylim is not None:
                self.peak_ax.set_xlim(self.peak_original_xlim)
                self.peak_ax.set_ylim(self.peak_original_ylim)
            else:
                # 使用数据范围
                data_type = self.peak_data_var.get()
                y_data = self.data_manager.get_data(data_type)
                if y_data is not None:
                    # X轴倒置：高波数在左，低波数在右
                    self.peak_ax.set_xlim(np.max(self.x_data), np.min(self.x_data))
                    self.peak_ax.set_ylim(np.min(y_data) * 0.95, np.max(y_data) * 1.05)

            # 添加到历史记录
            self.add_zoom_history(self.peak_ax.get_xlim(), self.peak_ax.get_ylim())

            self.peak_canvas.draw()
            logger.info("峰分析图形重置到原始视图")
        except Exception as e:
            logger.error(f"重置图形出错: {str(e)}")

    # ========== 日志管理方法 ==========

    def refresh_log(self):
        """刷新日志显示"""
        try:
            log_file = os.path.join('logs', 'ftir_processor.log')

            if not os.path.exists(log_file):
                self.log_text.delete('1.0', tk.END)
                self.log_text.insert('1.0', "日志文件不存在")
                return

            # 读取日志文件
            with open(log_file, 'r', encoding='utf-8') as f:
                log_content = f.read()

            # 保存所有日志行
            self.all_log_lines = log_content.split('\n')

            # 应用筛选
            self.filter_log()

            logger.info("日志已刷新")

        except Exception as e:
            messagebox.showerror("错误", f"刷新日志失败：{str(e)}")
            logger.error(f"刷新日志失败: {str(e)}")

    def filter_log(self):
        """根据级别和搜索关键词筛选日志"""
        try:
            if not hasattr(self, 'all_log_lines'):
                return

            level_filter = self.log_level_var.get()
            search_text = self.log_search_var.get().lower()

            # 清空显示
            self.log_text.delete('1.0', tk.END)

            # 筛选日志行
            for line in self.all_log_lines:
                # 级别筛选
                if level_filter != "全部":
                    if f" - {level_filter} - " not in line:
                        continue

                # 搜索筛选
                if search_text and search_text not in line.lower():
                    continue

                # 插入日志行并设置颜色
                self.insert_log_line(line)

            # 滚动到底部
            self.log_text.see(tk.END)

        except Exception as e:
            logger.error(f"筛选日志失败: {str(e)}")

    def insert_log_line(self, line):
        """插入日志行并设置颜色"""
        if not line.strip():
            self.log_text.insert(tk.END, line + '\n')
            return

        # 检测日志级别并设置颜色
        if ' - DEBUG - ' in line:
            self.log_text.insert(tk.END, line + '\n', 'DEBUG')
        elif ' - INFO - ' in line:
            self.log_text.insert(tk.END, line + '\n', 'INFO')
        elif ' - WARNING - ' in line:
            self.log_text.insert(tk.END, line + '\n', 'WARNING')
        elif ' - ERROR - ' in line:
            self.log_text.insert(tk.END, line + '\n', 'ERROR')
        elif ' - CRITICAL - ' in line:
            self.log_text.insert(tk.END, line + '\n', 'CRITICAL')
        else:
            self.log_text.insert(tk.END, line + '\n')

    def clear_log(self):
        """清空日志文件"""
        try:
            # 确认对话框
            result = messagebox.askyesno(
                "确认清空",
                "确定要清空日志文件吗？\n此操作不可恢复！",
                icon='warning'
            )

            if not result:
                return

            log_file = os.path.join('logs', 'ftir_processor.log')

            if os.path.exists(log_file):
                # 清空文件内容
                with open(log_file, 'w', encoding='utf-8') as f:
                    f.write('')

                # 刷新显示
                self.log_text.delete('1.0', tk.END)
                self.all_log_lines = []

                messagebox.showinfo("成功", "日志已清空")
                logger.info("日志文件已清空")
            else:
                messagebox.showwarning("警告", "日志文件不存在")

        except Exception as e:
            messagebox.showerror("错误", f"清空日志失败：{str(e)}")
            logger.error(f"清空日志失败: {str(e)}")

    def export_log(self):
        """导出日志到文件"""
        try:
            # 获取当前显示的日志内容
            log_content = self.log_text.get('1.0', tk.END)

            if not log_content.strip():
                messagebox.showwarning("警告", "没有可导出的日志内容")
                return

            # 选择保存位置
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_filename = f"ftir_log_{timestamp}.txt"

            file_path = filedialog.asksaveasfilename(
                initialdir=self.output_dir if os.path.exists(self.output_dir) else os.getcwd(),
                initialfile=default_filename,
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )

            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(log_content)

                messagebox.showinfo("成功", f"日志已导出到:\n{file_path}")
                logger.info(f"日志已导出到: {file_path}")

        except Exception as e:
            messagebox.showerror("错误", f"导出日志失败：{str(e)}")
            logger.error(f"导出日志失败: {str(e)}")

    def on_result_tree_click(self, event):
        """
        处理峰分析结果列表的单击事件，自动填充参数到输入框

        Args:
            event: Tkinter事件对象
        """
        # 获取点击位置的行
        item = self.result_tree.identify_row(event.y)

        if not item:
            return

        try:
            # 获取记录信息
            values = self.result_tree.item(item, 'values')
            if not values:
                return

            # values格式: (文件名, 编号, 波数, 峰高, 校正峰高, 区间下限, 区间上限, 面积, 校正面积)
            lower_limit = float(values[5])
            upper_limit = float(values[6])

            # 填充到输入框
            self.fill_range_to_inputs(lower_limit, upper_limit)

        except Exception as e:
            logger.error(f"自动填充参数失败: {str(e)}")

    def on_result_tree_right_click(self, event):
        """
        处理峰分析结果列表的右键点击事件

        Args:
            event: Tkinter事件对象
        """
        # 获取点击位置的行
        item = self.result_tree.identify_row(event.y)

        if not item:
            return

        # 选中该行
        self.result_tree.selection_set(item)

        # 创建右键菜单
        context_menu = tk.Menu(self.result_tree, tearoff=0)
        context_menu.add_command(label="删除此记录", command=lambda: self.delete_result_record(item))

        # 显示菜单
        try:
            context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            context_menu.grab_release()

    def delete_result_record(self, item):
        """
        删除峰分析结果列表中的指定记录

        Args:
            item: Treeview中的项目ID
        """
        try:
            # 获取记录信息
            values = self.result_tree.item(item, 'values')
            if not values:
                return

            # values格式: (文件名, 编号, 波数, 峰高, 校正峰高, 区间下限, 区间上限, 面积, 校正面积)
            file_name = values[0]
            peak_number = int(values[1])
            lower_limit = float(values[5])
            upper_limit = float(values[6])

            # 确认删除
            if not messagebox.askyesno("确认删除",
                                      f"确定要删除以下记录吗？\n\n"
                                      f"文件名: {file_name}\n"
                                      f"峰编号: {peak_number}\n"
                                      f"区间: {lower_limit:.2f} - {upper_limit:.2f}"):
                return

            # 从结果列表中删除
            self.result_tree.delete(item)

            # 从 analyzed_ranges 中删除对应的区间（兼容三元组和四元组格式）
            if hasattr(self, 'analyzed_ranges'):
                original_count = len(self.analyzed_ranges)
                # 查找并删除匹配的区间
                new_ranges = []
                for range_data in self.analyzed_ranges:
                    # 兼容旧格式（三元组）和新格式（四元组）
                    if len(range_data) == 3:
                        lower, upper, num = range_data
                        fname = None
                    elif len(range_data) == 4:
                        lower, upper, num, fname = range_data
                    else:
                        continue

                    # 检查是否匹配（需要同时匹配区间、峰编号和文件名）
                    is_match = (abs(lower - lower_limit) < 0.01 and
                               abs(upper - upper_limit) < 0.01 and
                               num == peak_number and
                               (fname is None or fname == file_name))

                    if not is_match:
                        new_ranges.append(range_data)

                self.analyzed_ranges = new_ranges
                logger.info(f"从 analyzed_ranges 删除区间: 原有{original_count}个，现有{len(self.analyzed_ranges)}个")

            # 【修复】检查输入框的值是否与删除的记录匹配，如果匹配则清空输入框
            # 这样可以避免删除记录后仍然显示黄色预览区域
            if hasattr(self, 'peak_lower_var') and hasattr(self, 'peak_upper_var'):
                try:
                    current_lower = float(self.peak_lower_var.get()) if self.peak_lower_var.get() else None
                    current_upper = float(self.peak_upper_var.get()) if self.peak_upper_var.get() else None

                    # 检查输入框的值是否与删除的记录匹配
                    if (current_lower is not None and current_upper is not None and
                        abs(current_lower - lower_limit) < 0.01 and
                        abs(current_upper - upper_limit) < 0.01):
                        # 输入框的值与删除的记录匹配，清空输入框
                        self.peak_lower_var.set("")
                        self.peak_upper_var.set("")
                        logger.info(f"删除记录后，已清空匹配的输入框值: {lower_limit:.2f}-{upper_limit:.2f}")
                except ValueError:
                    pass  # 输入框的值无效，忽略

            # 【修复】如果删除后结果表格为空，也清空输入框（避免显示黄色预览区域）
            if len(self.result_tree.get_children()) == 0:
                if hasattr(self, 'peak_lower_var') and hasattr(self, 'peak_upper_var'):
                    self.peak_lower_var.set("")
                    self.peak_upper_var.set("")
                    logger.info("删除最后一条记录后，已清空峰分析区域的上下限输入框")

            # 重新绘制图形（移除区间标记）
            if hasattr(self, 'peak_ax') and self.peak_ax is not None:
                logger.info("调用 update_peak_plot() 重新绘制图形")
                self.update_peak_plot()

            logger.info(f"已删除峰分析记录: 文件={file_name}, 峰编号={peak_number}, 区间={lower_limit:.2f}-{upper_limit:.2f}")

        except Exception as e:
            messagebox.showerror("错误", f"删除记录失败：{str(e)}")
            logger.error(f"删除峰分析记录失败: {str(e)}")

    def on_peak_plot_right_click(self, event):
        """
        处理图形区域的右键点击事件

        Args:
            event: Matplotlib事件对象
        """
        if event.xdata is None or event.ydata is None:
            return

        click_x = event.xdata

        # 检查是否点击在某个已分析的区间内
        clicked_range = None
        if hasattr(self, 'analyzed_ranges'):
            for range_data in self.analyzed_ranges:
                # 兼容旧格式（三元组）和新格式（四元组）
                if len(range_data) == 3:
                    lower, upper, peak_number = range_data
                    file_name = None
                elif len(range_data) == 4:
                    lower, upper, peak_number, file_name = range_data
                else:
                    continue

                # 考虑x轴可能倒置的情况
                if min(lower, upper) <= click_x <= max(lower, upper):
                    clicked_range = (lower, upper, peak_number, file_name)
                    break

        # 检查是否有当前选择的区间（但未分析）
        has_current_selection = False
        if self.peak_lower_var.get() and self.peak_upper_var.get():
            try:
                current_lower = float(self.peak_lower_var.get())
                current_upper = float(self.peak_upper_var.get())
                if min(current_lower, current_upper) <= click_x <= max(current_lower, current_upper):
                    has_current_selection = True
            except ValueError:
                pass

        # 创建右键菜单
        context_menu = tk.Menu(self.peak_ax.figure.canvas.get_tk_widget(), tearoff=0)

        if clicked_range:
            # 点击在已分析的区间内
            lower, upper, peak_number, file_name = clicked_range
            context_menu.add_command(
                label=f"删除区间分析 (峰#{peak_number})",
                command=lambda: self.delete_analyzed_range_from_plot(lower, upper, peak_number, file_name)
            )
            # 添加"填充参数"选项
            context_menu.add_separator()
            context_menu.add_command(
                label=f"填充参数到输入框",
                command=lambda: self.fill_range_to_inputs(lower, upper)
            )
        elif has_current_selection:
            # 点击在当前选择的区间内（但未分析）
            context_menu.add_command(
                label="取消选择",
                command=self.clear_peak_selection
            )
        else:
            # 没有点击在任何区间内
            context_menu.add_command(label="(无可用操作)", state=tk.DISABLED)

        # 显示菜单
        try:
            # 将matplotlib坐标转换为屏幕坐标
            canvas = self.peak_canvas.get_tk_widget()
            x_screen = canvas.winfo_rootx() + int(event.x)
            y_screen = canvas.winfo_rooty() + int(event.y)
            context_menu.tk_popup(x_screen, y_screen)
        finally:
            context_menu.grab_release()

    def delete_analyzed_range_from_plot(self, lower, upper, peak_number, file_name=None):
        """
        从图形中删除已分析的区间

        Args:
            lower: 区间下限
            upper: 区间上限
            peak_number: 峰编号
            file_name: 文件名（可选）
        """
        try:
            # 确认删除
            if not messagebox.askyesno("确认删除",
                                      f"确定要删除以下区间的分析吗？\n\n"
                                      f"峰编号: {peak_number}\n"
                                      f"区间: {lower:.2f} - {upper:.2f}"):
                return

            # 从 analyzed_ranges 中删除（兼容三元组和四元组格式）
            if hasattr(self, 'analyzed_ranges'):
                original_count = len(self.analyzed_ranges)
                new_ranges = []
                for range_data in self.analyzed_ranges:
                    # 兼容旧格式（三元组）和新格式（四元组）
                    if len(range_data) == 3:
                        l, u, n = range_data
                        fname = None
                    elif len(range_data) == 4:
                        l, u, n, fname = range_data
                    else:
                        continue

                    # 检查是否匹配
                    is_match = (abs(l - lower) < 0.01 and
                               abs(u - upper) < 0.01 and
                               n == peak_number and
                               (file_name is None or fname is None or fname == file_name))

                    if not is_match:
                        new_ranges.append(range_data)

                self.analyzed_ranges = new_ranges
                logger.info(f"从 analyzed_ranges 删除区间: 原有{original_count}个，现有{len(self.analyzed_ranges)}个")

            # 从结果列表中删除对应的记录
            deleted_from_tree = False
            for item in self.result_tree.get_children():
                values = self.result_tree.item(item, 'values')
                if values:
                    item_peak_number = int(values[1])
                    item_lower = float(values[5])
                    item_upper = float(values[6])

                    if (item_peak_number == peak_number and
                        abs(item_lower - lower) < 0.01 and
                        abs(item_upper - upper) < 0.01):
                        self.result_tree.delete(item)
                        deleted_from_tree = True
                        logger.info(f"从结果列表删除记录: 峰#{peak_number}")
                        break

            if not deleted_from_tree:
                logger.warning(f"未在结果列表中找到匹配的记录: 峰#{peak_number}")

            # 【修复】检查输入框的值是否与删除的记录匹配，如果匹配则清空输入框
            # 这样可以避免删除记录后仍然显示黄色预览区域
            if hasattr(self, 'peak_lower_var') and hasattr(self, 'peak_upper_var'):
                try:
                    current_lower = float(self.peak_lower_var.get()) if self.peak_lower_var.get() else None
                    current_upper = float(self.peak_upper_var.get()) if self.peak_upper_var.get() else None

                    # 检查输入框的值是否与删除的记录匹配
                    if (current_lower is not None and current_upper is not None and
                        abs(current_lower - lower) < 0.01 and
                        abs(current_upper - upper) < 0.01):
                        # 输入框的值与删除的记录匹配，清空输入框
                        self.peak_lower_var.set("")
                        self.peak_upper_var.set("")
                        logger.info(f"删除记录后，已清空匹配的输入框值: {lower:.2f}-{upper:.2f}")
                except ValueError:
                    pass  # 输入框的值无效，忽略

            # 【修复】如果删除后结果表格为空，也清空输入框（避免显示黄色预览区域）
            if len(self.result_tree.get_children()) == 0:
                if hasattr(self, 'peak_lower_var') and hasattr(self, 'peak_upper_var'):
                    self.peak_lower_var.set("")
                    self.peak_upper_var.set("")
                    logger.info("删除最后一条记录后，已清空峰分析区域的上下限输入框")

            # 重新绘制图形
            if hasattr(self, 'peak_ax') and self.peak_ax is not None:
                logger.info("调用 update_peak_plot() 重新绘制图形")
                self.update_peak_plot()

            logger.info(f"已从图形中删除区间分析: 峰编号={peak_number}, 区间={lower:.2f}-{upper:.2f}")

        except Exception as e:
            messagebox.showerror("错误", f"删除区间分析失败：{str(e)}")
            logger.error(f"删除区间分析失败: {str(e)}")

    def fill_range_to_inputs(self, lower, upper):
        """
        将区间范围填充到输入框

        Args:
            lower: 区间下限
            upper: 区间上限
        """
        try:
            # 填充到输入框
            self.peak_lower_var.set(f"{lower:.2f}")
            self.peak_upper_var.set(f"{upper:.2f}")

            logger.info(f"已将区间范围填充到输入框: 下限={lower:.2f}, 上限={upper:.2f}")

        except Exception as e:
            logger.error(f"填充区间范围失败: {str(e)}")


# 在文件末尾添加以下代码
def main():
    root = tk.Tk()
    root.title("FTIR_Processor")
    
    # 获取屏幕尺寸
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    
    # 计算16:9比例的窗口大小
    window_height = int(screen_height * 0.9)  # 使用90%的屏幕高度
    window_width = int(window_height * 16 / 9)  # 16:9比例
    
    # 确保窗口宽度不超过屏幕宽度
    if window_width > screen_width:
        window_width = int(screen_width * 0.9)
        window_height = int(window_width * 9 / 16)
    
    # 计算窗口位置，使其居中
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    
    # 设置窗口大小和位置
    root.geometry(f"{window_width}x{window_height}+{x}+{y}")
    
    # 设置窗口最大化
    root.state('zoomed')  # Windows系统使用'zoomed'
    # 如果是Linux或Mac系统，使用：
    # root.attributes('-zoomed', True)  # Linux
    # root.attributes('-fullscreen', True)  # Mac
    
    app = SpectralProcessorGUI(root)  # 必须保持引用以防止被垃圾回收
    logger.info("应用程序启动")
    root.mainloop()

if __name__ == "__main__":
    main()