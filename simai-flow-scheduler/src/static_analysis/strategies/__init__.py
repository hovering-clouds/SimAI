"""Analysis strategies — composable workflows of analysis passes."""
from .cassini_strategy import CassiniAnalyzer, CassiniAnalysisResult
from .default_strategy import (
    DefaultAnalyzer,
    DefaultAnalysisResult,
    DynamicPipelineAnalyzer,
    PipelineAnalyzer,
)
from .example_strategy import ExampleAnalyzer, ExampleAnalysisResult
from .puppeteer_strategy import PuppeteerAnalyzer, PuppeteerAnalysisResult
from .mfs_strategy import MfsAnalyzer, MfsAnalysisResult

__all__ = [
    "CassiniAnalyzer",
    "CassiniAnalysisResult",
    "DefaultAnalyzer",
    "DefaultAnalysisResult",
    "PipelineAnalyzer",
    "DynamicPipelineAnalyzer",
    "ExampleAnalyzer",
    "ExampleAnalysisResult",
    "PuppeteerAnalyzer",
    "PuppeteerAnalysisResult",
    "MfsAnalyzer",
    "MfsAnalysisResult",
]
