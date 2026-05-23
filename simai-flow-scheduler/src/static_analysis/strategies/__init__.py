"""Analysis strategies — composable workflows of analysis passes."""
from .cassini_strategy import CassiniAnalyzer, CassiniAnalysisResult
from .default_strategy import DefaultAnalyzer, DefaultAnalysisResult
from .example_strategy import ExampleAnalyzer, ExampleAnalysisResult
from .puppeteer_strategy import PuppeteerAnalyzer, PuppeteerAnalysisResult

__all__ = [
    "CassiniAnalyzer",
    "CassiniAnalysisResult",
    "DefaultAnalyzer",
    "DefaultAnalysisResult",
    "ExampleAnalyzer",
    "ExampleAnalysisResult",
    "PuppeteerAnalyzer",
    "PuppeteerAnalysisResult",
]
