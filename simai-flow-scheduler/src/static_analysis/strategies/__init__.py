"""Analysis strategies — composable workflows of analysis passes."""
from .default_strategy import DefaultAnalyzer, DefaultAnalysisResult
from .example_strategy import ExampleAnalyzer, ExampleAnalysisResult
from .puppeteer_strategy import PuppeteerAnalyzer, PuppeteerAnalysisResult

__all__ = [
    "DefaultAnalyzer",
    "DefaultAnalysisResult",
    "ExampleAnalyzer",
    "ExampleAnalysisResult",
    "PuppeteerAnalyzer",
    "PuppeteerAnalysisResult",
]
