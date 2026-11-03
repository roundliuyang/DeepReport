import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from jinja2 import Template
import os

from .chart_generator import ChartGenerator
from .citation_manager import CitationManager

logger = logging.getLogger(__name__)

class HTMLReportGenerator:
    """Generate HTML reports with charts and citations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.chart_generator = ChartGenerator(config)
        self.citation_manager = CitationManager(config)
        
        # Load HTML template
        self.template = self._load_template()
    
    def _load_template(self) -> Template:
        """Load HTML report template"""
        template_content = """
<!DOCTYPE html>
<html lang="{{ 'zh-CN' if chinese else 'en' }}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels@2"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f8f9fa;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 3rem 0;
            margin-bottom: 2rem;
        }
        .section {
            background: white;
            border-radius: 10px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .chart-container {
            position: relative;
            height: 400px;
            margin: 2rem 0;
        }
        .metric-card {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        }
        .data-table {
            overflow-x: auto;
        }
        .citation {
            font-size: 0.9em;
            color: #666;
            border-left: 3px solid #007bff;
            padding-left: 1rem;
            margin: 1rem 0;
        }
        .executive-summary {
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 1.5rem;
            margin-bottom: 2rem;
        }
        .recommendation {
            background: #f3e5f5;
            border-left: 4px solid #9c27b0;
            padding: 1rem;
            margin: 1rem 0;
        }
        .risk-high {
            background: #ffebee;
            border-left: 4px solid #f44336;
        }
        .risk-medium {
            background: #fff3e0;
            border-left: 4px solid #ff9800;
        }
        .risk-low {
            background: #e8f5e8;
            border-left: 4px solid #4caf50;
        }
        .nav-tabs .nav-link {
            color: #667eea;
            font-weight: 500;
        }
        .nav-tabs .nav-link.active {
            color: #764ba2;
            font-weight: 600;
        }
        .footer {
            background: #343a40;
            color: white;
            padding: 2rem 0;
            margin-top: 3rem;
        }
        /* Report reading layout */
        * { box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif; color: #253247; background: #f3f5f9; line-height: 1.8; overflow-wrap: anywhere; }
        .container { max-width: 1160px; margin: 0 auto; padding-left: 24px; padding-right: 24px; }
        .header { background: #253b60; padding: 48px 0; }
        .header h1 { font-size: clamp(26px, 3.2vw, 40px); font-weight: 650; line-height: 1.35; }
        .header .lead { opacity: .8; font-size: 16px; margin-top: 16px; }
        .section { padding: 32px; border: 1px solid #e3e8f0; border-radius: 16px; box-shadow: 0 4px 16px #24355004; }
        .section > h2 { font-size: 22px; font-weight: 650; margin-bottom: 24px; }
        .metric-card { background: #edf2fb; color: #294773; border: 1px solid #dbe4f3; }
        .header .metric-card { background: #ffffff10; border-color: #ffffff30; color: white; }
        .metric-card h5, .metric-card h4 { font-size: 15px; }
        .executive-summary { background: #eaf0fa; border-color: #5976ae; border-radius: 8px; padding: 24px 28px; }
        .executive-summary h3 { font-size: 20px; }
        .chart-container { height: 300px; margin: 24px 0; }
        .nav-tabs { display: flex; flex-wrap: wrap; gap: 6px; padding: 0 0 12px; border-bottom: 1px solid #e1e7ef; list-style: none; }
        .nav-tabs .nav-link { border: 0; border-radius: 8px; padding: 10px 16px; color: #5d6c83; background: transparent; font-size: 15px; line-height: 1.5; }
        .nav-tabs .nav-link:hover { background: #f1f4fa; color: #294773; }
        .nav-tabs .nav-link.active { background: #e8eefb; color: #294d8b; font-weight: 650; }
        .nav-link:focus-visible, summary:focus-visible, a:focus-visible { outline: 2px solid #4266a5; outline-offset: 3px; }
        .tab-content > .tab-pane { display: none; }
        .tab-content > .active { display: block; }
        #analysisTabsContent { padding-top: 18px; }
        .reading-column { max-width: 820px; margin: 0 auto; }
        .section-intro { color: #66748a; font-size: 14px; margin-bottom: 24px; }
        .method-steps { padding-left: 28px; }
        .method-steps li { padding: 0 0 20px 14px; }
        .method-steps li::marker { color: #5976ae; font-weight: 650; }
        .method-steps h3 { font-size: 18px; margin-bottom: 8px; }
        .method-steps p { color: #66748a; margin: 0; }
        .source-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }
        .source-stack { display: grid; gap: 20px; }
        .source-card { border: 1px solid #dfe6ef; border-radius: 12px; padding: 24px; background: #fff; min-width: 0; }
        .source-card h3 { font-size: 17px; line-height: 1.55; margin: 8px 0 14px; font-weight: 650; }
        .source-card p { font-size: 14px; color: #59667a; }
        .source-meta { color: #77859b; font-size: 12px; letter-spacing: .04em; margin-bottom: 8px; }
        .source-link { display: inline-block; font-size: 13px; color: #365f9d; text-decoration: none; border-bottom: 1px solid #cbd7e9; }
        .source-link:hover { color: #173d75; border-color: #173d75; }
        .source-preview { margin-top: 18px; }
        .source-details { border-top: 1px solid #e6ebf2; margin-top: 16px; padding-top: 12px; }
        .source-details summary { cursor: pointer; color: #365f9d; font-size: 14px; }
        .source-details[open] summary { margin-bottom: 16px; }
        .research-document + .research-document { border-top: 2px solid #e1e7ef; padding-top: 32px; margin-top: 40px; }
        .task-scope { margin-bottom: 20px; color: #66748a; font-size: 13px; }
        .task-scope summary { cursor: pointer; }
        .task-scope[open] summary { margin-bottom: 12px; }
        #analysisTabsContent .tab-pane { transition: none; }
        .task-heading { font-size: 19px; line-height: 1.6; margin-bottom: 20px; }
        .research-summary { padding: 20px 24px; background: #f0f4fb; border-left: 3px solid #6986bc; border-radius: 0 10px 10px 0; }
        .research-summary p:last-child { margin-bottom: 0; }
        .research-document p { font-size: 16px; line-height: 1.95; margin-bottom: 16px; }
        .research-subsection { padding-top: 30px; }
        .research-subsection h4 { font-size: 19px; font-weight: 650; line-height: 1.6; margin: 0 0 14px; display: flex; gap: 12px; }
        .section-number { font-size: 13px; color: #7f94b6; padding-top: 5px; flex-shrink: 0; }
        .evidence-links { display: flex; gap: 8px 14px; flex-wrap: wrap; align-items: baseline; padding-bottom: 20px; border-bottom: 1px solid #e8ecf2; }
        .evidence-links > span { font-size: 12px; color: #8b96a7; }
        .research-limitations { background: #faf8f2; border: 1px solid #eae4d5; border-radius: 10px; padding: 20px 24px; margin-top: 28px; }
        .research-limitations h4 { font-size: 16px; margin-bottom: 12px; }
        .research-limitations ul { padding-left: 20px; margin-bottom: 0; font-size: 14px; color: #726b5c; }
        .research-limitations li + li { margin-top: 8px; }
        .citation { padding: 16px 20px; background: #f8fafc; border-color: #ccd7e9; border-radius: 0 8px 8px 0; overflow-wrap: anywhere; }
        .recommendation { border-radius: 0 8px 8px 0; }
        .footer { background: #253b60; font-size: 13px; }
        @media (max-width: 600px) {
            .container { padding-left: 14px; padding-right: 14px; }
            .header { padding: 28px 0; }
            .section { padding: 20px 16px; margin-bottom: 18px; }
            .source-grid { grid-template-columns: 1fr; }
            .source-card { padding: 18px; }
            .nav-tabs { gap: 4px; }
            .nav-tabs .nav-link { padding: 9px 10px; font-size: 14px; }
            .research-summary, .research-limitations { padding: 16px; }
            .research-document p { font-size: 15px; }
            .research-subsection h4 { font-size: 17px; }
            .chart-container { height: 240px; }
        }
        @media print {
            .tab-content > .tab-pane { display: block !important; opacity: 1 !important; break-before: auto; }
            .nav-tabs, .footer { display: none; }
            .section { box-shadow: none; padding: 16px; }
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="container">
            <div class="row">
                <div class="col-md-8">
                    <h1 class="display-4">{{ title }}</h1>
                    <p class="lead">{{ subtitle }}</p>
                    <p class="mb-0">
                        <i class="fas fa-calendar-alt"></i> Generated: {{ generated_date }}
                        <span class="ms-3"><i class="fas fa-user"></i> {{ author }}</span>
                    </p>
                </div>
                <div class="col-md-4 text-end">
                    <div class="metric-card">
                        <h4>{{ '研究来源' if chinese else 'Research Sources' }}</h4>
                        <h2>{{ data_points }}</h2>
                        <small>{{ '已提取并采纳的来源数量' if chinese else 'Sources read and selected' }}</small>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <div class="container">
        <!-- Executive Summary -->
        {% if executive_summary %}
        <div class="executive-summary">
            <h3><i class="fas fa-lightbulb"></i> Executive Summary</h3>
            {{ executive_summary | safe }}
        </div>
        {% endif %}

        <!-- Key Metrics -->
        {% if key_metrics %}
        <div class="section">
            <h2><i class="fas fa-chart-line"></i> Key Metrics</h2>
            <div class="row">
                {% for metric in key_metrics %}
                <div class="col-md-3">
                    <div class="metric-card">
                        <h5>{{ metric.name }}</h5>
                        <h3>{{ metric.value }}</h3>
                        <small>{{ metric.description }}</small>
                    </div>
                </div>
                {% endfor %}
            </div>
        </div>
        {% endif %}

        <!-- Interactive Charts -->
        {% if charts %}
        <div class="section">
            <h2><i class="fas fa-chart-bar"></i> Data Visualization</h2>
            
            <!-- Chart Navigation Tabs -->
            <ul class="nav nav-tabs" id="chartTabs" role="tablist">
                {% for chart in charts %}
                <li class="nav-item" role="presentation">
                    <button class="nav-link {% if loop.first %}active{% endif %}" 
                            id="{{ chart.id }}-tab" data-bs-toggle="tab" 
                            data-bs-target="#{{ chart.id }}" type="button" role="tab">
                        {{ chart.title }}
                    </button>
                </li>
                {% endfor %}
            </ul>
            
            <div class="tab-content" id="chartTabsContent">
                {% for chart in charts %}
                <div class="tab-pane fade {% if loop.first %}show active{% endif %}" 
                     id="{{ chart.id }}" role="tabpanel">
                    <div class="chart-container">
                        <canvas id="{{ chart.id }}-canvas"></canvas>
                    </div>
                    <p class="text-muted">{{ chart.description }}</p>
                </div>
                {% endfor %}
            </div>
        </div>
        {% endif %}

        <!-- Analysis Sections -->
        {% if sections %}
        <div class="section">
            <h2>{{ "研究报告" if chinese else "Research Report" }}</h2>
            
            <ul class="nav nav-tabs" id="analysisTabs" role="tablist">
                {% for section in sections %}
                <li class="nav-item" role="presentation">
                    <button class="nav-link {% if loop.first %}active{% endif %}" 
                            id="{{ section.id }}-tab" data-bs-toggle="tab" 
                            data-bs-target="#{{ section.id }}" type="button" role="tab"
                            aria-controls="{{ section.id }}" aria-selected="{{ 'true' if loop.first else 'false' }}">
                        {{ {"methodology": "研究方法", "extracted_content": "提取的研究内容", "search_results": "搜索结果", "research_analysis": "分析", "final_report": "最终报告"}.get(section.id, section.title) if chinese else section.title }}
                    </button>
                </li>
                {% endfor %}
            </ul>
            
            <div class="tab-content" id="analysisTabsContent">
                {% for section in sections %}
                <div class="tab-pane fade {% if loop.first %}show active{% endif %}" 
                     id="{{ section.id }}" role="tabpanel" aria-labelledby="{{ section.id }}-tab">
                    <div class="mt-3">
                        {{ section.content | safe }}
                    </div>
                    
                    {% if section.data_table %}
                    <div class="data-table mt-3">
                        <h5>{{ section.data_table.title }}</h5>
                        <div class="table-responsive">
                            <table class="table table-striped">
                                <thead>
                                    <tr>
                                        {% for header in section.data_table.headers %}
                                        <th>{{ header }}</th>
                                        {% endfor %}
                                    </tr>
                                </thead>
                                <tbody>
                                    {% for row in section.data_table.rows %}
                                    <tr>
                                        {% for cell in row %}
                                        <td>{{ cell }}</td>
                                        {% endfor %}
                                    </tr>
                                    {% endfor %}
                                </tbody>
                            </table>
                        </div>
                    </div>
                    {% endif %}
                </div>
                {% endfor %}
            </div>
        </div>
        {% endif %}

        <!-- Risk Assessment -->
        {% if risk_assessment %}
        <div class="section">
            <h2><i class="fas fa-exclamation-triangle"></i> Risk Assessment</h2>
            {% for risk in risk_assessment %}
            <div class="recommendation risk-{{ risk.level }}">
                <h5>{{ risk.category }}</h5>
                <p><strong>Level:</strong> {{ risk.level | title }}</p>
                <p><strong>Impact:</strong> {{ risk.impact }}</p>
                <p><strong>Mitigation:</strong> {{ risk.mitigation }}</p>
            </div>
            {% endfor %}
        </div>
        {% endif %}

        <!-- Recommendations -->
        {% if recommendations %}
        <div class="section">
            <h2><i class="fas fa-thumbs-up"></i> Recommendations</h2>
            {% for rec in recommendations %}
            <div class="recommendation">
                <h5>{{ rec.title }}</h5>
                <p>{{ rec.description }}</p>
                <div class="mt-2">
                    <span class="badge bg-primary">{{ rec.priority }}</span>
                    <span class="badge bg-secondary">{{ rec.timeline }}</span>
                </div>
            </div>
            {% endfor %}
        </div>
        {% endif %}

        <!-- Citations -->
        {% if citations %}
        <div class="section">
            <h2><i class="fas fa-quote-left"></i> Citations & References</h2>
            {% for citation in citations %}
            <div class="citation">
                <strong>{{ citation.id }}.</strong> {{ citation.text }}
                {% if citation.url %}
                <br><a href="{{ citation.url }}" target="_blank" class="text-primary">{{ citation.url }}</a>
                {% endif %}
                {% if citation.date %}
                <br><small>Accessed: {{ citation.date }}</small>
                {% endif %}
            </div>
            {% endfor %}
        </div>
        {% endif %}

        <!-- Data Sources -->
        {% if data_sources %}
        <div class="section">
            <h2><i class="fas fa-database"></i> Data Sources</h2>
            <div class="row">
                {% for source in data_sources %}
                <div class="col-md-6">
                    <div class="card mb-3">
                        <div class="card-body">
                            <h5 class="card-title">{{ source.name }}</h5>
                            <p class="card-text">{{ source.description }}</p>
                            {% if source.url %}
                            <a href="{{ source.url }}" class="btn btn-outline-primary btn-sm" target="_blank">
                                <i class="fas fa-external-link-alt"></i> Visit Source
                            </a>
                            {% endif %}
                        </div>
                    </div>
                </div>
                {% endfor %}
            </div>
        </div>
        {% endif %}
    </div>

    <!-- Footer -->
    <div class="footer">
        <div class="container">
            <div class="row">
                <div class="col-md-6">
                    <h5>DeepReport</h5>
                    <p>AI-powered financial research and report generation system</p>
                </div>
                <div class="col-md-6 text-end">
                    <p class="mb-0">
                        Report ID: {{ report_id }}<br>
                        Version: {{ version }}
                    </p>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Chart configurations
        const chartConfigs = {{ chart_configs | tojson }};
        
        // Initialize charts when page loads
        document.addEventListener('DOMContentLoaded', function() {
            // Keep tabs usable when the external Bootstrap script is unavailable.
            if (typeof bootstrap === 'undefined') {
                document.querySelectorAll('[data-bs-toggle="tab"]').forEach(button => {
                    button.addEventListener('click', () => {
                        const nav = button.closest('[role="tablist"]');
                        nav.querySelectorAll('[role="tab"]').forEach(tab => {
                            const active = tab === button;
                            tab.classList.toggle('active', active);
                            tab.setAttribute('aria-selected', String(active));
                            const pane = document.querySelector(tab.dataset.bsTarget);
                            pane.classList.toggle('active', active);
                            pane.classList.toggle('show', active);
                        });
                    });
                });
            }
            // Initialize all charts
            Object.keys(chartConfigs).forEach(chartId => {
                const ctx = document.getElementById(chartId + '-canvas');
                if (ctx && typeof Chart !== "undefined") {
                    new Chart(ctx, chartConfigs[chartId]);
                }
            });
            
            // Add smooth scrolling
            document.querySelectorAll('a[href^="#"]').forEach(anchor => {
                anchor.addEventListener('click', function (e) {
                    e.preventDefault();
                    const target = document.querySelector(this.getAttribute('href'));
                    if (target) {
                        target.scrollIntoView({
                            behavior: 'smooth'
                        });
                    }
                });
            });
            
            // Add chart download functionality
            document.querySelectorAll('canvas').forEach(canvas => {
                canvas.addEventListener('dblclick', function() {
                    const link = document.createElement('a');
                    link.download = this.id.replace('-canvas', '') + '.png';
                    link.href = this.toDataURL();
                    link.click();
                });
            });
        });
    </script>
</body>
</html>
        """
        return Template(template_content, autoescape=True)
    
    async def generate_report(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate HTML report"""
        try:
            # Generate charts
            chart_configs = {}
            charts = []
            
            if report_data.get("charts_data"):
                for chart_data in report_data["charts_data"]:
                    chart_config = await self.chart_generator.generate_chart(chart_data)
                    if chart_config.get("success"):
                        chart_configs[chart_data["id"]] = chart_config["config"]
                        charts.append({
                            "id": chart_data["id"],
                            "title": chart_data["title"],
                            "description": chart_data.get("description", ""),
                            "type": chart_data.get("type", "line")
                        })
            
            # Process citations
            processed_citations = []
            if report_data.get("citations"):
                processed_citations = await self.citation_manager.process_citations(
                    report_data["citations"]
                )
            
            # Prepare template data
            template_data = {
                "chinese": any("\u4e00" <= c <= "\u9fff" for c in report_data.get("title", "")),
                "title": report_data.get("title", "Financial Research Report"),
                "subtitle": report_data.get("subtitle", "Comprehensive Analysis and Insights"),
                "generated_date": datetime.now().strftime("%B %d, %Y at %I:%M %p"),
                "author": report_data.get("author", "DeepReport AI"),
                "confidence_score": report_data.get("confidence_score"),
                "data_points": report_data.get("data_points", 0),
                "executive_summary": report_data.get("executive_summary", ""),
                "key_metrics": report_data.get("key_metrics", []),
                "charts": charts,
                "chart_configs": chart_configs,
                "sections": report_data.get("sections", []),
                "risk_assessment": report_data.get("risk_assessment", []),
                "recommendations": report_data.get("recommendations", []),
                "citations": processed_citations,
                "data_sources": report_data.get("data_sources", []),
                "report_id": report_data.get("report_id", f"DR-{datetime.now().strftime('%Y%m%d%H%M%S')}"),
                "version": report_data.get("version", "1.0")
            }
            
            # Generate HTML
            html_content = self.template.render(**template_data)
            
            return {
                "success": True,
                "html_content": html_content,
                "charts_count": len(charts),
                "citations_count": len(processed_citations),
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to generate HTML report: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def save_report(self, report_data: Dict[str, Any], output_path: str) -> bool:
        """Generate and save HTML report to file"""
        try:
            result = await self.generate_report(report_data)
            
            if result["success"]:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(result["html_content"])
                
                logger.info(f"Report saved to {output_path}")
                return True
            else:
                logger.error(f"Failed to generate report: {result['error']}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
            return False