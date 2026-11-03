"""
Deep Analyze Agent for financial data analysis and valuation using SmolAgents
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from smolagents import Tool
from smolagents.memory import AgentMemory
from smolagents.models import OpenAIModel

from .base_agent import BaseAgent, Task, TaskResult

logger = logging.getLogger(__name__)

class FinancialMetricsTool(Tool):
    """Tool for analyzing financial metrics"""
    
    def __init__(self):
        super().__init__()
        self.name = "analyze_financial_metrics"
        self.description = "Analyze financial metrics and performance indicators"
        self.inputs = {
            "financial_data": {
                "type": "array",
                "description": "Array of financial data objects"
            },
            "metrics_to_analyze": {
                "type": "array",
                "description": "Specific metrics to focus on",
                "nullable": True
            },
            "industry_context": {
                "type": "string",
                "description": "Industry context for analysis",
                "nullable": True
            }
        }
        self.output_type = "object"

    def forward(self, financial_data: List[Dict[str, Any]], metrics_to_analyze: List[str] = None, industry_context: str = None) -> Dict[str, Any]:
        """Analyze financial metrics (synchronous)"""
        try:
            if metrics_to_analyze is None:
                metrics_to_analyze = ["revenue", "profit", "growth", "efficiency"]

            analysis = {
                "metrics_analyzed": metrics_to_analyze,
                "data_points": len(financial_data),
                "key_findings": [],
                "trends": [],
                "anomalies": [],
                "recommendations": []
            }

            # Analyze each metric
            for metric in metrics_to_analyze:
                metric_analysis = self._analyze_single_metric_sync(financial_data, metric, industry_context)
                analysis["key_findings"].append(metric_analysis)

            # Identify trends
            analysis["trends"] = self._identify_trends_sync(financial_data, metrics_to_analyze)

            # Detect anomalies
            analysis["anomalies"] = self._detect_anomalies_sync(financial_data)

            # Generate recommendations
            analysis["recommendations"] = self._generate_recommendations_sync(analysis)

            return analysis

        except Exception as e:
            logger.error(f"Financial metrics analysis failed: {e}")
            return {"error": str(e)}
    
    async def _run_async(self, financial_data: List[Dict[str, Any]], metrics_to_analyze: List[str], industry_context: str) -> Dict[str, Any]:
        """Async helper for financial metrics analysis"""
        try:
            if metrics_to_analyze is None:
                metrics_to_analyze = ["revenue", "profit", "growth", "efficiency"]
            
            analysis = {
                "metrics_analyzed": metrics_to_analyze,
                "data_points": len(financial_data),
                "key_findings": [],
                "trends": [],
                "anomalies": [],
                "recommendations": []
            }
            
            # Analyze each metric
            for metric in metrics_to_analyze:
                metric_analysis = await self._analyze_single_metric(financial_data, metric, industry_context)
                analysis["key_findings"].append(metric_analysis)
            
            # Identify trends
            analysis["trends"] = await self._identify_trends(financial_data, metrics_to_analyze)
            
            # Detect anomalies
            analysis["anomalies"] = await self._detect_anomalies(financial_data)
            
            # Generate recommendations
            analysis["recommendations"] = await self._generate_recommendations(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Financial metrics analysis failed: {e}")
            return {"error": str(e)}
    
    def _analyze_single_metric_sync(self, data: List[Dict[str, Any]], metric: str, industry_context: str = None) -> Dict[str, Any]:
        """Analyze a single financial metric (synchronous)"""
        try:
            # Extract values for the metric
            values = []
            for item in data:
                if metric in item:
                    values.append(item[metric])
            
            if not values:
                return {"metric": metric, "error": "No data found for this metric"}
            
            # Basic statistical analysis
            avg_value = sum(values) / len(values)
            min_value = min(values)
            max_value = max(values)
            
            # Calculate growth if time series data
            growth_rates = []
            if len(values) > 1:
                for i in range(1, len(values)):
                    if values[i-1] != 0:
                        growth_rate = ((values[i] - values[i-1]) / values[i-1]) * 100
                        growth_rates.append(growth_rate)
            
            avg_growth = sum(growth_rates) / len(growth_rates) if growth_rates else 0
            
            return {
                "metric": metric,
                "average": avg_value,
                "min": min_value,
                "max": max_value,
                "growth_rate": avg_growth,
                "volatility": max_value - min_value,
                "data_points": len(values),
                "industry_context": industry_context
            }
            
        except Exception as e:
            return {"metric": metric, "error": str(e)}
    
    def _identify_trends_sync(self, data: List[Dict[str, Any]], metrics: List[str]) -> List[Dict[str, Any]]:
        """Identify trends in financial data (synchronous)"""
        trends = []
        
        for metric in metrics:
            try:
                values = [item.get(metric, 0) for item in data if metric in item]
                if len(values) >= 3:
                    # Simple trend detection
                    first_half = values[:len(values)//2]
                    second_half = values[len(values)//2:]
                    
                    first_avg = sum(first_half) / len(first_half)
                    second_avg = sum(second_half) / len(second_half)
                    
                    trend_direction = "increasing" if second_avg > first_avg else "decreasing"
                    trend_strength = abs(second_avg - first_avg) / first_avg if first_avg != 0 else 0
                    
                    trends.append({
                        "metric": metric,
                        "direction": trend_direction,
                        "strength": trend_strength,
                        "first_period_avg": first_avg,
                        "second_period_avg": second_avg
                    })
                    
            except Exception as e:
                logger.warning(f"Failed to analyze trend for {metric}: {e}")
        
        return trends
    
    def _detect_anomalies_sync(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect anomalies in financial data (synchronous)"""
        anomalies = []
        
        try:
            # Get all numeric fields
            numeric_fields = set()
            for item in data:
                for key, value in item.items():
                    if isinstance(value, (int, float)):
                        numeric_fields.add(key)
            
            # Check for anomalies in each field
            for field in numeric_fields:
                values = [item[field] for item in data if field in item]
                if len(values) >= 5:
                    # Simple anomaly detection using standard deviation
                    avg = sum(values) / len(values)
                    variance = sum((x - avg) ** 2 for x in values) / len(values)
                    std_dev = variance ** 0.5
                    
                    # Find values more than 2 standard deviations from mean
                    for i, value in enumerate(values):
                        if abs(value - avg) > 2 * std_dev:
                            anomalies.append({
                                "field": field,
                                "value": value,
                                "expected_range": [avg - 2*std_dev, avg + 2*std_dev],
                                "deviation": abs(value - avg) / std_dev,
                                "data_point_index": i
                            })
                            
        except Exception as e:
            logger.warning(f"Anomaly detection failed: {e}")
        
        return anomalies
    
    def _generate_recommendations_sync(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis (synchronous)"""
        recommendations = []
        
        try:
            # Analyze trends for recommendations
            for trend in analysis.get("trends", []):
                if trend["direction"] == "decreasing" and trend["strength"] > 0.1:
                    recommendations.append(f"Consider investigating declining {trend['metric']} trend")
                elif trend["direction"] == "increasing" and trend["strength"] > 0.1:
                    recommendations.append(f"Positive {trend['metric']} trend identified - consider expanding in this area")
            
            # Analyze anomalies for recommendations
            if analysis.get("anomalies"):
                recommendations.append("Review detected anomalies for potential data quality issues")
            
            # Default recommendations
            if not recommendations:
                recommendations.append("Continue monitoring financial metrics regularly")
                recommendations.append("Consider benchmarking against industry standards")
                
        except Exception as e:
            logger.warning(f"Failed to generate recommendations: {e}")
            recommendations.append("Manual review recommended due to analysis limitations")
        
        return recommendations

class SentimentAnalysisTool(Tool):
    """Tool for sentiment analysis of financial text"""
    
    def __init__(self):
        super().__init__()
        self.name = "analyze_sentiment"
        self.description = "Analyze sentiment in financial text data"
        self.inputs = {
            "texts": {
                "type": "array",
                "description": "Array of texts to analyze"
            },
            "context": {
                "type": "string",
                "description": "Context for sentiment analysis",
                "nullable": True
            }
        }
        self.output_type = "object"

    def forward(self, texts: List[str], context: str = None) -> Dict[str, Any]:
        """Analyze sentiment in financial texts (synchronous)"""
        try:
            if not texts:
                return {"error": "No texts provided for analysis"}

            # Simple keyword-based sentiment analysis (synchronous)
            positive_keywords = ["growth", "profit", "increase", "success", "strong", "positive", "opportunity"]
            negative_keywords = ["loss", "decline", "risk", "concern", "weak", "negative", "challenge"]

            analysis = {
                "total_texts": len(texts),
                "overall_sentiment": "neutral",
                "sentiment_distribution": {"positive": 0, "neutral": 0, "negative": 0},
                "confidence": 0.0,
                "key_themes": [],
                "notable_quotes": []
            }

            sentiment_scores = []

            for i, text in enumerate(texts):
                text_lower = text.lower()

                # Calculate sentiment score
                positive_count = sum(1 for keyword in positive_keywords if keyword in text_lower)
                negative_count = sum(1 for keyword in negative_keywords if keyword in text_lower)

                if positive_count > negative_count:
                    sentiment = "positive"
                    score = positive_count / (positive_count + negative_count) if (positive_count + negative_count) > 0 else 0.5
                elif negative_count > positive_count:
                    sentiment = "negative"
                    score = -negative_count / (positive_count + negative_count) if (positive_count + negative_count) > 0 else -0.5
                else:
                    sentiment = "neutral"
                    score = 0.0

                sentiment_scores.append(score)
                analysis["sentiment_distribution"][sentiment] += 1

                # Store notable quotes
                if abs(score) > 0.3:
                    analysis["notable_quotes"].append({
                        "text": text[:200] + "..." if len(text) > 200 else text,
                        "sentiment": sentiment,
                        "score": score
                    })

            # Calculate overall sentiment
            if sentiment_scores:
                avg_score = sum(sentiment_scores) / len(sentiment_scores)
                if avg_score > 0.1:
                    analysis["overall_sentiment"] = "positive"
                elif avg_score < -0.1:
                    analysis["overall_sentiment"] = "negative"

                analysis["confidence"] = abs(avg_score)

            # Extract key themes (synchronous)
            analysis["key_themes"] = self._extract_key_themes_sync(texts)

            return analysis

        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {"error": str(e)}
    
    async def _run_async(self, texts: List[str], context: str) -> Dict[str, Any]:
        """Async helper for sentiment analysis"""
        try:
            if not texts:
                return {"error": "No texts provided for analysis"}
            
            analysis = {
                "total_texts": len(texts),
                "overall_sentiment": "neutral",
                "sentiment_distribution": {"positive": 0, "neutral": 0, "negative": 0},
                "confidence": 0.0,
                "key_themes": [],
                "notable_quotes": []
            }
            
            # Simple keyword-based sentiment analysis
            positive_keywords = ["growth", "profit", "increase", "success", "strong", "positive", "opportunity"]
            negative_keywords = ["loss", "decline", "risk", "concern", "weak", "negative", "challenge"]
            
            sentiment_scores = []
            
            for i, text in enumerate(texts):
                text_lower = text.lower()
                
                # Calculate sentiment score
                positive_count = sum(1 for keyword in positive_keywords if keyword in text_lower)
                negative_count = sum(1 for keyword in negative_keywords if keyword in text_lower)
                
                if positive_count > negative_count:
                    sentiment = "positive"
                    score = positive_count / (positive_count + negative_count) if (positive_count + negative_count) > 0 else 0.5
                elif negative_count > positive_count:
                    sentiment = "negative"
                    score = -negative_count / (positive_count + negative_count) if (positive_count + negative_count) > 0 else -0.5
                else:
                    sentiment = "neutral"
                    score = 0.0
                
                sentiment_scores.append(score)
                analysis["sentiment_distribution"][sentiment] += 1
                
                # Store notable quotes
                if abs(score) > 0.3:
                    analysis["notable_quotes"].append({
                        "text": text[:200] + "..." if len(text) > 200 else text,
                        "sentiment": sentiment,
                        "score": score
                    })
            
            # Calculate overall sentiment
            if sentiment_scores:
                avg_score = sum(sentiment_scores) / len(sentiment_scores)
                if avg_score > 0.1:
                    analysis["overall_sentiment"] = "positive"
                elif avg_score < -0.1:
                    analysis["overall_sentiment"] = "negative"
                
                analysis["confidence"] = abs(avg_score)
            
            # Extract key themes
            analysis["key_themes"] = await self._extract_key_themes(texts)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {"error": str(e)}
    
    def _extract_key_themes_sync(self, texts: List[str]) -> List[str]:
        """Extract key themes from texts (synchronous)"""
        try:
            # Simple theme extraction using keyword frequency
            all_text = " ".join(texts).lower()
            
            financial_themes = [
                "revenue", "profit", "growth", "market", "investment", "risk",
                "technology", "innovation", "competition", "regulation", "economy"
            ]
            
            theme_counts = {}
            for theme in financial_themes:
                count = all_text.count(theme)
                if count > 0:
                    theme_counts[theme] = count
            
            # Return top themes
            sorted_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
            return [theme for theme, count in sorted_themes[:5]]
            
        except Exception as e:
            logger.warning(f"Failed to extract key themes: {e}")
            return []

class ValuationTool(Tool):
    """Tool for financial valuation analysis"""
    
    def __init__(self):
        super().__init__()
        self.name = "perform_valuation"
        self.description = "Perform financial valuation analysis"
        self.inputs = {
            "financial_data": {
                "type": "object",
                "description": "Financial data for valuation"
            },
            "valuation_method": {
                "type": "string",
                "description": "Valuation method to use",
                "nullable": True
            },
            "industry": {
                "type": "string",
                "description": "Industry for context",
                "nullable": True
            }
        }
        self.output_type = "object"

    def forward(self, financial_data: Dict[str, Any], valuation_method: str = "multiples", industry: str = None) -> Dict[str, Any]:
        """Perform financial valuation (synchronous)"""
        try:
            valuation = {
                "method": valuation_method,
                "industry": industry,
                "inputs": financial_data,
                "assumptions": [],
                "valuation_result": None,
                "sensitivity_analysis": {}
            }

            if valuation_method == "multiples":
                valuation = self._multiples_valuation_sync(financial_data, industry)
            elif valuation_method == "dcf":
                valuation = self._dcf_valuation_sync(financial_data)
            elif valuation_method == "comparable":
                valuation = self._comparable_valuation_sync(financial_data, industry)
            elif valuation_method == "asset_based":
                valuation = self._asset_based_valuation_sync(financial_data)

            return valuation

        except Exception as e:
            logger.error(f"Valuation analysis failed: {e}")
            return {"error": str(e)}
    
    async def _run_async(self, financial_data: Dict[str, Any], valuation_method: str, industry: str) -> Dict[str, Any]:
        """Async helper for valuation analysis"""
        try:
            valuation = {
                "method": valuation_method,
                "industry": industry,
                "inputs": financial_data,
                "assumptions": [],
                "valuation_result": None,
                "sensitivity_analysis": {}
            }
            
            if valuation_method == "multiples":
                valuation = await self._multiples_valuation(financial_data, industry)
            elif valuation_method == "dcf":
                valuation = await self._dcf_valuation(financial_data)
            elif valuation_method == "comparable":
                valuation = await self._comparable_valuation(financial_data, industry)
            elif valuation_method == "asset_based":
                valuation = await self._asset_based_valuation(financial_data)
            
            return valuation
            
        except Exception as e:
            logger.error(f"Valuation analysis failed: {e}")
            return {"error": str(e)}
    
    def _multiples_valuation_sync(self, data: Dict[str, Any], industry: str = None) -> Dict[str, Any]:
        """Perform multiples-based valuation (synchronous)"""
        try:
            # Industry multiples (simplified)
            industry_multiples = {
                "technology": {"pe": 25, "ps": 8, "ev_ebitda": 15},
                "finance": {"pe": 15, "ps": 3, "ev_ebitda": 10},
                "healthcare": {"pe": 20, "ps": 5, "ev_ebitda": 12},
                "consumer": {"pe": 18, "ps": 2, "ev_ebitda": 8}
            }
            
            multiples = industry_multiples.get(industry.lower(), {"pe": 18, "ps": 4, "ev_ebitda": 10})
            
            # Extract metrics from data
            net_income = data.get("net_income", 0)
            revenue = data.get("revenue", 0)
            ebitda = data.get("ebitda", 0)
            
            # Calculate valuations
            pe_valuation = net_income * multiples["pe"]
            ps_valuation = revenue * multiples["ps"]
            ev_ebitda_valuation = ebitda * multiples["ev_ebitda"]
            
            # Average valuation
            valuations = [pe_valuation, ps_valuation, ev_ebitda_valuation]
            valid_valuations = [v for v in valuations if v > 0]
            avg_valuation = sum(valid_valuations) / len(valid_valuations) if valid_valuations else 0
            
            return {
                "method": "multiples",
                "industry": industry,
                "multiples_used": multiples,
                "calculations": {
                    "pe_valuation": pe_valuation,
                    "ps_valuation": ps_valuation,
                    "ev_ebitda_valuation": ev_ebitda_valuation
                },
                "valuation_result": avg_valuation,
                "assumptions": [f"Used {industry} industry multiples" if industry else "Used generic multiples"]
            }
            
        except Exception as e:
            return {"method": "multiples", "error": str(e)}
    
    def _dcf_valuation_sync(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform discounted cash flow valuation (synchronous)"""
        try:
            # Simplified DCF calculation
            fcf = data.get("free_cash_flow", 0)
            growth_rate = data.get("growth_rate", 0.05)
            discount_rate = data.get("discount_rate", 0.10)
            terminal_growth_rate = data.get("terminal_growth_rate", 0.02)
            
            # 5-year projection
            projected_fcfs = []
            for i in range(1, 6):
                projected_fcf = fcf * (1 + growth_rate) ** i
                projected_fcfs.append(projected_fcf)
            
            # Discount projected FCFs
            discounted_fcfs = []
            for i, fcf in enumerate(projected_fcfs, 1):
                discounted_fcf = fcf / ((1 + discount_rate) ** i)
                discounted_fcfs.append(discounted_fcf)
            
            # Terminal value
            terminal_fcf = projected_fcfs[-1] * (1 + terminal_growth_rate)
            terminal_value = terminal_fcf / (discount_rate - terminal_growth_rate)
            discounted_terminal_value = terminal_value / ((1 + discount_rate) ** 5)
            
            # Total valuation
            total_valuation = sum(discounted_fcfs) + discounted_terminal_value
            
            return {
                "method": "dcf",
                "inputs": {
                    "initial_fcf": fcf,
                    "growth_rate": growth_rate,
                    "discount_rate": discount_rate,
                    "terminal_growth_rate": terminal_growth_rate
                },
                "projections": projected_fcfs,
                "discounted_cash_flows": discounted_fcfs,
                "terminal_value": terminal_value,
                "valuation_result": total_valuation,
                "assumptions": ["5-year projection", "perpetual growth model"]
            }
            
        except Exception as e:
            return {"method": "dcf", "error": str(e)}
    
    def _comparable_valuation_sync(self, data: Dict[str, Any], industry: str = None) -> Dict[str, Any]:
        """Perform comparable company analysis (synchronous)"""
        try:
            # Simplified comparable analysis
            metrics = {
                "revenue": data.get("revenue", 0),
                "net_income": data.get("net_income", 0),
                "assets": data.get("total_assets", 0)
            }
            
            # Mock comparable companies (in real implementation, this would come from market data)
            comparables = [
                {"name": "Company A", "multiple": 2.5, "metric": "revenue"},
                {"name": "Company B", "multiple": 15.0, "metric": "net_income"},
                {"name": "Company C", "multiple": 1.8, "metric": "assets"}
            ]
            
            valuations = []
            for comp in comparables:
                metric_value = metrics.get(comp["metric"], 0)
                valuation = metric_value * comp["multiple"]
                valuations.append(valuation)
            
            avg_valuation = sum(valuations) / len(valuations) if valuations else 0
            
            return {
                "method": "comparable",
                "industry": industry,
                "comparables": comparables,
                "individual_valuations": valuations,
                "valuation_result": avg_valuation,
                "assumptions": ["Limited comparable set", "Market-based multiples"]
            }
            
        except Exception as e:
            return {"method": "comparable", "error": str(e)}
    
    def _asset_based_valuation_sync(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform asset-based valuation (synchronous)"""
        try:
            total_assets = data.get("total_assets", 0)
            total_liabilities = data.get("total_liabilities", 0)
            intangible_assets = data.get("intangible_assets", 0)
            
            # Book value
            book_value = total_assets - total_liabilities
            
            # Adjusted book value (excluding intangibles)
            adjusted_book_value = book_value - intangible_assets
            
            # Use average of book value and adjusted book value
            valuation = (book_value + adjusted_book_value) / 2
            
            return {
                "method": "asset_based",
                "inputs": {
                    "total_assets": total_assets,
                    "total_liabilities": total_liabilities,
                    "intangible_assets": intangible_assets
                },
                "calculations": {
                    "book_value": book_value,
                    "adjusted_book_value": adjusted_book_value
                },
                "valuation_result": valuation,
                "assumptions": ["Conservative valuation", "Asset-focused approach"]
            }
            
        except Exception as e:
            return {"method": "asset_based", "error": str(e)}

class RiskAssessmentTool(Tool):
    """Tool for risk assessment"""
    
    def __init__(self):
        super().__init__()
        self.name = "assess_risks"
        self.description = "Assess financial and business risks"
        self.inputs = {
            "data": {
                "type": "object",
                "description": "Data for risk assessment"
            },
            "risk_categories": {
                "type": "array",
                "description": "Risk categories to assess",
                "nullable": True
            }
        }
        self.output_type = "object"
    
    def forward(self, data: Dict[str, Any], risk_categories: List[str] = None) -> Dict[str, Any]:
        """Assess risks based on data (synchronous)"""
        try:
            if risk_categories is None:
                risk_categories = ["financial", "operational", "market", "regulatory"]
            
            required = {
                "financial": ("debt_to_equity", "current_ratio", "profit_margin"),
                "operational": ("operational_efficiency", "employee_turnover"),
                "market": ("market_volatility", "competitive_pressure"),
                "regulatory": ("regulatory_changes",)
            }
            missing = {}
            for category in risk_categories:
                fields = required.get(category)
                if fields is None:
                    missing[category] = ["unsupported risk category"]
                    continue
                invalid = []
                for field in fields:
                    value = data.get(field)
                    if field == "regulatory_changes":
                        valid = value in ("low", "medium", "high")
                    else:
                        import math
                        valid = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
                    if not valid:
                        invalid.append(field)
                if invalid:
                    missing[category] = invalid
            if not risk_categories or missing:
                return {"error": "Insufficient data for risk assessment", "overall_risk_level": "unknown",
                        "missing_fields": missing, "risk_scores": {}, "risk_factors": [],
                        "mitigation_strategies": []}

            assessment = {
                "risk_categories": risk_categories,
                "overall_risk_level": "medium",
                "risk_scores": {},
                "risk_factors": [],
                "mitigation_strategies": []
            }
            
            # Assess each risk category
            total_score = 0
            for category in risk_categories:
                category_score = self._assess_risk_category_sync(data, category)
                assessment["risk_scores"][category] = category_score
                total_score += category_score
            
            # Calculate overall risk level
            avg_score = total_score / len(risk_categories) if risk_categories else 0
            if avg_score > 0.7:
                assessment["overall_risk_level"] = "high"
            elif avg_score > 0.4:
                assessment["overall_risk_level"] = "medium"
            else:
                assessment["overall_risk_level"] = "low"
            
            # Generate risk factors
            assessment["risk_factors"] = self._identify_risk_factors_sync(data, risk_categories)
            
            # Generate mitigation strategies
            assessment["mitigation_strategies"] = self._generate_mitigation_strategies_sync(assessment)
            
            return assessment
            
        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            return {"error": str(e)}
    
    async def _run_async(self, data: Dict[str, Any], risk_categories: List[str]) -> Dict[str, Any]:
        """Use the same validation and calculations as the synchronous tool."""
        return self.forward(data, risk_categories)

    def _assess_risk_category_sync(self, data: Dict[str, Any], category: str) -> float:
        """Assess risk for a specific category (synchronous)"""
        try:
            if category == "financial":
                return self._assess_financial_risk_sync(data)
            elif category == "operational":
                return self._assess_operational_risk_sync(data)
            elif category == "market":
                return self._assess_market_risk_sync(data)
            elif category == "regulatory":
                return self._assess_regulatory_risk_sync(data)
            else:
                return 0.5  # Default medium risk
                
        except Exception as e:
            logger.warning(f"Failed to assess {category} risk: {e}")
            return 0.5
    
    def _assess_financial_risk_sync(self, data: Dict[str, Any]) -> float:
        """Assess financial risk"""
        try:
            # Simple financial risk assessment
            debt_to_equity = data.get("debt_to_equity", 0)
            current_ratio = data.get("current_ratio", 1.0)
            profit_margin = data.get("profit_margin", 0)
            
            risk_score = 0.0
            
            # High debt increases risk
            if debt_to_equity > 2.0:
                risk_score += 0.3
            elif debt_to_equity > 1.0:
                risk_score += 0.15
            
            # Low liquidity increases risk
            if current_ratio < 1.0:
                risk_score += 0.3
            elif current_ratio < 1.5:
                risk_score += 0.15
            
            # Low profitability increases risk
            if profit_margin < 0:
                risk_score += 0.4
            elif profit_margin < 0.05:
                risk_score += 0.2
            
            return min(risk_score, 1.0)
            
        except Exception as e:
            logger.warning(f"Financial risk assessment failed: {e}")
            return 0.5
    
    def _assess_operational_risk_sync(self, data: Dict[str, Any]) -> float:
        """Assess operational risk"""
        try:
            # Simplified operational risk assessment
            operational_efficiency = data.get("operational_efficiency", 0.7)
            employee_turnover = data.get("employee_turnover", 0.15)
            
            risk_score = 0.0
            
            if operational_efficiency < 0.5:
                risk_score += 0.3
            elif operational_efficiency < 0.7:
                risk_score += 0.15
            
            if employee_turnover > 0.25:
                risk_score += 0.2
            elif employee_turnover > 0.15:
                risk_score += 0.1
            
            return min(risk_score, 1.0)
            
        except Exception as e:
            logger.warning(f"Operational risk assessment failed: {e}")
            return 0.5
    
    def _assess_market_risk_sync(self, data: Dict[str, Any]) -> float:
        """Assess market risk"""
        try:
            # Simplified market risk assessment
            market_volatility = data.get("market_volatility", 0.2)
            competitive_pressure = data.get("competitive_pressure", 0.5)
            
            risk_score = (market_volatility * 0.6) + (competitive_pressure * 0.4)
            return min(risk_score, 1.0)
            
        except Exception as e:
            logger.warning(f"Market risk assessment failed: {e}")
            return 0.5
    
    def _assess_regulatory_risk_sync(self, data: Dict[str, Any]) -> float:
        """Assess regulatory risk"""
        try:
            # Simplified regulatory risk assessment
            regulatory_changes = data.get("regulatory_changes", "low")
            
            if regulatory_changes == "high":
                return 0.8
            elif regulatory_changes == "medium":
                return 0.5
            else:
                return 0.2
                
        except Exception as e:
            logger.warning(f"Regulatory risk assessment failed: {e}")
            return 0.5
    
    def _identify_risk_factors_sync(self, data: Dict[str, Any], categories: List[str]) -> List[Dict[str, Any]]:
        """Identify specific risk factors (synchronous)"""
        risk_factors = []
        
        try:
            # Financial risk factors
            if "financial" in categories:
                debt_to_equity = data.get("debt_to_equity", 0)
                if debt_to_equity > 2.0:
                    risk_factors.append({
                        "category": "financial",
                        "factor": "High debt-to-equity ratio",
                        "severity": "high",
                        "description": f"Debt-to-equity ratio of {debt_to_equity} indicates high financial leverage"
                    })
            
            # Operational risk factors
            if "operational" in categories:
                efficiency = data.get("operational_efficiency", 0.7)
                if efficiency < 0.5:
                    risk_factors.append({
                        "category": "operational",
                        "factor": "Low operational efficiency",
                        "severity": "medium",
                        "description": f"Operational efficiency of {efficiency} indicates process inefficiencies"
                    })
            
        except Exception as e:
            logger.warning(f"Risk factor identification failed: {e}")
        
        return risk_factors
    
    def _generate_mitigation_strategies_sync(self, assessment: Dict[str, Any]) -> List[str]:
        """Generate risk mitigation strategies (synchronous)"""
        strategies = []
        
        try:
            risk_level = assessment.get("overall_risk_level", "medium")
            
            if risk_level == "high":
                strategies.append("Implement comprehensive risk management framework")
                strategies.append("Regular monitoring and reporting of key risk indicators")
                strategies.append("Diversification strategies to reduce concentration risk")
            elif risk_level == "medium":
                strategies.append("Enhanced monitoring of identified risk areas")
                strategies.append("Develop contingency plans for key risks")
            else:
                strategies.append("Maintain current risk management practices")
                strategies.append("Periodic risk assessment review")
            
        except Exception as e:
            logger.warning(f"Mitigation strategy generation failed: {e}")
            strategies.append("Review risk management approach")
        
        return strategies

class DeepAnalyzeAgent(BaseAgent):
    def __init__(self, model, memory=None, tools=None):
        super().__init__('DeepAnalyzeAgent', model, tools=tools, instructions=(
            'Analyze the supplied research evidence. Inspect passages using tools as needed. '
            'Choose indicators appropriate to the subject. Never invent missing data. '
            'Submit substantive research JSON through final_answer, with evidence IDs.'
        ))
