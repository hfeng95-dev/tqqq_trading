import os
import json
import traceback
from typing import Optional, Dict, Any, List
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

try:
    from google import genai
    HAS_NEW_GENAI = True
except Exception:
    HAS_NEW_GENAI = False

try:
    import google.generativeai as legacy_genai
    HAS_LEGACY_GENAI = True
except Exception:
    HAS_LEGACY_GENAI = False

SYSTEM_PROMPT = """
You are an elite quantitative portfolio manager and risk strategist specializing in 3x Leveraged ETFs (TQQQ / SQQQ) and Nasdaq-100 momentum regime switching.

YOUR CORE EXPERTISE:
1. Maximizing exposure to 3x Nasdaq upside during sustained Bull Market Regimes (QQQ > 200-day SMA).
2. Completely avoiding the catastrophic -70% to -85% drawdowns of 3x leveraged products during Bear Markets (e.g. 2022 Tech Bear, 2008 GFC, 2000 Dot-com, 2020 Covid Shock).
3. Mitigating Leveraged ETF Volatility Drag (decay in choppy sideways markets).
4. Timing aggressive V-shaped recovery bounces using MACD Histogram turns and Moving Average reclaim triggers.

Always provide razor-sharp, actionable, and mathematically grounded guidance with exact trigger levels, stop losses, and risk allocations.
"""

def get_configured_model(model_name: str = "gemini-3.8-flash"):
    """Legacy helper for backwards compatibility if legacy_genai is available."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or not HAS_LEGACY_GENAI:
        return None
    try:
        legacy_genai.configure(api_key=api_key)
        return legacy_genai.GenerativeModel(
            model_name=model_name,
            system_instruction=SYSTEM_PROMPT
        )
    except Exception:
        return None

def run_advisor_json(prompt: str, system_prompt: str, model_name: str = "gemini-3.8-flash") -> Optional[Dict[str, Any]]:
    """Executes Gemini JSON query with dual-SDK support (google-genai first, legacy fallback)."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    models_to_try = [model_name, "gemini-3.8-flash", "gemini-3.7-flash"]
    unique_models = []
    for m in models_to_try:
        if m and m not in unique_models:
            unique_models.append(m)

    if HAS_NEW_GENAI:
        try:
            client = genai.Client(api_key=api_key)
            for m in unique_models:
                try:
                    response = client.models.generate_content(
                        model=m,
                        contents=prompt,
                        config={
                            "system_instruction": system_prompt,
                            "response_mime_type": "application/json",
                            "temperature": 0.2
                        }
                    )
                    if response and response.text:
                        text = response.text.strip()
                        if text.startswith("```json"): text = text[7:]
                        elif text.startswith("```"): text = text[3:]
                        if text.endswith("```"): text = text[:-3]
                        return json.loads(text.strip())
                except Exception:
                    continue
        except Exception:
            pass

    if HAS_LEGACY_GENAI:
        try:
            legacy_genai.configure(api_key=api_key)
            for m in unique_models:
                try:
                    model = legacy_genai.GenerativeModel(
                        model_name=m,
                        system_instruction=system_prompt,
                        generation_config=legacy_genai.GenerationConfig(
                            response_mime_type="application/json",
                            temperature=0.2
                        )
                    )
                    response = model.generate_content(prompt)
                    if response and response.text:
                        text = response.text.strip()
                        if text.startswith("```json"): text = text[7:]
                        elif text.startswith("```"): text = text[3:]
                        if text.endswith("```"): text = text[:-3]
                        return json.loads(text.strip())
                except Exception:
                    continue
        except Exception:
            pass

    return None

def run_advisor_text(prompt: str, model_name: str = "gemini-3.8-flash") -> str:
    """Executes Gemini text query with dual-SDK support."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "⚠️ Gemini API is not configured. Please check GEMINI_API_KEY in .env."
    models_to_try = [model_name, "gemini-3.8-flash", "gemini-3.7-flash"]
    unique_models = []
    for m in models_to_try:
        if m and m not in unique_models:
            unique_models.append(m)

    if HAS_NEW_GENAI:
        try:
            client = genai.Client(api_key=api_key)
            for m in unique_models:
                try:
                    response = client.models.generate_content(
                        model=m,
                        contents=prompt
                    )
                    if response and response.text:
                        return response.text
                except Exception:
                    continue
        except Exception:
            pass

    if HAS_LEGACY_GENAI:
        try:
            legacy_genai.configure(api_key=api_key)
            for m in unique_models:
                try:
                    model = legacy_genai.GenerativeModel(model_name=m)
                    response = model.generate_content(prompt)
                    if response and response.text:
                        return response.text
                except Exception:
                    continue
        except Exception:
            pass

    return "⚠️ Error querying Gemini API across available models."

def analyze_market_regime(
    current_metrics: Dict[str, Any],
    recent_history: pd.DataFrame,
    model_name: str = "gemini-3.8-flash"
) -> Dict[str, Any]:
    """
    Evaluates current Nasdaq (QQQ / TQQQ) market regime and outputs structured AI advice.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return generate_mock_analysis(current_metrics)

    try:
        prompt = f"""
Analyze the current Nasdaq-100 market regime and provide tactical leveraged ETF (TQQQ / SQQQ) execution instructions:

CURRENT MARKET METRICS:
- Date: {current_metrics.get('date', 'Latest')}
- QQQ Price: ${current_metrics.get('qqq_price', 0):.2f}
- TQQQ Price: ${current_metrics.get('tqqq_price', 0):.2f}
- SQQQ Price: ${current_metrics.get('sqqq_price', 0):.2f}
- QQQ 200-day SMA: ${current_metrics.get('sma_200', 0):.2f} (Distance: {current_metrics.get('dist_200_sma_pct', 0):+.2f}%)
- QQQ 50-day SMA: ${current_metrics.get('sma_50', 0):.2f} (Distance: {current_metrics.get('dist_50_sma_pct', 0):+.2f}%)
- QQQ 10-day EMA: ${current_metrics.get('ema_10', 0):.2f}
- MACD Line: {current_metrics.get('macd', 0):.2f} | Signal: {current_metrics.get('macd_signal', 0):.2f} | Histogram: {current_metrics.get('macd_hist', 0):+.2f}
- RSI (14): {current_metrics.get('rsi', 50):.1f}
- Quantitative Strategy Stance: {current_metrics.get('current_stance', 'N/A')}

TASK:
1. Determine the Market Regime (Strong Bull Regime / Early Bull / Distribution / Severe Bear / V-Shaped Bounce).
2. Evaluate Leveraged Volatility Drag Risk (High / Medium / Low).
3. Provide the AI Recommended Allocation (% TQQQ, % SQQQ, % Cash).
4. Outline exact Key Triggers for entering, trailing, or aborting the position.
5. Identify Macro & Structural Risks.

OUTPUT JSON SCHEMA:
{{
  "regime": "Strong Bull Regime / Severe Bear / V-Shaped Recovery / Choppy Neutral",
  "recommended_stance": "LONG TQQQ (100%) / CASH & WAIT / SHORT / SQQQ HEDGE",
  "allocation": {{
    "TQQQ_pct": 100,
    "SQQQ_pct": 0,
    "Cash_pct": 0
  }},
  "volatility_decay_risk": "Low / Moderate / High",
  "regime_thesis": "Detailed 2-3 sentence institutional thesis explaining why this stance is mathematically optimal right now.",
  "execution_plan": {{
    "ideal_entry": "Key level or confirmation trigger to buy/add",
    "stop_loss_trigger": "Exact price/indicator level that forces an exit to cash",
    "trailing_management": "Rule for holding through bull trends (e.g. 10 EMA / 200 SMA)"
  }},
  "macro_risks": [
    "Risk factor 1",
    "Risk factor 2",
    "Risk factor 3"
  ],
  "institutional_verdict": "Final one-sentence verdict."
}}
"""
        parsed = run_advisor_json(prompt, SYSTEM_PROMPT, model_name)
        if parsed and isinstance(parsed, dict):
            return parsed
        return generate_mock_analysis(current_metrics)

    except Exception as e:
        print(f"Error in Gemini regime analysis: {e}")
        traceback.print_exc()
        return generate_mock_analysis(current_metrics)

def ask_gemini_copilot(
    user_query: str,
    market_context: Dict[str, Any],
    model_name: str = "gemini-3.8-flash"
) -> str:
    """Answers user queries about TQQQ trading strategy, risk management, and macro timing."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "⚠️ Gemini API is not configured. Please check GEMINI_API_KEY in .env."

    prompt = f"""
CURRENT MARKET CONTEXT:
- QQQ Price: ${market_context.get('qqq_price', 0):.2f}
- 200-day SMA: ${market_context.get('sma_200', 0):.2f} ({market_context.get('dist_200_sma_pct', 0):+.2f}%)
- 50-day SMA: ${market_context.get('sma_50', 0):.2f}
- MACD Histogram: {market_context.get('macd_hist', 0):+.2f}
- RSI (14): {market_context.get('rsi', 50):.1f}
- Current Quant Signal: {market_context.get('current_stance', 'N/A')}

USER QUESTION:
"{user_query}"

Provide a crisp, elite quantitative response with specific actionable numbers, mathematical risk management, and tactical execution instructions.
"""
    return run_advisor_text(prompt, model_name)


def generate_mock_analysis(current_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Fallback generator when offline."""
    qqq = current_metrics.get('qqq_price', 450)
    sma200 = current_metrics.get('sma_200', 430)
    is_bull = qqq > sma200
    
    return {
        "regime": "Strong Bull Regime" if is_bull else "Bear / Defensive Regime",
        "recommended_stance": "LONG TQQQ (100%)" if is_bull else "CASH & FLAT (100%)",
        "allocation": {
            "TQQQ_pct": 100 if is_bull else 0,
            "SQQQ_pct": 0,
            "Cash_pct": 0 if is_bull else 100
        },
        "volatility_decay_risk": "Low" if is_bull else "Elevated",
        "regime_thesis": f"QQQ is currently trading {'above' if is_bull else 'below'} its 200-day moving average (${sma200:.2f}), signaling a {'high-probability bull trend with minimal volatility drag' if is_bull else 'defensive environment where cash protects against 3x decay'}.",
        "execution_plan": {
            "ideal_entry": f"Hold full TQQQ position while QQQ stays above ${sma200:.2f}",
            "stop_loss_trigger": f"Close below 200 SMA (${sma200:.2f}) triggers 100% exit to CASH",
            "trailing_management": "Trail stops on daily 10 EMA / 200 SMA."
        },
        "macro_risks": [
            "Fed interest rate policy volatility",
            "Concentration in Mega-Cap Big Tech earnings",
            "Elevated VIX spikes causing 3x tracking slippage"
        ],
        "institutional_verdict": f"Maintain {'aggressive long exposure' if is_bull else 'disciplined capital preservation in cash'}."
    }
