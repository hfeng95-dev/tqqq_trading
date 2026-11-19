import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

try:
    from google import genai

    HAS_GENAI = True
except Exception:
    genai = None  # type: ignore
    HAS_GENAI = False

DEFAULT_MODELS: List[str] = ["gemini-3.8-flash", "gemini-3.7-flash"]

_CLIENT_CACHE: Dict[str, Any] = {}

SYSTEM_PROMPT = """
You are an elite quantitative portfolio manager and risk strategist specializing in 3x Leveraged ETFs (TQQQ / SQQQ) and Nasdaq-100 momentum regime switching.

YOUR CORE EXPERTISE:
1. Maximizing exposure to 3x Nasdaq upside during sustained Bull Market Regimes (QQQ > 200-day SMA).
2. Completely avoiding the catastrophic -70% to -85% drawdowns of 3x leveraged products during Bear Markets (e.g. 2022 Tech Bear, 2008 GFC, 2000 Dot-com, 2020 Covid Shock).
3. Mitigating Leveraged ETF Volatility Drag (decay in choppy sideways markets).
4. Timing aggressive V-shaped recovery bounces using MACD Histogram turns and Moving Average reclaim triggers.

Always provide razor-sharp, actionable, and mathematically grounded guidance with exact trigger levels, stop losses, and risk allocations.
""".strip()


def is_api_configured() -> bool:
    """Returns True if google-genai is installed and GEMINI_API_KEY is non-empty."""
    return HAS_GENAI and bool(os.environ.get("GEMINI_API_KEY", "").strip())


def get_available_models() -> List[str]:
    """Returns ordered model choices, prepending GEMINI_MODEL env override when set."""
    env_model = os.environ.get("GEMINI_MODEL", "").strip()
    candidates = [env_model, *DEFAULT_MODELS]
    unique: List[str] = []
    for m in candidates:
        if m and m not in unique:
            unique.append(m)
    return unique


def _resolve_models(model_name: Optional[str] = None) -> List[str]:
    env_model = os.environ.get("GEMINI_MODEL", "").strip()
    candidates = [model_name, env_model, *DEFAULT_MODELS]
    unique: List[str] = []
    for m in candidates:
        if m and m not in unique:
            unique.append(m)
    return unique


def _get_client(api_key: str) -> Any:
    if not HAS_GENAI or genai is None:
        return None
    client = _CLIENT_CACHE.get(api_key)
    if client is None:
        client = genai.Client(api_key=api_key)
        _CLIENT_CACHE[api_key] = client
    return client


def _is_non_retryable_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(
        token in msg
        for token in (
            "401",
            "403",
            "429",
            "api_key_invalid",
            "invalid api key",
            "permission_denied",
            "resource_exhausted",
            "quota",
            "unauthorized",
        )
    )


def _strip_json_fences(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _format_recent_history(recent_history: Optional[pd.DataFrame], max_rows: int = 10) -> str:
    if recent_history is None or recent_history.empty:
        return "No recent OHLCV history provided."

    hist = recent_history.dropna(subset=["Close"]) if "Close" in recent_history.columns else recent_history
    if hist.empty:
        return "No recent OHLCV history provided."

    summary_lines: List[str] = []
    closes = hist["Close"]
    if len(closes) >= 2:
        period_ret = (closes.iloc[-1] / closes.iloc[0] - 1.0) * 100.0
        summary_lines.append(
            f"- Window ({len(closes)} bars): Return {period_ret:+.2f}% | "
            f"High ${closes.max():.2f} | Low ${closes.min():.2f}"
        )
    if len(closes) >= 21:
        ret_20 = (closes.iloc[-1] / closes.iloc[-21] - 1.0) * 100.0
        summary_lines.append(f"- 20-Day QQQ Return: {ret_20:+.2f}%")

    tail_df = hist.tail(max_rows)
    summary_lines.append("- Last Bars (Date | Close | RSI | MACD_Hist | Position):")
    for dt, row in tail_df.iterrows():
        dt_str = dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt)
        close_v = float(row.get("Close", 0.0))
        rsi_v = row.get("RSI", float("nan"))
        macd_v = row.get("MACD_Hist", float("nan"))
        pos_v = float(row.get("Position", 0.0))
        rsi_s = f"{float(rsi_v):.1f}" if pd.notna(rsi_v) else "N/A"
        macd_s = f"{float(macd_v):+.2f}" if pd.notna(macd_v) else "N/A"
        summary_lines.append(
            f"  {dt_str} | ${close_v:.2f} | RSI {rsi_s} | MACD_H {macd_s} | Pos {pos_v:+.1f}"
        )

    return "\n".join(summary_lines)


def run_advisor_json(
    prompt: str,
    system_prompt: str = SYSTEM_PROMPT,
    model_name: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    """
    Executes a Gemini JSON query using google-genai.
    Returns (parsed_json_or_None, model_used_or_None, error_detail_or_None).
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None, None, "GEMINI_API_KEY is not configured in environment or .env"
    if not HAS_GENAI:
        return None, None, "google-genai SDK is not installed"

    unique_models = _resolve_models(model_name)
    last_error: Optional[str] = None

    try:
        client = _get_client(api_key)
    except Exception as exc:
        logger.error("Failed to initialize google-genai Client: %s", exc)
        return None, None, f"Client init failed: {exc}"

    for m in unique_models:
        try:
            response = client.models.generate_content(
                model=m,
                contents=prompt,
                config={
                    "system_instruction": system_prompt,
                    "response_mime_type": "application/json",
                    "temperature": 0.2,
                },
            )
            raw_text = getattr(response, "text", None)
            if not raw_text:
                last_error = f"Model {m} returned an empty response"
                continue
            parsed = json.loads(_strip_json_fences(raw_text))
            if isinstance(parsed, dict):
                return parsed, m, None
            last_error = f"Model {m} returned non-object JSON"
        except Exception as exc:
            last_error = f"{m}: {exc}"
            logger.warning("Gemini JSON call failed on %s: %s", m, exc)
            if _is_non_retryable_error(exc):
                break

    return None, None, last_error or "Gemini API call failed across configured models"


def run_advisor_text(
    prompt: str,
    model_name: Optional[str] = None,
    system_prompt: str = SYSTEM_PROMPT,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Executes a Gemini text query with SYSTEM_PROMPT via google-genai.
    Returns (text_or_None, model_used_or_None, error_detail_or_None).
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None, None, "GEMINI_API_KEY is not configured in environment or .env"
    if not HAS_GENAI:
        return None, None, "google-genai SDK is not installed"

    unique_models = _resolve_models(model_name)
    last_error: Optional[str] = None

    try:
        client = _get_client(api_key)
    except Exception as exc:
        logger.error("Failed to initialize google-genai Client: %s", exc)
        return None, None, f"Client init failed: {exc}"

    for m in unique_models:
        try:
            response = client.models.generate_content(
                model=m,
                contents=prompt,
                config={
                    "system_instruction": system_prompt,
                    "temperature": 0.3,
                },
            )
            raw_text = getattr(response, "text", None)
            if raw_text and raw_text.strip():
                return raw_text.strip(), m, None
            last_error = f"Model {m} returned an empty response"
        except Exception as exc:
            last_error = f"{m}: {exc}"
            logger.warning("Gemini text call failed on %s: %s", m, exc)
            if _is_non_retryable_error(exc):
                break

    return None, None, last_error or "Gemini API call failed across configured models"


def analyze_market_regime(
    current_metrics: Dict[str, Any],
    recent_history: Optional[pd.DataFrame] = None,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluates current Nasdaq (QQQ / TQQQ) market regime and outputs structured AI advice
    with explicit `is_fallback`, `model_used`, and `error_detail` metadata.
    """
    if not is_api_configured():
        reason = (
            "google-genai SDK is not installed"
            if not HAS_GENAI
            else "GEMINI_API_KEY is not configured in environment or .env"
        )
        return generate_mock_analysis(current_metrics, error_detail=reason)

    history_block = _format_recent_history(recent_history)
    buffer_pct = float(current_metrics.get("buffer_pct", 5.0))
    lower_buffer = float(current_metrics.get("lower_buffer", 0.0))
    dist_lower_buf = float(current_metrics.get("dist_lower_buffer_pct", 0.0))
    strat_name = current_metrics.get("strategy_name", "Alpha-Locked Master")

    prompt = f"""
Analyze the current Nasdaq-100 market regime and provide tactical leveraged ETF (TQQQ / SQQQ) execution instructions:

CURRENT MARKET METRICS:
- Date: {current_metrics.get('date', 'Latest')}
- Active Strategy: {strat_name}
- QQQ Price: ${float(current_metrics.get('qqq_price', 0)):.2f}
- TQQQ Price: ${float(current_metrics.get('tqqq_price', 0)):.2f}
- SQQQ Price: ${float(current_metrics.get('sqqq_price', 0)):.2f}
- QQQ 200-day SMA: ${float(current_metrics.get('sma_200', 0)):.2f} (Distance: {float(current_metrics.get('dist_200_sma_pct', 0)):+.2f}%)
- QQQ Switch Buffer: {buffer_pct:.1f}% -> Lower Buffer Floor: ${lower_buffer:.2f} (Distance: {dist_lower_buf:+.2f}%)
- QQQ 50-day SMA: ${float(current_metrics.get('sma_50', 0)):.2f} (Distance: {float(current_metrics.get('dist_50_sma_pct', 0)):+.2f}%)
- QQQ 10-day EMA: ${float(current_metrics.get('ema_10', 0)):.2f}
- QQQ 20-day Realized Vol: {float(current_metrics.get('rvol_20', 0)):.1f}%
- MACD Line: {float(current_metrics.get('macd', 0)):.2f} | Signal: {float(current_metrics.get('macd_signal', 0)):.2f} | Histogram: {float(current_metrics.get('macd_hist', 0)):+.2f}
- Wilder RSI (14): {float(current_metrics.get('rsi', 50)):.1f}
- Quantitative Strategy Stance: {current_metrics.get('current_stance', 'N/A')}

RECENT PRICE & SIGNAL TRAJECTORY:
{history_block}

TASK:
1. Determine the Market Regime (Strong Bull Regime / Early Bull / Distribution / Severe Bear / V-Shaped Bounce).
2. Evaluate Leveraged Volatility Drag Risk (Low / Moderate / High).
3. Provide the AI Recommended Allocation (% TQQQ, % SQQQ, % Cash).
4. Outline exact Key Triggers for entering, trailing, or aborting the position referencing the {buffer_pct:.1f}% buffer floor (${lower_buffer:.2f}).
5. Identify Macro & Structural Risks.

OUTPUT JSON SCHEMA:
{{
  "regime": "Strong Bull Regime / Severe Bear / V-Shaped Recovery / Choppy Neutral",
  "recommended_stance": "LONG TQQQ (100%) / PARTIAL LONG TQQQ (50%) / CASH & WAIT / SHORT / SQQQ HEDGE",
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
    parsed, used_model, error_detail = run_advisor_json(
        prompt, SYSTEM_PROMPT, model_name=model_name
    )
    if parsed is not None:
        parsed["is_fallback"] = False
        parsed["model_used"] = used_model or model_name
        parsed["error_detail"] = None
        return parsed

    return generate_mock_analysis(
        current_metrics,
        error_detail=error_detail or "Gemini API call returned no valid JSON",
    )


def ask_gemini_copilot(
    user_query: str,
    market_context: Dict[str, Any],
    model_name: Optional[str] = None,
    recent_history: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Answers user queries about TQQQ trading strategy, risk management, and macro timing.
    Returns structured dict with `text`, `is_fallback`, `model_used`, and `error_detail`.
    """
    buffer_pct = float(market_context.get("buffer_pct", 5.0))
    lower_buffer = float(market_context.get("lower_buffer", 0.0))
    dist_lower_buf = float(market_context.get("dist_lower_buffer_pct", 0.0))
    history_block = _format_recent_history(recent_history, max_rows=5)

    prompt = f"""
CURRENT MARKET CONTEXT:
- Date: {market_context.get('date', 'Latest')}
- Active Strategy: {market_context.get('strategy_name', 'N/A')}
- QQQ Price: ${float(market_context.get('qqq_price', 0)):.2f}
- TQQQ Price: ${float(market_context.get('tqqq_price', 0)):.2f} | SQQQ Price: ${float(market_context.get('sqqq_price', 0)):.2f}
- 200-day SMA: ${float(market_context.get('sma_200', 0)):.2f} ({float(market_context.get('dist_200_sma_pct', 0)):+.2f}%)
- Switch Buffer ({buffer_pct:.1f}%): Lower Floor ${lower_buffer:.2f} ({dist_lower_buf:+.2f}%)
- 50-day SMA: ${float(market_context.get('sma_50', 0)):.2f}
- MACD Histogram: {float(market_context.get('macd_hist', 0)):+.2f}
- Wilder RSI (14): {float(market_context.get('rsi', 50)):.1f}
- Current Quant Signal: {market_context.get('current_stance', 'N/A')}

RECENT TRAJECTORY:
{history_block}

USER QUESTION:
"{user_query}"

Provide a crisp, elite quantitative response with specific actionable numbers, mathematical risk management, and tactical execution instructions.
"""
    text, used_model, error_detail = run_advisor_text(
        prompt, model_name=model_name, system_prompt=SYSTEM_PROMPT
    )
    if text is not None:
        return {
            "text": text,
            "is_fallback": False,
            "model_used": used_model or model_name,
            "error_detail": None,
        }

    return {
        "text": f"⚠️ Offline / Fallback: Gemini Co-Pilot is currently unavailable ({error_detail}).",
        "is_fallback": True,
        "model_used": "offline-heuristic",
        "error_detail": error_detail,
    }


def generate_mock_analysis(
    current_metrics: Dict[str, Any],
    error_detail: Optional[str] = None,
) -> Dict[str, Any]:
    """Deterministic offline heuristic fallback with explicit fallback metadata."""
    qqq = float(current_metrics.get("qqq_price", 450.0))
    sma200 = float(current_metrics.get("sma_200", 430.0))
    buffer_pct = float(current_metrics.get("buffer_pct", 5.0))
    lower_buffer = float(
        current_metrics.get("lower_buffer", sma200 * (1.0 - buffer_pct / 100.0))
    )
    stance_raw = str(current_metrics.get("current_stance", "")).upper()

    if "SQQQ" in stance_raw or "SHORT" in stance_raw:
        regime = "Bear / Short Hedge Regime"
        rec_stance = "SHORT / LONG SQQQ (100%)"
        alloc = {"TQQQ_pct": 0, "SQQQ_pct": 100, "Cash_pct": 0}
        vol_risk = "High"
        verdict = "Maintain inverse SQQQ hedge while QQQ remains below its 200-day SMA."
    elif "50%" in stance_raw:
        regime = "Elevated-Volatility Bull / Partial Risk"
        rec_stance = "PARTIAL LONG TQQQ (50% / 50% CASH)"
        alloc = {"TQQQ_pct": 50, "SQQQ_pct": 0, "Cash_pct": 50}
        vol_risk = "Moderate"
        verdict = "Hold half-sized TQQQ exposure with 50% cash buffer to control 3x volatility drag."
    elif "LONG TQQQ" in stance_raw or qqq > sma200:
        regime = "Strong Bull Regime"
        rec_stance = "LONG TQQQ (100%)"
        alloc = {"TQQQ_pct": 100, "SQQQ_pct": 0, "Cash_pct": 0}
        vol_risk = "Low"
        verdict = "Maintain disciplined long TQQQ exposure while QQQ holds above its buffer floor."
    else:
        regime = "Bear / Defensive Regime"
        rec_stance = "CASH & FLAT (100%)"
        alloc = {"TQQQ_pct": 0, "SQQQ_pct": 0, "Cash_pct": 100}
        vol_risk = "High"
        verdict = "Preserve capital in cash until QQQ reclaims its moving-average trend structure."

    above_sma = qqq > sma200
    return {
        "regime": regime,
        "recommended_stance": rec_stance,
        "allocation": alloc,
        "volatility_decay_risk": vol_risk,
        "regime_thesis": (
            f"Offline heuristic assessment: QQQ (${qqq:.2f}) is trading "
            f"{'above' if above_sma else 'below'} its 200-day SMA (${sma200:.2f}) with a "
            f"{buffer_pct:.1f}% switch buffer floor at ${lower_buffer:.2f}. "
            f"Quant signal currently indicates {current_metrics.get('current_stance', rec_stance)}."
        ),
        "execution_plan": {
            "ideal_entry": f"Enter/hold TQQQ when QQQ confirms above 200 SMA (${sma200:.2f}) per active strategy rules.",
            "stop_loss_trigger": f"QQQ daily close below {buffer_pct:.1f}% buffer floor (${lower_buffer:.2f}) triggers defensive exit.",
            "trailing_management": f"Monitor QQQ 200 SMA (${sma200:.2f}), {buffer_pct:.1f}% floor (${lower_buffer:.2f}), and 10 EMA daily.",
        },
        "macro_risks": [
            "Interest-rate and inflation surprises pressuring long-duration Nasdaq valuations",
            "Mega-cap earnings concentration driving gap risk in 3x leveraged ETFs",
            "Volatility clustering causing daily compounding drag in choppy sideways regimes",
        ],
        "institutional_verdict": verdict,
        "is_fallback": True,
        "model_used": "offline-heuristic",
        "error_detail": error_detail or "Offline heuristic fallback used",
    }
