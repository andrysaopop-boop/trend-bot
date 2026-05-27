import ccxt
import pandas as pd

from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange


# =========================
# НАСТРОЙКИ
# =========================

SYMBOL = "SUI/USDT"
TIMEFRAME = "1h"
LIMIT = 500


# =========================
# ЗАГРУЗКА ДАННЫХ
# =========================

def fetch_ohlcv(symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
    """
    Получает свечи с нескольких бирж.
    Если одна биржа не работает, пробует следующую.
    Binance часто даёт 451 по региону, поэтому он в конце списка.
    """

    exchanges = [
        (
            "OKX",
            ccxt.okx({
                "enableRateLimit": True,
            })
        ),
        (
            "Bybit",
            ccxt.bybit({
                "enableRateLimit": True,
                "options": {
                    "defaultType": "spot"
                }
            })
        ),
        (
            "KuCoin",
            ccxt.kucoin({
                "enableRateLimit": True,
            })
        ),
        (
            "Binance",
            ccxt.binance({
                "enableRateLimit": True,
            })
        ),
    ]

    last_error = None

    for exchange_name, exchange in exchanges:
        try:
            print(f"Пробую биржу: {exchange_name}")

            candles = exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                limit=limit
            )

            if not candles or len(candles) < 100:
                print(f"{exchange_name}: мало свечей, пробую следующую биржу")
                continue

            df = pd.DataFrame(
                candles,
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.dropna().reset_index(drop=True)

            print(f"Данные получены с биржи: {exchange_name}")
            print(f"Получено свечей: {len(df)}")
            print("-" * 60)

            return df

        except Exception as error:
            last_error = error
            print(f"{exchange_name} не сработала:")
            print(error)
            print("-" * 60)

    raise RuntimeError(
        f"Не удалось получить данные ни с одной биржи. "
        f"Последняя ошибка: {last_error}"
    )


# =========================
# ИНДИКАТОРЫ
# =========================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет EMA, RSI, MACD, ADX, ATR.
    """

    df["ema20"] = EMAIndicator(
        close=df["close"],
        window=20
    ).ema_indicator()

    df["ema50"] = EMAIndicator(
        close=df["close"],
        window=50
    ).ema_indicator()

    df["ema200"] = EMAIndicator(
        close=df["close"],
        window=200
    ).ema_indicator()

    df["rsi"] = RSIIndicator(
        close=df["close"],
        window=14
    ).rsi()

    macd = MACD(
        close=df["close"],
        window_slow=26,
        window_fast=12,
        window_sign=9
    )

    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    adx = ADXIndicator(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=14
    )

    df["adx"] = adx.adx()

    atr = AverageTrueRange(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=14
    )

    df["atr"] = atr.average_true_range()

    return df


# =========================
# ПОДДЕРЖКА / СОПРОТИВЛЕНИЕ
# =========================

def find_support_resistance(
    df: pd.DataFrame,
    lookback: int = 80
) -> tuple[float, float]:
    """
    Простая версия:
    поддержка = минимум за последние lookback свечей
    сопротивление = максимум за последние lookback свечей
    """

    recent = df.tail(lookback)

    support = recent["low"].min()
    resistance = recent["high"].max()

    return round(float(support), 4), round(float(resistance), 4)


# =========================
# СТРУКТУРА ЦЕНЫ
# =========================

def price_structure_score(df: pd.DataFrame, lookback: int = 30) -> tuple[int, str]:
    """
    Проверяет структуру цены:
    - максимумы растут?
    - минимумы растут?
    - цена в конце выше, чем в начале?
    """

    recent = df.tail(lookback).reset_index(drop=True)

    half = lookback // 2

    first_part = recent.iloc[:half]
    second_part = recent.iloc[half:]

    first_close = recent["close"].iloc[0]
    last_close = recent["close"].iloc[-1]

    high_1 = first_part["high"].max()
    high_2 = second_part["high"].max()

    low_1 = first_part["low"].min()
    low_2 = second_part["low"].min()

    score = 0
    details = []

    if last_close > first_close:
        score += 1
        details.append("последняя цена выше начала периода")
    else:
        score -= 1
        details.append("последняя цена ниже начала периода")

    if high_2 > high_1:
        score += 1
        details.append("максимумы растут")
    else:
        score -= 1
        details.append("максимумы не растут")

    if low_2 > low_1:
        score += 1
        details.append("минимумы растут")
    else:
        score -= 1
        details.append("минимумы не растут")

    return score, ", ".join(details)


# =========================
# АНАЛИЗ ТРЕНДА
# =========================

def analyze_trend(df: pd.DataFrame) -> dict:
    """
    Главная логика анализа.
    Возвращает словарь с трендом, баллами, уровнями и причинами.
    """

    if len(df) < 50:
        raise ValueError(
            f"Слишком мало данных для анализа: {len(df)} строк. "
            f"Нужно хотя бы 50."
        )

    last = df.iloc[-1]
    prev = df.iloc[-2]

    score = 0
    reasons = []

    close = float(last["close"])
    prev_close = float(prev["close"])

    # =========================
    # EMA
    # =========================

    if close > last["ema20"]:
        score += 1
        reasons.append("Цена выше EMA20: краткосрочно есть сила")
    else:
        score -= 1
        reasons.append("Цена ниже EMA20: краткосрочно слабость")

    if close > last["ema50"]:
        score += 1
        reasons.append("Цена выше EMA50: среднесрочно покупатель держится")
    else:
        score -= 1
        reasons.append("Цена ниже EMA50: среднесрочно продавец давит")

    if last["ema20"] > last["ema50"]:
        score += 1
        reasons.append("EMA20 выше EMA50: краткосрочный тренд лучше")
    else:
        score -= 1
        reasons.append("EMA20 ниже EMA50: краткосрочный тренд слабый")

    if close > last["ema200"]:
        score += 1
        reasons.append("Цена выше EMA200: глобально структура сильнее")
    else:
        score -= 1
        reasons.append("Цена ниже EMA200: глобально актив под давлением")

    # =========================
    # MACD
    # =========================

    if last["macd"] > last["macd_signal"]:
        score += 1
        reasons.append("MACD выше сигнальной линии: импульс вверх")
    else:
        score -= 1
        reasons.append("MACD ниже сигнальной линии: импульс слабый")

    if last["macd_hist"] > prev["macd_hist"]:
        score += 1
        reasons.append("Гистограмма MACD растёт: импульс улучшается")
    else:
        score -= 1
        reasons.append("Гистограмма MACD падает: импульс тухнет")

    # =========================
    # RSI
    # =========================

    rsi = float(last["rsi"])

    if rsi >= 70:
        score -= 1
        reasons.append("RSI выше 70: перегрев, возможен откат")
    elif 60 <= rsi < 70:
        score += 1
        reasons.append("RSI 60–70: покупатель сильный, но ещё не жёсткий перегрев")
    elif 45 <= rsi < 60:
        reasons.append("RSI 45–60: нейтрально, явного перегрева нет")
    elif 30 <= rsi < 45:
        score -= 1
        reasons.append("RSI 30–45: слабость, покупатель не очень активен")
    else:
        score += 1
        reasons.append("RSI ниже 30: перепроданность, возможен отскок")

    # =========================
    # ADX
    # =========================

    adx = float(last["adx"])

    if adx < 20:
        reasons.append("ADX ниже 20: сильного тренда нет, вероятен боковик")
    elif 20 <= adx < 25:
        reasons.append("ADX 20–25: тренд слабый")
    elif 25 <= adx < 35:
        score += 1
        reasons.append("ADX 25–35: тренд есть")
    else:
        score += 2
        reasons.append("ADX выше 35: тренд сильный")

    # =========================
    # СТРУКТУРА ЦЕНЫ
    # =========================

    structure_score, structure_details = price_structure_score(df)

    if structure_score >= 2:
        score += 2
        reasons.append(f"Структура цены бычья: {structure_details}")
    elif structure_score <= -2:
        score -= 2
        reasons.append(f"Структура цены медвежья: {structure_details}")
    else:
        reasons.append(f"Структура цены смешанная: {structure_details}")

    # =========================
    # ОБЪЁМ
    # =========================

    avg_volume = float(df["volume"].tail(20).mean())
    current_volume = float(last["volume"])

    if current_volume > avg_volume and close > prev_close:
        score += 1
        reasons.append("Рост на объёме выше среднего: покупку подтверждают")
    elif current_volume > avg_volume and close < prev_close:
        score -= 1
        reasons.append("Падение на объёме выше среднего: продавец активен")
    else:
        reasons.append("Объём обычный: сильного подтверждения нет")

    # =========================
    # УРОВНИ
    # =========================

    support, resistance = find_support_resistance(df)

    distance_to_support = ((close - support) / close) * 100
    distance_to_resistance = ((resistance - close) / close) * 100

    # =========================
    # ИТОГОВЫЙ ТРЕНД
    # =========================

    if score >= 6:
        trend = "ВВЕРХ"
        confidence = "высокая"
    elif 3 <= score < 6:
        trend = "СЛАБЫЙ ВВЕРХ / ОТСКОК"
        confidence = "средняя"
    elif -2 <= score <= 2:
        trend = "БОКОВИК / НЕОПРЕДЕЛЁННОСТЬ"
        confidence = "средняя"
    elif -5 <= score < -2:
        trend = "СЛАБЫЙ ВНИЗ"
        confidence = "средняя"
    else:
        trend = "ВНИЗ"
        confidence = "высокая"

    # =========================
    # РЕШЕНИЕ
    # =========================

    if trend == "ВВЕРХ":
        decision = "Можно рассматривать покупку частями, но не всей суммой."
    elif trend == "СЛАБЫЙ ВВЕРХ / ОТСКОК":
        decision = "Можно смотреть маленькую пробную покупку, но ждать подтверждение."
    elif trend == "БОКОВИК / НЕОПРЕДЕЛЁННОСТЬ":
        decision = "Лучше ждать: или откат к поддержке, или пробой сопротивления."
    elif trend == "СЛАБЫЙ ВНИЗ":
        decision = "Покупать рано. Ждать поддержки или разворота."
    else:
        decision = "Не покупать сейчас. Тренд вниз, лучше ждать стабилизацию."

    return {
        "symbol": SYMBOL,
        "timeframe": TIMEFRAME,
        "close": round(close, 4),
        "trend": trend,
        "confidence": confidence,
        "score": score,
        "support": support,
        "resistance": resistance,
        "distance_to_support": round(distance_to_support, 2),
        "distance_to_resistance": round(distance_to_resistance, 2),
        "rsi": round(rsi, 2),
        "macd": round(float(last["macd"]), 4),
        "macd_signal": round(float(last["macd_signal"]), 4),
        "macd_hist": round(float(last["macd_hist"]), 4),
        "adx": round(adx, 2),
        "ema20": round(float(last["ema20"]), 4),
        "ema50": round(float(last["ema50"]), 4),
        "ema200": round(float(last["ema200"]), 4),
        "volume": round(current_volume, 2),
        "avg_volume": round(avg_volume, 2),
        "decision": decision,
        "reasons": reasons
    }


# =========================
# ВЫВОД ОТЧЁТА
# =========================

def print_report(result: dict) -> None:
    print()
    print("=" * 70)
    print("АНАЛИЗ ТРЕНДА")
    print("=" * 70)

    print(f"Актив: {result['symbol']}")
    print(f"Таймфрейм: {result['timeframe']}")
    print(f"Цена: {result['close']}")

    print("-" * 70)
    print(f"Тренд: {result['trend']}")
    print(f"Уверенность: {result['confidence']}")
    print(f"Баллы: {result['score']}")

    print("-" * 70)
    print(f"Поддержка: {result['support']}")
    print(f"Сопротивление: {result['resistance']}")
    print(f"До поддержки: {result['distance_to_support']}%")
    print(f"До сопротивления: {result['distance_to_resistance']}%")

    print("-" * 70)
    print("Индикаторы:")
    print(f"RSI: {result['rsi']}")
    print(f"MACD: {result['macd']}")
    print(f"MACD Signal: {result['macd_signal']}")
    print(f"MACD Hist: {result['macd_hist']}")
    print(f"ADX: {result['adx']}")
    print(f"EMA20: {result['ema20']}")
    print(f"EMA50: {result['ema50']}")
    print(f"EMA200: {result['ema200']}")
    print(f"Объём: {result['volume']}")
    print(f"Средний объём за 20 свечей: {result['avg_volume']}")

    print("-" * 70)
    print("Почему такой вывод:")
    for reason in result["reasons"]:
        print(f"- {reason}")

    print("-" * 70)
    print(f"Решение: {result['decision']}")

    print("=" * 70)
    print()


# =========================
# MAIN
# =========================

def main() -> None:
    try:
        df = fetch_ohlcv(SYMBOL, TIMEFRAME, LIMIT)

        df = add_indicators(df)

        df = df.dropna().reset_index(drop=True)

        print(f"Свечей после расчёта индикаторов: {len(df)}")

        if len(df) < 50:
            print("Ошибка: после индикаторов осталось слишком мало данных.")
            print("Увеличь LIMIT до 800 или 1000.")
            return

        result = analyze_trend(df)

        print_report(result)

    except KeyboardInterrupt:
        print("Остановлено пользователем.")

    except Exception as error:
        print()
        print("ОШИБКА:")
        print(error)
        print()
        print("Что можно сделать:")
        print("1. Проверь интернет.")
        print("2. Попробуй другой SYMBOL, например BTC/USDT.")
        print("3. Попробуй другой TIMEFRAME, например 4h или 1d.")
        print("4. Если биржа заблокирована, программа попробует другую.")
        print()


if __name__ == "__main__":
    main()