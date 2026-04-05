import os
import logging
import pandas as pd
import joblib
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    from groq import Groq
    _groq_available = True
except ImportError:
    _groq_available = False

from src.model_training import FEATURES


class SmartNoiseChatbot:
    def __init__(self):
        # Load API key at runtime (not import time) so st.secrets works on Streamlit Cloud
        groq_key = self._resolve_api_key()
        self.client     = Groq(api_key=groq_key) if (_groq_available and groq_key) else None
        self.model_name = "llama-3.1-8b-instant"
        # Conversation memory (last 6 turns kept for context)
        self.history: list[dict] = []

        try:
            self.df    = pd.read_csv('data/processed_data.csv')
            self.stats = self._precompute_stats()
        except Exception as e:
            self.df    = None
            self.stats = f"Data unavailable: {e}"

        try:
            self.ml_model = joblib.load('models/noise_model.pkl')
            self.meta     = joblib.load('models/model_meta.pkl')
        except Exception:
            self.ml_model = None
            self.meta     = {}

    @staticmethod
    def _resolve_api_key():
        """Try Streamlit secrets first, then env var."""
        try:
            import streamlit as st
            key = st.secrets.get("GROQ_API_KEY")
            if key:
                return str(key)
        except Exception:
            pass
        return os.environ.get("GROQ_API_KEY")

    # ------------------------------------------------------------------ #
    def _precompute_stats(self) -> str:
        d = self.df
        peak_hour  = d.groupby('hour')['Noise_Level_dB'].mean().idxmax()
        peak_noise = d.groupby('hour')['Noise_Level_dB'].mean().max()
        city_noise = d.groupby('City')['Noise_Level_dB'].mean().round(1)
        loudest    = city_noise.idxmax()
        quietest   = city_noise.idxmin()
        wknd_avg   = d[d['is_weekend'] == 1]['Noise_Level_dB'].mean()
        wkdy_avg   = d[d['is_weekend'] == 0]['Noise_Level_dB'].mean()
        rush_avg   = d[d['is_rush_hour'] == 1]['Noise_Level_dB'].mean()
        corr       = d['Noise_Level_dB'].corr(d['Traffic_Count'])
        city_lines = "\n".join(f"  - {c}: {v} dB" for c, v in city_noise.items())

        # Monthly trend summary
        monthly = d.groupby('month')['Noise_Level_dB'].mean()
        noisiest_month = monthly.idxmax()
        quietest_month = monthly.idxmin()
        month_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                       7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}

        return (
            f"Dataset: {len(d):,} records across {d['City'].nunique()} cities "
            f"({', '.join(sorted(d['City'].unique()))}).\n"
            f"Date range: {d['DateTime'].min()[:10]} to {d['DateTime'].max()[:10]}.\n"
            f"Peak noise hour: {peak_hour}:00 at {peak_noise:.1f} dB.\n"
            f"Rush-hour avg: {rush_avg:.1f} dB | Weekday avg: {wkdy_avg:.1f} dB | Weekend avg: {wknd_avg:.1f} dB.\n"
            f"Loudest city: {loudest} | Quietest city: {quietest}.\n"
            f"Noisiest month: {month_names[noisiest_month]} | Quietest month: {month_names[quietest_month]}.\n"
            f"Noise-Traffic correlation: {corr:.2f}.\n"
            f"Average noise by city:\n{city_lines}"
        )

    # ------------------------------------------------------------------ #
    def rule_based_intent(self, text: str) -> str:
        """Priority-ordered intent routing — more specific rules checked first."""
        t = text.lower()
        rules = [
            ("PREDICTION", ["predict", "forecast", "will be", "expect",
                             "future", "next hour", "estimate"]),
            ("HEALTH",     ["health", "danger", "safe", "risk", "effect",
                             "impact", "hearing", "damage"]),
            ("COMPARE",    ["compare", "vs", "versus", "difference between",
                             "better", "worse"]),
            ("TREND",      ["trend", "pattern", "over time", "monthly",
                             "seasonal", "increase", "decrease"]),
            ("ADVICE",     ["tip", "advice", "reduce", "mitigate", "solution",
                             "recommend", "improve", "lower noise"]),
            ("MODEL_INFO", ["model", "accuracy", "rmse", "r2", "performance",
                             "algorithm", "machine learning", "ml"]),
        ]
        for intent, keywords in rules:
            if any(w in t for w in keywords):
                return intent
        return "GENERAL"

    # ------------------------------------------------------------------ #
    def _extract_city(self, text: str) -> str:
        from src.api_integration import CITY_COORDS
        for c in CITY_COORDS:
            if c.lower() in text.lower():
                return c
        return "New York"

    def _build_messages(self, system_prompt: str, user_prompt: str) -> list:
        """Build message list with conversation history for multi-turn context."""
        messages = [{"role": "system", "content": system_prompt}]
        # Include last 6 history turns
        messages.extend(self.history[-6:])
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def _call_groq(self, system_prompt: str, user_prompt: str) -> str:
        if not self.client:
            return ("Groq API key not configured. "
                    "Set the GROQ_API_KEY environment variable to enable AI responses.")
        try:
            messages = self._build_messages(system_prompt, user_prompt)
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.4,
                max_tokens=700,
            )
            response = completion.choices[0].message.content
            # Store turn in history — cap at 20 items (10 turns) to prevent memory leak
            self.history.append({"role": "user",      "content": user_prompt})
            self.history.append({"role": "assistant", "content": response})
            if len(self.history) > 20:
                self.history = self.history[-20:]
            return response
        except Exception as e:
            return f"[Groq API Error]: {e}"

    # ------------------------------------------------------------------ #
    def handle_prediction(self, user_input: str) -> str:
        if not self.ml_model:
            return "The prediction model is currently unavailable."

        from src.api_integration import CITY_COORDS, get_realtime_features
        city  = self._extract_city(user_input)
        now   = datetime.now()
        hr    = now.hour

        try:
            # Use get_realtime_features — always returns the correct 18-feature vector
            feats = get_realtime_features(city)
            row   = pd.DataFrame([feats], columns=FEATURES)
            pred  = self.ml_model.predict(row)[0]

            # Extract display values from feature vector (new order: dow,is_wknd,is_rush,is_night,traffic,temp,wind,precip,humid,pm25,...)
            traffic = feats[4]
            temp    = feats[5]
            wind    = feats[6]
            precip  = feats[7]
            humid   = feats[8]
            pm25    = feats[9]

            level = ("very loud and potentially harmful" if pred > 75
                     else "elevated and noticeable" if pred > 65
                     else "moderate and acceptable")
            system_msg = (
                f"You are the Smart Urban Noise Analyzer AI. "
                f"The ML model predicted {pred:.1f} dB for {city} at {hr}:00 "
                f"({level} noise level). "
                f"Real weather context (Open-Meteo): temp={temp}°C, "
                f"wind={wind} km/h, humidity={humid}%, precipitation={precip} mm/hr. "
                f"Traffic estimate: {traffic} veh/hr. PM2.5 estimate: {pm25} µg/m³. "
                f"Explain the prediction, what it means for residents, and any recommended actions. "
                f"Be concise, friendly, and professional. Use bullet points where helpful."
            )
            return self._call_groq(system_msg, user_input)
        except Exception as e:
            logger.error("Chatbot prediction error for %s: %s", city, e)
            return f"Prediction error: {e}"

    def handle_compare(self, user_input: str) -> str:
        from src.api_integration import CITY_COORDS, get_realtime_features
        live_preds = {}
        if self.ml_model:
            for city in CITY_COORDS:
                try:
                    feats = get_realtime_features(city)
                    row   = pd.DataFrame([feats], columns=FEATURES)
                    live_preds[city] = round(self.ml_model.predict(row)[0], 1)
                except Exception as e:
                    logger.warning("Compare prediction failed for %s: %s", city, e)
                    live_preds[city] = "N/A"

        live_text = "\n".join(f"  - {c}: {v} dB (live estimate)" for c, v in live_preds.items())
        system_msg = (
            f"You are the Smart Urban Noise Analyzer. "
            f"Use ONLY this data to answer comparisons.\n\n"
            f"Historical averages:\n{self.stats}\n\n"
            f"Current live predictions:\n{live_text}\n\n"
            f"Be factual, concise, and use a comparison table if helpful."
        )
        return self._call_groq(system_msg, user_input)

    def handle_advice(self, user_input: str) -> str:
        system_msg = (
            "You are an urban noise pollution expert and city planner. "
            "Provide practical, evidence-based advice for reducing urban noise. "
            "Structure your response with clear categories (e.g., infrastructure, behavior, policy). "
            f"Reference this dataset context where relevant:\n{self.stats}"
        )
        return self._call_groq(system_msg, user_input)

    def handle_model_info(self, user_input: str) -> str:
        if self.meta:
            metrics_text = "\n".join(
                f"  {k}: RMSE={v['rmse']:.4f}, MAE={v['mae']:.4f}, R²={v['r2']:.4f}"
                for k, v in self.meta.get("metrics", {}).items()
            )
            context = (
                f"Best model: {self.meta.get('best_model_name', 'N/A')}\n"
                f"Features used ({len(self.meta.get('features', []))}): "
                f"{', '.join(self.meta.get('features', []))}\n"
                f"Training data: {len(self.df):,} records across 5 cities\n"
                f"Model comparison:\n{metrics_text}"
            )
        else:
            context = "Model metadata unavailable."
        system_msg = (
            "You are a data science assistant explaining ML model results to a non-technical audience. "
            "Explain metrics like RMSE and R² in plain language. "
            f"Use this information:\n{context}"
        )
        return self._call_groq(system_msg, user_input)

    def handle_trend(self, user_input: str) -> str:
        if self.df is not None:
            monthly = self.df.groupby('month')['Noise_Level_dB'].mean().round(1)
            hourly  = self.df.groupby('hour')['Noise_Level_dB'].mean().round(1)
            month_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                           7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
            monthly_text = ", ".join(f"{month_names[m]}: {v} dB" for m, v in monthly.items())
            hourly_text  = ", ".join(f"{h}:00={v}" for h, v in hourly.items())
            trend_context = (
                f"Monthly noise averages: {monthly_text}\n"
                f"Hourly noise averages: {hourly_text}\n"
                f"{self.stats}"
            )
        else:
            trend_context = self.stats

        system_msg = (
            "You are an urban data analyst. Identify and explain noise trends, "
            "patterns, and seasonal variations from the data below. "
            "Highlight the most significant findings.\n\n"
            f"{trend_context}"
        )
        return self._call_groq(system_msg, user_input)

    def handle_health(self, user_input: str) -> str:
        city = self._extract_city(user_input)
        city_avg = ""
        if self.df is not None:
            avg = self.df[self.df['City'] == city]['Noise_Level_dB'].mean()
            city_avg = f"Average noise in {city}: {avg:.1f} dB."

        system_msg = (
            "You are a public health expert specializing in environmental noise. "
            "Explain the health impacts of urban noise pollution based on WHO guidelines. "
            "Cover: hearing damage thresholds, sleep disruption, cardiovascular risk, "
            "cognitive effects on children, and stress. "
            f"Context from our dataset: {city_avg}\n{self.stats}\n"
            "Be informative but accessible. Use WHO dB thresholds as reference."
        )
        return self._call_groq(system_msg, user_input)

    def handle_general(self, user_input: str) -> str:
        system_msg = (
            "You are the Smart Urban Noise Analyzer chatbot for a graduation project. "
            "Answer questions using the following dataset insights. "
            "If the question is outside the scope of urban noise analysis, "
            "politely redirect to relevant topics.\n\n"
            f"{self.stats}"
        )
        return self._call_groq(system_msg, user_input)

    # ------------------------------------------------------------------ #
    def chat_loop(self):
        print("\n" + "=" * 55)
        print("  Smart Urban Noise Analyzer — Terminal Chatbot")
        print("=" * 55)
        print("Type 'exit' to quit.\n")
        dispatch = {
            "PREDICTION": self.handle_prediction,
            "COMPARE":    self.handle_compare,
            "ADVICE":     self.handle_advice,
            "MODEL_INFO": self.handle_model_info,
            "TREND":      self.handle_trend,
            "HEALTH":     self.handle_health,
            "GENERAL":    self.handle_general,
        }
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ('exit', 'quit'):
                print("Goodbye!")
                break
            intent   = self.rule_based_intent(user_input)
            response = dispatch[intent](user_input)
            print(f"\nAI: {response}\n")


if __name__ == "__main__":
    bot = SmartNoiseChatbot()
    bot.chat_loop()
