from flask import Flask, request, jsonify
from utils import load_model, load_encoder
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app, supports_credentials=True, origins=["http://localhost:3000"], allow_headers="*", methods=["GET", "POST", "OPTIONS"])

# Load models
sales_model = load_model('models/rf_sold_model.pkl')
waste_model = load_model('models/rf_waste_model.pkl')
staff_model = load_model('models/rf_staff_model.pkl')
price_model = load_model('models/rf_price_model.pkl')

# Load encoders
dish_encoder = load_encoder('models/dish_encoder.pkl')
day_encoder = load_encoder('models/day_encoder.pkl')

# Load historical dataset
df = pd.read_csv("hareesha-dataset.csv")
df["month"] = pd.to_datetime(df["date"]).dt.month
df["day"] = pd.to_datetime(df["date"]).dt.day
df["year"] = pd.to_datetime(df["date"]).dt.year

@app.route("/api/forecast/month", methods=["POST"])
def forecast_month():
    try:
        req = request.get_json()
        selected_month = int(req.get("month", 1))
        forecast_year = 2025

        dishes = df['dish_name'].unique()
        df_month = df[df["month"] == selected_month]
        if df_month.empty:
            return jsonify({"error": "No historical data for this month"}), 404

        breakdown = []
        total_sales = 0
        total_waste = 0

        for dish in dishes:
            df_dish = df_month[df_month['dish_name'] == dish]
            grouped = df_dish.groupby("day").agg({
                "prepared_count": "mean",
                "price_per_item": "mean",
                "staff_count": "mean",
                "is_peak_day": "mean",
                "day_of_week": lambda x: x.mode().iloc[0] if not x.mode().empty else "Monday"
            }).reset_index()

            for _, row in grouped.iterrows():
                day = int(row["day"])
                forecast_date = f"{forecast_year}-{str(selected_month).zfill(2)}-{str(day).zfill(2)}"

                dish_encoded = dish_encoder.transform([dish])[0]
                day_encoded = day_encoder.transform([row["day_of_week"]])[0]

                price_input = pd.DataFrame([{ 
                    "dish_name": dish_encoded, 
                    "year": forecast_year, 
                    "month": selected_month, 
                    "is_peak_day": row["is_peak_day"] 
                }])
                pred_price = price_model.predict(price_input)[0]

                sales_input = pd.DataFrame([{ 
                    "dish_name": dish_encoded, 
                    "day_of_week": day_encoded, 
                    "prepared_count": row["prepared_count"], 
                    "price_per_item": pred_price, 
                    "staff_count": row["staff_count"], 
                    "is_peak_day": row["is_peak_day"] 
                }])
                pred_sold = sales_model.predict(sales_input)[0]

                waste_input = pd.DataFrame([{ 
                    "dish_name": dish_encoded, 
                    "day_of_week": day_encoded, 
                    "prepared_count": row["prepared_count"], 
                    "sold_count": pred_sold, 
                    "price_per_item": pred_price, 
                    "staff_count": row["staff_count"], 
                    "is_peak_day": row["is_peak_day"] 
                }])
                pred_waste = waste_model.predict(waste_input)[0]

                staff_input = pd.DataFrame([{ 
                    "day_of_week": day_encoded, 
                    "is_peak_day": row["is_peak_day"], 
                    "prepared_count": row["prepared_count"], 
                    "sold_count": pred_sold, 
                    "price_per_item": pred_price 
                }])
                pred_staff = staff_model.predict(staff_input)[0]

                breakdown.append({
                    "date": forecast_date,
                    "dish_name": dish,
                    "predicted_price": round(pred_price, 2),
                    "predicted_sales": round(pred_sold, 2),
                    "predicted_waste": round(pred_waste, 2),
                    "predicted_staff": int(round(pred_staff))
                })

                total_sales += pred_sold * pred_price  # ✅ revenue in Rs.
                total_waste += pred_waste

        avg_sales = total_sales / len(breakdown)
        avg_waste = total_waste / len(breakdown)

        return jsonify({
            "breakdown": breakdown,
            "total_sales": round(total_sales, 2),
            "avg_daily_sales": round(avg_sales, 2),
            "total_food_waste": round(total_waste, 2),
            "avg_daily_waste": round(avg_waste, 2)
        })

    except Exception as e:
        print("❌ Forecasting error:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
