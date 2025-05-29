from flask import Flask, request, jsonify
from utils import load_model, load_encoder
import pandas as pd
from flask_cors import CORS
import bcrypt
import jwt
import datetime

from pymongo import MongoClient

# MongoDB connection
MONGO_URI = "mongodb+srv://tilankamihingu:tila1997@inventorywaste.eemvm.mongodb.net/?retryWrites=true&w=majority&appName=inventorywaste"
client = MongoClient(MONGO_URI)
db = client["inventorymanagement"]
users_collection = db["users"]

# JWT secret key
JWT_SECRET = "my_super_secret_2025_token"

# Initialize Flask
app = Flask(__name__)
CORS(app, supports_credentials=True, resources={
    r"/api/*": {
        "origins": "http://localhost:3000",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Load models
sales_model = load_model('models/rf_sold_model.pkl')
waste_model = load_model('models/rf_waste_model.pkl')
staff_model = load_model('models/rf_staff_model.pkl')
price_model = load_model('models/rf_price_model.pkl')

# Load encoders
dish_encoder = load_encoder('models/dish_encoder.pkl')
day_encoder = load_encoder('models/day_encoder.pkl')

# Load dataset
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

                total_sales += pred_sold * pred_price
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
        print("Forecasting error:", e)
        return jsonify({"error": str(e)}), 500

# @app.route("/api/forecast/dish-wise", methods=["GET"])
# def forecast_dish_wise():
#     try:
#         dish_summary = df.groupby("dish_name").agg({
#             "sold_count": "sum",
#             "waste_count": "sum",
#             "prepared_count": "sum"
#         }).reset_index()

#         dish_summary["waste_ratio"] = (
#             dish_summary["waste_count"] / dish_summary["prepared_count"]
#         ).fillna(0).round(2)

#         result = dish_summary.to_dict(orient="records")
#         return jsonify(result)

#     except Exception as e:
#         print("Dish-wise forecast error:", e)
#         return jsonify({"error": str(e)}), 500

@app.route("/api/forecast/dish-wise", methods=["POST"])
def forecast_dish_wise():
    try:
        req = request.get_json()
        selected_month = int(req.get("month", 1))

        df_month = df[df["month"] == selected_month]
        if df_month.empty:
            return jsonify([])

        summary = df_month.groupby("dish_name").agg({
            "sold_count": "sum",
            "waste_count": "sum",
            "prepared_count": "sum"
        }).reset_index()

        summary["waste_ratio"] = (
            summary["waste_count"] / summary["prepared_count"]
        ).fillna(0).round(2) * 100  # percentage

        # Rename columns to match FE expectations
        summary.rename(columns={
            "dish_name": "name",
            "sold_count": "predicted_sales",
            "waste_count": "predicted_waste"
        }, inplace=True)

        return jsonify(summary.to_dict(orient="records"))

    except Exception as e:
        print("Dish-wise prediction error:", e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/register", methods=["POST"])
def register():
    try:
        data = request.get_json()
        print("Incoming registration data:", data)

        email = data.get("email")
        password = data.get("password")

        if not email or not password:
            return jsonify({"error": "Email and password required"}), 400

        existing_user = users_collection.find_one({"email": email})
        if existing_user:
            return jsonify({"error": "User already exists"}), 400

        hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
        users_collection.insert_one({"email": email, "password": hashed_pw.decode()})

        return jsonify({"message": "User registered successfully"}), 201

    except Exception as e:
        print("Registration error:", e)
        return jsonify({"error": str(e)}), 500

@app.route("/api/login", methods=["POST"])
def login():
    try:
        data = request.get_json()
        email = data.get("email")
        password = data.get("password")

        user = users_collection.find_one({"email": email})
        if not user:
            return jsonify({"error": "Invalid email"}), 401

        if not bcrypt.checkpw(password.encode(), user["password"].encode()):
            return jsonify({"error": "Wrong password"}), 401

        token = jwt.encode({
            "email": email,
            "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=2)
        }, JWT_SECRET, algorithm="HS256")

        return jsonify({"token": token})

    except Exception as e:
        print("Login error:", e)
        return jsonify({"error": str(e)}), 500

@app.before_request
def handle_options_requests():
    if request.method == "OPTIONS":
        return '', 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
