House Price Predictor

A small web app that estimates the market price of a house in Rwanda (in million Rwandan francs, RWF) from a few simple characteristics. It was built for the Machine Learning module (ETT Y4 BTech, Kigali College, Rwanda Polytechnic).

Live app: https://houseprice-qmgn4d9eajhepfkzdxv4jj.streamlit.app/

Project purpose

A real-estate agency wants a quick tool that lets an agent type in the details of a house and instantly see an estimated price. This project trains a multiple linear regression model on past sales, saves it to a .sav file, and serves it through a Streamlit web app.

Dataset
Source: house_price_prediction_dataset.csv, provided by the agency (125 raw records).
Target: House_Price_Million_RWF (sale price in million RWF).
Predictors: Area_m2, Bedrooms, Bathrooms, House_Age_Years, Distance_to_City_km, Parking_Spaces, Neighborhood (Gasabo, Huye, Kicukiro, Kigali City, Musanze, Nyarugenge). House_ID is only a label and is not used.
The raw data had missing values, 5 duplicate rows and 5 impossible records (areas of 850 and 1200 m², prices of 950 and 1200 million RWF, and a house 85 km from the centre in a Kigali district). After cleaning, 114 houses remain (house_price_cleaned.csv). Every cleaning decision is explained in the notebook.
How the model was built
Cleaned the data: removed duplicates, dropped the row with a missing price, and removed the 5 impossible records.
Split the data 80% training / 20% testing with random_state=42.
Built a scikit-learn Pipeline: median imputation for numeric columns, most-frequent imputation and one-hot encoding (first category dropped) for Neighborhood, then LinearRegression.
Checked multicollinearity (Bedrooms and Bathrooms are strongly related, VIF about 7), residual plots, and assumption tests.
Saved the whole pipeline with joblib as house_price_model.sav, so it predicts directly from raw inputs.
Model performance
Set	R²	MAE (million RWF)	RMSE (million RWF)
Training	0.862	15.5	18.9
Test	0.846	17.2	19.8

The model explains about 85% of the price variation on unseen houses, with a typical error of about 20 million RWF. The test set is small (23 houses), so these figures are rough estimates.

Run the app locally
bash
git clone https://github.com/YOUR_USERNAME/house-price-predictor.git
cd house-price-predictor
pip install -r requirements.txt
python -m streamlit run app.py

Then open http://localhost:8501 in your browser.

Repository contents
File	Purpose
app.py	Streamlit web app
house_price_model.sav	Trained model (full pipeline)
house_price_analysis.ipynb	Cleaning, modelling, evaluation and saving the model
house_price_cleaned.csv	Cleaned dataset
requirements.txt	Python dependencies
Limitations
Small dataset (114 houses), so predictions carry a typical error of about 20 million RWF.
Only seven features are used; plot size, build quality and finishing also affect real prices.
The model assumes straight-line effects and is unreliable outside the training ranges (the app shows a warning in that case).
Author

Cikuru Elie, Machine Learning module.
