from fastapi import FastAPI, Form, File, UploadFile,HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import pickle
# import joblib
import json
# import config
import numpy as np
import requests
import io
# from typing import Optional
from PIL import Image
from torchvision import transforms
# from networkx import exception
import torch
from utils.model import ResNet9
# from utils import fertilizers
import firebase_admin
from firebase_admin import credentials, db,initialize_app
import asyncio
from pydantic import BaseModel
from typing import List
import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load environment variables
load_dotenv()

firebase_db_url = os.getenv('FIREBASE_DB_URL')

firebase_credentials = os.getenv('FIREBASE_CREDENTIALS') 
# Parse the Firebase credentials JSON string
firebase_credentials_dict = json.loads(firebase_credentials)
# Fix the `private_key` by replacing literal '\n' with actual newlines
firebase_credentials_dict['private_key'] = firebase_credentials_dict['private_key'].replace("\\n", "\n")


if not firebase_admin._apps:
    cred = credentials.Certificate(firebase_credentials_dict)
    initialize_app(cred, {'databaseURL': firebase_db_url})


# # Firebase Admin SDK setup
# cred = credentials.Certificate("firebaseCredentials.json")  # Path to your Firebase service account JSON file
# firebase_admin.initialize_app(cred, {
#     'databaseURL': 'https://krishishakti-a9606-default-rtdb.firebaseio.com/'
# })

dbmain = firestore.client()


crop_labels = [
'apple',
'banana',
'blackgram',
'chickpea',
'coconut',
'coffee',
'cotton',
'grapes',
'jute',
'kidneybeans',
'lentil',
'maize',
'mango',
'mothbeans',
'mungbean',
'muskmelon',
'orange',
'papaya',
'pigeonpeas',
'pomegranate',
'rice',
'watermelon']

disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)_Powdery_mildew',
                   'Cherry_(including_sour)_healthy',
                   'Corn_(maize)_Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)Common_rust',
                   'Corn_(maize)_Northern_Leaf_Blight',
                   'Corn_(maize)_healthy',
                   'Grape___Black_rot',
                   'Grape__Esca(Black_Measles)',
                   'Grape__Leaf_blight(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange__Haunglongbing(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,bell__Bacterial_spot',
                   'Pepper,bell__healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

crop_to_yield_mapping = {
    'papaya': 46,                 # Closest to "Sugarcane" (similar tropical crop)
    'rice': 40,                   # "Rice"
    'pomegranate': 36,            # Closest to "Peas & beans (Pulses)" (substitute category for fruits)
    'grapes': 12,                 # Closest to "Cowpea(Lobia)" (another climber plant)
    'coconut': 9,                 # "Coconut"
    'mango': 3,                   # "Banana" (tropical fruit)
    'mungbean': 27,               # "Moong(Green Gram)"
    'blackgram': 52,              # "Urad"
    'chickpea': 16,               # "Gram"
    'jute': 21,                   # "Jute"
    'mothbeans': 28,              # "Moth"
    'apple': 3,                   # Closest to "Banana" (fruit category)
    'banana': 3,                  # "Banana"
    'coffee': 46,                 # Closest to "Sugarcane" (high-value crop)
    'cotton': 11,                 # "Cotton(lint)"
    'kidneybeans': 36,            # Closest to "Peas & beans (Pulses)" 
    'lentil': 25,                 # "Masoor"
    'maize': 24,                  # "Maize"
    'muskmelon': 48,              # Closest to "Sweet potato" (similar seasonal crop)
    'orange': 3,                  # Closest to "Banana" (fruit category)
    'pigeonpeas': 1,              # "Arhar/Tur"
    'watermelon': 48              # Closest to "Sweet potato" (similar seasonal crop)
}

fertilizer_dic = {
        'NHigh': """The N value of soil is high and might give rise to weeds.
        <br/> Please consider the following suggestions:

        <br/><br/> 1. <i> Manure </i> – adding manure is one of the simplest ways to amend your soil with nitrogen. Be careful as there are various types of manures with varying degrees of nitrogen.

        <br/> 2. <i>Coffee grinds </i> – use your morning addiction to feed your gardening habit! Coffee grinds are considered a green compost material which is rich in nitrogen. Once the grounds break down, your soil will be fed with delicious, delicious nitrogen. An added benefit to including coffee grounds to your soil is while it will compost, it will also help provide increased drainage to your soil.

        <br/>3. <i>Plant nitrogen fixing plants</i> – planting vegetables that are in Fabaceae family like peas, beans and soybeans have the ability to increase nitrogen in your soil

        <br/>4. Plant ‘green manure’ crops like cabbage, corn and brocolli

        <br/>5. <i>Use mulch (wet grass) while growing crops</i> - Mulch can also include sawdust and scrap soft woods""",

        'Nlow': """The N value of your soil is low.
        <br/> Please consider the following suggestions:
        <br/><br/> 1. <i>Add sawdust or fine woodchips to your soil</i> – the carbon in the sawdust/woodchips love nitrogen and will help absorb and soak up and excess nitrogen.

        <br/>2. <i>Plant heavy nitrogen feeding plants</i> – tomatoes, corn, broccoli, cabbage and spinach are examples of plants that thrive off nitrogen and will suck the nitrogen dry.

        <br/>3. <i>Water</i> – soaking your soil with water will help leach the nitrogen deeper into your soil, effectively leaving less for your plants to use.

        <br/>4. <i>Sugar</i> – In limited studies, it was shown that adding sugar to your soil can help potentially reduce the amount of nitrogen is your soil. Sugar is partially composed of carbon, an element which attracts and soaks up the nitrogen in the soil. This is similar concept to adding sawdust/woodchips which are high in carbon content.

        <br/>5. Add composted manure to the soil.

        <br/>6. Plant Nitrogen fixing plants like peas or beans.

        <br/>7. <i>Use NPK fertilizers with high N value.

        <br/>8. <i>Do nothing</i> – It may seem counter-intuitive, but if you already have plants that are producing lots of foliage, it may be best to let them continue to absorb all the nitrogen to amend the soil for your next crops.""",

        'PHigh': """The P value of your soil is high.
        <br/> Please consider the following suggestions:

        <br/><br/>1. <i>Avoid adding manure</i> – manure contains many key nutrients for your soil but typically including high levels of phosphorous. Limiting the addition of manure will help reduce phosphorus being added.

        <br/>2. <i>Use only phosphorus-free fertilizer</i> – if you can limit the amount of phosphorous added to your soil, you can let the plants use the existing phosphorus while still providing other key nutrients such as Nitrogen and Potassium. Find a fertilizer with numbers such as 10-0-10, where the zero represents no phosphorous.

        <br/>3. <i>Water your soil</i> – soaking your soil liberally will aid in driving phosphorous out of the soil. This is recommended as a last ditch effort.

        <br/>4. Plant nitrogen fixing vegetables to increase nitrogen without increasing phosphorous (like beans and peas).

        <br/>5. Use crop rotations to decrease high phosphorous levels""",

        'Plow': """The P value of your soil is low.
        <br/> Please consider the following suggestions:

        <br/><br/>1. <i>Bone meal</i> – a fast acting source that is made from ground animal bones which is rich in phosphorous.

        <br/>2. <i>Rock phosphate</i> – a slower acting source where the soil needs to convert the rock phosphate into phosphorous that the plants can use.

        <br/>3. <i>Phosphorus Fertilizers</i> – applying a fertilizer with a high phosphorous content in the NPK ratio (example: 10-20-10, 20 being phosphorous percentage).

        <br/>4. <i>Organic compost</i> – adding quality organic compost to your soil will help increase phosphorous content.

        <br/>5. <i>Manure</i> – as with compost, manure can be an excellent source of phosphorous for your plants.

        <br/>6. <i>Clay soil</i> – introducing clay particles into your soil can help retain & fix phosphorus deficiencies.

        <br/>7. <i>Ensure proper soil pH</i> – having a pH in the 6.0 to 7.0 range has been scientifically proven to have the optimal phosphorus uptake in plants.

        <br/>8. If soil pH is low, add lime or potassium carbonate to the soil as fertilizers. Pure calcium carbonate is very effective in increasing the pH value of the soil.

        <br/>9. If pH is high, addition of appreciable amount of organic matter will help acidify the soil. Application of acidifying fertilizers, such as ammonium sulfate, can help lower soil pH""",

        'KHigh': """The K value of your soil is high</b>.
        <br/> Please consider the following suggestions:

        <br/><br/>1. <i>Loosen the soil</i> deeply with a shovel, and water thoroughly to dissolve water-soluble potassium. Allow the soil to fully dry, and repeat digging and watering the soil two or three more times.

        <br/>2. <i>Sift through the soil</i>, and remove as many rocks as possible, using a soil sifter. Minerals occurring in rocks such as mica and feldspar slowly release potassium into the soil slowly through weathering.

        <br/>3. Stop applying potassium-rich commercial fertilizer. Apply only commercial fertilizer that has a '0' in the final number field. Commercial fertilizers use a three number system for measuring levels of nitrogen, phosphorous and potassium. The last number stands for potassium. Another option is to stop using commercial fertilizers all together and to begin using only organic matter to enrich the soil.

        <br/>4. Mix crushed eggshells, crushed seashells, wood ash or soft rock phosphate to the soil to add calcium. Mix in up to 10 percent of organic compost to help amend and balance the soil.

        <br/>5. Use NPK fertilizers with low K levels and organic fertilizers since they have low NPK values.

        <br/>6. Grow a cover crop of legumes that will fix nitrogen in the soil. This practice will meet the soil’s needs for nitrogen without increasing phosphorus or potassium.
        """,

        'Klow': """The K value of your soil is low.
        <br/>Please consider the following suggestions:

        <br/><br/>1. Mix in muricate of potash or sulphate of potash
        <br/>2. Try kelp meal or seaweed
        <br/>3. Try Sul-Po-Mag
        <br/>4. Bury banana peels an inch below the soils surface
        <br/>5. Use Potash fertilizers since they contain high values potassium
        """
    }


crop_recommendation_model_path = 'crop_model.pkl'
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))

with open('yield_model.pkl', 'rb') as file:
    yield_model = pickle.load(file)

disease_model_path = 'plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(
    disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (POST, GET, etc.)
    allow_headers=["*"],  # Allows all headers
)

def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = '9d7cde1f6d07ec55650544be1631307e'
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        print(temperature, humidity)
        return temperature, humidity
    else:
        print("Nothing")
        return None
    
def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device)

def load_encoding_from_json(column_name):
    with open(f"encoding/{column_name}_encoding.json", 'r') as f:
        encoding_dict = json.load(f)
    return encoding_dict

def get_encoded_value(column_name, user_input):
    encoding_dict = load_encoding_from_json(column_name)
    
    if user_input in encoding_dict:
        return encoding_dict[user_input]
    else:
        print(f"Error: '{user_input}' is not a valid input for {column_name}.")
        return None

@app.get("/")
def root():
    return {"message": "Hello, World!"}

@app.post("/predictCrop")
async def predictRisk(n: str = Form(...),
    p: str = Form(...),
    k: str = Form(...),
    ph: str = Form(...),
    rainfall: str = Form(...),
    state: str = Form(...),
    city: str = Form(...),
    crop_year: str = Form(...),
    season: str = Form(...),
    area: str = Form(...),
    username: str = Form(...)):

    n = json.loads(n)
    p = json.loads(p)
    k = json.loads(k)
    ph = json.loads(ph)
    rainfall = json.loads(rainfall)
    city = json.loads(city)
    state = json.loads(state)
    crop_year = json.loads(crop_year)
    season = json.loads(season)
    area = json.loads(area)
    username = json.loads(username)
    temperature, humidity = weather_fetch(city)
    # temperature, humidity = 20, 82
    data = np.array([[n, p, k, temperature, humidity, ph, rainfall]])
    crop_probabilities = crop_recommendation_model.predict_proba(data)
    
    sorted_probability_dicts = []

    for prob in crop_probabilities:
        class_prob_pairs = [(crop_labels[i], prob[i]) for i in range(len(crop_labels))]
        
        sorted_class_prob_pairs = sorted(class_prob_pairs, key=lambda x: x[1], reverse=True)
        
        sorted_probability_dicts.append(sorted_class_prob_pairs)
        
    top_crops = []
    index = 0
    for crop in sorted_probability_dicts[0]:
            if index > 3: break
            top_crops.append(sorted_probability_dicts[0][index][0])
            index += 1
   
    crops_and_prediction = []
    for crop in top_crops:
        var = [[get_encoded_value('State',state), get_encoded_value('District',city), crop_to_yield_mapping[crop], 
                        int(crop_year), get_encoded_value('Season',season),float(area)]]
    
        expected_yield = yield_model.predict(var)[0]
        crops_and_prediction.append(expected_yield)
    
    ref = db.reference(f'predictions/{username}/topCrops')
    await asyncio.to_thread(ref.set, {
        '1': top_crops[0],
        '2': top_crops[1],
        '3': top_crops[2],
        '4': top_crops[3]
    })
    
    ref = db.reference(f'predictions/{username}/yieldPredictions')
    await asyncio.to_thread(ref.set, {
        '1': crops_and_prediction[0],
        '2': crops_and_prediction[1],
        '3': crops_and_prediction[2],
        '4': crops_and_prediction[3]
    })
    
@app.post('/predictFertilizer')     
async def predictFertilizer(n: str = Form(...),
    p: str = Form(...),
    k: str = Form(...),
    crop: str = Form(...),
    username: str = Form(...)):

    N = pd.to_numeric(json.loads(n))
    P = pd.to_numeric(json.loads(p))
    K = pd.to_numeric(json.loads(k))
    crop = json.loads(crop)

    print(crop)

    username = json.loads(username)

    print(crop)
    df = pd.read_csv('fertilizer.csv')

    nr = df[df['Crop'] == crop]['N'].iloc[0]
    pr = df[df['Crop'] == crop]['P'].iloc[0]
    kr = df[df['Crop'] == crop]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"

    response = str(fertilizer_dic[key])
    
    ref = db.reference(f'predictions/{username}/fertilizers')
    await asyncio.to_thread(ref.set, {
        'suggestions': response,
        'key': key
    })

    return JSONResponse(content={
        'suggestions': response,
        'key': key
    })
        
@app.post('/predictDisease')
async def predictDisease(image: UploadFile = File(...),
                         username: str = Form(...)):

    username = json.loads(username)

    transform = transforms.Compose([
        transforms.Resize(256),  # Resize the image to 224x224
        transforms.ToTensor(),           # Convert the image to a tensor
    ])
    image_content = await image.read()

    # Convert the image to a PIL Image
    # img = Image.open(io.BytesIO(image_content)).convert("RGB")  # Ensure it's RGB

            
    img = Image.open(io.BytesIO(image_content))
    img_t = transform(img)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = disease_model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    
    ref = db.reference(f'predictions/{username}/diseasePredictions')
    await asyncio.to_thread(ref.set, {
        'disease': prediction
    })

@app.post("/getPredictedCrops")
async def getPredictedCrop(
    username: str =  Form(...)
):
    print(username)
    username = json.loads(username)

    ref = db.reference(f'predictions/{username}/topCrops')
    top_crops = await asyncio.to_thread(ref.get)

    # Retrieve yield predictions
    ref = db.reference(f'predictions/{username}/yieldPredictions')
    yield_predictions = await asyncio.to_thread(ref.get)
    
    return JSONResponse(content={
            'crop_prediction': 
                {
                    'c1':top_crops[1],
                    'c2':top_crops[2],
                    'c3': top_crops[3],
                    'c4': top_crops[4]
                },
            'yield_prediction': 
                {
                    'y1': yield_predictions[1],
                    'y2': yield_predictions[2],
                    'y3': yield_predictions[3],
                    'y4': yield_predictions[4]
                }
        })


@app.post("/getPredictedFertilizers")
async def getPredictedFertilizers(
    username: str = Form(...)
):
    username = json.loads(username)
    
    ref = db.reference(f'predictions/{username}/fertilizers')
    fertilizers = await asyncio.to_thread(ref.get)
    
    if not fertilizers:
        raise HTTPException(status_code=404, detail="Fertilizer data not found")
    
    return JSONResponse(content={
        'suggestions': fertilizers['suggestions'],
        'key': fertilizers['key']
    })
    
@app.post("/getPredictedDisease") 
async def getPredictedDisease(
    username: str =  Form(...)
):
    
    username = json.loads(username)

    ref = db.reference(f'predictions/{username}/diseasePredictions')
    disease = await asyncio.to_thread(ref.get)
    
    return JSONResponse(content={
            'disease': disease['disease'],
        })


## Registering the farm
#  Pydantic models for request and farms
# class Farm(BaseModel):
#     farmId: str
#     farmName: str
#     latitude: str
#     longitude: str
#     userId: str

# class FarmsRequest(BaseModel):
#     userId: str
#     farms: List[Farm]

# @app.post("/register-farms")
# async def register_farms(request: FarmsRequest):
#     try:
#         # Simulate saving to Firestore
#         user_doc_ref = db.collection("farms").document(request.userId)
#         user_doc_ref.set({"farms": [farm.dict() for farm in request.farms]})
#         return {"message": "Farms registered successfully!", "userId": request.userId}
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail={"error": f"Failed to register farms: {str(e)}"}
#         )




class Farm(BaseModel):
    farmId: str
    farmName: str
    latitude: str
    longitude: str
    userId: str

class FarmsRequest(BaseModel):
    userId: str
    farms: List[Farm]

@app.post("/register-farms")
async def register_farms(request: FarmsRequest):
    try:
        # Simulate saving to Firestore
        user_doc_ref = dbmain.collection("farms").document(request.userId)
        user_doc_ref.set({"farms": [farm.dict() for farm in request.farms]})
        return {"message": "Farms registered successfully!", "userId": request.userId}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to register farms: {str(e)}"}
        )

## 

@app.get("/get-farms/{userId}")
async def get_farms(userId: str):
    try:
        user_doc_ref = dbmain.collection("farms").document(userId)
        user_doc = user_doc_ref.get()
        if user_doc.exists:
            return {"farms": user_doc.to_dict().get("farms", [])}
        else:
            return {"farms": []}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to get farms: {str(e)}"}
        )
    


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)