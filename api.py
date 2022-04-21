from typing import Optional, List, Union
from enum import Enum
from fastapi import FastAPI, HTTPException
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import requests
from pydantic import BaseModel
import numpy as np
from .settings import DATA_URL, KNN_INDEX_PATH

# Get the review data
response = requests.get(DATA_URL)
records = response.json()
df = pd.json_normalize(records)
df = df.rename({'gender of author': 'gender_of_author', 'gender of critic': 'gender_of_critic', 'same year': 'same_year'}, axis=1)

# Get the nearest neighbours index
index = np.memmap(KNN_INDEX_PATH, dtype='float32', mode='r+', shape=(len(df), len(df)))

class Review(BaseModel):
    id: int
    title: Union[str, None]
    author: Union[str, None]
    forum: Union[str, None]
    critic: Union[str, None]
    genre: Union[str, None]
    gender_of_author: Union[str, None]
    gender_of_critic: Union[str, None]
    year: Union[int, None]
    same_year: Union[str, None]
    tokens: Union[List[str], None]
    x: Union[float, None]
    y: Union[float,None]

    class Config:
        orm_mode = True


middleware = [Middleware(CORSMiddleware, 
                        allow_origins=['*'], 
                        allow_credentials=True, 
                        allow_methods=['*'], 
                        allow_headers=['*'])]

app = FastAPI(middleware=middleware)

@app.get("/api/reviews/", response_model=List[Review])
async def get_reviews():
    
    return list(df.to_dict('records'))
    
@app.get("/api/reviews/{review_id}", response_model=Review)
async def get_review(review_id: int):
    return df.iloc[review_id]

@app.get("/api/reviews/nearest/", response_model=List[Review])
async def get_nearest_neighbors(review_id: int, nn: int = 10):
    review_idxs = index[review_id][:nn]

    return list(df.iloc[review_idxs].to_dict('records'))