from typing import *
from fastapi import FastAPI, HTTPException
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import requests
from pydantic import BaseModel
import numpy as np
import json


def initialize_database():

    # Load the configuration
    with open("./configs/urls.json", "r+") as f:
        config = json.load(f)

    dfs = []
    shapes = {}

    # Iterate the materials
    for c in config["data"]:

        url = c["url"]

        # Get the review data
        response = requests.get(url)
        records = response.json()

        df = pd.json_normalize(records)
        df = df.rename(
            {
                "gender of author": "gender_of_author",
                "gender of critic": "gender_of_critic",
                "same year": "same_year",
            },
            axis=1,
        )

        df = df.where(pd.notnull(df), None)

        # Save shapes for each dataframe, for the index
        shapes.update({c["name"]: (len(df), len(df))})

        dfs.append(df)

    # Combine into single dataframe
    df = pd.concat(dfs, keys=[c["name"] for c in config["data"]])

    # Process by adding lengths of each review
    df["length"] = df.tokenized.apply(lambda x: len(x))

    index = {
        # Get the nearest neighbours index
        c["name"]: np.memmap(
            c["index"], dtype="int64", mode="r+", shape=shapes[c["name"]]
        )
        for c in config["data"]
    }

    return df, index


# Get the nearest neighbours index
# index = np.memmap(config["KNN_INDEX_PATH"], dtype='int64', mode='r+', shape=(len(df), len(df)))

####################################################################
df, index = initialize_database()
####################################################################


class Review(BaseModel):
    id: Union[int, None]
    title: Union[str, None]
    media: Union[str, None]
    author: Union[str, None]
    forum: Union[str, None]
    critic: Union[str, None]
    genre: Union[str, None]
    gender_of_author: Union[str, None]
    gender_of_critic: Union[str, None]
    year: Union[int, None]
    same_year: Union[str, None]
    tokenized: Union[List[str], None]
    counts: Union[List[Dict[str, Union[int, str]]], None]
    tfidf: Union[List[Dict[str, Union[float, str]]], None]
    x: Union[float, None]
    y: Union[float, None]
    x_sentence: Union[float, None]
    y_sentence: Union[float, None]
    length: Union[int, None]


class MultipleReview(BaseModel):
    id: Union[int, None]
    title: Union[str, None]
    media: Union[str, None]
    author: Union[str, None]
    forum: Union[str, None]
    critic: Union[str, None]
    genre: Union[str, None]
    gender_of_author: Union[str, None]
    gender_of_critic: Union[str, None]
    year: Union[int, None]
    same_year: Union[str, None]
    x: Union[float, None]
    y: Union[float, None]
    x_sentence: Union[float, None]
    y_sentence: Union[float, None]
    title_multiple: Union[str, None]
    author_multiple: Union[str, None]
    critic_multiple: Union[str, None]
    length: Union[int, None]

    class Config:
        # orm_mode = True
        from_attributes = True


middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["https://dh.gu.se", "https://kno.dh.gu.se"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
]

app = FastAPI(middleware=middleware)


@app.get("/api/reviews/{material}/", response_model=List[MultipleReview])
async def get_reviews(material: str, title: Optional[str] = None):

    if title:
        results = df.loc[material][df.title == title]
    else:
        results = df.loc[material].drop(["tokenized", "tfidf", "counts"], axis=1)

    return list(results.to_dict("records"))


@app.get("/api/reviews/{material}/{review_id}", response_model=Review)
async def get_review(material: str, review_id: int):
    print(material, review_id)

    return (
        df.loc[material]
        .iloc[review_id]
        .drop(["title_multiple", "author_multiple", "critic_multiple"])
    )


@app.get("/api/reviews/{material}/nearest/", response_model=List[MultipleReview])
async def get_nearest_neighbors(material: str, review_id: int, nn: int = 10):

    review_idxs = index[material][review_id, :nn]

    return list(
        df.loc[material]
        .iloc[review_idxs]
        .drop(["title_multiple", "author_multiple", "critic_multiple"], axis=1)
        .to_dict("records")
    )
