from fastapi import FastAPI
from pydantic import BaseModel

# creating app
app = FastAPI()

# decerator using get one read the data

@app.get("/")
def home():
    return "welcome home puppy ma, welcome home."



# 1. POST with JSON (create a new user)
class User(BaseModel):
    name: str
    email: str
    phn: int
    age: int
    weight:float

@app.post("/users/")
def create_user(user:User):
    return {"status": "User created", "user": user}
