
from fastapi import FastAPI
from pydantic import BaseModel

class User_input(BaseModel):
    nc : int
    #nc : float
    
    
app = FastAPI()

@app.post("/noclient")
def operate(input:User_input):
    
    
    
    #result = input.nc*2  
    result = ((input.nc * 137 + 187) % 1000)
    return result