from fastapi import FastAPI, Request, Form, Response
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

import main

from autocorrect import Speller
spell = Speller(lang='ru')
val = []

app = FastAPI()
templates = Jinja2Templates(directory="templates/")


def find(target):
    return main.get_products(target)


def correct(target):
    return spell(target)


@app.get("/")
async def form_post(request: Request):
    return templates.TemplateResponse('test3.html', context={'request': request})


@app.post("/")
async def form_post(request: Request, target: str = Form("")):
    if target == "":
        right = str("")
        return templates.TemplateResponse('test3.html', context={'request': request, 'right': right})

    right = correct(target)
    val.append(right)
    result = find(right)
    return templates.TemplateResponse('test2.html', context={'request': request, 'result': result, 'right': right})

@app.get("/history")
async def write_history(request: Request):
    return templates.TemplateResponse('search_history.html', context={'request': request, 'val': val})