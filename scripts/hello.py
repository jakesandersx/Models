from ultralytics import YOLO
import torch

model = YOLO('yolo11s.pt')

PATH = "C:/Users/MSJAIUser/PycharmProjects/Senior Research Project/cicada_yolo/data.yaml"

model.train(data=PATH, epochs=25)