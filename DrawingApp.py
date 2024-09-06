import tkinter as tk
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageGrab

model = load_model("mnist_model.keras")

def paint(event):
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    canvas.create_oval(x1, y1, x2, y2, fill="white", outline="white", width=40)
    
def clear_canvas():
    canvas.delete("all")
    
def predict_digit():
    #Get image
    canvas.update()
    x = root.winfo_rootx() + canvas.winfo_x()
    y = root.winfo_rooty() + canvas.winfo_y()
    x1 = x + canvas.winfo_width()
    y1 = y + canvas.winfo_height()
    image = ImageGrab.grab().crop((x, y, x1, y1))

    #Grayscale and resize
    image = image.convert('L')
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    #Turn into array and normalize
    image = np.array(image)
    image = image.astype('float32') / 255
    #Add batch size and channels
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)
    
    predictions = model.predict(image)[0]
    predictions = {i: prob for i, prob in enumerate(predictions)}
    #Sort predictions by greatest probability
    predictions = {i: prob for i, prob in 
                          sorted(predictions.items(), key=lambda item: item[1], reverse=True)
                          }
    #Turn to strings
    predictions = [f"{i}: {prob*100:.2f}%" for i, prob in predictions.items()]
    
    result_label.config(text="Predicted Probabilities:\n" + "\n".join(predictions))

root = tk.Tk()
root.title("MNIST Predictor")


canvas = tk.Canvas(root, width=400, height=400, bg="black", cursor="cross")
canvas.pack()
canvas.bind("<B1-Motion>", paint)

clear_button = tk.Button(root, text="Clear", command=clear_canvas)
clear_button.pack()

predict_button = tk.Button(root, text="Predict", command=predict_digit)
predict_button.pack()

result_label = tk.Label(root, text="Predictions: ")
result_label.pack()


tk.mainloop()