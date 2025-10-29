import tkinter as tk
from PIL import ImageTk, Image
from tkinter import filedialog


app = tk.Tk()
imageLabel = tk.Label(app)
startImage = "Pneumonia_Image.jpg"
imageLocation = ""

def createWindow():
    global app
    # Create the main application window
    app.title("AutoVision")
    app.geometry("400x400") # Set window size

    # Add a label
    label = tk.Label(app, text="AutoVision")
    label.pack()

    button = tk.Button(app, text="Select an image", command=on_click)
    button.pack()

    button2 = tk.Button(app, text="Quit", command=app.destroy)
    button2.pack()       

    display_image(startImage)
    imageLabel.pack()

    # Run the application
    app.mainloop()


def on_click():
    global imageLocation
    imgLocation = filedialog.askopenfilename(initialdir="/", title="Select Image", filetypes=(("png files","*.png"), ("jpg files", "*.jpg"), ("All files", "*.*")))
    display_image(imgLocation)
    imageLocation = imgLocation
    enable_send_picture_button()
    
    #img = ImageTk.PhotoImage(Image.open(imgLocation).resize((250, 250), Image.LANCZOS))                  
    #imgLabel = tk.Label(text= imgLocation, image = img)
    #imgLabel.pack()

def display_image(location : str):
    global imageLabel
    img = ImageTk.PhotoImage(Image.open(location).resize((250, 250), Image.LANCZOS))   
    imageLabel.config(image = img)
    imageLabel.pack()

buttonEnabled = False
def enable_send_picture_button():
    global app, buttonEnabled
    if(buttonEnabled == False):
        buttonEnabled = True
        button = tk.Button(app, text="Test Image", command=send_image)
        button.pack()

def send_image():
    global imageLocation
    print(imageLocation)
    


def main():
    createWindow()

if __name__ == "__main__":
    main()