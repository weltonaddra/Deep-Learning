from tkinter import *
from tkinter import ttk
from PIL import ImageTk, Image
from tkinter import filedialog
import os
from tkinter import StringVar
import threading
import torch

# from traitlets import This  # unused
from dataset import ChestXRayDataset
from trainer import ModelTrainer
from config import Config

# Model training
dataset = ChestXRayDataset(subset_fraction=1)
trainer = ModelTrainer(dataset)
 # This is the name of the file where the model state dict will be saved
savedPath = "AutovisionVer1.pt"
trained_model = None
if os.path.exists(savedPath):
    trainer.load_checkpoint(savedPath)
    trained_model = trainer.model



class AutoVisionGUI(Tk):
    def __init__(self):
        Tk.__init__(self)
        self.title("AutoVision")
        self.geometry("500x500")
        self.trained_model = trained_model

        menubar = Menu(self)
        self.config(menu=menubar)
        fileMenu = Menu(menubar,tearoff=0)  
        menubar.add_cascade(label="File", menu=fileMenu)
        fileMenu.add_command(label="Exit", command=quit)

        notebook = ttk.Notebook(self)
        tab1 = TrainFrame(notebook) 
        tab2 = TestFrame(notebook)
        notebook.add(tab1, text= "Train")
        notebook.add(tab2, text= "Test")
        notebook.pack(expand=True, fill="both")

        

        

# Menu displayed for testing a single image
class TestFrame(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.startImage = os.path.join("chest_xray", "selected", "NORMAL", "Pneumonia_Image.jpg")
        self.imageLocation = "" + self.startImage
        self.image = Label(self)

        name = Label(self, text="AutoVision", font=16, padx=10).pack()
        self.select_image_button = Button(self, text="Select an image", command = self.change_image)
        self.select_image_button.pack(pady=5)
        self.select_button = Button(self, text="Test Image", command=self.select_image, state=DISABLED, width=15, font=('Arial', 11))
        self.select_button.pack(pady=5)
        self.display_image()
        self.image.pack(pady=10)
        self.result_label = Label(self, text="Prediction: ", font=('Arial', 12))
        self.result_label.pack(pady=5)
        self.quit_button = Button(self, text="Quit", command=parent.quit)
        self.quit_button.pack(pady=5)
    
# Displays the image file stored in the img var 
    def display_image(self):
        self.photo_image = ImageTk.PhotoImage(Image.open(self.imageLocation).resize((250, 250), Image.LANCZOS))
        self.image.config(image = self.photo_image)

# Opens the file select menu 
    def change_image(self):
        self.imageLocation = filedialog.askopenfilename(initialdir="/", title="Select Image", filetypes=(("All files", "*.*"),("png files","*.png"), ("jpeg files", "*.jpeg"), ("jpg files", "*.jpg") ))
        self.check_select_button()
        self.display_image()

# If there is no image file selected it disables the test button 
# Else if there is an image location, it sets the button as on 
    def check_select_button(self):
        print(self.imageLocation)
        if(self.imageLocation == "" or trained_model is None or self.imageLocation == self.startImage):
            self.select_button.config(state=DISABLED)
        else:
            self.select_button.config(state=NORMAL)
        if(self.imageLocation == ""):
            self.imageLocation = "" + self.startImage
        

    def select_image(self):
        if trained_model is None:
            self.result_label.config(text="No trained model available")
            return
        img = Image.open(self.imageLocation)
        transformed_img = dataset.transforms['test'](img).unsqueeze(0).to(dataset.device)
        trained_model.eval()
        with torch.no_grad():
            outputs = trained_model(transformed_img)
            _, preds = torch.max(outputs, 1)
            pred = preds.item()
            class_name = dataset.class_names[pred]
            self.result_label.config(text=f"Prediction: {class_name}")
            
class TrainFrame(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.image = Label(self)
        self.nothingSelected = "No directory currently selected"
        self.trainDirectory = Config.DATA_DIR
        # self.trainDirectory.set(self.nothingSelected)
        self.imgPath = os.path.join("chest_xray", "selected", "NORMAL", "Pneumonia_Image.jpg")

        self.datasetSize = StringVar()
        self.trainState = StringVar()
        self.datasetSize.set(f"Train:  {dataset.dataset_sizes['train']}, Val: {dataset.dataset_sizes['val']}, Test: {dataset.dataset_sizes['test']}")
        self.trainState.set("AI Training")

        self.animating = False
        self.animation_step = 0
        self.animation_steps = ["AI Training", "AI Training.", "AI Training..", "AI Training..."]

        Label(self, text="AutoVision", font=16, padx=10).pack()
        self.label_training = Label(self, textvariable=self.trainState, font=1)
        self.label_training.pack()
        Label(self, text="  ").pack() #Empty space
        Label(self, text="Image Sizes", font=10).pack()
        Label(self, textvariable= self.datasetSize, font=1).pack(pady=(0,20)) #Dataset size


        self.train_button = Button(self, text="Train Model", command=self.train_using_directory)
        self.train_button.pack(pady=5, anchor='center')

        self.quit_button = Button(self, text="Quit", command=parent.quit)
        self.quit_button.pack(pady=5, anchor='center')

        self.image.pack(pady=10)
        self.display_image(self.imgPath)

        self.choose_directory_label = Label(self, textvariable=self.trainDirectory, anchor="w")
        self.choose_directory_label.pack(side=BOTTOM, fill=X, pady=10)


    def display_image(self, imgPath=os.path.join("Pneumonia_Image.jpg")):
        self.photo_image = ImageTk.PhotoImage(Image.open(self.imgPath).resize((225, 225), Image.LANCZOS))
        self.image.config(image = self.photo_image)

    def select_directory(self):
        self.trainDirectory.set(filedialog.askdirectory(title="Select a folder to train from"))
        self.update_choose_directory_label()
        self.check_train_button()
        self.display_image()
        self.check_train_button()


    def check_train_button(self):
        if(self.trainDirectory.get() ==  self.nothingSelected):
            self.train_button.config(state=DISABLED)
        else:
            self.train_button.config(state=NORMAL)

    def update_choose_directory_label(self):
        if(self.trainDirectory.get() == ""):
            self.trainDirectory.set( "" + self.nothingSelected)

    def start_animation(self):
        if not self.animating:
            self.animating = True
            self.animate_loading()

    def animate_loading(self):
        if self.animating:
            self.trainState.set(self.animation_steps[self.animation_step])
            self.animation_step = (self.animation_step + 1) % len(self.animation_steps)
            self.after(500, self.animate_loading)

    def stop_animation(self):
        self.animating = False
        self.trainState.set("AI Training")

    # Update this function to connect it to the AI
    def train_using_directory(self):
        global trained_model
        self.train_button.config(state=DISABLED)
        self.start_animation()
        def do_training():
            print(self.trainDirectory)
            trained_model = trainer.train_model()
            self.master.trained_model = trained_model
            self.after(0, self.stop_animation)
            self.after(0, lambda: self.train_button.config(state=NORMAL))
        thread = threading.Thread(target=do_training)
        thread.start()

 



if __name__ == "__main__":
 
    app = AutoVisionGUI()
    app.mainloop()
