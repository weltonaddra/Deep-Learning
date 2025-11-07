from tkinter import *
from tkinter import ttk
from PIL import ImageTk, Image
from tkinter import filedialog

class AutoVisionGUI(Tk):
    def __init__(self):
        Tk.__init__(self)
        self.title("AutoVision")
        self.geometry("400x400")

        menubar = Menu(self)
        self.config(menu=menubar)
        fileMenu = Menu(menubar,tearoff=0)  
        menubar.add_cascade(label="File", menu=fileMenu)
        fileMenu.add_command(label="Exit", command=quit)


        notebook = ttk.Notebook(self)
        tab1 = TestFrame(notebook)
        tab2 = TrainFrame(notebook)
        notebook.add(tab1, text= "Test")
        notebook.add(tab2, text= "Train")
        notebook.pack(expand=True, fill="both")



# Menu displayed for testing a single image
class TestFrame(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.startImage = "Pneumonia_Image.jpg"
        self.imageLocation = "" + self.startImage
        self.image = Label(self)

        name = Label(self, text="AutoVision", font=16, padx=10).pack()
        self.select_image_button = Button(self, text="Select an image", command = self.change_image).pack()
        self.quit_button = Button(self, text="Quit", command=parent.quit).pack()
        self.display_image()
        self.image.pack()
        self.select_button = Button(self, text="Test Image", command=self.select_image, state=DISABLED)
        self.select_button.pack()
    

    def display_image(self):
        img = ImageTk.PhotoImage(Image.open(self.imageLocation).resize((250, 250), Image.LANCZOS))   
        self.image.config(image = img)

    def change_image(self):
        self.imageLocation = filedialog.askopenfilename(initialdir="/", title="Select Image", filetypes=(("png files","*.png"), ("jpg files", "*.jpg"), ("All files", "*.*")))
        self.check_select_button()
        self.display_image()

    def check_select_button(self):
        print(self.imageLocation)
        if(self.imageLocation == ""):
            self.select_button.config(state=DISABLED)
            self.imageLocation = "" + self.startImage
        else:
            self.select_button.config(state=NORMAL)
    

    # Update this function to connect it to the AI
    def select_image(self):
        print(self.imageLocation)



# Menu displayed for training the AI
# Directory and folder are used interchangably. Folder is displayed on the user side. Directory is code side.
class TrainFrame(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.image = Label(self)
        self.nothingSelected = "No directory currently selected"
        self.trainDirectory = StringVar()
        self.trainDirectory.set(self.nothingSelected)

        Label(self, text="AutoVision", font=16, padx=10).pack()
        Label(self, text="AI Training").pack()
        self.select_image_button = Button(self, text="Select an folder to train from", command = self.select_directory).pack()
        self.quit_button = Button(self, text="Quit", command=parent.quit).pack()
        self.display_image()
        self.image.pack()
        self.train_button = Button(self, text="Train using selected folder", command=self.train_using_directory(), state=DISABLED)
        self.train_button.pack()
        self.choose_directory_label = Label(self, textvariable=self.trainDirectory, anchor="sw").pack(side=BOTTOM, anchor="sw")


    def display_image(self):
        img = ImageTk.PhotoImage(Image.open("Pneumonia_Image.jpg").resize((225, 225), Image.LANCZOS))   
        self.image.config(image = img)

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


    # Update this function to connect it to the AI
    def train_using_directory(self):
        print(self.trainDirectory)
 

    


if __name__ == "__main__":
    app = AutoVisionGUI()
    app.mainloop()