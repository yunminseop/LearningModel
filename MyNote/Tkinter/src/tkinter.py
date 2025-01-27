import numpy as np
import tensorflow as tf
import tkinter as tk
from PIL import Image, ImageTk
from gtts import gTTS
import playsound
import os

cnn = tf.keras.models.load_model("my_cnn_for_deploy.h5")

class_names_en = ['airplain', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

class_id = 0
tk_img=''

def process_image():
    global class_id, tk_img

    fname = tk.filedialog.askopenfilename()
    img = Image.open(fname)
    tk_img=img.resize([128,128])
    tk_img=ImageTk.PhotoImage(tk_img)
    
    canvas.create_image((canvas.winfo_width()/2, canvas.winfo_height()/2), image=tk_img, anchor="center")

    x_test = []
    x = np.asarray(img.resize([32,32]))/255.0
    x_test.append(x)
    x_test = np.asarray(x_test)
    res = cnn.predict(x_test)
    class_id = np.argmax(res)
    label_en['text'] = '영어: '+ class_names_en[class_id]
    
    def tts_english():
        tts = gTTS(text=class_names_en[class_id], lang="en")
        if os.path.isfile("word.mp3"): os.remove("word.mp3")
        tts.save("word.mp3")
        playsound.playsound("word.mp3", True)

    
    def quit_program():
        win.destroy()

    win = tk.Tk()
    win.title('En study')
    win.geometry("512, 500")

    process_button=tk.Button(win, text="영상 선택", command=process_image)
    quit_button = tk.Button(win, text="끝내기", command=quit_program)
    canvas = tk.Canvas(win, width=256, height=256, bg="cyan", bd=4)

    label_en = tk.Label(win,width=16, height=1, bg="yellow", bd=4, text="영어", anchor='w')
    tts_en = tk.Button(win, text="듣기", command=tts_english)

    process_button.grid(row=0, column=0)
    quit_button.grid(row=1, column=0)
    canvas.grid(row=0, column=1)
    label_en.grid(row=1, column=1, sticky='e')
    tts_en.grid(ros=1, column=2, sticky='w')

    win.mainloop()