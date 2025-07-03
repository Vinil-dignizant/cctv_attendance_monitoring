import tkinter as tk
from tkinter import ttk

def configure_styles():
    style = ttk.Style()
    
    # Main window style
    style.configure('TFrame', background='#f0f0f0')
    style.configure('TLabel', background='#f0f0f0', font=('Helvetica', 10))
    style.configure('TButton', font=('Helvetica', 10), padding=5)
    style.configure('TNotebook', background='#f0f0f0')
    style.configure('TNotebook.Tab', font=('Helvetica', 10, 'bold'), padding=[10, 5])
    
    # Treeview style
    style.configure('Treeview', 
                   font=('Helvetica', 9), 
                   rowheight=25,
                   fieldbackground='#ffffff')
    style.configure('Treeview.Heading', 
                   font=('Helvetica', 10, 'bold'),
                   background='#e1e1e1')
    style.map('Treeview', background=[('selected', '#0078d7')])
    
    # Status colors
    style.configure('Status.TLabel', font=('Helvetica', 10))
    style.configure('Status.Running.TLabel', foreground='green')
    style.configure('Status.Stopped.TLabel', foreground='red')
    style.configure('Status.Error.TLabel', foreground='orange')