# gui/components/person_management.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from app.db.models import FaceFeature, FaceImage, Person
import cv2
import numpy as np
from typing import List, Optional
from datetime import datetime
from app.db.crud import (
    create_person, add_face_feature, add_face_image,
    get_all_persons, get_person_by_id, delete_person
)
from app.db.database import get_db
from scripts.add_persons_db import get_feature


import os
import io

class PersonManagementView(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.current_image = None
        self.current_person = None
        self.setup_ui()
        self.load_persons()

    def setup_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        
        # Main container with notebook
        self.notebook = ttk.Notebook(self)
        self.notebook.grid(row=0, column=0, sticky="nsew")
        
        # Person List Tab
        self.list_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.list_tab, text="Person List")
        self.setup_list_tab()
        
        # Add/Edit Person Tab
        self.edit_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.edit_tab, text="Add/Edit Person")
        self.setup_edit_tab()

    def setup_list_tab(self):
        self.list_tab.columnconfigure(0, weight=1)
        self.list_tab.rowconfigure(1, weight=1)
        
        # Toolbar
        toolbar = ttk.Frame(self.list_tab)
        toolbar.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        ttk.Button(
            toolbar,
            text="Add New",
            command=lambda: self.notebook.select(1)
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            toolbar,
            text="Refresh",
            command=self.load_persons
        ).pack(side=tk.LEFT, padx=5)
        
        # Person Treeview
        self.person_tree = ttk.Treeview(
            self.list_tab,
            columns=("id", "name", "employee_id", "department", "faces"),
            show="headings"
        )
        self.person_tree.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        # Configure columns
        self.person_tree.heading("id", text="ID")
        self.person_tree.heading("name", text="Name")
        self.person_tree.heading("employee_id", text="Employee ID")
        self.person_tree.heading("department", text="Department")
        self.person_tree.heading("faces", text="Face Count")
        
        self.person_tree.column("id", width=50, anchor=tk.CENTER)
        self.person_tree.column("name", width=150)
        self.person_tree.column("employee_id", width=100)
        self.person_tree.column("department", width=100)
        self.person_tree.column("faces", width=80, anchor=tk.CENTER)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(
            self.list_tab,
            orient=tk.VERTICAL,
            command=self.person_tree.yview
        )
        self.person_tree.configure(yscroll=scrollbar.set)
        scrollbar.grid(row=1, column=1, sticky="ns")
        
        # Context menu
        self.tree_menu = tk.Menu(self, tearoff=0)
        self.tree_menu.add_command(
            label="View Details",
            command=self.view_person_details
        )
        self.tree_menu.add_command(
            label="Delete",
            command=self.delete_selected_person
        )
        # Add Edit option to context menu
        self.tree_menu.add_command(
            label="Edit",
            command=self.edit_selected_person
        )
        self.person_tree.bind("<Button-3>", self.show_tree_menu)

    def setup_edit_tab(self):
        self.edit_tab.columnconfigure(0, weight=1)
        self.edit_tab.rowconfigure(1, weight=1)
        
        # Form frame
        form_frame = ttk.Frame(self.edit_tab)
        form_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        
        # Person details
        ttk.Label(form_frame, text="Name:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        self.name_var = tk.StringVar()
        ttk.Entry(form_frame, textvariable=self.name_var).grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        
        ttk.Label(form_frame, text="Employee ID:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
        self.emp_id_var = tk.StringVar()
        ttk.Entry(form_frame, textvariable=self.emp_id_var).grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        
        ttk.Label(form_frame, text="Department:").grid(row=2, column=0, sticky="e", padx=5, pady=5)
        self.dept_var = tk.StringVar()
        ttk.Entry(form_frame, textvariable=self.dept_var).grid(row=2, column=1, sticky="ew", padx=5, pady=5)
        
        ttk.Label(form_frame, text="Email:").grid(row=3, column=0, sticky="e", padx=5, pady=5)
        self.email_var = tk.StringVar()
        ttk.Entry(form_frame, textvariable=self.email_var).grid(row=3, column=1, sticky="ew", padx=5, pady=5)
        
        # Image frame
        img_frame = ttk.LabelFrame(self.edit_tab, text="Face Image")
        img_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        img_frame.columnconfigure(0, weight=1)
        img_frame.rowconfigure(0, weight=1)
        
        self.image_label = ttk.Label(img_frame)
        self.image_label.grid(row=0, column=0, sticky="nsew")
        
        # Buttons
        btn_frame = ttk.Frame(self.edit_tab)
        btn_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
        
        ttk.Button(
            btn_frame,
            text="Load Image",
            command=self.load_image
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            btn_frame,
            text="Save",
            command=self.save_person
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            btn_frame,
            text="Cancel",
            command=lambda: self.notebook.select(0)
        ).pack(side=tk.RIGHT, padx=5)

    def load_persons(self, db=None):
        """Load all persons from database"""
        close_db = False
        if db is None:
            db = next(get_db())
            close_db = True
        
        try:
            persons = get_all_persons(db)
            
            # Clear existing data
            for item in self.person_tree.get_children():
                self.person_tree.delete(item)
            
            # Add persons to treeview
            for person in persons:
                self.person_tree.insert("", tk.END, values=(
                    person.id,
                    person.name,
                    person.employee_id,
                    person.department,
                    len(person.face_features)
                ))
        finally:
            if close_db:
                db.close()

    def load_image(self):
        """Load image from file"""
        file_path = filedialog.askopenfilename(
            title="Select Face Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        
        if file_path:
            try:
                image = cv2.imread(file_path)
                if image is None:
                    raise ValueError("Failed to load image")
                    
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Store the original image
                self.current_image = image
                
                # Display thumbnail
                img = Image.fromarray(image)
                img.thumbnail((300, 300))
                self.imgtk = ImageTk.PhotoImage(image=img)
                self.image_label.config(image=self.imgtk)
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
                self.current_image = None
                self.image_label.config(image=None)

    def save_person(self):
        """Save person data to database - handles both new and existing persons"""
        name = self.name_var.get().strip()
        if not name:
            messagebox.showerror("Error", "Name is required")
            return
        
        db = None
        try:
            db = next(get_db())
            
            if self.current_person:
                # Re-load the person in the current session
                current_person = db.query(Person).get(self.current_person.id)
                if not current_person:
                    messagebox.showerror("Error", "Person not found in database")
                    return
                    
                # Update existing person
                current_person.name = name
                current_person.email = self.email_var.get().strip() or None
                current_person.department = self.dept_var.get().strip() or None
                current_person.employee_id = self.emp_id_var.get().strip() or None
                
                # Only update face features if new image was loaded
                if self.current_image is not None and self.current_image.size > 0:
                    # Delete old features and images
                    db.query(FaceFeature).filter(FaceFeature.person_id == current_person.id).delete()
                    db.query(FaceImage).filter(FaceImage.person_id == current_person.id).delete()
                    
                    # Add new features and images
                    face_emb = get_feature(self.current_image)
                    add_face_feature(db, current_person.id, face_emb)
                    
                    # Save new image
                    os.makedirs("datasets/face_images", exist_ok=True)
                    image_path = f"datasets/face_images/{current_person.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(image_path, cv2.cvtColor(self.current_image, cv2.COLOR_RGB2BGR))
                    
                    # Save thumbnail
                    thumbnail = cv2.resize(self.current_image, (100, 100))
                    add_face_image(db, current_person.id, image_path, thumbnail)
                
                db.commit()
                messagebox.showinfo("Success", "Person updated successfully")
            else:
                # Check if image is provided for new person
                if self.current_image is None or self.current_image.size == 0:
                    messagebox.showerror("Error", "Please load a face image for new person")
                    return
                    
                # Create new person
                person = create_person(
                    db=db,
                    name=name,
                    email=self.email_var.get().strip() or None,
                    department=self.dept_var.get().strip() or None,
                    employee_id=self.emp_id_var.get().strip() or None
                )
                
                # Process face image
                face_emb = get_feature(self.current_image)
                
                # Save face feature
                add_face_feature(db, person.id, face_emb)
                
                # Save image (store path only)
                os.makedirs("datasets/face_images", exist_ok=True)
                image_path = f"datasets/face_images/{person.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(image_path, cv2.cvtColor(self.current_image, cv2.COLOR_RGB2BGR))
                
                # Save thumbnail in database
                thumbnail = cv2.resize(self.current_image, (100, 100))
                add_face_image(db, person.id, image_path, thumbnail)
                
                db.commit()
                messagebox.showinfo("Success", "Person added successfully")
            
            # Clear form and refresh
            self.clear_form()
            self.current_person = None
            self.notebook.select(0)
            self.load_persons()  # Now this works without passing db
                
        except Exception as e:
            if db:
                db.rollback()
            messagebox.showerror("Error", f"Failed to save person: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            if db:
                db.close()

    def clear_form(self):
        """Clear the form fields"""
        self.name_var.set("")
        self.emp_id_var.set("")
        self.dept_var.set("")
        self.email_var.set("")
        self.current_image = None
        self.image_label.config(image=None)

    def view_person_details(self):
        """View details of selected person"""
        selected = self.person_tree.focus()
        if not selected:
            return
        
        person_id = self.person_tree.item(selected)['values'][0]
        db = next(get_db())
        person = get_person_by_id(db, person_id)
        
        if person:
            # Create details window
            details_win = tk.Toplevel(self)
            details_win.title(f"Person Details - {person.name}")
            
            # Display person info
            ttk.Label(details_win, text=f"Name: {person.name}").pack(padx=10, pady=5, anchor="w")
            ttk.Label(details_win, text=f"Employee ID: {person.employee_id or 'N/A'}").pack(padx=10, pady=5, anchor="w")
            ttk.Label(details_win, text=f"Department: {person.department or 'N/A'}").pack(padx=10, pady=5, anchor="w")
            
            # Display face images if available
            if person.face_images:
                img_frame = ttk.LabelFrame(details_win, text="Face Images")
                img_frame.pack(padx=10, pady=10, fill="both", expand=True)
                
                for img in person.face_images:
                    try:
                        # Create thumbnail from binary data
                        thumbnail = Image.frombytes(
                            'RGB', 
                            (100, 100), 
                            img.thumbnail
                        )
                        imgtk = ImageTk.PhotoImage(image=thumbnail)
                        
                        label = ttk.Label(img_frame, image=imgtk)
                        label.image = imgtk  # Keep reference
                        label.pack(side=tk.LEFT, padx=5, pady=5)
                    except:
                        continue

    def edit_selected_person(self):
        """Edit selected person"""
        selected = self.person_tree.focus()
        if not selected:
            return
        
        person_id = self.person_tree.item(selected)['values'][0]
        db = next(get_db())
        try:
            person = db.query(Person).get(person_id)
            if person:
                # Switch to edit tab
                self.notebook.select(1)
                
                # Populate form fields
                self.name_var.set(person.name)
                self.emp_id_var.set(person.employee_id or "")
                self.dept_var.set(person.department or "")
                self.email_var.set(person.email or "")
                
                # Store just the ID, not the entire object
                self.current_person = person
                
                # Load first face image if available
                if person.face_images:
                    try:
                        image_path = person.face_images[0].image_path
                        image = cv2.imread(image_path)
                        if image is not None:
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            self.current_image = image
                            
                            # Display thumbnail
                            img = Image.fromarray(image)
                            img.thumbnail((300, 300))
                            self.imgtk = ImageTk.PhotoImage(image=img)
                            self.image_label.config(image=self.imgtk)
                    except Exception as e:
                        print(f"Error loading image: {e}")
        finally:
            db.close()

    def delete_selected_person(self):
        """Delete selected person"""
        selected = self.person_tree.focus()
        if not selected:
            return
        
        person_id = self.person_tree.item(selected)['values'][0]
        person_name = self.person_tree.item(selected)['values'][1]
        
        if messagebox.askyesno(
            "Confirm Delete",
            f"Are you sure you want to delete {person_name}?"
        ):
            db = next(get_db())
            if delete_person(db, person_id):
                messagebox.showinfo("Success", "Person deleted successfully")
                self.load_persons()
            else:
                messagebox.showerror("Error", "Failed to delete person")

    def show_tree_menu(self, event):
        """Show context menu for treeview"""
        item = self.person_tree.identify_row(event.y)
        if item:
            self.person_tree.selection_set(item)
            self.tree_menu.tk_popup(event.x_root, event.y_root)