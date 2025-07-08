# gui/components/camera_management.py
import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Dict, Optional
from app.db.crud import (
    create_camera, get_all_cameras, get_camera, 
    update_camera, delete_camera
)
from app.db.database import get_db

class CameraManagementView(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.setup_ui()
        self.load_cameras()

    def setup_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        
        # Toolbar
        toolbar = ttk.Frame(self)
        toolbar.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        ttk.Button(
            toolbar,
            text="Add Camera",
            command=self.show_add_dialog
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            toolbar,
            text="Refresh",
            command=self.load_cameras
        ).pack(side=tk.LEFT, padx=5)
        
        # Treeview
        self.tree = ttk.Treeview(
            self,
            columns=("id", "name", "location", "url", "type", "enabled"),
            show="headings"
        )
        self.tree.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        # Configure columns
        self.tree.heading("id", text="Camera ID")
        self.tree.heading("name", text="Name")
        self.tree.heading("location", text="Location")
        self.tree.heading("url", text="URL")
        self.tree.heading("type", text="Event Type")
        self.tree.heading("enabled", text="Enabled")
        
        # Context menu
        self.context_menu = tk.Menu(self, tearoff=0)
        self.context_menu.add_command(
            label="Edit",
            command=self.edit_selected
        )
        self.context_menu.add_command(
            label="Delete",
            command=self.delete_selected
        )
        self.tree.bind("<Button-3>", self.show_context_menu)

    def load_cameras(self):
        """Load cameras from database"""
        db = next(get_db())
        cameras = get_all_cameras(db)
        
        # Clear existing data
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Add cameras to treeview
        for cam in cameras:
            self.tree.insert("", tk.END, values=(
                cam.camera_id,
                cam.camera_name,
                cam.location,
                cam.url,
                cam.event_type,
                "Yes" if cam.is_enabled else "No"
            ))

    def show_add_dialog(self):
        """Show dialog to add new camera"""
        dialog = tk.Toplevel(self)
        dialog.title("Add New Camera")
        
        # Form fields
        ttk.Label(dialog, text="Camera ID:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        camera_id = ttk.Entry(dialog)
        camera_id.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        ttk.Label(dialog, text="Name:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        name = ttk.Entry(dialog)
        name.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        
        ttk.Label(dialog, text="Location:").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        location = ttk.Entry(dialog)
        location.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        
        ttk.Label(dialog, text="URL:").grid(row=3, column=0, padx=5, pady=5, sticky="e")
        url = ttk.Entry(dialog)
        url.grid(row=3, column=1, padx=5, pady=5, sticky="ew")
        
        ttk.Label(dialog, text="Event Type:").grid(row=4, column=0, padx=5, pady=5, sticky="e")
        event_type = ttk.Combobox(dialog, values=["login", "logout"])
        event_type.grid(row=4, column=1, padx=5, pady=5, sticky="ew")
        event_type.current(0)
        
        enabled = tk.BooleanVar(value=True)
        ttk.Checkbutton(dialog, text="Enabled", variable=enabled).grid(row=5, column=1, padx=5, pady=5, sticky="w")
        
        # Buttons
        btn_frame = ttk.Frame(dialog)
        btn_frame.grid(row=6, column=0, columnspan=2, pady=10)
        
        ttk.Button(
            btn_frame,
            text="Save",
            command=lambda: self.save_camera(
                dialog, camera_id.get(), name.get(),
                location.get(), url.get(),
                event_type.get(), enabled.get()
            )
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            btn_frame,
            text="Cancel",
            command=dialog.destroy
        ).pack(side=tk.LEFT, padx=5)

    def save_camera(self, dialog, camera_id, name, location, url, event_type, enabled):
        """Save new camera to database"""
        if not camera_id:
            messagebox.showerror("Error", "Camera ID is required")
            return
            
        try:
            db = next(get_db())
            create_camera(
                db=db,
                camera_id=camera_id,
                camera_name=name or None,
                location=location or None,
                url=url or None,
                event_type=event_type,
                is_enabled=enabled
            )
            dialog.destroy()
            self.load_cameras()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save camera: {str(e)}")

    def show_context_menu(self, event):
        """Show context menu for selected camera"""
        item = self.tree.identify_row(event.y)
        if item:
            self.tree.selection_set(item)
            self.context_menu.tk_popup(event.x_root, event.y_root)

    def edit_selected(self):
        """Edit selected camera"""
        selected = self.tree.selection()
        if not selected:
            return
            
        camera_id = self.tree.item(selected[0])['values'][0]
        db = next(get_db())
        camera = get_camera(db, camera_id)
        
        if not camera:
            messagebox.showerror("Error", "Camera not found")
            return
            
        # Create edit dialog
        dialog = tk.Toplevel(self)
        dialog.title(f"Edit Camera {camera_id}")
        dialog.resizable(False, False)
        
        # Form fields with existing values
        ttk.Label(dialog, text="Camera ID:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        ttk.Label(dialog, text=camera_id).grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        ttk.Label(dialog, text="Name:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        name = ttk.Entry(dialog)
        name.insert(0, camera.camera_name or "")
        name.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        
        ttk.Label(dialog, text="Location:").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        location = ttk.Entry(dialog)
        location.insert(0, camera.location or "")
        location.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        
        ttk.Label(dialog, text="URL:").grid(row=3, column=0, padx=5, pady=5, sticky="e")
        url = ttk.Entry(dialog)
        url.insert(0, camera.url or "")
        url.grid(row=3, column=1, padx=5, pady=5, sticky="ew")
        
        ttk.Label(dialog, text="Event Type:").grid(row=4, column=0, padx=5, pady=5, sticky="e")
        event_type = ttk.Combobox(dialog, values=["login", "logout"])
        event_type.set(camera.event_type)
        event_type.grid(row=4, column=1, padx=5, pady=5, sticky="ew")
        
        enabled = tk.BooleanVar(value=camera.is_enabled)
        ttk.Checkbutton(dialog, text="Enabled", variable=enabled).grid(row=5, column=1, padx=5, pady=5, sticky="w")
        
        # Buttons
        btn_frame = ttk.Frame(dialog)
        btn_frame.grid(row=6, column=0, columnspan=2, pady=10)
        
        ttk.Button(
            btn_frame,
            text="Save",
            command=lambda: self.update_camera(
                dialog, camera_id, 
                name.get(),
                location.get(),
                url.get(),
                event_type.get(),
                enabled.get()
            )
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            btn_frame,
            text="Cancel",
            command=dialog.destroy
        ).pack(side=tk.LEFT, padx=5)

    def update_camera(self, dialog, camera_id, name, location, url, event_type, enabled):
        """Update camera in database"""
        try:
            db = next(get_db())
            if update_camera(
                db, 
                camera_id,
                camera_name=name or None,
                location=location or None,
                url=url or None,
                event_type=event_type,
                is_enabled=enabled
            ):
                dialog.destroy()
                self.load_cameras()
            else:
                messagebox.showerror("Error", "Failed to update camera")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update camera: {str(e)}")

    def delete_selected(self):
        """Delete selected camera"""
        selected = self.tree.selection()
        if not selected:
            return
            
        camera_id = self.tree.item(selected[0])['values'][0]
        
        if messagebox.askyesno(
            "Confirm Delete",
            f"Are you sure you want to delete camera {camera_id}?"
        ):
            try:
                db = next(get_db())
                if delete_camera(db, camera_id):
                    self.load_cameras()
                else:
                    messagebox.showerror("Error", "Failed to delete camera")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete camera: {str(e)}")