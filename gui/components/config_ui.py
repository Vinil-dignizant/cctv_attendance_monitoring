import tkinter as tk
from tkinter import ttk, messagebox
from app.config.config_manager import ConfigManager

class ConfigView(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.config = ConfigManager()
        self.setup_ui()

    def setup_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        
        # Form fields
        fields = [
            ("DB_HOST", "Database Host"),
            ("DB_PORT", "Database Port"),
            ("DB_NAME", "Database Name"),
            ("DB_USER", "Database User"),
            ("DB_PASSWORD", "Database Password", True),  # Password field
            ("PGTZ", "PostgreSQL Timezone"),
            ("TZ", "Application Timezone")
        ]
        
        self.vars = {}
        
        for i, field in enumerate(fields):
            key, label, *options = field
            is_password = options and options[0]
            
            ttk.Label(self, text=label).grid(row=i, column=0, padx=5, pady=5, sticky="e")
            
            self.vars[key] = tk.StringVar(value=self.config.get_config().get(key, ""))
            
            if is_password:
                entry = ttk.Entry(self, textvariable=self.vars[key], show="*")
            else:
                entry = ttk.Entry(self, textvariable=self.vars[key])
                
            entry.grid(row=i, column=1, padx=5, pady=5, sticky="ew")
        
        # Buttons
        btn_frame = ttk.Frame(self)
        btn_frame.grid(row=len(fields)+1, column=0, columnspan=2, pady=10)
        
        ttk.Button(
            btn_frame,
            text="Save",
            command=self.save_config
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            btn_frame,
            text="Test Connection",
            command=self.test_connection
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            btn_frame,
            text="Reset to Defaults",
            command=self.reset_defaults
        ).pack(side=tk.RIGHT, padx=5)

    def save_config(self):
        """Save configuration to file"""
        new_config = {key: var.get() for key, var in self.vars.items()}
        self.config.update_config(new_config)
        messagebox.showinfo("Success", "Configuration saved successfully")

    def test_connection(self):
        """Test database connection with current settings"""
        from sqlalchemy import create_engine, text
        from sqlalchemy.exc import OperationalError, SQLAlchemyError
        
        try:
            # Get current configuration values
            db_url = (
                f"postgresql://{self.vars['DB_USER'].get()}:{self.vars['DB_PASSWORD'].get()}@"
                f"{self.vars['DB_HOST'].get()}:{self.vars['DB_PORT'].get()}/{self.vars['DB_NAME'].get()}"
            )
            
            # Create engine with short timeout
            engine = create_engine(db_url, connect_args={
                'connect_timeout': 5,
                'options': f"-c timezone={self.vars['PGTZ'].get()}"
            })
            
            # Test connection with text() wrapper
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                
            messagebox.showinfo("Success", "Database connection successful")
        except OperationalError as e:
            messagebox.showerror("Error", f"Connection failed: {str(e)}")
        except SQLAlchemyError as e:
            messagebox.showerror("Error", f"Database error: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"Unexpected error: {str(e)}")

    def reset_defaults(self):
        """Reset configuration to defaults"""
        if messagebox.askyesno("Confirm", "Reset to default configuration?"):
            self.config.update_config(self.config.DEFAULT_CONFIG)
            for key, var in self.vars.items():
                var.set(self.config.DEFAULT_CONFIG.get(key, ""))
            messagebox.showinfo("Success", "Configuration reset to defaults")