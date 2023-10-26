import customtkinter as ctk


class Segment(ctk.CTkFrame):
    def __init__(self, parent, label_text, current_text):
        super().__init__(master=parent, fg_color='transparent')

        # Grid layout
        self.rowconfigure(0, weight=1)
        self.columnconfigure((0,4), weight=2, uniform='a')
        self.columnconfigure((1,2,3), weight=1, uniform='a')
        ctk.CTkLabel(self, text=label_text).grid(row=0, column=1, sticky='e')
        ctk.CTkEntry(self, width=50).grid(row=0, column=2, padx=20, sticky='we')
        ctk.CTkLabel(self, text=current_text).grid(row=0, column=3, sticky='w')
        self.pack(expand=True, fill='x')
