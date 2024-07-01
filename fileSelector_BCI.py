import sys
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QListWidget, QFileDialog

class FolderSelectorApp(QWidget):
    def __init__(self, title='Select Data Folder'):
        super().__init__()
        
        self.selected_folders = []  
        self.initUI()
        self.set_app_title(title) 
    def initUI(self):
        self.setWindowTitle('Folder Selector')
        self.setGeometry(100, 100, 800, 600)
        layout = QVBoxLayout()
        self.select_button = QPushButton('Select Folders', self)
        self.select_button.clicked.connect(self.select_folders)
        layout.addWidget(self.select_button)
        self.exit_button = QPushButton('Exit', self)
        self.exit_button.clicked.connect(self.exit_app)
        layout.addWidget(self.exit_button)
        self.folder_list = QListWidget(self)
        layout.addWidget(self.folder_list)
        self.setLayout(layout)
    
    def select_folders(self):
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.FileMode.Directory)  
        file_dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)  
        folder_path = file_dialog.getExistingDirectory(self, "Select folders")
        if folder_path:
            self.folder_list.clear()
            self.selected_folders.append(folder_path)  
            self.folder_list.addItem(folder_path)
    
    def exit_app(self):
        self.close()

    def set_app_title(self, title):
        self.setWindowTitle(title)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = FolderSelectorApp()
    ex.show()
    sys.exit(app.exec())
