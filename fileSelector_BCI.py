import sys
from PyQt6.QtWidgets import QLabel, QApplication, QCheckBox, QWidget, QGroupBox, QInputDialog, QHBoxLayout, QVBoxLayout, QPushButton, QListWidget, QFileDialog
#import scripts as functions
from processBCI_Bergamo_Utilities import *
from data_dict_create_module import main as mainImport
from PyQt6.QtGui import QPixmap


class FolderSelectorApp(QWidget):
    def __init__(self, title='Select Data Folder'):
        super().__init__()
        self.selected_folders = []
        self.recentSession = None
        self.oldSession = None
        self.initUI()
        self.set_app_title(title) 
        self.bases = 'test test2 test3'
    def initUI(self):
        self.setWindowTitle('Folder Selector')
        self.setGeometry(100, 100, 800, 600)
        topButtons = QVBoxLayout()
        topPlots = QHBoxLayout()
        midPlots = QHBoxLayout()
        self.SessionSelectBox = QGroupBox('Select Sessions')

        self.oldDataCheckBox = QCheckBox('Using Old Suite2p ROIs')
        self.oldDataCheckBox.stateChanged.connect(self.OldDataSelect)
        topButtons.addWidget(self.oldDataCheckBox)

        
        self.recentSelectButton = QPushButton('Select Most Recent Session', self)
        self.recentSelectButton.clicked.connect(self.selectRecentFolder)
        topButtons.addWidget(self.recentSelectButton)

        self.oldSessionButton = QPushButton('Select Session With Old Suite2p ROIs', self)
        self.oldSessionButton.setEnabled(False)
        self.oldSessionButton.clicked.connect(self.selectOlderFolder)
        topButtons.addWidget(self.oldSessionButton)

        
        self.runPipelineButton = QPushButton('Run Pipeline With Selected Files')
        self.runPipelineButton.clicked.connect(self.runPipeline)
        topButtons.addWidget(self.runPipelineButton)

        self.selectedFolders = QListWidget(self)
        self.selectedFolders.setFixedHeight(100)
        topButtons.addWidget(self.selectedFolders)
        self.setLayout(topButtons)

        ###### top
        self.sessionSummaryLabel = QLabel('Waiting for session summary')
        topPlots.addWidget(self.sessionSummaryLabel)
        
        self.cnsLabel_rois = QLabel('Waiting for CNS ROI Images')
        topPlots.addWidget(self.cnsLabel_rois)

        topButtons.addLayout(topPlots)
        ###### top

        ###### mid
        self.cnsLabel_trialAvg = QLabel('Waiting for CNS Trial Traces')
        midPlots.addWidget(self.cnsLabel_trialAvg)

        self.cnsLabel_entireSession = QLabel('Waiting for CNS Raw Traces')
        midPlots.addWidget(self.cnsLabel_entireSession)

        topButtons.addLayout(midPlots)
        ###### mid

        self.cnsListLabel = QLabel('Waiting for CNS List')
        topButtons.addWidget(self.cnsListLabel)

        self.exit_button = QPushButton('Exit', self)
        self.exit_button.clicked.connect(self.exit_app)
        topButtons.addWidget(self.exit_button)

        self.showMaximized()
        

    def runPipeline(self):

        #get bases
        self.bases = getBases(self.recentSession)
        basesTemp = ','.join(self.bases)
        
        #ask for base inds
        basesInd, ok = QInputDialog.getText(self, 'Base Selection (as tuple, BCI session first!)', basesTemp)
        basesInd = np.fromstring(basesInd[1:-1], sep=',') #turns tuple into array
        folder = self.recentSession+ '/'
        
        #Handle old vs new roi choice
        if self.oldSession is not None:
            old_folder = self.oldSession+ '/'
            print('Inds Selected', basesInd)
            print('Loading Old ROIs')
            # loadSuite2pROIS(basesInd, self.recentSession, old_folder)
        else:
            old_folder = None
            print('Inds Selected', basesInd)
            print('New FOV')
            # loadSuite2pROIS(basesInd, self.recentSession, old_folder)
        
        #Generate and Save Plots
        if ok:
            print('Suite2p ROIs saved!', basesInd)

            #create saved-data location
            data = mainImport(folder) 
            generateSessionSummary(data, folder)
            
            scalingFactor = 0.3
            figure1 = QPixmap(folder + 'SessionSummary.png')
            height = figure1.height()
            width = figure1.width()
            figure1 = figure1.scaled(width*(scalingFactor+0.1),height*scalingFactor)
            self.sessionSummaryLabel.setPixmap(figure1)

            cnsList = findConditionedNeurons(data, folder)
            cellROIs = QPixmap(folder + 'cns_visual.png')
            height = cellROIs.height()
            width = cellROIs.width()
            cellROIs = cellROIs.scaled(width*(scalingFactor+0.1),height*scalingFactor)


            rawTraces = QPixmap(folder + 'cns_EntireSession.png')
            height = rawTraces.height()
            width = rawTraces.width()
            rawTraces = rawTraces.scaled(width*(scalingFactor+0.1),height*scalingFactor)


            avgTrial =  QPixmap(folder + 'cns_AvgTrial.png')
            height = avgTrial.height()
            width = avgTrial.width()
            avgTrial = avgTrial.scaled(width*(scalingFactor+0.1),height*scalingFactor)


            self.cnsLabel_rois.setPixmap(cellROIs)
            self.cnsLabel_trialAvg.setPixmap(avgTrial)
            self.cnsLabel_entireSession.setPixmap(rawTraces)
            self.cnsListLabel.setText(str(cnsList))

    def selectRecentFolder(self):
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.FileMode.Directory)  
        file_dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)  
        folder_path = file_dialog.getExistingDirectory(self, "Select Recent Session Date")
        if folder_path:
            self.selectedFolders.clear()
            self.recentSession = folder_path
            self.selectedFolders.addItem(folder_path)

    def selectOlderFolder(self):
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.FileMode.Directory)  
        file_dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)  
        folder_path = file_dialog.getExistingDirectory(self, "Select Old Session Date")
        if folder_path:
            self.oldSession = folder_path
            self.selectedFolders.addItem(folder_path)
    
    def OldDataSelect(self, checkedOff):
        self.oldSessionButton.setEnabled(checkedOff == 2)

    def exit_app(self):
        self.close()

    def set_app_title(self, title):
        self.setWindowTitle(title)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = FolderSelectorApp('BCI Summary')
    ex.show()
    sys.exit(app.exec())
